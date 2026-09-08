import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

from src.contracts.detection import BoundingBox, Detection, DetectionResult

@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    last_seen_frame: int
    last_seen_timestamp: float
    hits: int = 1
    lost_frames: int = 0

def compute_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    xA = max(boxA.xmin, boxB.xmin)
    yA = max(boxA.ymin, boxB.ymin)
    xB = min(boxA.xmax, boxB.xmax)
    yB = min(boxA.ymax, boxB.ymax)

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea <= 0.0:
        return 0.0

    boxAArea = boxA.area
    boxBArea = boxB.area
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0

class ByteTrackerAdapter:
    """
    Adapter theo vết đối tượng (Object Tracker) theo cơ chế liên kết dữ liệu IoU hai tầng.
    Duy trì định danh track_id liên tục qua các frame để phục vụ FSM đánh giá rủi ro.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self._next_id = 1
        self._active_tracks: Dict[int, TrackedObject] = {}

    def update(self, det_result: DetectionResult) -> DetectionResult:
        """
        Cập nhật danh sách tracks với các detections mới.
        Gán track_id cho các detection và trả về DetectionResult đã được theo vết.
        """
        detections = det_result.detections
        frame_id = det_result.frame_id
        timestamp = det_result.timestamp_mono

        matched_tracks = set()
        matched_detections = set()
        updated_detections: List[Detection] = []

        # 1. Khớp nối các detection hiện tại với active tracks qua IoU cao
        for d_idx, det in enumerate(detections):
            best_iou = 0.0
            best_track_id = None

            for t_id, track in self._active_tracks.items():
                if t_id in matched_tracks:
                    continue
                # Ưu tiên cùng class_name
                if track.class_name != det.class_name:
                    continue

                iou = compute_iou(det.bbox, track.bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = t_id

            if best_track_id is not None:
                matched_tracks.add(best_track_id)
                matched_detections.add(d_idx)
                # Cập nhật track hiện tại
                track = self._active_tracks[best_track_id]
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.last_seen_frame = frame_id
                track.last_seen_timestamp = timestamp
                track.hits += 1
                track.lost_frames = 0

                updated_detections.append(Detection(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    track_id=best_track_id
                ))

        # 2. Tạo track mới cho các detections chưa được khớp
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_id = self._next_id
                self._next_id += 1

                self._active_tracks[new_id] = TrackedObject(
                    track_id=new_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    last_seen_frame=frame_id,
                    last_seen_timestamp=timestamp,
                    hits=1,
                    lost_frames=0
                )

                updated_detections.append(Detection(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    track_id=new_id
                ))

        # 3. Tăng lost_frames cho các tracks không xuất hiện trong frame này
        lost_track_ids = []
        for t_id, track in self._active_tracks.items():
            if t_id not in matched_tracks:
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    lost_track_ids.append(t_id)

        # Xóa các tracks đã mất quá lâu
        for t_id in lost_track_ids:
            del self._active_tracks[t_id]

        return DetectionResult(
            detections=updated_detections,
            frame_id=det_result.frame_id,
            timestamp_mono=det_result.timestamp_mono,
            latency_ms=det_result.latency_ms,
            preprocess_transform=det_result.preprocess_transform
        )

    def reset(self) -> None:
        """Xóa toàn bộ tracks."""
        self._active_tracks.clear()
        self._next_id = 1
