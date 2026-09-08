import os
import time
from typing import Optional, List, Dict, Any
import numpy as np
import torch

from src.contracts.frame_packet import FramePacket
from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.detection.class_filter import ClassFilter

class YoloDetector:
    """
    Adapter cho mô hình YOLOv8/YOLO11 sử dụng thư viện Ultralytics.
    Chạy suy luận, lọc theo tập lớp chỉ định và bảo toàn chính xác tọa độ ảnh gốc.
    """

    def __init__(
        self,
        weights_path: str = "models/weights/yolov8n.pt",
        class_filter: Optional[ClassFilter] = None,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "mps",
        input_size: int = 640
    ):
        self.weights_path = weights_path
        self.class_filter = class_filter or ClassFilter()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Xác định thiết bị tính toán
        if device == "mps" and not torch.backends.mps.is_available():
            print("[YoloDetector] MPS không khả dụng, chuyển sang CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số YOLO: {self.weights_path}")

        print(f"[YoloDetector] Đang nạp mô hình từ {self.weights_path} lên {self.device}...")
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)
        self._model.to(self.device)
        print("[YoloDetector] Nạp mô hình YOLO thành công.")

    def detect(self, packet: FramePacket) -> DetectionResult:
        """Thực hiện phát hiện đối tượng trên FramePacket."""
        t0 = time.monotonic()
        w_orig, h_orig = packet.original_size

        if self._model is None:
            return DetectionResult(
                detections=[],
                frame_id=packet.frame_id,
                timestamp_mono=packet.timestamp_mono,
                latency_ms=0.0
            )

        # Chạy suy luận qua Ultralytics
        results = self._model.predict(
            source=packet.image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if results and len(results) > 0:
            res = results[0]
            names = res.names
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    cls_id = int(cls_arr[i])
                    class_name = names.get(cls_id, str(cls_id))
                    conf = float(conf_arr[i])

                    # Lọc theo enabled_classes
                    if not self.class_filter.is_enabled(class_name):
                        continue

                    # Clamp tọa độ bảo đảm nằm gọn trong kích thước ảnh gốc
                    xmin, ymin, xmax, ymax = xyxy_arr[i]
                    raw_bbox = BoundingBox(
                        xmin=float(xmin),
                        ymin=float(ymin),
                        xmax=float(xmax),
                        ymax=float(ymax)
                    )
                    clamped_bbox = raw_bbox.clamp(w_orig, h_orig)

                    if clamped_bbox.area > 0:
                        detections.append(Detection(
                            class_name=class_name,
                            confidence=conf,
                            bbox=clamped_bbox,
                            track_id=None
                        ))

        latency = (time.monotonic() - t0) * 1000.0

        return DetectionResult(
            detections=detections,
            frame_id=packet.frame_id,
            timestamp_mono=packet.timestamp_mono,
            latency_ms=latency,
            preprocess_transform={
                "original_size": (w_orig, h_orig),
                "input_size": self.input_size
            }
        )
