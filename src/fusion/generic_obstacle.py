"""Phát hiện vật cản gần trong hành lang chỉ từ độ sâu metric, không cần lớp COCO.

YOLO chỉ biết một số lớp; tường, mặt bàn, cạnh tủ, cột... không được cảnh báo.
Trên TUM fr1_desk, 77% frame có vật cản < 1 m trong hành lang nhưng không có HIGH.
Lớp này chỉ chạy với depth metric (cần mét tuyệt đối) và bỏ qua dải sàn sát đáy ảnh.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.detection import BoundingBox
from src.fusion.walking_corridor import WalkingCorridor


def build_generic_obstacle_detector(fusion_cfg: dict, corridor: WalkingCorridor
                                    ) -> Optional["GenericObstacleDetector"]:
    cfg = fusion_cfg.get("generic_obstacle", {})
    if not cfg.get("enabled", False):
        return None
    return GenericObstacleDetector(
        corridor=corridor,
        near_distance_m=float(cfg.get("near_distance_m", 1.0)),
        min_area_ratio=float(cfg.get("min_area_ratio", 0.15)),
        floor_ignore_ratio=float(cfg.get("floor_ignore_ratio", 0.25)),
    )


@dataclass(frozen=True)
class GenericObstacle:
    bbox: BoundingBox
    distance_m: float
    area_ratio: float


class GenericObstacleDetector:
    def __init__(
        self,
        corridor: WalkingCorridor,
        near_distance_m: float = 1.0,
        min_area_ratio: float = 0.15,
        floor_ignore_ratio: float = 0.25,
        min_pixels: int = 200,
    ):
        if near_distance_m <= 0 or not 0 < min_area_ratio <= 1 or not 0 <= floor_ignore_ratio < 1:
            raise ValueError("Tham số vật cản chung không hợp lệ")
        self.corridor = corridor
        self.near_distance_m = near_distance_m
        self.min_area_ratio = min_area_ratio
        self.floor_ignore_ratio = floor_ignore_ratio
        self.min_pixels = min_pixels
        self._mask_cache: Tuple[Tuple[int, int], Optional[np.ndarray]] = ((0, 0), None)

    def _corridor_mask(self, width: int, height: int) -> np.ndarray:
        if self._mask_cache[0] != (width, height):
            mask = np.zeros((height, width), np.uint8)
            polygon = self.corridor.polygon(width, height)
            if polygon:
                cv2.fillPoly(mask, [np.round(np.array(polygon)).astype(np.int32)], 1)
            mask[int(round(height * (1.0 - self.floor_ignore_ratio))):, :] = 0
            self._mask_cache = ((width, height), mask.astype(bool))
        return self._mask_cache[1]

    def find(self, depth_map: Optional[DepthMap],
             detected_boxes: Iterable[BoundingBox]) -> Optional[GenericObstacle]:
        if depth_map is None or depth_map.representation != DepthRepresentation.METRIC_DEPTH:
            return None
        height, width = depth_map.shape
        region = self._corridor_mask(width, height).copy()
        # Vùng đã có vật được nhận diện do RiskFSM đánh giá theo lớp; không báo trùng.
        for box in detected_boxes:
            b = box.clamp(width, height)
            region[int(b.ymin):int(np.ceil(b.ymax)), int(b.xmin):int(np.ceil(b.xmax))] = False
        region &= depth_map.valid_mask
        total = int(np.count_nonzero(region))
        if total < self.min_pixels:
            return None
        near = region & (depth_map.values < self.near_distance_m)
        count = int(np.count_nonzero(near))
        area_ratio = count / total
        if count < self.min_pixels or area_ratio < self.min_area_ratio:
            return None
        ys, xs = np.nonzero(near)
        bbox = BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        distance = float(np.percentile(depth_map.values[near], 20))
        return GenericObstacle(bbox=bbox, distance_m=distance, area_ratio=area_ratio)
