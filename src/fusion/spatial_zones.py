from typing import Tuple
from src.contracts.detection import BoundingBox
from src.contracts.risk import Direction

class SpatialZones:
    """
    Phân vùng không gian ngang của khung hình:
    - Vùng Trái (0% - left_ratio, mặc định 35%)
    - Vùng Giữa (left_ratio - 1.0 - right_ratio, mặc định 30%)
    - Vùng Phải (1.0 - right_ratio - 100%, mặc định 35%)
    Tuân thủ mục 4.2 của Kế hoạch.
    """

    def __init__(self, left_ratio: float = 0.35, center_ratio: float = 0.30, right_ratio: float = 0.35):
        total = left_ratio + center_ratio + right_ratio
        self.left_thresh = left_ratio / total
        self.right_thresh = (left_ratio + center_ratio) / total

    def determine_direction(self, bbox: BoundingBox, frame_width: float) -> Direction:
        """
        Xác định hướng của đối tượng dựa trên tâm và mức độ chồng lấn với vùng giữa.
        """
        if frame_width <= 0:
            return Direction.CENTER

        cx_norm = bbox.center_x / frame_width

        # Kiểm tra xem bbox có chiếm đáng kể phần đường đi ở giữa không
        left_boundary = self.left_thresh * frame_width
        right_boundary = self.right_thresh * frame_width

        # Nếu vật thể rất rộng và bao trùm cả vùng giữa
        overlap_left = max(bbox.xmin, left_boundary)
        overlap_right = min(bbox.xmax, right_boundary)
        if overlap_right > overlap_left:
            center_overlap_width = overlap_right - overlap_left
            if center_overlap_width / (right_boundary - left_boundary) > 0.4:
                return Direction.CENTER

        if cx_norm < self.left_thresh:
            return Direction.LEFT
        elif cx_norm > self.right_thresh:
            return Direction.RIGHT
        else:
            return Direction.CENTER
