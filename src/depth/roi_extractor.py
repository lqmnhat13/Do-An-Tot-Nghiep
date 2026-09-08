import numpy as np
from typing import Tuple, Optional
from src.contracts.detection import BoundingBox
from src.contracts.depth_map import DepthMap
from src.depth.depth_representation import DepthRepresentationHelper

class ROIExtractor:
    """
    Trích xuất độ sâu cho vùng quan tâm (ROI) của từng Bounding Box và ước lượng khoảng cách.
    Tuân thủ mục 4.1: Co viền (erosion), loại trừ background, kiểm tra valid_mask,
    tính percentile độ gần theo đúng chiều near_is_larger và ước tính khoảng cách theo mét.
    """

    def __init__(
        self,
        erosion_ratio: float = 0.15,
        percentile: float = 80.0, # 80th percentile cho near_is_larger=True
        min_valid_pixels: int = 15,
        min_valid_ratio: float = 0.25
    ):
        self.erosion_ratio = erosion_ratio
        self.percentile = percentile
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_ratio = min_valid_ratio
        self._cached_frame_id = -1
        self._cached_norm_depth: Optional[np.ndarray] = None

    def extract_proximity(self, bbox: BoundingBox, depth_map: DepthMap) -> Tuple[float, str, str]:
        """
        Trích xuất mức độ gần tương đối và khoảng cách ước lượng cho một BoundingBox.
        Trả về:
            relative_proximity: float [0.0, 1.0] (1.0 là gần nhất)
            proximity_desc: str (ví dụ: "rất gần (~0.8m)", "gần (~1.5m)", "vừa phải (~2.5m)", "xa")
            reason: str giải thích căn cứ tính toán
        """
        h_depth, w_depth = depth_map.shape

        # 1. Co viền bbox (erosion) để giảm nhiễu nền xung quanh vật thể
        bw = bbox.width
        bh = bbox.height
        pad_x = bw * self.erosion_ratio
        pad_y = bh * self.erosion_ratio

        x1 = int(round(max(0, bbox.xmin + pad_x)))
        y1 = int(round(max(0, bbox.ymin + pad_y)))
        x2 = int(round(min(w_depth, bbox.xmax - pad_x)))
        y2 = int(round(min(h_depth, bbox.ymax - pad_y)))

        if x2 <= x1 or y2 <= y1:
            # Bbox nhỏ sau khi co viền, dùng bbox nguyên bản
            x1 = int(round(max(0, bbox.xmin)))
            y1 = int(round(max(0, bbox.ymin)))
            x2 = int(round(min(w_depth, bbox.xmax)))
            y2 = int(round(min(h_depth, bbox.ymax)))

        if x2 <= x1 or y2 <= y1:
            return 0.0, "chưa rõ", "BBox nằm ngoài phạm vi ảnh hoặc diện tích bằng 0"

        roi_values = depth_map.values[y1:y2, x1:x2]
        roi_mask = depth_map.valid_mask[y1:y2, x1:x2]

        total_roi_pixels = roi_values.size
        valid_pixels_count = int(np.count_nonzero(roi_mask))

        # 2. Kiểm tra chất lượng dữ liệu vùng ROI
        if valid_pixels_count < self.min_valid_pixels or (valid_pixels_count / max(1, total_roi_pixels)) < self.min_valid_ratio:
            return 0.0, "chưa rõ", f"Không đủ pixel hợp lệ trong ROI ({valid_pixels_count}/{total_roi_pixels})"

        # 3. Chuẩn hóa giá trị độ sâu của toàn khung hình về [0, 1] (tối ưu cache theo frame_id)
        if self._cached_frame_id != depth_map.frame_id or self._cached_norm_depth is None:
            self._cached_frame_id = depth_map.frame_id
            self._cached_norm_depth = DepthRepresentationHelper.normalize_relative_depth(
                depth_map.values,
                depth_map.valid_mask,
                near_is_larger=depth_map.near_is_larger
            )

        roi_norm_vals = self._cached_norm_depth[y1:y2, x1:x2][roi_mask]

        # 4. Lấy phân vị (percentile) độ gần
        score = float(np.percentile(roi_norm_vals, self.percentile))
        score = max(0.0, min(1.0, score))

        # 5. Ước lượng khoảng cách mét xấp xỉ trong nhà
        est_dist = max(0.5, min(4.5, 0.4 + (1.0 - score) * 3.6))

        # 6. Phân nhóm mô tả độ gần kèm khoảng cách mét
        if score >= 0.75:
            desc = f"rất gần (~{est_dist:.1f}m)"
        elif score >= 0.50:
            desc = f"gần (~{est_dist:.1f}m)"
        elif score >= 0.30:
            desc = f"vừa phải (~{est_dist:.1f}m)"
        else:
            desc = f"ở xa (> 3m)"

        reason = f"Độ gần {score*100:.0f}% (~{est_dist:.1f}m, {valid_pixels_count}px)"
        return score, desc, reason
