import numpy as np
from typing import Tuple, Optional, Sequence
from src.contracts.detection import BoundingBox
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.depth.depth_representation import DepthRepresentationHelper

# (khoảng cách mét, điểm độ gần): khớp ngưỡng mặc định của RiskFSM 0.75/0.45/0.25
# với 1.5/2.5/3.5 m, để FSM/Aggregator/HUD giữ nguyên hợp đồng điểm [0, 1].
DEFAULT_METRIC_BREAKPOINTS = ((0.0, 1.0), (1.5, 0.75), (2.5, 0.45), (3.5, 0.25), (6.0, 0.0))


def metric_breakpoints(high_m: float, medium_m: float, low_m: float,
                       score_high: float, score_medium: float, score_low: float
                       ) -> Tuple[Tuple[float, float], ...]:
    """Dựng bảng quy đổi mét -> điểm sao cho khoảng cách <= high_m <=> điểm >= score_high."""
    if not 0.0 < high_m < medium_m < low_m:
        raise ValueError("Ngưỡng khoảng cách phải thoả 0 < high < medium < low")
    far_m = low_m + (low_m - medium_m) * 2.5
    return ((0.0, 1.0), (high_m, score_high), (medium_m, score_medium),
            (low_m, score_low), (far_m, 0.0))


class ROIExtractor:
    """
    Trích xuất độ sâu cho vùng quan tâm (ROI) của từng Bounding Box và ước lượng độ gần.
    Tuân thủ mục 4.1: Co viền (erosion), loại trừ background, kiểm tra valid_mask,
    tính percentile độ gần theo đúng chiều near_is_larger.
    - relative_depth: chuẩn hoá min-max theo frame (điểm là thứ hạng gần/xa trong frame).
    - metric_depth: khoảng cách (mét) của phần gần nhất của vật, quy đổi sang điểm
      bằng metric_breakpoints; không phụ thuộc phần còn lại của khung hình.
    """

    def __init__(
        self,
        erosion_ratio: float = 0.15,
        percentile: float = 80.0, # 80th percentile cho near_is_larger=True
        min_valid_pixels: int = 15,
        min_valid_ratio: float = 0.25,
        metric_breakpoints: Sequence[Tuple[float, float]] = DEFAULT_METRIC_BREAKPOINTS
    ):
        self.erosion_ratio = erosion_ratio
        self.percentile = percentile
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_ratio = min_valid_ratio
        self._metric_x = np.array([p[0] for p in metric_breakpoints], dtype=np.float64)
        self._metric_y = np.array([p[1] for p in metric_breakpoints], dtype=np.float64)
        if np.any(np.diff(self._metric_x) <= 0) or np.any(np.diff(self._metric_y) > 0):
            raise ValueError("metric_breakpoints phải tăng dần theo mét và giảm dần theo điểm")
        self._cached_frame_id = -1
        self._cached_norm_depth: Optional[np.ndarray] = None

    def score_from_distance(self, distance_m: float) -> float:
        return float(np.interp(distance_m, self._metric_x, self._metric_y))

    @staticmethod
    def describe(score: float, metric: bool = False) -> str:
        """Mô tả bằng lời (không đọc số mét); bản metric dựa trên khoảng cách đã quy đổi."""
        if metric:
            if score >= 0.75:
                return "ở rất gần"
            if score >= 0.50:
                return "ở gần"
            if score >= 0.30:
                return "cách vài bước"
            return "ở xa"
        if score >= 0.75:
            return "độ gần tương đối cao"
        if score >= 0.50:
            return "độ gần tương đối trung bình"
        if score >= 0.30:
            return "độ gần tương đối thấp"
        return "ở xa"

    def extract_proximity(self, bbox: BoundingBox, depth_map: DepthMap) -> Tuple[float, str, str]:
        """
        Trích xuất mức độ gần tương đối cho một BoundingBox.
        Trả về:
            relative_proximity: float [0.0, 1.0] (1.0 là gần nhất)
            proximity_desc: str (ví dụ: "độ gần tương đối cao", "độ gần tương đối trung bình", "độ gần tương đối thấp", "ở xa", "chưa rõ")
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

        # 3a. Độ sâu metric: dùng khoảng cách phần gần nhất của vật, không chuẩn hoá theo frame
        if depth_map.representation == DepthRepresentation.METRIC_DEPTH:
            roi_meters = roi_values[roi_mask]
            distance = float(np.percentile(roi_meters, 100.0 - self.percentile))
            if not np.isfinite(distance):
                return 0.0, "chưa rõ", "Khoảng cách ROI không hợp lệ"
            score = self.score_from_distance(distance)
            reason = f"Khoảng cách ước lượng {distance:.1f} m ({valid_pixels_count}px)"
            return score, self.describe(score, metric=True), reason

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

        # 5. Phân nhóm mô tả độ gần tương đối (không khẳng định khoảng cách mét tuyệt đối)
        reason = f"Độ gần tương đối {score*100:.0f}% ({valid_pixels_count}px)"
        return score, self.describe(score), reason
