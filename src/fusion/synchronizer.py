from dataclasses import dataclass
from typing import Optional, Dict
import threading
import time

from src.contracts.detection import DetectionResult
from src.contracts.depth_map import DepthMap
from src.contracts.risk import DataQuality

@dataclass
class SynchronizedPair:
    detection_result: DetectionResult
    depth_map: Optional[DepthMap]
    data_quality: DataQuality
    pair_skew_ms: float
    effective_timestamp: float # Thời điểm của nguồn cũ hơn

class Synchronizer:
    """
    Bộ đồng bộ thời gian và kiểm soát dữ liệu cũ/lệch giữa Detection và Depth.
    Tuân thủ mục 3.3 của Kế hoạch:
    - Ưu tiên ghép cùng frame_id.
    - Kiểm soát max_pair_skew_ms và tuổi thọ dữ liệu theo cấu hình.
    - Thread-safe giữa các worker; không để depth cũ đến muộn ghi đè depth mới.
    - Khi có độ lệch vượt ngưỡng, đánh dấu DEGRADED; khi quá tuổi, đánh dấu STALE.
    """

    def __init__(
        self,
        max_pair_skew_ms: float = 600.0,
        max_detection_age_ms: float = 1000.0,
        max_depth_age_ms: float = 1200.0,
        max_alert_age_ms: float = 1500.0
    ):
        self.max_pair_skew_ms = max_pair_skew_ms
        self.max_detection_age_ms = max_detection_age_ms
        self.max_depth_age_ms = max_depth_age_ms
        self.max_alert_age_ms = max_alert_age_ms

        self._lock = threading.Lock()
        self._latest_depth: Optional[DepthMap] = None
        self._depth_history: Dict[int, DepthMap] = {} # Lưu tạm depth theo frame_id

    def register_depth(self, depth_map: DepthMap) -> None:
        """Đăng ký kết quả depth map mới nhất (thread-safe, không cho mẫu cũ ghi đè mẫu mới)."""
        with self._lock:
            if self._latest_depth is None or depth_map.capture_timestamp >= self._latest_depth.capture_timestamp:
                self._latest_depth = depth_map
            self._depth_history[depth_map.frame_id] = depth_map
            # Giữ kích thước history nhỏ gọn
            if len(self._depth_history) > 20:
                oldest_id = min(self._depth_history.keys())
                del self._depth_history[oldest_id]

    def synchronize(self, det_result: DetectionResult, current_mono: Optional[float] = None) -> SynchronizedPair:
        """
        Ghép cặp kết quả detection với depth map tương ứng.
        Đánh giá DataQuality (VALID, DEGRADED, STALE, UNAVAILABLE) theo các ngưỡng cấu hình.
        """
        now = current_mono if current_mono is not None else time.monotonic()
        det_age_ms = (now - det_result.timestamp_mono) * 1000.0

        # Lấy snapshot depth map dưới lock
        with self._lock:
            # 1. Tìm depth map có cùng frame_id (ghép đôi hoàn hảo)
            matched_depth = self._depth_history.get(det_result.frame_id)

            # 2. Nếu không có cùng frame_id, lấy depth map mới nhất đã có
            if matched_depth is None and self._latest_depth is not None:
                matched_depth = self._latest_depth

        if matched_depth is None:
            # Chưa từng có depth map nào được sinh ra
            return SynchronizedPair(
                detection_result=det_result,
                depth_map=None,
                data_quality=DataQuality.UNAVAILABLE,
                pair_skew_ms=0.0,
                effective_timestamp=det_result.timestamp_mono
            )

        skew_ms = abs(det_result.timestamp_mono - matched_depth.capture_timestamp) * 1000.0
        depth_age_ms = (now - matched_depth.capture_timestamp) * 1000.0
        effective_ts = min(det_result.timestamp_mono, matched_depth.capture_timestamp)
        overall_age_ms = (now - effective_ts) * 1000.0

        # Đánh giá chất lượng dữ liệu:
        # - STALE: Bất kỳ nguồn nào hoặc tuổi kết hợp vượt ngưỡng tuổi tối đa cho phép
        # - DEGRADED: Độ lệch thời gian giữa hai nguồn vượt max_pair_skew_ms
        # - VALID: Cả hai nguồn đều mới và lệch dưới max_pair_skew_ms
        if (
            det_age_ms > self.max_detection_age_ms
            or depth_age_ms > self.max_depth_age_ms
            or overall_age_ms > self.max_alert_age_ms
        ):
            quality = DataQuality.STALE
        elif skew_ms > self.max_pair_skew_ms:
            quality = DataQuality.DEGRADED
        else:
            quality = DataQuality.VALID

        return SynchronizedPair(
            detection_result=det_result,
            depth_map=matched_depth,
            data_quality=quality,
            pair_skew_ms=skew_ms,
            effective_timestamp=effective_ts
        )
