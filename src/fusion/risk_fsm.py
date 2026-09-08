from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import time

from src.contracts.detection import Detection, BoundingBox
from src.contracts.depth_map import DepthMap
from src.contracts.risk import DataQuality, RiskLevel, Direction, RiskAssessment
from src.detection.class_filter import ClassFilter
from src.depth.roi_extractor import ROIExtractor
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import SynchronizedPair

@dataclass
class TrackState:
    track_id: int
    current_risk: RiskLevel = RiskLevel.UNDETERMINED
    consecutive_high_count: int = 0
    consecutive_safe_count: int = 0
    last_alert_time: float = 0.0
    last_depth_frame_id: Optional[int] = None
    last_assessment: Optional[RiskAssessment] = None

class RiskFSM:
    """
    Finite State Machine (FSM) đánh giá nguy cơ va chạm cho từng đối tượng được theo vết.
    Tuân thủ mục 4.3 của Kế hoạch:
    - Hysteresis cho tăng và hạ mức nguy cơ.
    - Không đếm lặp cùng một kết quả depth.
    - Đường kích hoạt nhanh khi nguy cơ cao trực diện (instant_high_risk_in_center).
    - Cooldown chống phát lặp, nhưng cho phép vượt cooldown khi mức nguy cơ tăng cấp.
    """

    def __init__(
        self,
        class_filter: ClassFilter,
        spatial_zones: SpatialZones,
        roi_extractor: ROIExtractor,
        confirmations_to_escalate: int = 1,
        confirmations_to_deescalate: int = 3,
        instant_high_risk_in_center: bool = True,
        cooldown_sec: float = 2.5,
        allow_escalation_override: bool = True
    ):
        self.class_filter = class_filter
        self.spatial_zones = spatial_zones
        self.roi_extractor = roi_extractor
        self.confirmations_to_escalate = confirmations_to_escalate
        self.confirmations_to_deescalate = confirmations_to_deescalate
        self.instant_high_risk_in_center = instant_high_risk_in_center
        self.cooldown_sec = cooldown_sec
        self.allow_escalation_override = allow_escalation_override

        self._tracks: Dict[int, TrackState] = {}

    def update(self, sync_pair: SynchronizedPair, current_mono: Optional[float] = None) -> List[RiskAssessment]:
        """
        Đánh giá nguy cơ cho toàn bộ đối tượng trong SynchronizedPair.
        Trả về danh sách RiskAssessment của các đối tượng trong khung hình.
        """
        now = current_mono if current_mono is not None else time.monotonic()
        det_result = sync_pair.detection_result
        depth_map = sync_pair.depth_map
        data_quality = sync_pair.data_quality
        orig_w = det_result.preprocess_transform.get("original_size", (640, 480))[0]

        assessments: List[RiskAssessment] = []
        current_frame_track_ids = set()

        for det in det_result.detections:
            t_id = det.track_id or -1
            current_frame_track_ids.add(t_id)

            if t_id not in self._tracks:
                self._tracks[t_id] = TrackState(track_id=t_id)
            state = self._tracks[t_id]

            # 1. Xác định hướng không gian
            direction = self.spatial_zones.determine_direction(det.bbox, orig_w)

            # 2. Đánh giá độ gần từ ROI
            proximity = 0.0
            prox_desc = "chưa rõ"
            reason = "Chưa có dữ liệu độ sâu"

            is_new_depth_sample = False
            if depth_map is not None:
                if state.last_depth_frame_id != depth_map.frame_id:
                    is_new_depth_sample = True
                    state.last_depth_frame_id = depth_map.frame_id

                proximity, prox_desc, reason = self.roi_extractor.extract_proximity(det.bbox, depth_map)

            # 3. Đánh giá mức nguy cơ tức thời (instant candidate risk)
            candidate_risk = self._evaluate_candidate_risk(
                det=det,
                direction=direction,
                proximity=proximity,
                data_quality=data_quality
            )

            # 4. Áp dụng State Machine & Hysteresis
            final_risk = self._apply_hysteresis(state, candidate_risk, direction, proximity, is_new_depth_sample)
            state.current_risk = final_risk

            # 5. Đóng gói kết quả RiskAssessment
            vi_name = self.class_filter.get_vietnamese_name(det.class_name)
            assessment = RiskAssessment(
                track_id=det.track_id,
                class_name=vi_name,
                direction=direction,
                risk_level=final_risk,
                data_quality=data_quality,
                relative_proximity=proximity,
                proximity_desc=prox_desc,
                reason=reason,
                source_timestamp=sync_pair.effective_timestamp,
                expires_at=now + 1.0 # Hết hạn sau 1.0s nếu không được refresh
            )
            state.last_assessment = assessment
            assessments.append(assessment)

        # Dọn dẹp tracks cũ không còn xuất hiện
        if len(self._tracks) > 50:
            active_ids = {t for t in self._tracks if t in current_frame_track_ids}
            if len(active_ids) < len(self._tracks):
                self._tracks = {t: s for t, s in self._tracks.items() if t in current_frame_track_ids}

        return assessments

    def _evaluate_candidate_risk(
        self,
        det: Detection,
        direction: Direction,
        proximity: float,
        data_quality: DataQuality
    ) -> RiskLevel:
        # Nếu hoàn toàn không có dữ liệu độ sâu -> UNDETERMINED
        if data_quality == DataQuality.UNAVAILABLE or proximity <= 0.0:
            return RiskLevel.UNDETERMINED

        # Nếu không nằm trong nhóm alert_classes -> NO_ALERT
        if not self.class_filter.is_alert_candidate(det.class_name):
            return RiskLevel.NO_ALERT

        # Đánh giá theo độ gần và hướng
        if proximity >= 0.55:
            if direction == Direction.CENTER:
                return RiskLevel.HIGH
            else:
                return RiskLevel.HIGH if proximity >= 0.75 else RiskLevel.MEDIUM
        elif proximity >= 0.35:
            if direction == Direction.CENTER:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        elif proximity >= 0.20:
            return RiskLevel.LOW
        else:
            return RiskLevel.NO_ALERT

    def _apply_hysteresis(
        self,
        state: TrackState,
        candidate: RiskLevel,
        direction: Direction,
        proximity: float,
        is_new_depth_sample: bool
    ) -> RiskLevel:
        # Đường kích hoạt tức thì nếu vật cản ở ngay chính diện phía trước
        if (
            self.instant_high_risk_in_center
            and candidate == RiskLevel.HIGH
            and direction == Direction.CENTER
        ):
            state.consecutive_high_count = self.confirmations_to_escalate
            state.consecutive_safe_count = 0
            return RiskLevel.HIGH

        if candidate == RiskLevel.HIGH:
            if is_new_depth_sample or state.consecutive_high_count == 0:
                state.consecutive_high_count += 1
            state.consecutive_safe_count = 0
            if state.consecutive_high_count >= self.confirmations_to_escalate:
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM if state.current_risk != RiskLevel.HIGH else RiskLevel.HIGH

        elif candidate == RiskLevel.MEDIUM:
            state.consecutive_high_count = 0
            state.consecutive_safe_count = 0
            return RiskLevel.MEDIUM

        elif candidate in (RiskLevel.NO_ALERT, RiskLevel.LOW):
            state.consecutive_high_count = 0
            if is_new_depth_sample:
                state.consecutive_safe_count += 1

            if state.current_risk in (RiskLevel.HIGH, RiskLevel.MEDIUM):
                if state.consecutive_safe_count >= self.confirmations_to_deescalate:
                    return candidate
                else:
                    return state.current_risk
            return candidate

        return candidate

    def can_trigger_alert(self, track_id: int, target_risk: RiskLevel, current_mono: float) -> bool:
        """
        Kiểm tra xem track_id có được phép phát cảnh báo âm thanh không.
        Xử lý cooldown và ghi đè (override) khi nguy cơ tăng cấp.
        """
        if track_id not in self._tracks:
            return True

        state = self._tracks[track_id]
        time_since_alert = current_mono - state.last_alert_time

        # Nếu đang trong cooldown
        if time_since_alert < self.cooldown_sec:
            # Cho phép ghi đè nếu nguy cơ tăng lên HIGH mà trước đó chưa cảnh báo HIGH
            if self.allow_escalation_override and target_risk == RiskLevel.HIGH and state.current_risk != RiskLevel.HIGH:
                return True
            return False

        return True

    def mark_alert_triggered(self, track_id: int, current_mono: float) -> None:
        """Đánh dấu thời điểm đã phát cảnh báo cho track_id."""
        if track_id in self._tracks:
            self._tracks[track_id].last_alert_time = current_mono
