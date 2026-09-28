from typing import List, Optional
import time

from src.contracts.risk import RiskAssessment, RiskLevel, Direction, DataQuality
from src.contracts.audio import AudioTask, AudioPriority
from src.fusion.risk_fsm import RiskFSM

DIRECTION_TEXT_VI = {
    Direction.LEFT: "bên trái",
    Direction.CENTER: "phía trước",
    Direction.RIGHT: "bên phải"
}

class AlertAggregator:
    """
    Bộ tổng hợp và lựa chọn cảnh báo nguy cơ quan trọng nhất.
    Tuân thủ mục 4.3 của Kế hoạch:
    - Tổng hợp 1 cảnh báo ngắn quan trọng nhất khi có vật cản.
    - Báo hướng và độ gần tương đối, không khẳng định khoảng cách mét.
    - Lọc bỏ dữ liệu stale/degraded trước khi xếp hạng hoặc kích hoạt cooldown.
    - Cảnh báo có thời hạn hiệu lực rõ ràng (expires_at) ràng buộc với tuổi dữ liệu nguồn.
    """

    def __init__(
        self,
        risk_fsm: RiskFSM,
        alert_lifetime_sec: float = 3.0,
        global_alert_interval_sec: float = 0.0,
        max_source_age_sec: Optional[float] = None,
        max_detection_age_ms: Optional[float] = None,
        max_depth_age_ms: Optional[float] = None,
        max_alert_age_ms: Optional[float] = None,
    ):
        self.risk_fsm = risk_fsm
        self.alert_lifetime_sec = alert_lifetime_sec
        self.global_alert_interval_sec = global_alert_interval_sec
        self._last_global_alert_time: float = 0.0
        self._last_alert_risk_level: Optional[RiskLevel] = None

        if max_source_age_sec is not None:
            self.max_source_age_sec = max_source_age_sec
        else:
            ages = [v for v in [max_detection_age_ms, max_depth_age_ms, max_alert_age_ms] if v is not None]
            if ages:
                self.max_source_age_sec = min(ages) / 1000.0
            else:
                # Mặc định theo configs/fusion_rules.yaml: min(1000, 1200, 1500) ms = 1.0s
                self.max_source_age_sec = 1.0

        self._last_selected_reason: str = ""

    def aggregate(self, assessments: List[RiskAssessment], current_mono: Optional[float] = None) -> Optional[AudioTask]:
        now = current_mono if current_mono is not None else time.monotonic()

        # 1. Lọc các đối tượng:
        # - data_quality phải là VALID (không dùng dữ liệu DEGRADED, STALE hoặc UNAVAILABLE)
        # - risk_level phải là HIGH hoặc MEDIUM
        # - Chưa hết hạn assessment (a.is_expired(now))
        # - Chưa quá hạn dữ liệu nguồn (now - a.source_timestamp <= max_source_age_sec)
        # - Được phép phát theo quy tắc cooldown của FSM
        active_candidates: List[RiskAssessment] = []
        for a in assessments:
            if a.data_quality != DataQuality.VALID:
                continue
            if a.risk_level not in (RiskLevel.HIGH, RiskLevel.MEDIUM):
                continue
            if a.is_expired(now):
                continue
            if (now - a.source_timestamp) > self.max_source_age_sec:
                continue

            t_id = a.track_id if a.track_id is not None else -1
            if self.risk_fsm.can_trigger_alert(t_id, a.risk_level, now):
                active_candidates.append(a)

        if not active_candidates:
            return None

        # 2. Xếp hạng độ ưu tiên khẩn cấp
        # Thứ tự: HIGH ở CENTER > HIGH ở LEFT/RIGHT > MEDIUM ở CENTER > MEDIUM ở LEFT/RIGHT
        def sort_key(a: RiskAssessment):
            level_score = 2 if a.risk_level == RiskLevel.HIGH else 1
            dir_score = 2 if a.direction == Direction.CENTER else 1
            return (level_score, dir_score, a.relative_proximity)

        active_candidates.sort(key=sort_key, reverse=True)
        top_risk = active_candidates[0]

        # 3. Kiểm tra khoảng nghỉ âm thanh toàn cục (Global Alert Interval)
        if self.global_alert_interval_sec > 0.0:
            time_since_global = now - self._last_global_alert_time
            if time_since_global < self.global_alert_interval_sec:
                # Cho phép vượt khoảng nghỉ nếu là HIGH khẩn cấp mà cảnh báo trước đó không phải HIGH
                is_escalation = (top_risk.risk_level == RiskLevel.HIGH and self._last_alert_risk_level != RiskLevel.HIGH)
                if not is_escalation:
                    return None

        # 4. Tạo câu nói tiếng Việt ngắn gọn, không khẳng định khoảng cách mét
        dir_text = DIRECTION_TEXT_VI.get(top_risk.direction, "phía trước")
        prox_text = top_risk.proximity_desc # ví dụ: "độ gần tương đối cao", "độ gần tương đối trung bình"

        if top_risk.risk_level == RiskLevel.HIGH:
            prefix = "Chú ý, có"
        else:
            prefix = "Có"

        if prox_text and prox_text != "chưa rõ":
            if top_risk.direction == Direction.CENTER:
                alert_text = f"{prefix} {top_risk.class_name} phía trước, {prox_text}"
            else:
                alert_text = f"{prefix} {top_risk.class_name} {dir_text}, {prox_text}"
        else:
            if top_risk.direction == Direction.CENTER:
                alert_text = f"{prefix} {top_risk.class_name} phía trước"
            else:
                alert_text = f"{prefix} {top_risk.class_name} {dir_text}"

        t_id = top_risk.track_id if top_risk.track_id is not None else -1
        self._last_selected_reason = (
            f"Chọn track {t_id} ({top_risk.class_name}) - "
            f"Mức: {top_risk.risk_level.value}, Hướng: {top_risk.direction.value}, "
            f"Độ gần tương đối: {top_risk.relative_proximity:.2f} ({top_risk.proximity_desc})"
        )

        # 5. Đánh dấu đã phát cảnh báo cho track này và toàn cục để kích hoạt cooldown
        self.risk_fsm.mark_alert_triggered(t_id, now, risk_level=top_risk.risk_level)
        self._last_global_alert_time = now
        self._last_alert_risk_level = top_risk.risk_level

        priority = AudioPriority.HIGH_RISK if top_risk.risk_level == RiskLevel.HIGH else AudioPriority.SYSTEM_STATUS

        # 6. Hạn của AudioTask là thời điểm sớm nhất giữa hạn assessment, hạn nguồn và task lifetime
        task_lifetime_limit = now + self.alert_lifetime_sec
        source_age_limit = top_risk.source_timestamp + self.max_source_age_sec
        assessment_limit = top_risk.expires_at

        task_expires_at = min(task_lifetime_limit, source_age_limit, assessment_limit)

        return AudioTask(
            priority=priority,
            text=alert_text,
            sound_file="assets/audio/alert_high.wav" if top_risk.risk_level == RiskLevel.HIGH else None,
            created_at=now,
            expires_at=task_expires_at,
            interruptible=True,
            task_id=f"alert_track_{t_id}_{int(now*1000)}"
        )

    @property
    def last_reason(self) -> str:
        return self._last_selected_reason
