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
    - Đọc kèm khoảng cách ước lượng trực quan và hướng.
    - Cảnh báo có thời hạn hiệu lực rõ ràng (expires_at).
    """

    def __init__(self, risk_fsm: RiskFSM, alert_lifetime_sec: float = 3.0):
        self.risk_fsm = risk_fsm
        self.alert_lifetime_sec = alert_lifetime_sec
        self._last_selected_reason: str = ""

    def aggregate(self, assessments: List[RiskAssessment], current_mono: Optional[float] = None) -> Optional[AudioTask]:
        now = current_mono if current_mono is not None else time.monotonic()

        # 1. Lọc các đối tượng có nguy cơ cần cảnh báo (HIGH hoặc MEDIUM)
        active_candidates: List[RiskAssessment] = []
        for a in assessments:
            if a.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
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

        # 3. Tạo câu nói tiếng Việt tự nhiên kèm khoảng cách
        dir_text = DIRECTION_TEXT_VI.get(top_risk.direction, "phía trước")
        prox_text = top_risk.proximity_desc # ví dụ: "rất gần (~0.8m)", "gần (~1.3m)"

        if top_risk.direction == Direction.CENTER:
            alert_text = f"Có {top_risk.class_name} {prox_text} phía trước"
        else:
            alert_text = f"Có {top_risk.class_name} {dir_text}, {prox_text}"

        t_id = top_risk.track_id if top_risk.track_id is not None else -1
        self._last_selected_reason = (
            f"Chọn track {t_id} ({top_risk.class_name}) - "
            f"Mức: {top_risk.risk_level.value}, Hướng: {top_risk.direction.value}, "
            f"Độ gần: {top_risk.relative_proximity:.2f} ({top_risk.proximity_desc})"
        )

        # 4. Đánh dấu đã phát cảnh báo cho track này để kích hoạt cooldown
        self.risk_fsm.mark_alert_triggered(t_id, now)

        priority = AudioPriority.HIGH_RISK if top_risk.risk_level == RiskLevel.HIGH else AudioPriority.SYSTEM_STATUS

        return AudioTask(
            priority=priority,
            text=alert_text,
            sound_file="assets/audio/alert_high.wav" if top_risk.risk_level == RiskLevel.HIGH else None,
            created_at=now,
            expires_at=now + self.alert_lifetime_sec,
            interruptible=True,
            task_id=f"alert_track_{t_id}_{int(now*1000)}"
        )

    @property
    def last_reason(self) -> str:
        return self._last_selected_reason
