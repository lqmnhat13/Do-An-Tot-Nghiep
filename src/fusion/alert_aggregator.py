from typing import Dict, Iterable, List, Optional, Set
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
    - Chỉ tổng hợp cảnh báo HIGH; MEDIUM/LOW không kích hoạt âm thanh hoặc cooldown.
    - Câu nói chỉ gồm tên vật và hướng ("ghế phía trước"): mọi câu được đọc đều là HIGH nên
      mức gần là thừa, và câu ngắn để người dùng nghe xong trước khi đi tới vật.
    - Không đọc khoảng cách mét.
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
        revoke_grace_sec: float = 0.4,
    ):
        self.risk_fsm = risk_fsm
        self.alert_lifetime_sec = alert_lifetime_sec
        self.global_alert_interval_sec = global_alert_interval_sec
        self._last_global_alert_time: float = 0.0
        self._last_alert_risk_level: Optional[RiskLevel] = None
        # Câu cảnh báo chỉ bị dừng khi dữ liệu VALID liên tục trong revoke_grace_sec không còn
        # thấy vật HIGH cùng tên (chống cắt câu vì YOLO bỏ sót vài frame).
        self.revoke_grace_sec = revoke_grace_sec
        self._last_high_seen: Dict[str, float] = {}
        self._last_not_valid_time = float("-inf")

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

    @staticmethod
    def format_alert_text(class_name: str, direction: Direction) -> str:
        return f"{class_name} {DIRECTION_TEXT_VI.get(direction, 'phía trước')}"

    @classmethod
    def alert_texts(cls, class_names: Iterable[str]) -> List[str]:
        """Mọi câu cảnh báo có thể đọc, để TTSEngine dựng sẵn clip."""
        return [cls.format_alert_text(name, d) for name in dict.fromkeys(class_names) for d in Direction]

    def _is_alertable(self, a: RiskAssessment, now: float) -> bool:
        """HIGH trên dữ liệu VALID, chưa hết hạn và nguồn chưa quá tuổi (chưa xét cooldown)."""
        return (a.data_quality == DataQuality.VALID
                and a.risk_level == RiskLevel.HIGH
                and not a.is_expired(now)
                and (now - a.source_timestamp) <= self.max_source_age_sec)

    def supported_alert_keys(self, assessments: List[RiskAssessment], data_quality: DataQuality,
                             current_mono: Optional[float] = None) -> Optional[Set[str]]:
        """Tên vật còn là HIGH trong revoke_grace_sec gần nhất.

        None nghĩa là chưa đủ bằng chứng để dừng câu nào: frame này hoặc một frame trong
        khoảng grace không VALID. Dữ liệu cũ/suy giảm không bao giờ được coi là "đã hết nguy hiểm".
        """
        now = current_mono if current_mono is not None else time.monotonic()
        if data_quality != DataQuality.VALID:
            self._last_not_valid_time = now
            return None
        for a in assessments:
            if self._is_alertable(a, now):
                self._last_high_seen[a.class_name] = now
        cutoff = now - self.revoke_grace_sec
        self._last_high_seen = {k: t for k, t in self._last_high_seen.items() if t >= cutoff}
        if self._last_not_valid_time >= cutoff:
            return None
        return set(self._last_high_seen)

    def release_global_interval(self) -> None:
        """Câu cảnh báo trước bị huỷ/không phát: không giữ khoảng nghỉ toàn cục cho nó."""
        self._last_global_alert_time = 0.0

    def aggregate(self, assessments: List[RiskAssessment], current_mono: Optional[float] = None) -> Optional[AudioTask]:
        now = current_mono if current_mono is not None else time.monotonic()

        # 1. Lọc các đối tượng HIGH trên dữ liệu VALID còn mới (_is_alertable) và được phép
        # phát theo quy tắc cooldown của FSM.
        active_candidates: List[RiskAssessment] = []
        for a in assessments:
            if not self._is_alertable(a, now):
                continue

            t_id = a.track_id if a.track_id is not None else -1
            if self.risk_fsm.can_trigger_alert(t_id, a.risk_level, now):
                active_candidates.append(a)

        if not active_candidates:
            return None

        # 2. Xếp hạng độ ưu tiên khẩn cấp
        # Thứ tự: HIGH ở CENTER > HIGH ở LEFT/RIGHT, sau đó theo độ gần.
        def sort_key(a: RiskAssessment):
            dir_score = 2 if a.direction == Direction.CENTER else 1
            return (dir_score, a.relative_proximity)

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

        # 4. Câu ngắn: tên vật trước, rồi hướng; không đọc khoảng cách mét
        alert_text = self.format_alert_text(top_risk.class_name, top_risk.direction)

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

        priority = AudioPriority.HIGH_RISK

        # 6. Hạn của AudioTask là thời điểm sớm nhất giữa hạn assessment, hạn nguồn và task lifetime
        task_lifetime_limit = now + self.alert_lifetime_sec
        source_age_limit = top_risk.source_timestamp + self.max_source_age_sec
        assessment_limit = top_risk.expires_at

        task_expires_at = min(task_lifetime_limit, source_age_limit, assessment_limit)

        return AudioTask(
            priority=priority,
            text=alert_text,
            sound_file="assets/audio/alert_high.wav",
            created_at=now,
            expires_at=task_expires_at,
            interruptible=True,
            task_id=f"alert_track_{t_id}_{int(now*1000)}",
            alert_key=top_risk.class_name
        )

    @property
    def last_reason(self) -> str:
        return self._last_selected_reason
