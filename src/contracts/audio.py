from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import time

class AudioPriority(IntEnum):
    HIGH_RISK = 1      # Cảnh báo nguy cơ va chạm khẩn cấp (ngắt mọi âm thanh khác)
    SYSTEM_STATUS = 2  # Cảnh báo lỗi hệ thống, mất camera, dữ liệu suy giảm
    ON_DEMAND = 3      # Đọc kết quả OCR, trả lời VQA theo yêu cầu
    INFO = 4           # Thông báo trạng thái thông thường

@dataclass
class AudioTask:
    """
    Nhiệm vụ phát âm thanh gửi tới Audio Coordinator.
    Có thứ tự ưu tiên, thời hạn hiệu lực và khả năng bị ngắt.
    """
    priority: AudioPriority
    text: str
    sound_file: Optional[str] = None # Đường dẫn WAV phát tức thì nếu có
    created_at: float = field(default_factory=time.monotonic)
    expires_at: float = 0.0 # 0 nghĩa là không hết hạn
    interruptible: bool = True
    task_id: str = ""

    def is_expired(self, current_mono: Optional[float] = None) -> bool:
        if self.expires_at <= 0.0:
            return False
        now = current_mono if current_mono is not None else time.monotonic()
        return now > self.expires_at

    def __lt__(self, other: "AudioTask") -> bool:
        # Trong PriorityQueue, số nhỏ hơn có độ ưu tiên cao hơn
        if self.priority != other.priority:
            return self.priority < other.priority
        # Nếu cùng độ ưu tiên, việc tạo sớm hơn được ưu tiên trước
        return self.created_at < other.created_at
