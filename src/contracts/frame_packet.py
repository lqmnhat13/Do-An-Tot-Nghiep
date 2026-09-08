from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np
import time

@dataclass(frozen=True)
class FramePacket:
    """
    Gói dữ liệu khung hình camera bất biến (immutable) được phân phối đến các worker.
    Tuân thủ mục 3.3 của Kế hoạch: bảo toàn kích thước gốc và timestamp monotonic.
    """
    frame_id: int
    source_id: str
    image: np.ndarray # BGR frame (numpy array)
    original_size: Tuple[int, int] # (width, height)
    timestamp_mono: float # time.monotonic() tại thời điểm thu nhận
    timestamp_sensor: Optional[float] = None # Thời điểm chụp từ cảm biến/camera nếu có
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.original_size[0]

    @property
    def height(self) -> int:
        return self.original_size[1]

    def age_ms(self, current_mono: Optional[float] = None) -> float:
        """Tính tuổi của frame (milliseconds)."""
        now = current_mono if current_mono is not None else time.monotonic()
        return (now - self.timestamp_mono) * 1000.0
