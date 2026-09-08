from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DataQuality(str, Enum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"

class RiskLevel(str, Enum):
    UNDETERMINED = "UNDETERMINED"
    NO_ALERT = "NO_ALERT"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class Direction(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"

@dataclass(frozen=True)
class RiskAssessment:
    """
    Kết quả đánh giá nguy cơ va chạm của một đối tượng.
    Tách biệt DataQuality và RiskLevel theo mục 4.3 của Kế hoạch.
    """
    track_id: Optional[int]
    class_name: str
    direction: Direction
    risk_level: RiskLevel
    data_quality: DataQuality
    relative_proximity: float # [0..1]
    proximity_desc: str # "rất gần", "gần", "vừa phải", "xa", "chưa rõ"
    reason: str
    source_timestamp: float
    expires_at: float

    def is_expired(self, current_mono: float) -> bool:
        return current_mono > self.expires_at
