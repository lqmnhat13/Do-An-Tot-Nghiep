from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class BoundingBox:
    """Bounding box dạng (xmin, ymin, xmax, ymax) với tọa độ pixel."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.xmin)), int(round(self.ymin)), int(round(self.xmax)), int(round(self.ymax)))

    def clamp(self, max_width: float, max_height: float) -> "BoundingBox":
        """Giới hạn tọa độ trong phạm vi ảnh (0..max_width, 0..max_height)."""
        return BoundingBox(
            xmin=max(0.0, min(self.xmin, max_width)),
            ymin=max(0.0, min(self.ymin, max_height)),
            xmax=max(0.0, min(self.xmax, max_width)),
            ymax=max(0.0, min(self.ymax, max_height))
        )

@dataclass(frozen=True)
class Detection:
    """Một đối tượng phát hiện được kèm nhãn, độ tin cậy và định danh track_id."""
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None

@dataclass(frozen=True)
class DetectionResult:
    """Toàn bộ kết quả phát hiện trong một khung hình kèm metadata đồng bộ."""
    detections: List[Detection]
    frame_id: int
    timestamp_mono: float
    latency_ms: float
    preprocess_transform: Dict[str, Any] = field(default_factory=dict)
