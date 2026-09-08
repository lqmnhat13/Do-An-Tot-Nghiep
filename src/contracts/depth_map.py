from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np

class DepthRepresentation(str, Enum):
    RELATIVE_DEPTH = "relative_depth"
    INVERSE_DEPTH = "inverse_depth"
    METRIC_DEPTH = "metric_depth"

@dataclass(frozen=True)
class DepthMap:
    """
    Hợp đồng DepthMap chuẩn xác theo mục 4.1 của Kế hoạch.
    Bảo toàn ý nghĩa vật lý/toán học, chiều gần/xa và valid_mask.
    """
    representation: DepthRepresentation
    unit: str # "relative" hoặc "meter"
    near_is_larger: bool # True đối với relative/inverse depth (lớn hơn = gần hơn)
    values: np.ndarray # Mảng float32 2D chứa giá trị độ sâu
    valid_mask: np.ndarray # Mảng boolean 2D đánh dấu các pixel hợp lệ (không phải NaN, Inf, hay invalid)
    frame_id: int
    capture_timestamp: float # Monotonic timestamp của frame nguồn
    preprocess_transform: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    @property
    def height(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[1]
