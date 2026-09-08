from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import time

@dataclass
class OCRRequest:
    request_id: str
    image: np.ndarray
    created_at: float = field(default_factory=time.monotonic)
    timeout_sec: float = 5.0

@dataclass
class OCRResult:
    request_id: str
    success: bool
    text: str
    detected_boxes: List[Any] = field(default_factory=list)
    quality_status: str = "OK" # "OK" | "BLURRY" | "DARK" | "TOO_BRIGHT"
    latency_sec: float = 0.0
    error_message: Optional[str] = None

@dataclass
class VQARequest:
    request_id: str
    image: np.ndarray
    question: str
    created_at: float = field(default_factory=time.monotonic)
    timeout_sec: float = 8.0

@dataclass
class VQAResult:
    request_id: str
    success: bool
    answer: str
    latency_sec: float = 0.0
    error_message: Optional[str] = None
