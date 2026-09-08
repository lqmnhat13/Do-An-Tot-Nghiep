import time
from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np

class PerformanceMetrics:
    """
    Theo dõi các chỉ số hiệu năng hệ thống: FPS, P50/P95 Latency và tỷ lệ drop frame.
    Tuân thủ mục 9.1 của Kế hoạch: báo cáo P50/P95 trung thực thay vì chỉ lấy số đẹp.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._latencies: Dict[str, deque] = {
            "detection_ms": deque(maxlen=window_size),
            "depth_ms": deque(maxlen=window_size),
            "end_to_end_ms": deque(maxlen=window_size),
            "ocr_sec": deque(maxlen=20),
            "vqa_sec": deque(maxlen=20)
        }
        self._frame_times: deque = deque(maxlen=window_size)
        self._last_frame_ts = 0.0

    def record_frame(self) -> None:
        now = time.monotonic()
        if self._last_frame_ts > 0.0:
            dt = now - self._last_frame_ts
            if dt > 0.0:
                self._frame_times.append(dt)
        self._last_frame_ts = now

    def record_latency(self, metric_name: str, value: float) -> None:
        if metric_name not in self._latencies:
            self._latencies[metric_name] = deque(maxlen=self.window_size)
        self._latencies[metric_name].append(value)

    def get_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        avg_dt = float(np.mean(self._frame_times))
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def get_percentiles(self, metric_name: str) -> Dict[str, float]:
        q = self._latencies.get(metric_name)
        if not q or len(q) == 0:
            return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
        arr = np.array(q)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr))
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "fps": round(self.get_fps(), 1),
            "detection": self.get_percentiles("detection_ms"),
            "depth": self.get_percentiles("depth_ms"),
            "end_to_end": self.get_percentiles("end_to_end_ms")
        }
