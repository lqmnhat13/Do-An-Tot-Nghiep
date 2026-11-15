"""Dò điểm dừng nói dựa trên năng lượng (RMS) của từng block âm thanh."""

from typing import Optional

import numpy as np


class SilenceEndpointer:
    """Quyết định khi nào ngừng thu âm.

    Trả lý do dừng khi:
    - ``"end_of_speech"``: đã nghe thấy tiếng nói rồi im lặng liên tục ``silence_sec``;
    - ``"max_duration"``: tổng thời lượng đạt ``max_sec``;
    - ``"no_speech"``: không có tiếng nói nào trong ``no_speech_sec`` đầu tiên.
    Thời gian được đếm theo số mẫu đã feed, không theo đồng hồ.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        rms_threshold: float = 0.01,
        silence_sec: float = 1.2,
        max_sec: float = 8.0,
        no_speech_sec: float = 5.0,
    ):
        self.sample_rate = int(sample_rate)
        self.rms_threshold = float(rms_threshold)
        self._silence_limit = int(round(silence_sec * self.sample_rate))
        self._max_samples = int(round(max_sec * self.sample_rate))
        self._no_speech_samples = int(round(no_speech_sec * self.sample_rate))
        self._total = 0
        self._silence_run = 0
        self.heard_speech = False

    def feed(self, chunk: np.ndarray) -> Optional[str]:
        """Nhận một block mono; trả ``None`` để tiếp tục hoặc lý do dừng."""
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        self._total += samples.size
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms >= self.rms_threshold:
            self.heard_speech = True
            self._silence_run = 0
        else:
            self._silence_run += samples.size

        if self.heard_speech and self._silence_run >= self._silence_limit:
            return "end_of_speech"
        if self._total >= self._max_samples:
            return "max_duration"
        if not self.heard_speech and self._total >= self._no_speech_samples:
            return "no_speech"
        return None
