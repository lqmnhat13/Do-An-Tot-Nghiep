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

    Ngưỡng tự thích nghi: micro MacBook trong phòng yên tĩnh có nền RMS ~0,014 (đỉnh 0,031),
    nên ngưỡng cố định 0,01 coi tiếng nền là tiếng nói. ``CAL_BLOCKS`` block đầu (sau tiếng bíp)
    dùng để đo nền (median RMS); sau đó ngưỡng = ``min(max(rms_threshold, 3 x nền), 0,06)``.
    ``rms_threshold`` vì vậy là ngưỡng TỐI THIỂU; trần 0,06 đảm bảo vẫn nhận ra tiếng nói
    (p90 ~0,136) nếu người dùng nói ngay trong lúc đo nền.
    """

    CAL_BLOCKS = 10        # ~0,3 s với block 30 ms
    FLOOR_FACTOR = 3.0
    MAX_THRESHOLD = 0.06
    SPEECH_BLOCKS = 3      # số block to liên tiếp mới tính là tiếng nói (bỏ qua tiếng click)

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
        self._cal_rms = []
        self._threshold: Optional[float] = None
        self._loud_run = 0
        self.heard_speech = False
        # Vị trí (số mẫu tính từ đầu) cuối block to gần nhất sau hiệu chuẩn; None nếu chưa có.
        self.last_speech_end_samples: Optional[int] = None

    def feed(self, chunk: np.ndarray) -> Optional[str]:
        """Nhận một block mono; trả ``None`` để tiếp tục hoặc lý do dừng."""
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        self._total += samples.size
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if self._threshold is None:
            # Đang đo nền: không tính tiếng nói, không cộng im lặng (âm thanh vẫn được ghi).
            self._cal_rms.append(rms)
            if len(self._cal_rms) >= self.CAL_BLOCKS:
                floor = float(np.median(self._cal_rms))
                self._threshold = min(max(self.rms_threshold, self.FLOOR_FACTOR * floor),
                                      self.MAX_THRESHOLD)
        elif rms >= self._threshold:
            self.last_speech_end_samples = self._total
            self._loud_run += 1
            if self._loud_run >= self.SPEECH_BLOCKS:
                self.heard_speech = True
            if self.heard_speech:
                self._silence_run = 0
        else:
            self._loud_run = 0
            self._silence_run += samples.size

        if self.heard_speech and self._silence_run >= self._silence_limit:
            return "end_of_speech"
        if self._total >= self._max_samples:
            return "max_duration"
        if not self.heard_speech and self._total >= self._no_speech_samples:
            return "no_speech"
        return None
