import cv2
import time
import numpy as np
from typing import Optional, Tuple

class VideoPlayback:
    """
    Trình phát lại video từ tệp cho mục đích kiểm thử có thể tái lập và đánh giá offline.
    Tuân thủ mục 9.3 của Kế hoạch: hỗ trợ replay thời gian thực (giữ đúng nhịp frame).
    """

    def __init__(self, video_path: str, loop: bool = True, target_fps: Optional[float] = None):
        self.video_path = video_path
        self.loop = loop
        self.target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._native_fps = 30.0
        self._total_frames = 0
        self._current_frame_idx = 0
        self._open()

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không thể mở tệp video: {self.video_path}")
        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame_idx = 0

    @property
    def fps(self) -> float:
        return self.target_fps if self.target_fps is not None else self._native_fps

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._current_frame_idx = 0
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return False, None
            else:
                return False, None

        self._current_frame_idx += 1
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
