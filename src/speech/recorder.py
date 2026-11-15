"""Thu âm micro 16 kHz mono cho câu hỏi VQA bằng giọng nói."""

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.speech.endpointing import SilenceEndpointer


@dataclass(frozen=True)
class RecordingResult:
    """Kết quả một lần thu âm.

    ``reason``: ``"end_of_speech" | "max_duration" | "no_speech" | "stopped"``.
    """

    audio: np.ndarray
    reason: str
    heard_speech: bool


class MicrophoneError(RuntimeError):
    """Không mở/đọc được micro (thiếu PortAudio, không có thiết bị, bị từ chối quyền)."""


class MicRecorder:
    """Đọc ``sounddevice.InputStream`` theo block cho tới khi endpointer báo dừng
    hoặc ``stop_event`` được set (kiểm tra giữa mỗi block ``block_sec``)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_sec: float = 0.03,
        device: Optional[int] = None,
        endpointer_kwargs: Optional[dict] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.block_size = max(1, int(round(block_sec * self.sample_rate)))
        self.device = device
        self.endpointer_kwargs = dict(endpointer_kwargs or {})

    def record(self, stop_event: threading.Event) -> RecordingResult:
        # Import trễ: CI/máy không có PortAudio vẫn import được module này.
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise MicrophoneError(f"Không tải được sounddevice: {exc}") from exc

        device_errors = (OSError, getattr(sd, "PortAudioError", OSError))
        endpointer = SilenceEndpointer(sample_rate=self.sample_rate, **self.endpointer_kwargs)
        blocks = []
        reason = None
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                blocksize=self.block_size,
            ) as stream:
                while reason is None:
                    if stop_event.is_set():
                        reason = "stopped"
                        break
                    data, _overflowed = stream.read(self.block_size)
                    chunk = np.asarray(data, dtype=np.float32)
                    if chunk.ndim == 2:
                        chunk = chunk[:, 0]
                    chunk = chunk.reshape(-1)
                    blocks.append(chunk)
                    reason = endpointer.feed(chunk)
        except device_errors as exc:
            raise MicrophoneError(f"Lỗi micro: {exc}") from exc

        audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
        return RecordingResult(
            audio=audio.astype(np.float32, copy=False),
            reason=reason,
            heard_speech=endpointer.heard_speech,
        )
