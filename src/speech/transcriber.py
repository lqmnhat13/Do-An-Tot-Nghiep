"""Nhận dạng giọng nói tiếng Việt offline bằng PhoWhisper (chạy local, không gửi âm thanh ra mạng)."""

import threading
import unicodedata
from typing import Optional

import numpy as np
import torch

from src.runtime.model_loading import offline_load_error, pretrained_kwargs

MIN_AUDIO_SEC = 0.1  # ngắn hơn mức này coi như không có tiếng nói, không gọi model


class PhoWhisperTranscriber:
    """Bọc ``WhisperProcessor`` + ``WhisperForConditionalGeneration``.

    Model chỉ nạp ở lần ``transcribe`` đầu tiên (lazy). Nạp lỗi thì ghi ``load_error``
    và mọi lần gọi sau ném ``RuntimeError`` ngay, không thử nạp lại.
    Một ``threading.Lock`` bao cả nạp lẫn sinh để an toàn khi gọi từ nhiều luồng.
    """

    def __init__(self, model_name: str = "vinai/PhoWhisper-small", device: str = "mps"):
        self.model_name = model_name
        if device == "mps" and not torch.backends.mps.is_available():
            print("[PhoWhisper] MPS không khả dụng, chuyển sang CPU.")
            device = "cpu"
        self.device = device
        self._processor = None
        self._model = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> None:
        # Gọi khi đang giữ self._lock.
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        print(f"[PhoWhisper] Đang nạp {self.model_name} lên {self.device}...")
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            load_kwargs = pretrained_kwargs()
            processor = WhisperProcessor.from_pretrained(self.model_name, **load_kwargs)
            # float32 cả trên MPS: fp16 trên MPS từng gây lỗi trong dự án.
            model = WhisperForConditionalGeneration.from_pretrained(self.model_name, **load_kwargs)
            model = model.to(self.device)
            model.eval()
        except Exception as exc:
            self._load_error = offline_load_error("PhoWhisper", self.model_name, exc)
            print(f"[PhoWhisper] {self._load_error}")
            raise RuntimeError(self._load_error) from exc
        self._processor = processor
        self._model = model
        print("[PhoWhisper] Nạp model thành công.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Chuyển âm thanh mono float32 thành văn bản tiếng Việt (NFC, đã strip)."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < MIN_AUDIO_SEC * sample_rate:
            return ""
        with self._lock:
            self._ensure_loaded()
            features = self._processor(
                audio, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device)
            with torch.no_grad():
                ids = self._model.generate(features, language="vi", task="transcribe")
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return unicodedata.normalize("NFC", text).strip()
