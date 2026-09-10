import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.runtime.model_loading import offline_load_error, pretrained_kwargs


class VQABackend(ABC):
    """Backend thay thế được, nhận nguyên ảnh và câu hỏi của người dùng."""

    @abstractmethod
    def answer(self, image: np.ndarray, question: str) -> str:
        """Trả về text mô tả/trả lời, hoặc raise khi backend không khả dụng."""


class LegacyCaptionBackend(VQABackend):
    """Backend BLIP caption + MarianMT kế thừa từ VQAService ban đầu."""

    def __init__(
        self,
        device: str = "mps",
        caption_model_name: str = "Salesforce/blip-image-captioning-base",
        translation_model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        lazy_load: bool = True
    ):
        if device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.caption_model_name = caption_model_name
        self.translation_model_name = translation_model_name
        self._caption_processor = None
        self._caption_model = None
        self._trans_tokenizer = None
        self._trans_model = None
        self._models_lock = threading.Lock()
        self._load_attempted = False
        self._load_error: Optional[str] = None

        if not lazy_load:
            self._load_models()

    def _load_models(self) -> bool:
        with self._models_lock:
            if self._caption_model is not None and self._trans_model is not None:
                return True
            if self._load_attempted:
                return False

            self._load_attempted = True
            try:
                print(
                    "[LegacyCaptionBackend] Đang nạp BLIP "
                    f"({self.caption_model_name}) lên {self.device}..."
                )
                from transformers import (
                    BlipProcessor,
                    BlipForConditionalGeneration,
                    MarianTokenizer,
                    MarianMTModel,
                )
                load_kwargs = pretrained_kwargs()
                self._caption_processor = BlipProcessor.from_pretrained(
                    self.caption_model_name, **load_kwargs
                )
                self._caption_model = BlipForConditionalGeneration.from_pretrained(
                    self.caption_model_name, **load_kwargs
                ).to(self.device)
                self._caption_model.eval()

                print(
                    "[LegacyCaptionBackend] Đang nạp MarianMT "
                    f"({self.translation_model_name})..."
                )
                self._trans_tokenizer = MarianTokenizer.from_pretrained(
                    self.translation_model_name, **load_kwargs
                )
                self._trans_model = MarianMTModel.from_pretrained(
                    self.translation_model_name, **load_kwargs
                ).to(self.device)
                self._trans_model.eval()
                self._load_error = None
                print("[LegacyCaptionBackend] Nạp model thành công.")
                return True
            except Exception as exc:
                self._caption_processor = None
                self._caption_model = None
                self._trans_tokenizer = None
                self._trans_model = None
                model_names = f"{self.caption_model_name}, {self.translation_model_name}"
                self._load_error = offline_load_error("BLIP/MarianMT", model_names, exc)
                print(f"[LegacyCaptionBackend] {self._load_error}")
                return False

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def answer(self, image: np.ndarray, question: str) -> str:
        # Backend legacy là captioner nên chưa dùng question để điều khiển sinh text.
        del question
        if not self._load_models():
            raise RuntimeError(self._load_error or "Backend BLIP/MarianMT không khả dụng")

        img_rgb = image[:, :, ::-1] if len(image.shape) == 3 else image
        inputs = self._caption_processor(
            images=Image.fromarray(img_rgb), return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self._caption_model.generate(**inputs, max_new_tokens=45)
        caption = self._caption_processor.decode(
            output[0], skip_special_tokens=True
        ).strip()
        sentence = (
            f"{caption}."
            if caption.lower().startswith(("there is", "there are", "this is", "a photo of"))
            else f"There is {caption}."
        )

        trans_inputs = self._trans_tokenizer(sentence, return_tensors="pt")
        trans_inputs = {key: value.to(self.device) for key, value in trans_inputs.items()}
        with torch.no_grad():
            translated = self._trans_model.generate(
                **trans_inputs, max_length=64, num_beams=3, repetition_penalty=1.2
            )
        text = self._trans_tokenizer.decode(
            translated[0], skip_special_tokens=True
        ).strip()
        return text[:-1].strip() if text.endswith(".") else text


def create_vqa_backend(
    backend_name: str,
    *,
    device: str,
    caption_model_name: str,
    translation_model_name: str,
    lazy_load: bool
) -> Optional[VQABackend]:
    normalized = backend_name.strip().lower()
    if normalized in ("none", "disabled", "rule_vqa"):
        return None
    if normalized in ("legacy_caption", "blip_vlm"):
        return LegacyCaptionBackend(
            device=device,
            caption_model_name=caption_model_name,
            translation_model_name=translation_model_name,
            lazy_load=lazy_load
        )
    print(f"[VQAService] Backend không được hỗ trợ: {backend_name}. Dùng fallback detection context.")
    return None
