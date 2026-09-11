import threading
import json
from pathlib import Path
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


class MLXVLMBackend(VQABackend):
    """Question-aware MLX inference, strictly local and always loaded on demand.

    One lock covers both loading and generation: processors/Metal state must not
    be used concurrently. No worker is owned by this backend; cancellation and
    suppression of late results remain the caller's responsibility.
    """

    SYSTEM_PROMPT = (
        "Trả lời trực tiếp câu hỏi bằng tiếng Việt, trong 1–2 câu ngắn, hoàn chỉnh "
        "và có dấu kết câu. Chỉ nêu chi tiết nhìn thấy rõ trong ảnh, liên quan đến "
        "câu hỏi. Không suy đoán cảm xúc, ý định, danh tính hoặc thông tin ngoài ảnh. "
        "Không mặc định các giả thiết trong câu hỏi là đúng. Nếu không đủ bằng chứng, "
        "hãy nói: 'Không xác định rõ từ ảnh.' Không khẳng định đường đi an toàn. "
        "Không liệt kê dài, không giải thích quá trình suy luận."
    )

    def __init__(self, model_path: str, max_image_size: int = 512,
                 max_tokens: int = 64):
        self.model_path = model_path
        self.max_image_size = max_image_size
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._model = self._processor = None
        self._generate = self._apply_chat_template = None
        self._config = None
        self._load_attempted = False
        self._load_error: Optional[str] = None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _load_models(self) -> None:
        # Caller holds _lock. Never acquire it recursively.
        if self._model is not None:
            return
        if self._load_attempted:
            raise RuntimeError(self._load_error)
        self._load_attempted = True
        try:
            if not self.model_path:
                raise ValueError("Chưa cấu hình vqa.mlx_vlm.model_path")
            path = Path(self.model_path).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            path = path.resolve()
            # Reject repo IDs/missing paths BEFORE importing MLX or calling load:
            # mlx_vlm.load otherwise treats missing directories as Hub repo IDs.
            if not path.is_dir() or not (path / "config.json").is_file():
                raise FileNotFoundError(f"Thiếu thư mục model/config.json: {path}")
            if not any(path.glob("*.safetensors")):
                raise FileNotFoundError(f"Thiếu trọng số safetensors: {path}")
            with (path / "config.json").open(encoding="utf-8") as handle:
                config = json.load(handle)
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template

            model, processor = load(
                str(path), local_files_only=True, trust_remote_code=False
            )
            self._model, self._processor = model, processor
            self._config = config
            self._generate = generate
            self._apply_chat_template = apply_chat_template
        except Exception as exc:
            self._load_error = (
                f"MLX-VLM không khả dụng: {exc}. Cài mlx-vlm và chạy "
                "scripts/download_models.py --mlx-vlm khi có mạng."
            )
            raise RuntimeError(self._load_error) from exc

    def answer(self, image: np.ndarray, question: str) -> str:
        with self._lock:
            # Hard caps still apply when users increase YAML values.
            size = max(28, min(int(self.max_image_size), 512))
            tokens = max(1, min(int(self.max_tokens), 64))
            self._load_models()
            rgb = image[:, :, ::-1] if image.ndim == 3 else image
            pil_image = Image.fromarray(rgb).convert("RGB")
            pil_image.thumbnail((size, size), Image.Resampling.LANCZOS)
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt = self._apply_chat_template(
                self._processor, self._config, messages, num_images=1
            )
            output = self._generate(
                self._model, self._processor, prompt, image=[pil_image],
                max_tokens=tokens, temperature=0.0, verbose=False
            )
            return self._validated_answer(output, tokens)

    @staticmethod
    def _validated_answer(output, token_limit: int) -> str:
        """Reject potentially unfinished output; never invent a continuation.

        Older MLX-VLM versions may return text or omit finish_reason. Token count
        is then a conservative limit signal. Punctuation is only a heuristic,
        not proof of grammatical completeness or visual grounding.
        """
        text = output if isinstance(output, str) else getattr(output, "text", None)
        if not isinstance(text, str):
            raise RuntimeError("MLX-VLM trả về kết quả không hợp lệ")
        reason = getattr(output, "finish_reason", None)
        count = getattr(output, "generation_tokens", None)
        hit_limit = reason == "length" or (
            reason != "stop" and isinstance(count, int) and count >= token_limit
        )
        if hit_limit:
            # Even a complete-looking prefix can lose a qualification in the
            # missing tail. Discard the whole answer and let the service fallback.
            raise RuntimeError("MLX-VLM chạm giới hạn token; bỏ câu trả lời có thể bị cắt")
        text = text.strip()
        ending = text.rstrip('\"\u201d\u2019\u0027)')
        if not ending or not ending.endswith((".", "!", "?")) or ending.endswith("..."):
            raise RuntimeError("MLX-VLM trả lời rỗng hoặc chưa có dấu kết câu rõ ràng")
        return text


def create_vqa_backend(
    backend_name: str,
    *,
    device: str,
    caption_model_name: str,
    translation_model_name: str,
    lazy_load: bool,
    mlx_vlm_config: Optional[dict] = None
) -> Optional[VQABackend]:
    normalized = backend_name.strip().lower()
    if normalized in ("none", "disabled", "rule_vqa"):
        return None
    if normalized == "mlx_vlm":
        config = mlx_vlm_config or {}
        return MLXVLMBackend(
            model_path=config.get("model_path", ""),
            max_image_size=config.get("max_image_size", 512),
            max_tokens=config.get("max_tokens", 64),
        )
    if normalized in ("legacy_caption", "blip_vlm"):
        return LegacyCaptionBackend(
            device=device,
            caption_model_name=caption_model_name,
            translation_model_name=translation_model_name,
            lazy_load=lazy_load
        )
    print(f"[VQAService] Backend không được hỗ trợ: {backend_name}. Dùng fallback detection context.")
    return None
