import time
from typing import Any, Dict, Optional

from src.contracts.request import VQARequest, VQAResult
from src.vqa.backend import VQABackend, create_vqa_backend
from src.vqa.formatting import format_vqa_answer
from src.vqa.safety import SafetyGuardrail


class VQAService:
    """Điều phối guardrail, backend VQA và detection-context fallback."""

    def __init__(
        self,
        device: str = "mps",
        use_vlm: bool = True,
        caption_model_name: str = "Salesforce/blip-image-captioning-base",
        translation_model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        lazy_load: bool = True,
        backend_name: str = "legacy_caption",
        backend: Optional[VQABackend] = None,
        safety_guardrail: Optional[SafetyGuardrail] = None
    ):
        # Giữ cấu hình cũ để tương thích consumer hiện tại.
        self.device = device
        self.use_vlm = use_vlm
        self.caption_model_name = caption_model_name
        self.translation_model_name = translation_model_name
        self.backend_name = backend_name
        self.safety_guardrail = safety_guardrail or SafetyGuardrail()

        if backend is not None:
            self.backend = backend
        elif use_vlm:
            self.backend = create_vqa_backend(
                backend_name,
                device=device,
                caption_model_name=caption_model_name,
                translation_model_name=translation_model_name,
                lazy_load=lazy_load
            )
        else:
            self.backend = None

    @property
    def load_error(self) -> Optional[str]:
        return getattr(self.backend, "load_error", None)

    def answer(
        self,
        request: VQARequest,
        visual_context: Optional[Dict[str, Any]] = None
    ) -> VQAResult:
        started_at = time.monotonic()

        # Guardrail luôn chạy trước và không phụ thuộc backend.
        refusal = self.safety_guardrail.check(request.question)
        if refusal is not None:
            return VQAResult(
                request_id=request.request_id,
                success=True,
                answer=refusal,
                latency_sec=time.monotonic() - started_at
            )

        backend_text = ""
        if self.backend is not None:
            try:
                backend_text = self.backend.answer(request.image, request.question)
            except Exception as exc:
                print(
                    f"[VQAService] Backend {self.backend_name} lỗi: {exc}. "
                    "Dùng fallback detection context."
                )

        return VQAResult(
            request_id=request.request_id,
            success=True,
            answer=format_vqa_answer(backend_text, visual_context),
            latency_sec=time.monotonic() - started_at
        )
