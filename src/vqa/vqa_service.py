import time
import os
import threading
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from PIL import Image

from src.contracts.request import VQARequest, VQAResult

class VQAService:
    """
    Dịch vụ hỏi đáp thị giác và mô tả khung cảnh (VQA & Scene Captioning).
    Kết hợp mô hình thị giác ngôn ngữ BLIP (Vision-Language Model) và
    dịch máy nơ-ron MarianMT sang tiếng Việt, kèm rào chắn an toàn (Safety Guardrail).
    """

    def __init__(
        self,
        device: str = "mps",
        use_vlm: bool = True,
        caption_model_name: str = "Salesforce/blip-image-captioning-base",
        translation_model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        lazy_load: bool = True
    ):
        if device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.use_vlm = use_vlm
        self.caption_model_name = caption_model_name
        self.translation_model_name = translation_model_name

        self._caption_processor = None
        self._caption_model = None
        self._trans_tokenizer = None
        self._trans_model = None
        self._models_lock = threading.Lock()
        self._is_loading = False

        if not lazy_load and self.use_vlm:
            self._load_models()

    def _load_models(self) -> bool:
        with self._models_lock:
            if self._caption_model is not None and self._trans_model is not None:
                return True

            try:
                print(f"[VQAService] Đang nạp mô hình mô tả ảnh BLIP ({self.caption_model_name}) lên {self.device}...")
                from transformers import BlipProcessor, BlipForConditionalGeneration, MarianTokenizer, MarianMTModel

                self._caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
                self._caption_model = BlipForConditionalGeneration.from_pretrained(self.caption_model_name).to(self.device)
                self._caption_model.eval()

                print(f"[VQAService] Đang nạp mô hình dịch tiếng Việt MarianMT ({self.translation_model_name})...")
                self._trans_tokenizer = MarianTokenizer.from_pretrained(self.translation_model_name)
                self._trans_model = MarianMTModel.from_pretrained(self.translation_model_name).to(self.device)
                self._trans_model.eval()

                print("[VQAService] Nạp mô hình VQA & Translation thành công.")
                return True
            except Exception as e:
                print(f"[VQAService] Lỗi nạp mô hình VLM: {e}")
                return False

    def answer(self, request: VQARequest, visual_context: Optional[Dict[str, Any]] = None) -> VQAResult:
        t0 = time.monotonic()
        q_lower = request.question.strip().lower()

        # 1. RÀO CHẮN AN TOÀN (Safety Guardrail)
        safety_keywords = ["an toàn", "đi được", "bước tiếp", "có va chạm", "đi thẳng", "qua đường", "có nguy hiểm"]
        if any(kw in q_lower for kw in safety_keywords):
            return VQAResult(
                request_id=request.request_id,
                success=True,
                answer="Hệ thống không thể xác nhận đường đi có an toàn hay không. Xin hãy cẩn thận dùng gậy dẫn đường và kiểm tra xung quanh.",
                latency_sec=time.monotonic() - t0
            )

        # 2. XỬ LÝ MÔ TẢ KHUNG CẢNH BẰNG VLM (BLIP + MARIANMT)
        vlm_success = False
        vi_description = ""

        if self.use_vlm:
            if self._caption_model is None or self._trans_model is None:
                self._load_models()

            if self._caption_model is not None and self._trans_model is not None:
                try:
                    # Chuyển đổi BGR sang RGB cho PIL
                    img_rgb = request.image[:, :, ::-1] if len(request.image.shape) == 3 else request.image
                    pil_img = Image.fromarray(img_rgb)

                    # A. Sinh mô tả chi tiết bằng BLIP Image Captioning
                    inputs = self._caption_processor(images=pil_img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        out = self._caption_model.generate(**inputs, max_new_tokens=45)
                    en_caption = self._caption_processor.decode(out[0], skip_special_tokens=True).strip()

                    # B. Chuẩn hóa câu để dịch tự nhiên
                    if not en_caption.lower().startswith(("there is", "there are", "this is", "a photo of")):
                        sentence_en = f"There is {en_caption}."
                    else:
                        sentence_en = f"{en_caption}."

                    # C. Dịch sang tiếng Việt bằng MarianMT
                    trans_inputs = self._trans_tokenizer(sentence_en, return_tensors="pt")
                    trans_inputs = {k: v.to(self.device) for k, v in trans_inputs.items()}

                    with torch.no_grad():
                        trans_out = self._trans_model.generate(
                            **trans_inputs,
                            max_length=64,
                            num_beams=3,
                            repetition_penalty=1.2
                        )
                    vi_raw = self._trans_tokenizer.decode(trans_out[0], skip_special_tokens=True).strip()

                    # Làm sạch câu dịch
                    if vi_raw.endswith("."):
                        vi_raw = vi_raw[:-1].strip()
                    vi_description = vi_raw
                    vlm_success = True

                except Exception as e:
                    print(f"[VQAService] Lỗi chạy suy luận VLM: {e}")

        # 3. KẾT HỢP CHI TIẾT TỌA ĐỘ VÀ VẬT CẢN TỪ YOLO & DEPTH (SPATIAL GROUNDING)
        spatial_details = []
        if visual_context:
            spatial_objects = visual_context.get("spatial_objects", [])
            for obj in spatial_objects[:3]:
                name = obj.get("name", "")
                direction = obj.get("direction", "")
                prox = obj.get("proximity", "")
                dir_vi = "ở giữa" if direction == "CENTER" else ("bên trái" if direction == "LEFT" else "bên phải")
                spatial_details.append(f"{name} {dir_vi} ({prox})")

        # 4. TỔNG HỢP CÂU TRẢ LỜI ĐẦY ĐỦ, MÔ TẢ SINH ĐỘNG
        if vlm_success and vi_description:
            final_text = f"Khung cảnh: {vi_description}."
            if spatial_details:
                final_text += f" Cụ thể có: {', '.join(spatial_details)}."
            return VQAResult(
                request_id=request.request_id,
                success=True,
                answer=final_text,
                latency_sec=time.monotonic() - t0
            )

        # Fallback dựa trên nhãn phát hiện nếu VLM không khả dụng
        if spatial_details:
            final_text = f"Trước mặt bạn phát hiện có: {', '.join(spatial_details)}."
            return VQAResult(
                request_id=request.request_id,
                success=True,
                answer=final_text,
                latency_sec=time.monotonic() - t0
            )

        return VQAResult(
            request_id=request.request_id,
            success=True,
            answer="Chưa phát hiện được khung cảnh hoặc vật thể rõ ràng phía trước.",
            latency_sec=time.monotonic() - t0
        )
