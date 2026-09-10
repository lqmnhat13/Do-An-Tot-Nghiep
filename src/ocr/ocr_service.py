import time
import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from src.contracts.request import OCRRequest, OCRResult
from src.ocr.image_quality import ImageQualityChecker
from src.runtime.model_loading import is_offline_mode, offline_load_error

class OCRService:
    """
    Dịch vụ nhận diện văn bản tiếng Việt theo yêu cầu (On-Demand OCR).
    Tuân thủ mục 5.2 của Kế hoạch:
    - Kiểm tra chất lượng ảnh trước khi chạy.
    - Sắp xếp các đoạn văn bản theo đúng thứ tự đọc (từ trên xuống, từ trái sang).
    - Trả lời trung thực khi không đọc được, không suy diễn thêm chữ.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = True,
        min_confidence: float = 0.35,
        quality_checker: Optional[ImageQualityChecker] = None
    ):
        self.languages = languages or ["vi", "en"]
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.quality_checker = quality_checker or ImageQualityChecker()
        self._reader = None
        self._load_attempted = False
        self._load_error: Optional[str] = None

    def _ensure_reader(self) -> None:
        if self._reader is not None or self._load_attempted:
            return

        self._load_attempted = True
        print("[OCRService] Đang khởi tạo EasyOCR engine cho tiếng Việt...")
        try:
            import easyocr
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                download_enabled=not is_offline_mode()
            )
            self._load_error = None
            print("[OCRService] EasyOCR khởi tạo thành công.")
        except Exception as exc:
            self._reader = None
            self._load_error = offline_load_error("EasyOCR", ",".join(self.languages), exc)
            print(f"[OCRService] {self._load_error} Dùng fallback OCR không khả dụng.")

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def process(self, request: OCRRequest) -> OCRResult:
        t0 = time.monotonic()

        # 1. Kiểm tra chất lượng ảnh đầu vào
        is_ok, msg, metrics = self.quality_checker.assess(request.image)
        if not is_ok:
            return OCRResult(
                request_id=request.request_id,
                success=False,
                text=msg,
                quality_status="BAD_QUALITY",
                latency_sec=time.monotonic() - t0,
                error_message=msg
            )

        try:
            self._ensure_reader()
            if self._reader is None:
                message = self._load_error or "EasyOCR không khả dụng."
                return OCRResult(
                    request_id=request.request_id,
                    success=False,
                    text=message,
                    latency_sec=time.monotonic() - t0,
                    error_message=message
                )
            # 2. Chạy nhận dạng OCR
            raw_results = self._reader.readtext(request.image)

            # 3. Lọc và định dạng
            valid_items = []
            for item in raw_results:
                bbox_poly, text, conf = item
                if conf >= self.min_confidence and len(text.strip()) > 0:
                    # Tính tâm y và x để sắp xếp thứ tự đọc
                    pts = np.array(bbox_poly)
                    top_y = float(np.min(pts[:, 1]))
                    left_x = float(np.min(pts[:, 0]))
                    valid_items.append((top_y, left_x, text.strip(), conf, bbox_poly))

            if not valid_items:
                return OCRResult(
                    request_id=request.request_id,
                    success=True,
                    text="Không phát hiện thấy chữ trong ảnh.",
                    latency_sec=time.monotonic() - t0
                )

            # 4. Sắp xếp thứ tự đọc (ưu tiên dòng trên trước, các từ trên cùng dòng theo thứ tự trái sang phải)
            # Nhóm các từ có độ cao y chênh lệch dưới 15px vào cùng một dòng
            valid_items.sort(key=lambda x: (int(x[0] // 20), x[1]))

            extracted_lines = [item[2] for item in valid_items]
            full_text = " ".join(extracted_lines)

            return OCRResult(
                request_id=request.request_id,
                success=True,
                text=full_text,
                detected_boxes=[item[4] for item in valid_items],
                quality_status="OK",
                latency_sec=time.monotonic() - t0
            )

        except Exception as e:
            return OCRResult(
                request_id=request.request_id,
                success=False,
                text=f"Lỗi trong quá trình đọc chữ: {str(e)}",
                latency_sec=time.monotonic() - t0,
                error_message=str(e)
            )
