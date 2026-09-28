import time
import os
import cv2
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

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Tiền xử lý ảnh: phóng đại nếu ảnh nhỏ và tăng tương phản (CLAHE) để nhận diện chữ li ti."""
        h, w = image.shape[:2]
        scale = 1.0
        # Nếu ảnh nhỏ hơn 900px chiều ngang hoặc 700px chiều dọc, phóng to 1.5x để ký tự thuốc đủ nét
        if w < 900 or h < 700:
            scale = 1.5
            proc_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            proc_img = image.copy()

        # Tăng tương phản nếu ảnh hơi mờ hoặc thiếu tương phản
        try:
            if len(proc_img.shape) == 3:
                lab = cv2.cvtColor(proc_img, cv2.COLOR_BGR2LAB)
                l, a, b_ch = cv2.split(lab)
                if float(np.std(l)) < 45.0:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    lab = cv2.merge((l, a, b_ch))
                    proc_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

        return proc_img, scale

    @staticmethod
    def _cluster_and_sort_lines(raw_items: List[Dict[str, Any]]) -> str:
        """
        Phân cụm các từ thành từng dòng dựa trên độ phủ hình học (Vertical Overlap)
        và sắp xếp từ trái sang phải, từ trên xuống dưới.
        """
        if not raw_items:
            return ""

        # Sắp xếp sơ bộ theo vị trí trung tâm trục Y
        sorted_items = sorted(raw_items, key=lambda x: x["center_y"])
        lines = []

        for item in sorted_items:
            best_line = None
            best_overlap = 0.0

            for line in lines:
                overlap = max(0.0, min(item["bottom"], line["bottom"]) - max(item["top"], line["top"]))
                min_h = min(item["height"], line["line_height"])
                ratio = overlap / min_h if min_h > 0 else 0.0
                if ratio > 0.4 and ratio > best_overlap:
                    best_overlap = ratio
                    best_line = line

            if best_line is not None:
                best_line["words"].append(item)
                best_line["top"] = min(best_line["top"], item["top"])
                best_line["bottom"] = max(best_line["bottom"], item["bottom"])
                best_line["line_height"] = best_line["bottom"] - best_line["top"]
            else:
                lines.append({
                    "top": item["top"],
                    "bottom": item["bottom"],
                    "line_height": item["height"],
                    "words": [item]
                })

        # Sắp xếp các dòng từ trên xuống dưới
        lines.sort(key=lambda l: l["top"])

        # Nối các từ trong cùng dòng theo thứ tự từ trái sang phải
        result_lines = []
        for line in lines:
            line["words"].sort(key=lambda w: w["left"])
            line_text = " ".join(w["text"] for w in line["words"]).strip()
            if line_text:
                if line_text[-1] in ".!?:;":
                    result_lines.append(line_text)
                else:
                    result_lines.append(line_text + ".")

        return " ".join(result_lines)

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

            # 2. Tiền xử lý ảnh (upscale + tăng tương phản)
            processed_image, scale = self._preprocess_image(request.image)

            # 3. Chạy nhận dạng EasyOCR
            raw_results = self._reader.readtext(processed_image)

            # 4. Lọc và chuẩn hoá toạ độ về ảnh gốc
            valid_items = []
            detected_boxes = []
            for item in raw_results:
                bbox_poly, text, conf = item
                clean_text = text.strip()
                if conf >= self.min_confidence and len(clean_text) > 0:
                    pts = np.array(bbox_poly, dtype=np.float32)
                    if scale != 1.0:
                        pts = pts / scale

                    top_y = float(np.min(pts[:, 1]))
                    bottom_y = float(np.max(pts[:, 1]))
                    left_x = float(np.min(pts[:, 0]))
                    right_x = float(np.max(pts[:, 0]))
                    height = max(bottom_y - top_y, 1.0)
                    center_y = (top_y + bottom_y) / 2.0

                    orig_box = pts.tolist()
                    detected_boxes.append(orig_box)
                    valid_items.append({
                        "top": top_y,
                        "bottom": bottom_y,
                        "left": left_x,
                        "right": right_x,
                        "center_y": center_y,
                        "height": height,
                        "text": clean_text,
                        "conf": conf,
                        "box": orig_box
                    })

            if not valid_items:
                return OCRResult(
                    request_id=request.request_id,
                    success=True,
                    text="Không phát hiện thấy chữ trong ảnh.",
                    latency_sec=time.monotonic() - t0
                )

            # 5. Phân cụm dòng hình học thông minh và ghép chữ
            full_text = self._cluster_and_sort_lines(valid_items)

            return OCRResult(
                request_id=request.request_id,
                success=True,
                text=full_text,
                detected_boxes=detected_boxes,
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
