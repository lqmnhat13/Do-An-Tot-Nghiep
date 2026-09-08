import cv2
import numpy as np
from typing import Tuple, Dict

class ImageQualityChecker:
    """
    Đánh giá sơ bộ chất lượng khung hình trước khi tốn tài nguyên chạy OCR.
    Tuân thủ mục 5.2 của Kế hoạch:
    - Kiểm tra rung mờ (Laplacian variance).
    - Kiểm tra độ sáng trung bình (tránh quá tối hoặc quá lóa).
    - Hướng dẫn chụp lại thay vì đoán mò khi ảnh chất lượng kém.
    """

    def __init__(
        self,
        blur_threshold: float = 50.0,
        min_brightness: float = 40.0,
        max_brightness: float = 235.0,
        min_dimension: int = 200
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_dimension = min_dimension

    def assess(self, image: np.ndarray) -> Tuple[bool, str, Dict[str, float]]:
        """
        Đánh giá chất lượng ảnh.
        Trả về:
            is_acceptable: bool
            status_message: str (tiếng Việt)
            metrics: dict thông số đo được
        """
        h, w = image.shape[:2]
        if h < self.min_dimension or w < self.min_dimension:
            return False, "Kích thước ảnh quá nhỏ", {"width": float(w), "height": float(h)}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Đo độ sắc nét bằng Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Đo độ sáng trung bình
        brightness = float(np.mean(gray))

        metrics = {
            "blur_score": laplacian_var,
            "brightness": brightness,
            "width": float(w),
            "height": float(h)
        }

        # 3. Phân loại lỗi
        if brightness < self.min_brightness:
            return False, "Ảnh quá tối, vui lòng hướng camera về nguồn sáng và thử lại", metrics

        if brightness > self.max_brightness:
            return False, "Ảnh bị lóa sáng mạnh, vui lòng chỉnh góc chụp và thử lại", metrics

        if laplacian_var < self.blur_threshold:
            return False, "Ảnh bị mờ hoặc rung tay, vui lòng giữ yên và chụp lại", metrics

        return True, "Ảnh đạt chất lượng đọc", metrics
