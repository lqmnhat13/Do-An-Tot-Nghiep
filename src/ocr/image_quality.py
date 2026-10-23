import cv2
import numpy as np
from typing import Tuple, Dict

class ImageQualityChecker:
    """
    Đánh giá sơ bộ chất lượng khung hình trước khi tốn tài nguyên chạy OCR.
    Tuân thủ mục 5.2 của Kế hoạch:
    - Kiểm tra rung mờ (Laplacian variance) trên ảnh đã thu về analysis_max_side,
      để ảnh độ phân giải cao nhưng nét không bị đánh giá oan là mờ.
    - Kiểm tra độ sáng trung bình (quá tối); lóa sáng = rất sáng VÀ mất tương phản,
      vì trang giấy trắng chụp tốt vốn có độ sáng trung bình cao.
    - Hướng dẫn chụp lại thay vì đoán mò khi ảnh chất lượng kém.
    Metrics có "too_small" = 1.0 khi ảnh quá nhỏ để thử đọc.
    """

    def __init__(
        self,
        blur_threshold: float = 50.0,
        min_brightness: float = 40.0,
        max_brightness: float = 235.0,
        min_dimension: int = 200,
        analysis_max_side: int = 1024,
        min_glare_contrast: float = 15.0
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_dimension = min_dimension
        self.analysis_max_side = analysis_max_side
        self.min_glare_contrast = min_glare_contrast

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
            return False, "Kích thước ảnh quá nhỏ", {"width": float(w), "height": float(h), "too_small": 1.0}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Đo độ sắc nét bằng Laplacian variance ở độ phân giải chuẩn hoá
        scale = min(1.0, self.analysis_max_side / float(max(h, w)))
        small = gray if scale >= 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        laplacian_var = float(cv2.Laplacian(small, cv2.CV_64F).var())

        # 2. Đo độ sáng trung bình và tương phản
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        metrics = {
            "blur_score": laplacian_var,
            "brightness": brightness,
            "contrast": contrast,
            "width": float(w),
            "height": float(h)
        }

        # 3. Phân loại lỗi
        if brightness < self.min_brightness:
            return False, "Ảnh quá tối, vui lòng hướng camera về nguồn sáng và thử lại", metrics

        if brightness > self.max_brightness and contrast < self.min_glare_contrast:
            return False, "Ảnh bị lóa sáng mạnh, vui lòng chỉnh góc chụp và thử lại", metrics

        if laplacian_var < self.blur_threshold:
            return False, "Ảnh bị mờ hoặc rung tay, vui lòng giữ yên và chụp lại", metrics

        return True, "Ảnh đạt chất lượng đọc", metrics
