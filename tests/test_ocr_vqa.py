import unittest
import numpy as np
import time

from src.contracts.request import OCRRequest, VQARequest
from src.ocr.image_quality import ImageQualityChecker
from src.vqa.vqa_service import VQAService

class TestOCRVQA(unittest.TestCase):
    def test_image_quality_checker_dark(self):
        checker = ImageQualityChecker(min_brightness=40.0)
        # Ảnh đen hoàn toàn
        dark_img = np.zeros((300, 300, 3), dtype=np.uint8)
        is_ok, msg, metrics = checker.assess(dark_img)
        self.assertFalse(is_ok)
        self.assertIn("tối", msg)

    def test_image_quality_checker_blur(self):
        checker = ImageQualityChecker(blur_threshold=50.0)
        # Ảnh xám trơn (không có cạnh -> laplacian variance = 0)
        flat_img = np.ones((300, 300, 3), dtype=np.uint8) * 128
        is_ok, msg, metrics = checker.assess(flat_img)
        self.assertFalse(is_ok)
        self.assertIn("mờ", msg)

    def test_vqa_safety_guardrail(self):
        vqa = VQAService(use_vlm=False)
        # Hỏi về an toàn đi tiếp -> Phải từ chối xác nhận an toàn!
        req = VQARequest(request_id="vqa_1", image=np.zeros((10, 10, 3), dtype=np.uint8), question="Tôi có đi tiếp an toàn không?")
        res = vqa.answer(req)
        self.assertTrue(res.success)
        self.assertIn("không thể xác nhận đường đi có an toàn hay không", res.answer)

    def test_vqa_with_context(self):
        vqa = VQAService(use_vlm=False)
        req = VQARequest(request_id="vqa_2", image=np.zeros((10, 10, 3), dtype=np.uint8), question="Mô tả khung cảnh phía trước")
        res = vqa.answer(req, visual_context={
            "spatial_objects": [{"name": "người", "direction": "CENTER", "proximity": "gần"}]
        })
        self.assertTrue(res.success)
        self.assertIn("người", res.answer)

if __name__ == "__main__":
    unittest.main()
