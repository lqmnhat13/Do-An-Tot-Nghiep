import unittest
import numpy as np
import time

from src.contracts.request import OCRRequest, VQARequest
from src.ocr.image_quality import ImageQualityChecker
from src.vqa.backend import VQABackend
from src.vqa.vqa_service import VQAService


class FakeVQABackend(VQABackend):
    def __init__(self):
        self.calls = []

    def answer(self, image, question):
        self.calls.append((image, question))
        return f"Trả lời cho: {question}"


class FailingVQABackend(VQABackend):
    def answer(self, image, question):
        raise RuntimeError("backend test failure")

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

    def test_backend_receives_question_unchanged_and_distinct_calls(self):
        backend = FakeVQABackend()
        vqa = VQAService(backend=backend)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        first_question = "  Trên bàn có gì?  "
        second_question = "Cửa ra vào ở đâu?"

        first = vqa.answer(VQARequest("vqa_a", image, first_question))
        second = vqa.answer(VQARequest("vqa_b", image, second_question))

        self.assertEqual(
            [call[1] for call in backend.calls],
            [first_question, second_question]
        )
        self.assertIn(first_question.strip(), first.answer)
        self.assertIn(second_question, second.answer)

    def test_safety_guardrail_runs_before_backend(self):
        backend = FakeVQABackend()
        vqa = VQAService(backend=backend)
        request = VQARequest(
            "vqa_safe",
            np.zeros((10, 10, 3), dtype=np.uint8),
            "Tôi có đi tiếp an toàn không?"
        )

        result = vqa.answer(request)

        self.assertEqual(backend.calls, [])
        self.assertIn("không thể xác nhận", result.answer)

    def test_backend_failure_uses_detection_context_fallback(self):
        vqa = VQAService(backend=FailingVQABackend())
        request = VQARequest(
            "vqa_failure",
            np.zeros((10, 10, 3), dtype=np.uint8),
            "Mô tả phía trước"
        )

        result = vqa.answer(request, visual_context={
            "spatial_objects": [
                {"name": "người", "direction": "CENTER", "proximity": "gần"}
            ]
        })

        self.assertTrue(result.success)
        self.assertIn("người", result.answer)

if __name__ == "__main__":
    unittest.main()
