import unittest
import cv2
import numpy as np
import time

from src.contracts.request import OCRRequest, VQARequest
from src.ocr.image_quality import ImageQualityChecker
from src.ocr.ocr_service import OCRService
from src.vqa.backend import VQABackend
from src.vqa.vqa_service import VQAService
from src.vqa.formatting import format_vqa_answer


class FakeVQABackend(VQABackend):
    def __init__(self):
        self.calls = []

    def answer(self, image, question):
        self.calls.append((image, question))
        return f"Trả lời cho: {question}"


class FailingVQABackend(VQABackend):
    def answer(self, image, question):
        raise RuntimeError("backend test failure")


class FixedVQABackend(VQABackend):
    def __init__(self, answer):
        self.text = answer

    def answer(self, image, question):
        return self.text

class TestOCRVQA(unittest.TestCase):
    def test_formatting_preserves_terminal_punctuation_and_legacy_caption(self):
        for text in ("Ghế màu đỏ.", "Không xác định rõ!", "Đây là ghế?",
                     '“Không xác định rõ.”'):
            with self.subTest(text=text):
                self.assertEqual(format_vqa_answer(text), "Khung cảnh: " + text)
        self.assertEqual(format_vqa_answer("Một chiếc ghế"),
                         "Khung cảnh: Một chiếc ghế.")
        self.assertEqual(format_vqa_answer("Ghế màu đỏ.", {"spatial_objects": [
            {"name": "ghế", "direction": "LEFT", "proximity": "gần"}
        ]}), "Khung cảnh: Ghế màu đỏ. Cụ thể có: ghế bên trái (gần).")

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

    def test_white_document_is_not_glare(self):
        checker = ImageQualityChecker(max_brightness=240.0)
        page = np.full((1400, 1000, 3), 250, dtype=np.uint8)
        for y in range(100, 1300, 120):
            cv2.putText(page, "TOTAL RM 95.48", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        is_ok, msg, metrics = checker.assess(page)
        self.assertGreater(metrics["brightness"], 240.0)
        self.assertNotIn("lóa", msg)

    def test_washed_out_image_is_glare(self):
        checker = ImageQualityChecker(max_brightness=240.0)
        is_ok, msg, _ = checker.assess(np.full((480, 640, 3), 252, dtype=np.uint8))
        self.assertFalse(is_ok)
        self.assertIn("lóa", msg)

    def test_blur_is_measured_at_normalized_resolution(self):
        # Ảnh nét độ phân giải cao: Laplacian ở độ phân giải gốc thấp hơn nhiều so với khi thu nhỏ.
        rng = np.random.default_rng(0)
        texture = cv2.resize(rng.integers(0, 255, (300, 400), dtype=np.uint8), (3200, 2400),
                             interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        full_res = float(cv2.Laplacian(texture, cv2.CV_64F).var())
        _, _, metrics = ImageQualityChecker().assess(image)
        self.assertGreater(metrics["blur_score"], full_res)

    def test_flagged_image_is_still_read(self):
        class FlaggingChecker:
            def assess(self, image):
                return False, "Ảnh bị mờ hoặc rung tay, vui lòng giữ yên và chụp lại", {}

        class Reader:
            def __init__(self, items):
                self.items = items

            def readtext(self, image):
                return self.items

        image = np.full((300, 300, 3), 200, dtype=np.uint8)
        service = OCRService(quality_checker=FlaggingChecker())
        service._load_attempted = True
        service._reader = Reader([([[0, 0], [90, 0], [90, 20], [0, 20]], "HẠN DÙNG", 0.9),
                                  ([[0, 40], [90, 40], [90, 60], [0, 60]], "rác", 0.4)])
        res = service.process(OCRRequest("q1", image))
        self.assertTrue(res.success)
        self.assertIn("HẠN DÙNG", res.text)
        self.assertNotIn("rác", res.text)  # ảnh kém chất lượng đòi độ tin cậy cao hơn

        service._reader = Reader([])
        res = service.process(OCRRequest("q2", image))
        self.assertFalse(res.success)
        self.assertEqual(res.quality_status, "BAD_QUALITY")
        self.assertIn("mờ", res.text)

    def test_too_small_image_is_rejected_without_reading(self):
        service = OCRService()
        service._load_attempted = True
        service._reader = None
        res = service.process(OCRRequest("q3", np.zeros((50, 50, 3), dtype=np.uint8)))
        self.assertEqual(res.quality_status, "BAD_QUALITY")
        self.assertIn("nhỏ", res.text)

    def test_guardrail_refuses_movement_and_hazard_questions(self):
        from src.vqa.safety import SafetyGuardrail
        guard = SafetyGuardrail()
        for question in (
            "Tôi có thể đi tiếp không?", "Đi về phía trước được không?", "Phía trước có vật cản không?",
            "Bước tiếp có sao không?", "Tôi qua đường được không?", "Có nguy hiểm gì phía trước không?",
            "Đường phía trước có an toàn không?", "Mình có nên rẽ trái không?", "Có bậc thang phía trước không?",
            "Phía trước có xe đang chạy tới không?", "Biển báo ghi gì, tôi có đi thẳng được không?",
            "Can I walk forward?", "Is there an obstacle ahead?",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(guard.check(question))

    def test_guardrail_allows_reading_and_counting_questions(self):
        # Các câu thật từ OpenViVQA từng bị chặn nhầm bởi khớp từ khoá đơn.
        from src.vqa.safety import SafetyGuardrail
        guard = SafetyGuardrail()
        for question in (
            "Biển báo ghi là đi thẳng hay rẽ trái?", "đi thẳng theo bảng báo thì tới được đâu ?",
            "có mấy người đang đi qua đường ?", "có bao nhiêu người đang băng qua đường ?",
            "tại cổng trường tuyên truyền điều gì để gìn giữ an toàn giao thông ?",
            "Tờ giấy này viết về an toàn thực phẩm phải không?", "Đây là ngã tư nào?",
            "những bậc thang này dẫn đến đâu ?", "quán ăn này nằm trên đường nào của quận gò vấp ?",
            "Phía trước có gì?", "Mô tả khung cảnh phía trước.",
        ):
            with self.subTest(question=question):
                self.assertIsNone(guard.check(question))

    def test_vqa_safety_guardrail(self):
        vqa = VQAService(use_vlm=False)
        # Hỏi về an toàn đi tiếp -> Phải từ chối xác nhận an toàn!
        req = VQARequest(request_id="vqa_1", image=np.zeros((10, 10, 3), dtype=np.uint8), question="Tôi có đi tiếp an toàn không?")
        res = vqa.answer(req)
        self.assertTrue(res.success)
        self.assertIn("không thể xác nhận đường đi có an toàn hay không", res.answer)
        english = vqa.answer(VQARequest(
            "vqa_english", np.zeros((10, 10, 3), dtype=np.uint8),
            "Is it safe to walk?"
        ))
        self.assertIn("không thể xác nhận", english.answer)

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

    def test_unsafe_model_answer_is_refused_without_stale_context(self):
        request = VQARequest("unsafe", np.zeros((10, 10, 3), dtype=np.uint8),
                             "Mô tả phía trước")
        for text in ("Bạn có thể đi tiếp an toàn.",
                     "Lối đi phía trước an toàn.",
                     "Không có vật cản phía trước.",
                     "An toàn để đi tiếp.",
                     "Bạn có thể đi tiếp.",
                     "You can proceed."):
            with self.subTest(text=text):
                service = VQAService(backend=FixedVQABackend(text))
                result = service.answer(request, {"spatial_objects": [
                    {"name": "ghế", "direction": "CENTER", "proximity": "gần"}
                ]})
                self.assertIn("không thể xác nhận đường đi", result.answer)
                self.assertNotIn("ghế", result.answer)

    def test_model_can_report_unsafe_path_without_refusal(self):
        service = VQAService(backend=FixedVQABackend(
            "Lối đi phía trước không an toàn vì có cành cây chắn."
        ))
        result = service.answer(VQARequest(
            "blocked", np.zeros((10, 10, 3), dtype=np.uint8), "Mô tả phía trước"
        ))
        self.assertIn("không an toàn", result.answer)

if __name__ == "__main__":
    unittest.main()
