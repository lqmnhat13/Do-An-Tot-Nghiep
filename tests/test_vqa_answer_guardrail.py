"""Kiểm thử guardrail câu trả lời VQA và định dạng câu trả lời (không cần MLX/MPS/camera)."""
import unicodedata
import unittest

from src.vqa.backend import MLXVLMBackend
from src.vqa.formatting import VQA_UNAVAILABLE_ANSWER, format_vqa_answer
from src.vqa.safety import SafetyGuardrail

# Câu trả lời khẳng định lối đi trống/an toàn: bắt buộc bị thay bằng REFUSAL.
MUST_BLOCK = (
    "Không có vật gì phía trước.",
    "Không có gì cản đường.",
    "Không có gì chắn lối đi.",
    "Trước mặt bạn là một khoảng trống.",
    "Đường thông thoáng.",
    "Bạn có thể tiến lên.",
    "Lối đi rộng rãi, không có người.",
    "Phía trước chỉ có sàn nhà trống.",
    "Hành lang vắng, không có ai.",
    "Không có gì đáng chú ý phía trước.",
    # Đã bị chặn từ trước, phải giữ nguyên.
    "Phía trước không có gì.",
    "Bạn có thể đi tiếp.",
    "Phía trước là hành lang trống, không có đồ vật.",
    "Không thấy vật cản nào.",
    "Không có chướng ngại vật.",
    # Biến thể gần.
    "Bạn có thể tiến tới.",
    "Bạn cứ đi thẳng.",
    "Hãy đi thẳng.",
    "Đi thẳng là được.",
    "Không có gì ở phía trước.",
    "Lối đi trống.",
    "Đường trống.",
    "Không có vật cản.",
    "An toàn để đi.",
    "Đường phía trước an toàn, bạn có thể đi tiếp.",
    "Biển ghi lối ra, đường phía trước an toàn.",
    "Có biển báo, an toàn để đi tiếp.",
    "Không thể xác nhận, nhưng đường đi an toàn.",
    "Có một cái ghế. Phía trước trống.",
    "The path is clear, you can walk.",
    # Vòng 2: dấu phẩy trong khoảng chủ ngữ–vị ngữ.
    "Phía trước là hành lang dài, trống.",
    "Phía trước là hành lang dài, không có ai.",
    "Hành lang dài, không có người.",
    "Phía trước là hành lang, không có vật gì.",
    # Vòng 2: các câu từng bị bỏ lọt.
    "Phía trước chỉ có sàn nhà.",
    "Phía trước chỉ có tường trắng và sàn nhà.",
    "Không phát hiện vật cản.",
    "Không có trở ngại.",
    "Không có vấn đề gì.",
    "Không nhìn thấy gì.",
    "Phía trước chẳng có gì.",
    "Phía trước không còn gì.",
    "Không có xe nào phía trước.",
    "Bạn có thể yên tâm đi.",
    "Bạn đi được.",
    "Đi được.",
    "Đi tiếp được.",
    "Cứ đi đi.",
    "Lối đi không bị cản.",
    "Đường không bị cản trở.",
    "Lối đi tự do.",
    "Phòng trống, không có đồ vật.",
    "Căn phòng trống không.",
    "Phía trước là một khoảng không gian rộng.",
    "Trước mặt là khoảng không.",
    "Phía trước ổn.",
    "Phía trước bình thường.",
    "Mọi thứ đều ổn.",
    "Hoàn toàn trống.",
    "Trống trơn.",
    "Không nguy hiểm.",
    "Phía trước là một cái sân trống.",
    "Nothing ahead.",
    "It is safe.",
    "No obstacles.",
    # Vòng 3: lách qua dạng "... nói/đọc/ghi: ..." không có biển/nhãn.
    "Có thể nói: phía trước trống.",
    "Tóm lại có thể nói: đường đi an toàn.",
    "Nhìn chung, có thể nói: lối đi trống.",
    "Có thể nói là: không có gì phía trước.",
    "Người ta nói: đường an toàn.",
    "Tôi nói: phía trước trống.",
    "Nội dung: bạn có thể đi tiếp.",
    "Phía trước có nội dung: đường thông thoáng.",
    "Tôi đọc: “phía trước không có vật cản”",
    "Tôi thấy ghi: phía trước trống.",
    "Trong ảnh có chữ: đi thẳng được.",
    # Chữ trên biển là lời khuyên di chuyển hoặc khẳng định lối đi an toàn.
    "Biển ghi: đi thẳng được.",
    "Bảng hiệu ghi: Phía trước an toàn.",
    # Vòng 3: nơi chốn mở rộng.
    "Trên sàn không có gì.",
    "Có một cái ngõ không có gì.",
    "Có một vỉa hè không có người đi.",
    "Một cái cầu thang không có ai.",
    "Có một cánh cửa, không có gì phía sau.",
)

# Mô tả đồ vật hoặc đọc nội dung chữ: không được chặn.
MUST_ALLOW = (
    "Biển báo ghi: Chú ý an toàn.",
    "Tờ giấy nói về an toàn thực phẩm.",
    "Biển ghi: An toàn là trên hết.",
    "Phía trước có một cái ghế và một cái bàn.",
    "Biển ghi: Lối thoát hiểm.",
    "Chai nước trống đặt trên bàn.",
    "Có một người đang đi về phía bạn.",
    "Ghế trống ở bên trái.",
    "Phía trước có một hành lang dài và một cánh cửa.",
    "Có một chiếc ghế trống bên trái.",
    "Trên bàn có một chai nước trống.",
    "Không thể xác nhận đường đi an toàn.",
    "Lối đi không an toàn.",
    # Vòng 2.
    "Phía trước có một cái ghế trống.",
    "Có một cái bàn và hai cái ghế.",
    "Cửa đang mở.",
    "Đường phố có nhiều xe máy.",
    "Có một người đàn ông mặc áo xanh.",
    "Phía trước có cầu thang đi lên.",
    "Biển ghi: Không phận sự miễn vào.",
    "Biển báo ghi: Cấm hút thuốc.",
    "Trên bàn không có gì.",
    "Có một chai nước, không có nắp.",
    "Phòng trống có một cái bàn.",
    # Vòng 3.
    "Có một cái bảng ghi: Giờ mở cửa 8 giờ.",
    "Biển báo ghi: Cấm vào, có một cái xe đẩy bên cạnh.",
    "Tờ giấy ghi: Đường Lê Lợi.",
    "Chiếc ghế bên trái không có người ngồi.",
    "Có hai người đang nói chuyện.",
)


class TestVQAAnswerGuardrail(unittest.TestCase):
    def setUp(self):
        self.guard = SafetyGuardrail()

    def test_clear_path_claims_are_refused(self):
        for answer in MUST_BLOCK:
            with self.subTest(answer=answer):
                self.assertEqual(self.guard.check_answer(answer), SafetyGuardrail.REFUSAL)

    def test_decomposed_unicode_is_normalized(self):
        for answer in ("Không có vật gì phía trước.", "Đường thông thoáng.", "Bạn có thể tiến lên.",
                       "Không có gì cản đường.", "Trước mặt bạn là một khoảng trống."):
            with self.subTest(answer=answer):
                nfd = unicodedata.normalize("NFD", answer)
                self.assertEqual(self.guard.check_answer(nfd), SafetyGuardrail.REFUSAL)
        self.assertIsNotNone(self.guard.check(unicodedata.normalize("NFD", "Tôi có thể đi tiếp không?")))

    def test_descriptions_and_sign_reading_are_allowed(self):
        for answer in MUST_ALLOW:
            with self.subTest(answer=answer):
                self.assertIsNone(self.guard.check_answer(answer))


class TestVQAAnswerFormatting(unittest.TestCase):
    CONTEXT = {"spatial_objects": [{"name": "ghế", "direction": "LEFT", "proximity": "gần"}]}

    def test_answer_is_spoken_as_is_without_prefix(self):
        self.assertEqual(format_vqa_answer("Ghế màu đỏ."), "Ghế màu đỏ.")
        self.assertEqual(format_vqa_answer("Một chiếc ghế"), "Một chiếc ghế.")
        self.assertEqual(format_vqa_answer("Ghế màu đỏ.", self.CONTEXT),
                         "Ghế màu đỏ.")

    def test_dont_know_answer_uses_detection_fallback(self):
        for text in ("Không xác định rõ từ ảnh.", "Không xác định rõ từ ảnh",
                     "không xác định rõ.", "“Không xác định rõ từ ảnh.”", "KHÔNG XÁC ĐỊNH RÕ!"):
            with self.subTest(text=text):
                self.assertEqual(format_vqa_answer(text, self.CONTEXT),
                                 "Trước mặt bạn phát hiện có: ghế bên trái (gần).")
                self.assertEqual(format_vqa_answer(text), VQA_UNAVAILABLE_ANSWER)

    def test_prompt_requests_at_most_25_words(self):
        self.assertIn("tối đa 25 từ", MLXVLMBackend.SYSTEM_PROMPT)
        self.assertNotIn("tối đa 40 từ", MLXVLMBackend.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
