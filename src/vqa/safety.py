import re
from typing import Optional


class SafetyGuardrail:
    """Guardrail độc lập, luôn chạy trước backend model."""

    REFUSAL = (
        "Hệ thống không thể xác nhận đường đi có an toàn hay không. "
        "Xin hãy cẩn thận dùng gậy dẫn đường và kiểm tra xung quanh."
    )
    # Phân loại theo ý định thay vì khớp từ khoá đơn lẻ, vì từ khoá đơn ("đi thẳng",
    # "qua đường", "an toàn") chặn nhầm câu hỏi đọc biển báo hoặc đếm người
    # (9/3545 câu OpenViVQA bị chặn nhầm) mà lại bỏ lọt "Tôi có thể đi tiếp không?".
    _MOVE = r"(?:đi|bước|qua đường|sang đường|băng qua|di chuyển|tiến lên|tiến tới|rẽ)"
    _ASK = r"(?:có thể|được không|được chứ|được chưa|có được|có sao không|có ổn không|nên|có nên|liệu)"
    MOVE = re.compile(rf"\b{_MOVE}\b")
    # Xin phép di chuyển: từ hỏi phép đứng sát động từ di chuyển ("có thể đi tiếp",
    # "đi về phía trước được không"), không tính câu hỏi chỉ đường/đếm người.
    MOVE_PERMISSION = re.compile(
        rf"\b{_ASK}\s+(?:\w+\s+){{0,2}}{_MOVE}\b|\b{_MOVE}(?:\s+\w+){{0,4}}\s+{_ASK}\b")
    WH_QUESTION = re.compile(
        r"\b(?:nào|đâu|gì|ai|mấy|bao nhiêu|bao xa|tại sao|vì sao|thế nào|như thế nào)\b")
    FIRST_PERSON = re.compile(r"\b(?:tôi|mình|tớ|tao)\b")
    THIRD_PERSON = re.compile(r"\b(?:người|họ|khách|xe)\b")
    HAZARD = re.compile(
        r"\b(?:an toàn|vật cản|chướng ngại|nguy hiểm|va chạm|va vào|đâm vào|vấp ngã|té ngã|"
        r"ngã(?! (?:ba|tư|năm|sáu|bảy)))\b")
    # Hỏi sự có mặt của bậc/hố/cầu thang dạng "có ... không" (không chặn câu mô tả "bậc thang dẫn đến đâu").
    HAZARD_PRESENCE = re.compile(
        r"\bcó\b[^?]*\b(?:bậc|cầu thang|hố|ổ gà)\b[^?]*\bkhông\b"
        r"|\bcó\b[^?]*\bxe\b[^?]*\b(?:tới|đến|chạy|lại gần|đi tới)\b[^?]*\bkhông\b")
    SAFETY_TOPIC = re.compile(r"\ban toàn (?:giao thông|thực phẩm|lao động|vệ sinh|thông tin|điện|cháy nổ)\b")
    READING = re.compile(
        r"\b(?:ghi|viết|đọc|chữ|nội dung|tuyên truyền|khẩu hiệu|biển (?:báo|hiệu)|bảng (?:báo|hiệu)|"
        r"tờ giấy|nhãn|thông báo)\b")
    ENGLISH = re.compile(
        r"\b(?:safe to|is it safe|can i (?:walk|cross|go|move|proceed)|"
        r"(?:any|is there an?) (?:obstacle|hazard|danger))")

    def check(self, question: str) -> Optional[str]:
        normalized = " ".join(question.strip().lower().split())
        if self.ENGLISH.search(normalized):
            return self.REFUSAL
        reading = bool(self.READING.search(normalized))
        first_person = bool(self.FIRST_PERSON.search(normalized))
        # 1. Người dùng hỏi bản thân có đi/bước/qua đường được không.
        if self.MOVE.search(normalized) and first_person:
            return self.REFUSAL
        if (self.MOVE_PERMISSION.search(normalized) and not reading
                and not self.WH_QUESTION.search(normalized)
                and not self.THIRD_PERSON.search(normalized)):
            return self.REFUSAL
        # 2. Hỏi có vật cản/nguy hiểm/an toàn không (trừ câu hỏi nội dung chữ hoặc chủ đề chung).
        hazard = self.HAZARD.search(normalized) or self.HAZARD_PRESENCE.search(normalized)
        if hazard and (first_person or not (reading or self.SAFETY_TOPIC.search(normalized))):
            return self.REFUSAL
        return None

    def check_answer(self, answer: str) -> Optional[str]:
        normalized = answer.strip().lower()
        for sentence in re.split(r"[.!?]", normalized):
            if re.search(
                r"\b(?:có thể|nên|hãy|cứ|được)\s+"
                r"(?:đi|bước|qua đường|di chuyển|tiếp tục)\b", sentence
            ) or re.search(
                r"\bkhông (?:có|thấy) (?:vật cản|chướng ngại|nguy hiểm)\b",
                sentence
            ) or re.search(
                r"\b(?:safe to (?:walk|cross|proceed)|you can (?:walk|cross|proceed)|"
                r"no obstacles? ahead)\b", sentence
            ):
                return self.REFUSAL
            if "an toàn" in sentence and not re.search(
                r"\b(?:không|chưa|chẳng)\s+an toàn\b|"
                r"\bkhông thể xác nhận[^.!?]{0,30}\ban toàn\b",
                sentence
            ):
                return self.REFUSAL
        return None
