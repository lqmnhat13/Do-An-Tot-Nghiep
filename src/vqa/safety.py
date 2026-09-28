import re
from typing import Optional


class SafetyGuardrail:
    """Guardrail độc lập, luôn chạy trước backend model."""

    REFUSAL = (
        "Hệ thống không thể xác nhận đường đi có an toàn hay không. "
        "Xin hãy cẩn thận dùng gậy dẫn đường và kiểm tra xung quanh."
    )
    KEYWORDS = (
        "an toàn", "đi được", "bước tiếp", "có va chạm",
        "đi thẳng", "qua đường", "có nguy hiểm",
        "safe to walk", "safe to cross", "can i walk", "can i cross",
    )

    def check(self, question: str) -> Optional[str]:
        normalized = question.strip().lower()
        if any(keyword in normalized for keyword in self.KEYWORDS):
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
