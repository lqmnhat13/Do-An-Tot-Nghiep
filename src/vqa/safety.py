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
    )

    def check(self, question: str) -> Optional[str]:
        normalized = question.strip().lower()
        if any(keyword in normalized for keyword in self.KEYWORDS):
            return self.REFUSAL
        return None
