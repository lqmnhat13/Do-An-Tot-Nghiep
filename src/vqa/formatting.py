import re
from typing import Any, Dict, List, Optional

# Câu "không biết" mà system prompt yêu cầu model dùng ("Không xác định rõ từ ảnh.");
# so khớp bỏ qua hoa/thường, dấu câu và ngoặc kép.
_DONT_KNOW = re.compile(r"không xác định(?: rõ)?(?: từ (?:hình )?ảnh)?")


def _spatial_details(visual_context: Optional[Dict[str, Any]]) -> List[str]:
    if not visual_context:
        return []
    details = []
    for obj in visual_context.get("spatial_objects", [])[:3]:
        name = obj.get("name", "")
        direction = obj.get("direction", "")
        proximity = obj.get("proximity", "")
        direction_vi = (
            "ở giữa" if direction == "CENTER"
            else "bên trái" if direction == "LEFT"
            else "bên phải"
        )
        details.append(f"{name} {direction_vi} ({proximity})")
    return details


def format_vqa_answer(
    backend_text: str,
    visual_context: Optional[Dict[str, Any]] = None
) -> str:
    """Đọc nguyên câu trả lời của model; khi không có thì dùng danh sách vật nhận diện."""
    details = _spatial_details(visual_context)
    description = backend_text.strip()
    if _DONT_KNOW.fullmatch(" ".join(re.sub(r"[^\w\s]", " ", description.lower()).split())):
        # Không nói "Không xác định rõ" rồi lại liệt kê vật thể: dùng fallback detection.
        description = ""
    if description:
        # Preserve model punctuation; legacy captions still need a final period.
        ending = description.rstrip('\"\u201d\u2019\u0027)')
        suffix = "" if ending.endswith((".", "!", "?", "…")) else "."
        # Chỉ đọc câu trả lời; danh sách vật nhận diện chỉ dùng khi model không trả lời được.
        return f"{description}{suffix}"
    if details:
        return f"Trước mặt bạn phát hiện có: {', '.join(details)}."
    # Không dùng câu kiểu "chưa phát hiện vật thể": người khiếm thị dễ hiểu là phía trước trống.
    return VQA_UNAVAILABLE_ANSWER


VQA_UNAVAILABLE_ANSWER = (
    "Chưa mô tả được khung cảnh lúc này. Vui lòng thử lại và tiếp tục dùng gậy dò đường."
)
