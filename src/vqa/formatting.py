from typing import Any, Dict, List, Optional


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
    """Ghép output backend với grounding, hoặc dùng grounding làm fallback."""
    details = _spatial_details(visual_context)
    description = backend_text.strip()
    if description:
        final_text = f"Khung cảnh: {description}."
        if details:
            final_text += f" Cụ thể có: {', '.join(details)}."
        return final_text
    if details:
        return f"Trước mặt bạn phát hiện có: {', '.join(details)}."
    return "Chưa phát hiện được khung cảnh hoặc vật thể rõ ràng phía trước."
