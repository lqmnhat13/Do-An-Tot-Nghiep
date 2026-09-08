from typing import Set, Dict, Optional, List

# 21 lớp COCO trong nhà theo mục 1.1 của Kế hoạch
DEFAULT_ENABLED_CLASSES = {
    # Người và thú nuôi
    "person", "cat", "dog",
    # Nội thất
    "chair", "couch", "bed", "dining table",
    # Thiết bị
    "tv", "laptop", "keyboard", "mouse", "cell phone",
    # Đồ gia dụng
    "bottle", "cup", "bowl", "book", "clock", "vase",
    # Nhà bếp và phòng tắm
    "microwave", "oven", "toaster", "sink", "refrigerator", "toilet"
}

# Các đối tượng có kích thước lớn hoặc di chuyển, có nguy cơ gây va chạm khi đi lại
DEFAULT_ALERT_CLASSES = {
    "person", "cat", "dog",
    "chair", "couch", "bed", "dining table",
    "sink", "refrigerator", "toilet"
}

VIETNAMESE_NAMES = {
    "person": "người",
    "cat": "mèo",
    "dog": "chó",
    "chair": "ghế",
    "couch": "ghế sofa",
    "bed": "giường",
    "dining table": "bàn ăn",
    "tv": "ti vi",
    "laptop": "máy tính",
    "keyboard": "bàn phím",
    "mouse": "chuột máy tính",
    "cell phone": "điện thoại",
    "bottle": "chai nước",
    "cup": "cốc",
    "bowl": "bát",
    "book": "sách",
    "clock": "đồng hồ",
    "vase": "bình hoa",
    "microwave": "lò vi sóng",
    "oven": "lò nướng",
    "toaster": "máy nướng bánh",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
    "toilet": "bồn cầu"
}

class ClassFilter:
    """
    Bộ lọc lớp đối tượng phát hiện theo cấu hình của dự án.
    Tách biệt rõ ràng giữa enabled_classes (cho phép hiển thị/mô tả)
    và alert_classes (kích hoạt cảnh báo nguy cơ va chạm).
    """

    def __init__(
        self,
        enabled_classes: Optional[List[str]] = None,
        alert_classes: Optional[List[str]] = None,
        custom_vi_names: Optional[Dict[str, str]] = None
    ):
        self.enabled_classes: Set[str] = set(enabled_classes) if enabled_classes else set(DEFAULT_ENABLED_CLASSES)
        self.alert_classes: Set[str] = set(alert_classes) if alert_classes else set(DEFAULT_ALERT_CLASSES)
        self.vi_names: Dict[str, str] = dict(VIETNAMESE_NAMES)
        if custom_vi_names:
            self.vi_names.update(custom_vi_names)

    def is_enabled(self, class_name: str) -> bool:
        """Kiểm tra đối tượng có thuộc tập được phép xử lý không."""
        return class_name.lower() in self.enabled_classes

    def is_alert_candidate(self, class_name: str) -> bool:
        """Kiểm tra đối tượng có nằm trong nhóm có thể phát cảnh báo không."""
        return class_name.lower() in self.alert_classes

    def get_vietnamese_name(self, class_name: str) -> str:
        """Lấy tên tiếng Việt tự nhiên của lớp đối tượng."""
        return self.vi_names.get(class_name.lower(), class_name)
