import unittest
from src.ocr.ocr_service import OCRService


class TestOCRClustering(unittest.TestCase):
    def test_cluster_and_sort_lines_orders_correctly(self):
        # Giả lập 2 dòng văn bản:
        # Dòng 1: "Thành phần:" (trái) và "Paracetamol 500mg" (phải)
        # Dòng 2: "Liều dùng: 1 viên" (dưới)
        # Bị đảo thứ tự phát hiện từ EasyOCR
        items = [
            {
                "top": 102.0, "bottom": 126.0, "left": 220.0, "right": 350.0,
                "center_y": 114.0, "height": 24.0, "text": "Paracetamol 500mg", "conf": 0.9, "box": []
            },
            {
                "top": 98.0, "bottom": 124.0, "left": 40.0, "right": 180.0,
                "center_y": 111.0, "height": 26.0, "text": "Thành phần:", "conf": 0.95, "box": []
            },
            {
                "top": 145.0, "bottom": 170.0, "left": 40.0, "right": 250.0,
                "center_y": 157.5, "height": 25.0, "text": "Liều dùng: 1 viên", "conf": 0.92, "box": []
            },
        ]
        result = OCRService._cluster_and_sort_lines(items)
        # Kỳ vọng: Dòng 1 ghép đúng "Thành phần: Paracetamol 500mg.", dòng 2 tiếp theo "Liều dùng: 1 viên."
        self.assertEqual(result, "Thành phần: Paracetamol 500mg. Liều dùng: 1 viên.")

    def test_cluster_empty_items(self):
        self.assertEqual(OCRService._cluster_and_sort_lines([]), "")


if __name__ == "__main__":
    unittest.main()
