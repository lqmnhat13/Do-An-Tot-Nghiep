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

    def test_separates_distant_signs_on_same_row(self):
        items = [
            {"top": 100.0, "bottom": 140.0, "left": 30.0, "right": 130.0,
             "center_y": 120.0, "height": 40.0, "text": "GIÁ 2"},
            {"top": 102.0, "bottom": 142.0, "left": 430.0, "right": 610.0,
             "center_y": 122.0, "height": 40.0, "text": "SẢN PHẨM"},
        ]
        self.assertEqual(
            OCRService._cluster_and_sort_lines(items), "GIÁ 2. SẢN PHẨM."
        )

    def test_tall_letters_on_adjacent_rows_do_not_reverse_reading_order(self):
        items = [
            {"top": 488.0, "bottom": 560.0, "left": 990.0, "right": 1170.0,
             "center_y": 524.0, "height": 72.0, "text": "BÁN"},
            {"top": 555.0, "bottom": 615.0, "left": 995.0, "right": 1171.0,
             "center_y": 585.0, "height": 60.0, "text": "CHẠY"},
        ]
        self.assertEqual(OCRService._cluster_and_sort_lines(items), "BÁN. CHẠY.")

    def test_different_font_sizes_on_one_sign_stay_in_reading_order(self):
        items = [
            {"top": 648.0, "bottom": 680.0, "left": 642.0, "right": 700.0,
             "center_y": 664.0, "height": 32.0, "text": "3300"},
            {"top": 655.0, "bottom": 717.0, "left": 485.0, "right": 617.0,
             "center_y": 686.0, "height": 62.0, "text": "Milam"},
        ]
        self.assertEqual(OCRService._cluster_and_sort_lines(items), "Milam 3300.")


if __name__ == "__main__":
    unittest.main()
