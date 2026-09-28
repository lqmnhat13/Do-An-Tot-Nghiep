import importlib.util
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import cv2
import numpy as np

from src.ocr.ocr_service import OCRService


class TestVisionOCR(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform == "darwin" and importlib.util.find_spec("Vision") is not None,
        "Apple Vision requires macOS and PyObjC",
    )
    def test_reads_text_from_bgr_frame_in_worker_thread(self):
        from src.ocr.vision_engine import VisionOCRReader

        image = np.full((300, 700, 3), 255, dtype=np.uint8)
        cv2.putText(image, "ONE WAY", (45, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 0, 0), 7)
        reader = VisionOCRReader(["en"])
        with ThreadPoolExecutor(max_workers=1) as executor:
            items = executor.submit(reader.readtext, image).result(timeout=10)
        self.assertTrue(any("ONE WAY" in text for _, text, _ in items))
        for box, _, confidence in items:
            self.assertTrue(0 <= confidence <= 1)
            self.assertTrue(all(0 <= x <= 700 and 0 <= y <= 300 for x, y in box))

    def test_vision_load_failure_uses_easyocr(self):
        reader = object()
        with patch("src.ocr.vision_engine.VisionOCRReader",
                   side_effect=RuntimeError("unavailable")), \
             patch("easyocr.Reader", return_value=reader) as easyocr_reader:
            service = OCRService(engine="vision", use_gpu=False)
            service._ensure_reader()
        self.assertIs(service._reader, reader)
        easyocr_reader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
