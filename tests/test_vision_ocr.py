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

    def test_vision_runtime_failure_falls_back_to_easyocr(self):
        from src.contracts.request import OCRRequest

        class BrokenVision:
            def __init__(self, languages):
                pass

            def readtext(self, image):
                raise ValueError("NSInvalidArgumentException - key does not exist")

        class FakeEasyOCR:
            def readtext(self, image):
                return [([[0, 0], [100, 0], [100, 30], [0, 30]], "XIN CHÀO", 0.9)]

        class PassQuality:
            def assess(self, image):
                return True, "ok", {}

        image = np.full((300, 700, 3), 255, dtype=np.uint8)
        with patch("src.ocr.vision_engine.VisionOCRReader", BrokenVision), \
             patch("easyocr.Reader", return_value=FakeEasyOCR()) as easyocr_reader:
            service = OCRService(engine="vision", use_gpu=False, quality_checker=PassQuality())
            first = service.process(OCRRequest(request_id="1", image=image))
            second = service.process(OCRRequest(request_id="2", image=image))
        self.assertTrue(first.success)
        self.assertIn("XIN CHÀO", first.text)
        self.assertIn("XIN CHÀO", second.text)
        easyocr_reader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
