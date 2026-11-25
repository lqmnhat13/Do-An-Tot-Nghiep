import importlib.util
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import cv2
import numpy as np

import time

from src.contracts.request import OCRRequest
from src.ocr.ocr_service import OCRService
from src.ocr.vision_worker import OCRCancelled, VisionUnavailable


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

    @unittest.skipUnless(
        sys.platform == "darwin" and importlib.util.find_spec("Vision") is not None,
        "Apple Vision requires macOS and PyObjC",
    )
    def test_service_reads_with_real_vision_subprocess(self):
        image = np.full((300, 700, 3), 255, dtype=np.uint8)
        cv2.putText(image, "ONE WAY", (45, 170), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 7)
        service = OCRService(engine="vision", languages=["en"], quality_checker=PassQuality(),
                             read_timeout_sec=10.0)
        try:
            result = service.process(OCRRequest(request_id="1", image=image))
        finally:
            service.close()
        self.assertTrue(result.success)
        self.assertIn("ONE WAY", result.text)
        self.assertIsNone(service._reader)  # không phải nạp EasyOCR

    def test_vision_unavailable_uses_easyocr_from_then_on(self):
        FakeVisionClient.script = [VisionUnavailable("No module named 'Vision'")]
        with patch("src.ocr.ocr_service.VisionWorkerClient", FakeVisionClient), \
             patch("easyocr.Reader", return_value=FakeEasyOCR()) as easyocr_reader:
            service = make_service()
            first = service.process(OCRRequest(request_id="1", image=IMAGE))
            second = service.process(OCRRequest(request_id="2", image=IMAGE))
        self.assertIn("XIN CHÀO", first.text)
        self.assertIn("XIN CHÀO", second.text)
        self.assertEqual(service._vision.calls, 1)  # đã tắt Vision, không gọi lại
        easyocr_reader.assert_called_once()

    def test_vision_failure_falls_back_for_that_read_only(self):
        """Treo/lỗi một lần không chuyển hẳn sang EasyOCR: lần sau vẫn thử Vision trước."""
        vision_item = ([[0, 0], [100, 0], [100, 30], [0, 30]], "LỐI RA", 0.95)
        for failure in (TimeoutError("Vision không phản hồi đúng hạn"),
                        RuntimeError("Vision lỗi khi đọc: ValueError: key does not exist")):
            with self.subTest(failure=type(failure).__name__):
                FakeVisionClient.script = [failure, [vision_item]]
                with patch("src.ocr.ocr_service.VisionWorkerClient", FakeVisionClient), \
                     patch("easyocr.Reader", return_value=FakeEasyOCR()) as easyocr_reader:
                    service = make_service()
                    t0 = time.monotonic()
                    first = service.process(OCRRequest(request_id="1", image=IMAGE))
                    self.assertLess(time.monotonic() - t0, 1.0)
                    second = service.process(OCRRequest(request_id="2", image=IMAGE))
                self.assertIn("XIN CHÀO", first.text)
                self.assertIn("LỐI RA", second.text)
                easyocr_reader.assert_called_once()

    def test_cancelled_read_returns_quietly_without_easyocr(self):
        FakeVisionClient.script = [OCRCancelled()]
        with patch("src.ocr.ocr_service.VisionWorkerClient", FakeVisionClient), \
             patch("easyocr.Reader") as easyocr_reader:
            service = make_service()
            result = service.process(OCRRequest(request_id="1", image=IMAGE))
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "cancelled")
        easyocr_reader.assert_not_called()

    def test_warm_up_cancel_close_reach_vision_client(self):
        with patch("src.ocr.ocr_service.VisionWorkerClient", FakeVisionClient):
            service = make_service()
            service.warm_up()
            service.cancel()
            service.close()
        self.assertEqual(service._vision.events, ["warm_up", "cancel", "close"])
        easy = OCRService(engine="easyocr")
        easy.warm_up()
        easy.cancel()
        easy.close()  # không có Vision: không làm gì


class FakeVisionClient:
    script = []

    def __init__(self, languages, **kwargs):
        self.kwargs = kwargs
        self.script = list(FakeVisionClient.script)
        self.calls = 0
        self.events = []
        self.disabled = False

    def readtext(self, image):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, VisionUnavailable):
            self.disabled = True
        if isinstance(item, BaseException):
            raise item
        return item

    def warm_up(self):
        self.events.append("warm_up")

    def cancel(self):
        self.events.append("cancel")

    def close(self):
        self.events.append("close")


class FakeEasyOCR:
    def readtext(self, image):
        return [([[0, 0], [100, 0], [100, 30], [0, 30]], "XIN CHÀO", 0.9)]


class PassQuality:
    def assess(self, image):
        return True, "ok", {}


IMAGE = np.full((300, 700, 3), 255, dtype=np.uint8)


def make_service():
    return OCRService(engine="vision", use_gpu=False, quality_checker=PassQuality())


if __name__ == "__main__":
    unittest.main()
