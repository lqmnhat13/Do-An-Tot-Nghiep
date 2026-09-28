import threading
import unittest
from unittest.mock import MagicMock

import numpy as np

from scripts.run_ocr_camera import OCRCameraSession, build_parser
from src.contracts.request import OCRResult


class TestOCRCameraCLI(unittest.TestCase):
    def test_parser_accepts_camera_ocr_options(self):
        args = build_parser().parse_args([
            "--camera", "1",
            "--device", "cpu",
            "--min-confidence", "0.4",
            "--no-speech",
        ])
        self.assertEqual(args.camera, 1)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.min_confidence, 0.4)
        self.assertTrue(args.no_speech)

    def test_parser_accepts_image_option(self):
        args = build_parser().parse_args([
            "--image", "test.jpg",
            "--no-speech",
        ])
        self.assertEqual(args.image, "test.jpg")
        self.assertTrue(args.no_speech)

    def test_trigger_passes_camera_frame_to_ocr(self):
        service = MagicMock()
        service.process.return_value = OCRResult(
            request_id="result",
            success=True,
            text="Hà Nội",
            quality_status="GOOD",
            latency_sec=0.15,
        )
        tts = MagicMock()
        session = OCRCameraSession(service, tts)
        frame = np.ones((4, 5, 3), dtype=np.uint8)

        self.assertTrue(session.trigger(frame))
        session._worker.join(timeout=0.5)

        self.assertFalse(session._worker.is_alive())
        request = service.process.call_args.args[0]
        np.testing.assert_array_equal(request.image, frame)
        self.assertIsNot(request.image, frame)
        tts.speak.assert_called_once_with("Hà Nội")
        self.assertEqual(session.status, "READY")
        self.assertEqual(session.last_text, "Hà Nội")

    def test_busy_request_is_rejected_and_cancelled_result_is_suppressed(self):
        entered = threading.Event()
        release = threading.Event()
        service = MagicMock()

        def blocking_process(request):
            entered.set()
            release.wait(timeout=1.0)
            return OCRResult(
                request_id=request.request_id,
                success=True,
                text="Kết quả đã bị hủy",
                quality_status="GOOD",
                latency_sec=0.3,
            )

        service.process.side_effect = blocking_process
        tts = MagicMock()
        session = OCRCameraSession(service, tts)
        frame = np.zeros((4, 5, 3), dtype=np.uint8)

        self.assertTrue(session.trigger(frame))
        entered.wait(timeout=0.5)
        self.assertFalse(session.trigger(frame))  # Bận -> reject

        session.cancel()
        self.assertEqual(session.status, "CANCELLED")
        tts.stop.assert_called_once()

        release.set()
        session._worker.join(timeout=0.5)

        # Vì đã bị cancel, không được gọi TTS speak
        tts.speak.assert_not_called()
        self.assertEqual(session.last_text, "")


if __name__ == "__main__":
    unittest.main()
