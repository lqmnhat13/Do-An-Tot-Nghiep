import threading
import unittest
from unittest.mock import MagicMock

import numpy as np

from scripts.run_vqa_camera import VQACameraSession, build_parser
from src.contracts.request import VQAResult


class TestVQACameraCLI(unittest.TestCase):
    def test_parser_accepts_camera_vqa_options(self):
        args = build_parser().parse_args([
            "--camera", "1",
            "--device", "cpu",
            "--backend", "disabled",
            "--question", "Trong ảnh có gì?",
            "--no-speech",
        ])
        self.assertEqual(args.camera, 1)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.backend, "disabled")
        self.assertEqual(args.question, "Trong ảnh có gì?")
        self.assertTrue(args.no_speech)

    def test_trigger_passes_camera_frame_and_question_to_vqa(self):
        service = MagicMock()
        service.answer.return_value = VQAResult(
            request_id="result", success=True, answer="Có một cái bàn.", latency_sec=0.2
        )
        tts = MagicMock()
        session = VQACameraSession(service, "Trong ảnh có gì?", tts)
        frame = np.ones((4, 5, 3), dtype=np.uint8)

        self.assertTrue(session.trigger(frame))
        session._worker.join(timeout=0.5)

        self.assertFalse(session._worker.is_alive())
        request = service.answer.call_args.args[0]
        self.assertEqual(request.question, "Trong ảnh có gì?")
        np.testing.assert_array_equal(request.image, frame)
        self.assertIsNot(request.image, frame)
        tts.speak.assert_called_once_with("Có một cái bàn.")
        self.assertEqual(session.status, "READY")

    def test_busy_request_is_rejected_and_cancelled_result_is_suppressed(self):
        entered = threading.Event()
        release = threading.Event()
        service = MagicMock()

        def blocking_answer(request):
            entered.set()
            release.wait(timeout=1.0)
            return VQAResult(
                request_id=request.request_id,
                success=True,
                answer="Kết quả đã bị hủy",
                latency_sec=0.3,
            )

        service.answer.side_effect = blocking_answer
        tts = MagicMock()
        session = VQACameraSession(service, "Mô tả ảnh", tts)
        frame = np.zeros((3, 3, 3), dtype=np.uint8)

        self.assertTrue(session.trigger(frame))
        self.assertTrue(entered.wait(timeout=0.5))
        self.assertFalse(session.trigger(frame))
        session.cancel()
        release.set()
        session._worker.join(timeout=0.5)

        self.assertFalse(session._worker.is_alive())
        tts.stop.assert_called_once()
        tts.speak.assert_not_called()
        self.assertNotEqual(session.last_answer, "Kết quả đã bị hủy")


if __name__ == "__main__":
    unittest.main()
