"""Test cho các lỗi phát hiện khi kiểm thử nghiệm thu trên dữ liệu thật (2026-11-09).

1. Phím Q dùng câu hỏi gây ảo giác ("Mô tả khung cảnh phía trước").
2. Câu dự phòng VQA nghe như "phía trước trống"; validator loại số kiểu Việt Nam
   (25.000 đồng, 12.05.2027); câu trả lời tiếng Anh; câu khẳng định lối đi trống lọt qua.
3. Mất camera / depth treo không được báo bằng giọng nói; camera thật bị thay bằng dummy.
4. OCR (Apple Vision) có thể treo vô hạn và khoá luôn VQA.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine
from src.camera.bounded_buffer import BoundedBuffer
from src.camera.camera_manager import CameraManager
from src.contracts.audio import AudioPriority
from src.contracts.frame_packet import FramePacket
from src.contracts.request import OCRRequest
from src.fusion.alert_aggregator import AlertAggregator
from src.ocr.ocr_service import OCRService
from src.runtime.system_coordinator import SystemCoordinator
from src.ui.app_runner import AppRunner
from src.vqa.backend import MLXVLMBackend
from src.vqa.formatting import format_vqa_answer
from src.vqa.safety import SafetyGuardrail


# ---------------------------------------------------------------- 1. Câu hỏi phím Q
class TestVQAKeyQuestion(unittest.TestCase):
    def test_q_key_asks_the_evaluated_question(self):
        coordinator = MagicMock()
        runner = AppRunner(coordinator, hud_renderer=MagicMock(), enable_gui=True)
        keys = iter([ord('q'), 27])
        with patch("src.ui.app_runner.cv2") as cv2:
            cv2.waitKey.side_effect = lambda _: next(keys)
            runner.run()
        coordinator.trigger_vqa.assert_called_once_with("Phía trước có gì?")


# ---------------------------------------------------------------- 2. VQA an toàn và tiếng Việt
class TestVQAAnswerHandling(unittest.TestCase):
    def test_vietnamese_numbers_are_not_sentence_breaks(self):
        for text in ("Giá là 25.000 đồng.", "Hộp sữa hết hạn ngày 12.05.2027.",
                     "Khoai tây chiên giá 1.29 đô la.", "Thuốc 2,5 mg mỗi viên."):
            with self.subTest(text=text):
                self.assertEqual(MLXVLMBackend._validated_answer(text, 128), text)

    def test_four_real_sentences_are_still_rejected(self):
        with self.assertRaises(RuntimeError):
            MLXVLMBackend._validated_answer("Có ghế. Có bàn. Có tủ. Có đèn.", 128)

    def test_english_answer_is_rejected(self):
        with self.assertRaises(RuntimeError):
            MLXVLMBackend._validated_answer(
                "Bathroom with a sink, a towel hanging on a rack, and a mirror on the wall.", 128)

    def test_vietnamese_answer_with_brand_name_is_kept(self):
        text = "Đây là chai nước Coca Cola."
        self.assertEqual(MLXVLMBackend._validated_answer(text, 128), text)

    def test_fallback_without_answer_never_suggests_empty_path(self):
        text = format_vqa_answer("", None).lower()
        self.assertNotIn("chưa phát hiện", text)
        self.assertNotIn("không có", text)
        self.assertIn("thử lại", text)

    def test_answers_claiming_clear_path_are_refused(self):
        guard = SafetyGuardrail()
        for answer in ("Lối đi thông thoáng.", "Phía trước trống, không có gì.",
                       "Hành lang phía trước trống trải.", "The path is clear."):
            with self.subTest(answer=answer):
                self.assertIsNotNone(guard.check_answer(answer))

    def test_empty_objects_are_not_path_claims(self):
        guard = SafetyGuardrail()
        for answer in ("Có một chiếc ghế trống bên trái.", "Trên bàn có một chai nước trống."):
            with self.subTest(answer=answer):
                self.assertIsNone(guard.check_answer(answer))


# ---------------------------------------------------------------- 3. Hỏng phải được báo
class _FailingCapture:
    def __init__(self, *args, **kwargs):
        pass

    def isOpened(self):
        return False

    def read(self):
        return False, None

    def set(self, *args):
        pass

    def release(self):
        pass


class TestCameraLossIsNotHidden(unittest.TestCase):
    def test_real_camera_never_falls_back_to_synthetic_frames(self):
        with patch("src.camera.camera_manager.cv2.VideoCapture", _FailingCapture):
            camera = CameraManager(source=0, reconnect_delay_sec=0.01, max_reconnect_attempts=2)
            buffer = BoundedBuffer(2)
            camera.subscribe(buffer)
            camera.start()
            try:
                time.sleep(0.3)
                self.assertFalse(camera._is_dummy)
                self.assertFalse(camera.is_alive())
                self.assertIsNone(buffer.get(timeout=0.05))
            finally:
                camera.stop()


class TestHealthAnnouncements(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)
        self.camera = MagicMock()
        self.camera.is_alive.return_value = True
        self.depth = MagicMock()
        self.depth.load_error = None
        self.system = SystemCoordinator(self.camera, MagicMock(), MagicMock(), self.depth,
                                        MagicMock(), MagicMock(), AlertAggregator(MagicMock()),
                                        self.audio, MagicMock(), MagicMock())
        self.t0 = 1000.0
        self.system._started_at = self.t0
        self.posted = []
        original = self.audio.post_task

        def record(task):
            self.posted.append(task)
            return original(task)
        self.audio.post_task = record

    def tearDown(self):
        self.system.stop()

    def fresh_frame(self, now):
        self.camera.get_latest_frame.return_value = FramePacket(
            1, "cam", np.zeros((8, 8, 3), np.uint8), (8, 8), now)

    def healthy_at(self, now):
        self.fresh_frame(now)
        self.system.watchdog._heartbeats = {"detection_worker": now, "depth_worker": now}
        self.system._last_valid_pair_time = now

    def status_texts(self):
        return [t.text for t in self.posted if t.priority == AudioPriority.SYSTEM_STATUS]

    def test_no_announcement_while_healthy(self):
        now = self.t0 + 20
        with patch("src.runtime.watchdog.time.monotonic", return_value=now):
            self.healthy_at(now)
            self.assertIsNone(self.system._check_health(now))
        self.assertEqual(self.status_texts(), [])

    def test_camera_loss_is_spoken_and_repeated(self):
        now = self.t0 + 20
        self.camera.is_alive.return_value = False
        self.fresh_frame(now - 5)
        self.assertEqual(self.system._check_health(now), "camera")
        self.system._check_health(now + 1)          # không lặp ngay
        self.system._check_health(now + 11)         # nhắc lại sau khoảng lặp
        texts = self.status_texts()
        self.assertEqual(len(texts), 2)
        self.assertIn("camera", texts[0].lower())
        self.assertIn("tạm ngừng", texts[0].lower())

    def test_camera_loss_is_spoken_even_in_ocr_mode(self):
        self.system.set_on_demand_active(True, "OCR")
        self.system._finish_on_demand(self.system._active_on_demand_token)
        now = self.t0 + 20
        self.camera.get_latest_frame.return_value = None
        self.system._check_health(now)
        self.assertEqual(len(self.status_texts()), 1)

    def test_stale_depth_is_spoken_in_warning_mode(self):
        now = self.t0 + 20
        with patch("src.runtime.watchdog.time.monotonic", return_value=now):
            self.healthy_at(now)
            self.system._last_valid_pair_time = now - 5
            self.assertEqual(self.system._check_health(now), "depth")
        self.assertIn("độ sâu", self.status_texts()[0].lower())

    def test_stalled_depth_worker_is_spoken(self):
        now = self.t0 + 20
        with patch("src.runtime.watchdog.time.monotonic", return_value=now):
            self.healthy_at(now)
            self.system.watchdog._heartbeats["depth_worker"] = now - 10
            self.assertEqual(self.system._check_health(now), "depth")

    def test_depth_model_load_failure_is_spoken(self):
        now = self.t0 + 20
        self.depth.load_error = "Không nạp được Depth Anything"
        with patch("src.runtime.watchdog.time.monotonic", return_value=now):
            self.healthy_at(now)
            self.assertEqual(self.system._check_health(now), "depth")

    def test_recovery_is_spoken_once(self):
        now = self.t0 + 20
        self.camera.is_alive.return_value = False
        self.camera.get_latest_frame.return_value = None
        self.system._check_health(now)
        self.camera.is_alive.return_value = True
        with patch("src.runtime.watchdog.time.monotonic", return_value=now + 2):
            self.healthy_at(now + 2)
            self.system._check_health(now + 2)
            self.system._check_health(now + 3)
        texts = self.status_texts()
        self.assertEqual(len(texts), 2)
        self.assertIn("trở lại", texts[1])

    def test_startup_grace_period(self):
        self.camera.get_latest_frame.return_value = None
        self.assertIsNone(self.system._check_health(self.t0 + 1))
        self.assertEqual(self.status_texts(), [])

    def test_health_thread_runs_and_stops(self):
        self.system.HEALTH_CHECK_INTERVAL_SEC = 0.01
        with patch.object(self.system, "_check_health") as check:
            self.system.start()
            time.sleep(0.15)
            self.system.stop()
            self.assertGreater(check.call_count, 2)
        self.assertFalse(self.system._health_thread.is_alive())


# ---------------------------------------------------------------- 4. OCR không được treo vô hạn
class _HangingVision:
    release = threading.Event()

    def __init__(self, languages):
        pass

    def readtext(self, image):
        self.release.wait(timeout=5)
        return []


class _FakeEasyOCR:
    def readtext(self, image):
        return [([[0, 0], [100, 0], [100, 30], [0, 30]], "LỐI THOÁT HIỂM", 0.9)]


class _PassQuality:
    def assess(self, image):
        return True, "ok", {}


class TestOCRTimeouts(unittest.TestCase):
    def test_hanging_vision_falls_back_to_easyocr_within_timeout(self):
        image = np.full((300, 700, 3), 255, np.uint8)
        try:
            with patch("src.ocr.vision_engine.VisionOCRReader", _HangingVision), \
                 patch("easyocr.Reader", return_value=_FakeEasyOCR()):
                service = OCRService(engine="vision", use_gpu=False, quality_checker=_PassQuality(),
                                     read_timeout_sec=0.2)
                t0 = time.monotonic()
                result = service.process(OCRRequest(request_id="1", image=image))
            self.assertLess(time.monotonic() - t0, 2.0)
            self.assertTrue(result.success)
            self.assertIn("LỐI THOÁT HIỂM", result.text)
        finally:
            _HangingVision.release.set()

    def test_hung_on_demand_task_is_reported_and_releases_worker(self):
        engine = MagicMock(spec=TTSEngine)
        engine.is_speaking.return_value = False
        audio = AudioCoordinator(engine)
        audio.wait_until_idle = MagicMock(return_value=True)
        camera = MagicMock()
        camera.get_latest_frame.return_value = FramePacket(
            1, "cam", np.zeros((8, 8, 3), np.uint8), (8, 8), time.monotonic())
        ocr = MagicMock()
        hang = threading.Event()
        ocr.process.side_effect = lambda req: hang.wait(timeout=5)
        system = SystemCoordinator(camera, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                   MagicMock(), AlertAggregator(MagicMock()), audio, ocr, MagicMock())
        system.OCR_TIMEOUT_SEC = 0.2
        posted = []
        original = audio.post_task
        audio.post_task = lambda task: posted.append(task.text) or original(task)
        try:
            self.assertTrue(system.trigger_ocr())
            system._on_demand_thread.join(timeout=2)
            self.assertFalse(system._on_demand_thread.is_alive())
            self.assertTrue(any("quá thời gian" in t for t in posted))
            self.assertTrue(system.trigger_vqa())
        finally:
            hang.set()
            system.stop()


if __name__ == "__main__":
    unittest.main()
