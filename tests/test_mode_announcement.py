"""Thông báo chế độ ngắn, không bị cảnh báo vật cản cắt ngang; phím bị từ chối luôn được nói ra."""
import threading
import time
import unittest
from unittest.mock import MagicMock

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine
from src.contracts.audio import AudioPriority, AudioTask
from src.contracts.risk import DataQuality, Direction, RiskAssessment, RiskLevel
from src.fusion.alert_aggregator import AlertAggregator
from src.runtime.system_coordinator import (
    BUSY_TEXT, MODE_ANNOUNCEMENTS, STOPPING_TEXT, SystemCoordinator)


def high(name="người"):
    now = time.monotonic()
    return RiskAssessment(1, name, Direction.CENTER, RiskLevel.HIGH, DataQuality.VALID,
                          .9, "ở rất gần", "test", now, now + 1.0)


class TestModeAnnouncement(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)  # không start: câu nằm trong hàng chờ
        self.fsm = MagicMock()
        self.fsm.can_trigger_alert.return_value = True
        self.system = SystemCoordinator(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                        self.fsm, AlertAggregator(self.fsm), self.audio,
                                        MagicMock(), MagicMock())

    def tearDown(self):
        self.system.stop()

    def queued(self):
        return [e.task for e in list(self.audio._queue.queue)]

    def test_announcements_are_short_and_not_interruptible(self):
        self.system.set_on_demand_active(True, "OCR")
        self.assertEqual([t.text for t in self.queued()], [MODE_ANNOUNCEMENTS["OCR"]])
        self.audio.interrupt()
        self.system.enter_warning_mode()
        tasks = self.queued()
        self.assertEqual([t.text for t in tasks], [MODE_ANNOUNCEMENTS["WARNING"]])
        self.assertFalse(tasks[0].interruptible)
        for text in MODE_ANNOUNCEMENTS.values():
            self.assertLessEqual(len(text.split()), 7)

    def test_alert_waits_for_warning_announcement_without_consuming_cooldown(self):
        self.system.set_on_demand_active(True, "OCR")
        self.audio.interrupt()
        self.system.enter_warning_mode()
        self.system._post_safety_alert([high()], DataQuality.VALID)
        self.fsm.mark_alert_triggered.assert_not_called()
        self.assertEqual(len(self.queued()), 1)
        # Thông báo phát xong (rời hàng chờ) -> cảnh báo được tạo ngay ở frame kế tiếp.
        self.audio._queue.get_nowait()
        self.system._post_safety_alert([high()], DataQuality.VALID)
        self.fsm.mark_alert_triggered.assert_called_once()
        self.assertEqual([t.text for t in self.queued()], ["người phía trước"])

    def test_alert_gate_is_bounded_if_announcement_never_finishes(self):
        self.system.MODE_ANNOUNCE_MAX_SEC = 0.0
        self.system.set_on_demand_active(True, "OCR")
        self.audio.interrupt()
        self.system.enter_warning_mode()
        self.system._post_safety_alert([high()], DataQuality.VALID)
        self.fsm.mark_alert_triggered.assert_called_once()

    def test_playing_announcement_not_cut_by_high_alert(self):
        self.audio._current_task = AudioTask(AudioPriority.ON_DEMAND, MODE_ANNOUNCEMENTS["WARNING"],
                                             interruptible=False)
        self.audio._current_entry = object()
        self.assertTrue(self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, "người phía trước")))
        self.engine.stop.assert_not_called()

    def test_is_pending_or_playing(self):
        task = AudioTask(AudioPriority.ON_DEMAND, "x")
        self.assertFalse(self.audio.is_pending_or_playing(task))
        self.audio.post_task(task)
        self.assertTrue(self.audio.is_pending_or_playing(task))
        self.audio._queue.get_nowait()
        self.audio._current_task = task
        self.assertTrue(self.audio.is_pending_or_playing(task))


class TestRejectedKeyFeedback(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)
        self.release = threading.Event()
        self.entered = threading.Event()
        self.ocr = MagicMock()

        def slow_ocr(req):
            self.entered.set()
            self.release.wait(timeout=5)
            return MagicMock(text="Chữ", latency_sec=.1)
        self.ocr.process.side_effect = slow_ocr
        camera = MagicMock()
        from src.contracts.frame_packet import FramePacket
        import numpy as np
        camera.get_latest_frame.side_effect = lambda: FramePacket(
            1, "t", np.zeros((8, 8, 3), np.uint8), (8, 8), time.monotonic())
        self.system = SystemCoordinator(camera, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                        MagicMock(), AlertAggregator(MagicMock()), self.audio,
                                        self.ocr, MagicMock())

    def tearDown(self):
        self.release.set()
        if self.system._on_demand_thread is not None:
            self.system._on_demand_thread.join(timeout=2)
        self.system.stop()

    def texts(self):
        return [e.task.text for e in list(self.audio._queue.queue)]

    def test_space_while_task_running_says_busy(self):
        self.assertTrue(self.system.trigger_ocr())
        self.assertTrue(self.entered.wait(timeout=2))
        self.assertFalse(self.system.trigger_ocr())
        self.assertIn(BUSY_TEXT, self.texts())

    def test_key_after_c_while_cancelled_worker_still_running_says_stopping(self):
        self.assertTrue(self.system.trigger_ocr())
        self.assertTrue(self.entered.wait(timeout=2))
        self.system.enter_warning_mode()
        self.assertFalse(self.system.trigger_ocr())
        self.assertIn(STOPPING_TEXT, self.texts())
        self.assertFalse(self.system.trigger_vqa(None))
        self.assertEqual(self.system.get_runtime_snapshot()["selected_mode"], "WARNING")

    def test_rejection_while_listening_is_silent(self):
        self.assertTrue(self.system.trigger_ocr())
        self.assertTrue(self.entered.wait(timeout=2))
        with self.system._state_lock:
            self.system._listening = True
        self.audio.interrupt()
        self.assertFalse(self.system.trigger_ocr())
        self.assertEqual(self.texts(), [])


if __name__ == "__main__":
    unittest.main()
