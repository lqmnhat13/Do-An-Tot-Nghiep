import threading
import unittest
from unittest.mock import MagicMock, patch

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine
from src.contracts.audio import AudioPriority, AudioTask


class TestAudioModeGating(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)

    def tearDown(self):
        self.audio.stop()
        if self.audio._thread:
            self.assertFalse(self.audio._thread.is_alive())

    def test_muted_safety_is_rejected_and_system_status_remains(self):
        self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, 'Cảnh báo đã chờ'))
        self.audio.set_safety_alerts_enabled(False)
        self.assertTrue(self.audio._queue.empty())
        self.assertFalse(self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, 'Cảnh báo mới')))
        self.assertTrue(self.audio.post_task(AudioTask(AudioPriority.SYSTEM_STATUS, 'Lỗi camera')))
        self.audio.start()
        self.assertTrue(self.audio.wait_until_idle(timeout=1))
        self.engine.speak.assert_called_once_with('Lỗi camera')

    def test_interrupt_invalidates_dequeued_task_and_preserves_new_count(self):
        dequeued, release, spoken = threading.Event(), threading.Event(), threading.Event()
        original = self.audio._play_entry
        def blocked(entry):
            if entry.task.text == 'Cảnh báo cũ':
                dequeued.set()
                release.wait(timeout=2)
            original(entry)
        self.engine.speak.side_effect = lambda text: spoken.set()
        with patch.object(self.audio, '_play_entry', side_effect=blocked):
            self.audio.start()
            try:
                self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, 'Cảnh báo cũ', sound_file='old.wav'))
                self.assertTrue(dequeued.wait(timeout=1))
                self.audio.interrupt()
                self.audio.post_task(AudioTask(AudioPriority.ON_DEMAND, 'Kết quả mới'))
                self.assertEqual(self.audio._pending_tasks_count, 1)
                release.set()
                self.assertTrue(spoken.wait(timeout=1))
                self.assertTrue(self.audio.wait_until_idle(timeout=1))
            finally:
                release.set()
        self.engine.play_sound.assert_not_called()
        self.engine.speak.assert_called_once_with('Kết quả mới')
        self.assertEqual(self.audio._pending_tasks_count, 0)

    def test_disabling_and_reenabling_cannot_revive_dequeued_high(self):
        dequeued, release = threading.Event(), threading.Event()
        original = self.audio._play_entry
        def blocked(entry):
            dequeued.set()
            release.wait(timeout=2)
            original(entry)
        with patch.object(self.audio, '_play_entry', side_effect=blocked):
            self.audio.start()
            try:
                self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, 'Không phát lại'))
                self.assertTrue(dequeued.wait(timeout=1))
                self.audio.set_safety_alerts_enabled(False)
                self.audio.set_safety_alerts_enabled(True)
                release.set()
                self.assertTrue(self.audio.wait_until_idle(timeout=1))
            finally:
                release.set()
        self.engine.speak.assert_not_called()

    def test_mode_switch_during_chime_cancels_speech(self):
        chiming, release = threading.Event(), threading.Event()
        def wait_for_chime(*args, **kwargs):
            chiming.set()
            release.wait(timeout=2)
        self.engine.wait_until_done.side_effect = wait_for_chime
        self.audio.start()
        try:
            self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, 'Không được đọc', sound_file='alert.wav'))
            self.assertTrue(chiming.wait(timeout=1))
            self.audio.set_safety_alerts_enabled(False)
            self.audio.interrupt()
            self.audio.post_task(AudioTask(AudioPriority.ON_DEMAND, 'Chế độ OCR'))
            release.set()
            self.assertTrue(self.audio.wait_until_idle(timeout=1))
        finally:
            release.set()
        self.engine.speak.assert_called_once_with('Chế độ OCR')
        self.engine.stop.assert_called()

    def test_mode_switch_stops_current_warning(self):
        speaking = threading.Event()
        def speak(text):
            self.engine.is_speaking.return_value = True
            speaking.set()
        def stop():
            self.engine.is_speaking.return_value = False
        self.engine.speak.side_effect = speak
        self.engine.stop.side_effect = stop
        self.audio.start()
        self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, 'Đang cảnh báo'))
        self.assertTrue(speaking.wait(timeout=1))
        self.audio.set_safety_alerts_enabled(False)
        self.assertTrue(self.audio.wait_until_idle(timeout=1))
        self.engine.stop.assert_called()
        self.assertIsNone(self.audio._current_task)

    def test_concurrent_post_interrupt_and_gate_have_no_deadlock(self):
        errors = []
        barrier = threading.Barrier(3)
        def check_speak(text):
            if text.startswith('HIGH') and not self.audio._safety_alerts_enabled:
                errors.append('HIGH spoke while muted')
        self.engine.speak.side_effect = check_speak
        def post():
            try:
                barrier.wait(timeout=1)
                for i in range(80):
                    self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, f'HIGH {i}'))
            except Exception as exc:
                errors.append(exc)
        def switch():
            try:
                barrier.wait(timeout=1)
                for _ in range(80):
                    self.audio.set_safety_alerts_enabled(False)
                    self.audio.interrupt()
                    self.audio.set_safety_alerts_enabled(True)
            except Exception as exc:
                errors.append(exc)
        self.audio.start()
        threads = [threading.Thread(target=post), threading.Thread(target=switch)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=1)
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(self.audio.wait_until_idle(timeout=1))
