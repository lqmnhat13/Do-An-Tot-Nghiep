import unittest
import time
from unittest.mock import MagicMock
from src.contracts.audio import AudioTask, AudioPriority
from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine

class TestAudioPriority(unittest.TestCase):
    def setUp(self):
        self.mock_engine = MagicMock(spec=TTSEngine)
        self.mock_engine.is_speaking.return_value = False
        self.coordinator = AudioCoordinator(tts_engine=self.mock_engine, dedup_window_sec=1.0)

    def tearDown(self):
        self.coordinator.stop()

    def test_priority_ordering(self):
        # Đẩy task INFO trước, sau đó đẩy task HIGH_RISK
        t_info = AudioTask(priority=AudioPriority.INFO, text="Bình thường")
        t_high = AudioTask(priority=AudioPriority.HIGH_RISK, text="Cực kỳ nguy hiểm")

        self.coordinator.post_task(t_info)
        self.coordinator.post_task(t_high)

        # Lấy từ queue ra: HIGH_RISK phải ra trước!
        first = self.coordinator._queue.get()
        second = self.coordinator._queue.get()

        self.assertEqual(first.priority, AudioPriority.HIGH_RISK)
        self.assertEqual(second.priority, AudioPriority.INFO)

    def test_drop_expired_task(self):
        now = time.monotonic()
        # Task đã hết hạn 1 giây trước
        expired_task = AudioTask(
            priority=AudioPriority.HIGH_RISK,
            text="Cảnh báo quá hạn",
            created_at=now - 2.0,
            expires_at=now - 1.0
        )
        accepted = self.coordinator.post_task(expired_task)
        self.assertFalse(accepted)
        self.assertEqual(self.coordinator.dropped_expired_count, 1)

    def test_drop_task_expired_during_chime_before_speak(self):
        """Task hết hạn trong thời gian phát chime không được gọi speak()."""
        now = time.monotonic()
        # Task có hạn 0.05s
        task = AudioTask(
            priority=AudioPriority.HIGH_RISK,
            text="Cảnh báo hết hạn lúc chime",
            sound_file="assets/audio/alert_high.wav",
            created_at=now,
            expires_at=now + 0.04
        )

        def mock_wait(*args, **kwargs):
            # Giả lập phát chime mất 0.06s khiến task bị quá hạn
            time.sleep(0.06)

        self.mock_engine.wait_until_done.side_effect = mock_wait
        self.coordinator.start()

        self.coordinator.post_task(task)
        time.sleep(0.15)

        self.mock_engine.play_sound.assert_called_with("assets/audio/alert_high.wav")
        self.mock_engine.speak.assert_not_called()
        self.assertEqual(self.coordinator.dropped_expired_count, 1)

    def test_valid_task_speaks_after_chime(self):
        """Task còn hạn hợp lệ sau chime phải được gọi speak()."""
        now = time.monotonic()
        task = AudioTask(
            priority=AudioPriority.HIGH_RISK,
            text="Cảnh báo hợp lệ",
            sound_file="assets/audio/alert_high.wav",
            created_at=now,
            expires_at=now + 2.0 # Còn hạn 2s
        )

        self.coordinator.start()
        self.coordinator.post_task(task)
        time.sleep(0.1)

        self.mock_engine.play_sound.assert_called_with("assets/audio/alert_high.wav")
        self.mock_engine.speak.assert_called_with("Cảnh báo hợp lệ")

    def test_preemption(self):
        # Giả lập đang có task ON_DEMAND đang đọc dở
        current_task = AudioTask(priority=AudioPriority.ON_DEMAND, text="Đang đọc bài báo dài")
        self.coordinator._current_task = current_task
        self.mock_engine.is_speaking.return_value = True

        # Gửi cảnh báo HIGH_RISK tới -> phải gọi stop() ngay!
        high_task = AudioTask(priority=AudioPriority.HIGH_RISK, text="Vật cản khẩn cấp")
        self.coordinator.post_task(high_task)

        self.mock_engine.stop.assert_called()
        self.assertEqual(self.coordinator.interrupted_count, 1)

    def test_non_interruptible_task_cannot_be_preempted(self):
        # Task VQA/OCR có interruptible = False thì cảnh báo không được ngắt nó
        current_task = AudioTask(priority=AudioPriority.ON_DEMAND, text="Đang trả lời VQA", interruptible=False)
        self.coordinator._current_task = current_task
        self.mock_engine.is_speaking.return_value = True

        high_task = AudioTask(priority=AudioPriority.HIGH_RISK, text="Vật cản khẩn cấp")
        self.coordinator.post_task(high_task)

        # Do task hiện tại interruptible=False, stop() không được gọi
        self.mock_engine.stop.assert_not_called()
        self.assertEqual(self.coordinator.interrupted_count, 0)

    def test_wait_until_idle(self):
        self.mock_engine.is_speaking.return_value = False
        # Ban đầu rỗng -> wait_until_idle trả về True ngay
        self.assertTrue(self.coordinator.wait_until_idle(timeout=0.2))

        # Giả lập đang bận phát
        self.coordinator._pending_tasks_count = 1
        self.mock_engine.is_speaking.return_value = True
        self.assertFalse(self.coordinator.wait_until_idle(timeout=0.1))

        # Giả lập phát xong
        self.coordinator._pending_tasks_count = 0
        self.mock_engine.is_speaking.return_value = False
        self.assertTrue(self.coordinator.wait_until_idle(timeout=0.2))

if __name__ == "__main__":
    unittest.main()
