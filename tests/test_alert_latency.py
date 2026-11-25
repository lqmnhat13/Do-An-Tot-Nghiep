"""Độ trễ cảnh báo: câu ngắn đọc từ clip dựng sẵn, và dừng câu cảnh báo khi vật không còn nguy hiểm."""
import os
import tempfile
import time
import unittest
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine, _ClipOutput, load_wav_mono
from src.contracts.audio import AudioPriority, AudioTask
from src.contracts.risk import DataQuality, Direction, RiskAssessment, RiskLevel
from src.fusion.alert_aggregator import AlertAggregator
from src.runtime.system_coordinator import SystemCoordinator


def assessment(name="ghế", track_id=1, direction=Direction.CENTER, level=RiskLevel.HIGH,
               quality=DataQuality.VALID, ts=None):
    now = time.monotonic() if ts is None else ts
    return RiskAssessment(track_id, name, direction, level, quality, .9, "ở rất gần", "test",
                          now, now + 1.0)


def write_wav(path, samples=100, rate=22050):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.arange(samples, dtype=np.int16).tobytes())


class TestShortAlertText(unittest.TestCase):
    def setUp(self):
        self.fsm = MagicMock()
        self.fsm.can_trigger_alert.return_value = True

    def test_text_is_name_then_direction_only(self):
        agg = AlertAggregator(self.fsm)
        now = time.monotonic()
        task = agg.aggregate([assessment("ghế", ts=now)], current_mono=now)
        self.assertEqual(task.text, "ghế phía trước")
        self.assertEqual(task.alert_key, "ghế")
        task = AlertAggregator(self.fsm).aggregate(
            [assessment("vật cản", direction=Direction.LEFT, ts=now)], current_mono=now)
        self.assertEqual(task.text, "vật cản bên trái")

    def test_alert_texts_cover_every_name_and_direction(self):
        texts = AlertAggregator.alert_texts(["ghế", "người", "ghế"])
        self.assertEqual(len(texts), 6)
        self.assertIn("người bên phải", texts)
        self.assertIn("ghế phía trước", texts)


class TestSupportedAlertKeys(unittest.TestCase):
    def setUp(self):
        self.agg = AlertAggregator(MagicMock(), revoke_grace_sec=0.4)

    def test_non_valid_frame_never_supports_revocation(self):
        for quality in (DataQuality.STALE, DataQuality.DEGRADED, DataQuality.UNAVAILABLE):
            self.assertIsNone(self.agg.supported_alert_keys([], quality, current_mono=10.0))

    def test_key_kept_within_grace_then_dropped(self):
        self.assertEqual(self.agg.supported_alert_keys(
            [assessment("ghế", ts=10.0)], DataQuality.VALID, current_mono=10.0), {"ghế"})
        self.assertEqual(self.agg.supported_alert_keys([], DataQuality.VALID, current_mono=10.3), {"ghế"})
        self.assertEqual(self.agg.supported_alert_keys([], DataQuality.VALID, current_mono=10.5), set())

    def test_only_fresh_valid_high_counts(self):
        candidates = [assessment("ghế", level=RiskLevel.MEDIUM, ts=10.0),
                      assessment("bàn", ts=7.0),  # nguồn quá cũ
                      assessment("người", ts=10.0)]
        self.assertEqual(self.agg.supported_alert_keys(candidates, DataQuality.VALID, current_mono=10.0),
                         {"người"})

    def test_recent_non_valid_data_blocks_revocation(self):
        """Phải có đủ khoảng grace dữ liệu VALID liên tục mới được kết luận vật đã hết nguy hiểm."""
        self.agg.supported_alert_keys([assessment("ghế", ts=10.0)], DataQuality.VALID, current_mono=10.0)
        self.agg.supported_alert_keys([], DataQuality.STALE, current_mono=10.6)
        self.assertIsNone(self.agg.supported_alert_keys([], DataQuality.VALID, current_mono=10.7))
        self.assertEqual(self.agg.supported_alert_keys([], DataQuality.VALID, current_mono=11.1), set())

    def test_release_global_interval_allows_next_alert(self):
        fsm = MagicMock()
        fsm.can_trigger_alert.return_value = True
        agg = AlertAggregator(fsm, global_alert_interval_sec=3.5)
        self.assertIsNotNone(agg.aggregate([assessment("ghế", ts=10.0)], current_mono=10.0))
        self.assertIsNone(agg.aggregate([assessment("người", 2, ts=10.5)], current_mono=10.5))
        agg.release_global_interval()
        self.assertIsNotNone(agg.aggregate([assessment("người", 2, ts=10.6)], current_mono=10.6))


class TestRetainSafetyAlerts(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)

    def high(self, key):
        return AudioTask(AudioPriority.HIGH_RISK, f"{key} phía trước", alert_key=key)

    def test_current_alert_stopped_when_key_unsupported(self):
        self.audio._current_task = self.high("ghế")
        self.audio._current_entry = object()
        self.assertFalse(self.audio.retain_safety_alerts({"ghế"}))
        self.engine.stop.assert_not_called()
        self.assertTrue(self.audio.retain_safety_alerts({"người"}))
        self.engine.stop.assert_called_once()
        self.assertIsNone(self.audio._current_task)
        self.assertIsNone(self.audio._current_entry)

    def test_queued_unsupported_alerts_dropped_others_kept(self):
        self.audio.post_task(self.high("ghế"))
        self.audio.post_task(self.high("người"))
        self.audio.post_task(AudioTask(AudioPriority.SYSTEM_STATUS, "Mất camera"))
        self.assertTrue(self.audio.retain_safety_alerts({"người"}))
        remaining = []
        while not self.audio._queue.empty():
            remaining.append(self.audio._queue.get_nowait().task.text)
        self.assertEqual(sorted(remaining), ["Mất camera", "người phía trước"])
        self.assertEqual(self.audio._pending_tasks_count, 2)

    def test_alert_without_key_is_never_revoked(self):
        self.audio._current_task = AudioTask(AudioPriority.HIGH_RISK, "Vật cản")
        self.audio._current_entry = object()
        self.assertFalse(self.audio.retain_safety_alerts(set()))
        self.engine.stop.assert_not_called()


class TestCoordinatorRevokesPassedObstacle(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)
        self.fsm = MagicMock()
        self.fsm.can_trigger_alert.return_value = True
        self.agg = AlertAggregator(self.fsm, global_alert_interval_sec=3.5, revoke_grace_sec=0.0)
        self.system = SystemCoordinator(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                        self.fsm, self.agg, self.audio, MagicMock(), MagicMock())

    def tearDown(self):
        self.system.stop()

    def queued_texts(self):
        return [e.task.text for e in list(self.audio._queue.queue)]

    def test_alert_dropped_when_valid_data_shows_obstacle_gone(self):
        self.system._post_safety_alert([assessment("ghế")], DataQuality.VALID)
        self.assertEqual(self.queued_texts(), ["ghế phía trước"])
        # Dữ liệu cũ/suy giảm không bao giờ được coi là "đã an toàn".
        self.system._post_safety_alert([], DataQuality.STALE)
        self.assertEqual(self.queued_texts(), ["ghế phía trước"])
        self.system._post_safety_alert([], DataQuality.VALID)
        self.assertEqual(self.queued_texts(), [])
        # Câu bị huỷ không giữ khoảng nghỉ toàn cục: vật mới gần được báo ngay.
        self.system._post_safety_alert([assessment("người", 2)], DataQuality.VALID)
        self.assertEqual(self.queued_texts(), ["người phía trước"])

    def test_without_quality_no_revocation(self):
        self.system._post_safety_alert([assessment("ghế")])
        self.system._post_safety_alert([])
        self.assertEqual(self.queued_texts(), ["ghế phía trước"])


class TestTTSEngineClips(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = TTSEngine(voice="Linh", speech_rate_wpm=240, clip_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clip_path_depends_on_voice_rate_and_text(self):
        other_rate = TTSEngine(voice="Linh", speech_rate_wpm=200, clip_dir=self.tmp.name)
        self.assertNotEqual(self.engine.clip_path("ghế phía trước"), other_rate.clip_path("ghế phía trước"))
        self.assertNotEqual(self.engine.clip_path("ghế phía trước"), self.engine.clip_path("người phía trước"))
        self.assertIsNone(TTSEngine().clip_path("ghế phía trước"))

    @patch("src.audio.tts_engine.subprocess.Popen")
    @patch("src.audio.tts_engine._ClipOutput")
    def test_cached_clip_played_in_process(self, output_cls, popen):
        write_wav(self.engine.clip_path("ghế phía trước"))
        self.assertTrue(self.engine.speak("ghế phía trước"))
        output = output_cls.return_value
        output.play.assert_called_once()
        popen.assert_not_called()
        output.is_active.return_value = True
        self.assertTrue(self.engine.is_speaking())
        self.engine.stop()
        output.stop.assert_called()
        self.engine.speak("ghế phía trước")
        output_cls.assert_called_once()  # một stream dùng lại cho mọi lần phát

    @patch("src.audio.tts_engine.subprocess.Popen")
    @patch("src.audio.tts_engine._ClipOutput")
    def test_missing_clip_or_output_failure_falls_back_to_say(self, output_cls, popen):
        self.engine.speak("Đọc chữ")
        output_cls.return_value.play.assert_not_called()
        self.assertEqual(popen.call_args[0][0], ["say", "-v", "Linh", "-r", "240", "Đọc chữ"])
        write_wav(self.engine.clip_path("ghế phía trước"))
        output_cls.return_value.play.side_effect = RuntimeError("no output device")
        self.assertTrue(self.engine.speak("ghế phía trước"))
        self.assertEqual(popen.call_args[0][0][-1], "ghế phía trước")

    @patch("src.audio.tts_engine.subprocess.run")
    def test_prerender_writes_missing_clips_atomically(self, run):
        def fake_say(cmd, **kwargs):
            out = cmd[cmd.index("-o") + 1]
            if cmd[-1] == "lỗi":
                raise RuntimeError("say failed")
            write_wav(out)
        run.side_effect = fake_say
        write_wav(self.engine.clip_path("đã có"))
        self.assertEqual(self.engine.prerender(["ghế phía trước", "đã có", "lỗi"]), 1)
        self.assertTrue(os.path.exists(self.engine.clip_path("ghế phía trước")))
        self.assertFalse(os.path.exists(self.engine.clip_path("lỗi")))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(sorted(os.listdir(self.tmp.name)),
                         sorted(os.path.basename(self.engine.clip_path(t)) for t in ("ghế phía trước", "đã có")))


class TestClipOutput(unittest.TestCase):
    def test_callback_feeds_clip_then_silence(self):
        out_dev = _ClipOutput()
        out_dev._ensure_stream = lambda: None
        out_dev.play(np.array([1, 2, 3, 4, 5], dtype=np.int16))
        self.assertTrue(out_dev.is_active())
        out = np.full((3, 1), -1, dtype=np.int16)
        out_dev._callback(out, 3, None, None)
        self.assertEqual(out[:, 0].tolist(), [1, 2, 3])
        out_dev._callback(out, 3, None, None)
        self.assertEqual(out[:, 0].tolist(), [4, 5, 0])
        self.assertFalse(out_dev.is_active())
        out_dev._callback(out, 3, None, None)
        self.assertEqual(out[:, 0].tolist(), [0, 0, 0])

    def test_stop_silences_immediately(self):
        out_dev = _ClipOutput()
        out_dev._ensure_stream = lambda: None
        out_dev.play(np.ones(1000, dtype=np.int16))
        out_dev.stop()
        self.assertFalse(out_dev.is_active())
        self.assertTrue(out_dev.wait(0.01))
        out = np.full((4, 1), -1, dtype=np.int16)
        out_dev._callback(out, 4, None, None)
        self.assertEqual(out[:, 0].tolist(), [0, 0, 0, 0])

    def test_dead_stream_does_not_hang_audio_thread(self):
        """Mất thiết bị (callback không chạy nữa): quá hạn thì coi như đã phát xong."""
        out_dev = _ClipOutput()
        out_dev._ensure_stream = lambda: None
        out_dev.play(np.ones(441, dtype=np.int16))  # 10 ms
        out_dev._deadline = time.monotonic() - 0.01
        self.assertFalse(out_dev.is_active())
        self.assertTrue(out_dev.wait(None))

    def test_load_wav_resamples_to_mono_output_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            write_wav(path, samples=22050, rate=22050)
            data = load_wav_mono(path)
        self.assertEqual(data.dtype, np.int16)
        self.assertEqual(len(data), 44100)


if __name__ == "__main__":
    unittest.main()
