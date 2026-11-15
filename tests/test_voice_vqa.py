"""Test luồng hỏi VQA bằng giọng nói (phím Q -> bíp -> thu micro -> PhoWhisper -> VQA).

Micro, PhoWhisper, `say`/`afplay` đều là fake: chạy được trên CI Ubuntu không có micro/MPS.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine
from src.contracts.audio import AudioPriority, AudioTask
from src.contracts.frame_packet import FramePacket
from src.fusion.alert_aggregator import AlertAggregator
from src.runtime.system_coordinator import (
    DEFAULT_VQA_QUESTION, NO_SPEECH_TEXT, VOICE_UNAVAILABLE_TEXT, SystemCoordinator)
from src.speech.recorder import MicrophoneError, RecordingResult
from src.ui.app_runner import AppRunner
from src.ui.hud_renderer import HUDRenderer
from src.vqa.safety import SafetyGuardrail
from src.vqa.vqa_service import VQAService

CHIME = "assets/audio/chime.wav"
SPEECH = np.full(1600, 0.1, dtype=np.float32)


def make_packet(frame_id):
    return FramePacket(frame_id, "test", np.full((16, 16, 3), frame_id, dtype=np.uint8),
                       (16, 16), time.monotonic())


class FakeRecorder:
    """Fake MicRecorder: trả kết quả cố định, ném lỗi, hoặc thu cho tới khi stop_event set."""

    def __init__(self, result=None, error=None, block=False, on_record=None):
        self.endpointer_kwargs = {"max_sec": 8.0}
        self.result = result or RecordingResult(SPEECH, "end_of_speech", True)
        self.error = error
        self.block = block
        self.on_record = on_record
        self.started = threading.Event()
        self.stop_event = None
        self.call_time = None
        self.returned = threading.Event()

    def record(self, stop_event):
        self.stop_event = stop_event
        self.call_time = time.monotonic()
        self.started.set()
        try:
            if self.on_record is not None:
                self.on_record()
            if self.error is not None:
                raise self.error
            if self.block:
                stop_event.wait(timeout=5)
                # Người dùng đã nói trước khi bị dừng: kết quả có tiếng nói thật.
                return RecordingResult(SPEECH, "stopped", True)
            return self.result
        finally:
            self.returned.set()


def silence_prints(test):
    """Log runtime dự kiến (lỗi micro giả lập, phím bấm) không làm bẩn output test."""
    patcher = patch("builtins.print")
    patcher.start()
    test.addCleanup(patcher.stop)


class TestVoiceVQA(unittest.TestCase):
    def setUp(self):
        silence_prints(self)
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)
        self.posted = []
        original_post = self.audio.post_task

        def record_post(task):
            accepted = original_post(task)
            if accepted:
                self.posted.append((task.text, task.sound_file))
            return accepted

        self.audio.post_task = record_post
        self.audio.start()
        self.camera = MagicMock()
        self.packet = make_packet(1)
        self.camera.get_latest_frame.return_value = self.packet
        self.vqa = MagicMock()
        self.vqa.answer.return_value = MagicMock(answer="Áo màu đỏ.", latency_sec=.1)
        self.transcriber = MagicMock()
        self.transcriber.transcribe.return_value = "Cái áo này màu gì?"
        self.recorder = FakeRecorder()
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        if hasattr(self, "system"):
            self.system.stop()
        self.audio.stop()

    def build(self, recorder="default", transcriber="default", vqa=None):
        self.system = SystemCoordinator(
            self.camera, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            AlertAggregator(MagicMock()), self.audio, MagicMock(), vqa or self.vqa,
            voice_recorder=self.recorder if recorder == "default" else recorder,
            transcriber=self.transcriber if transcriber == "default" else transcriber)
        return self.system

    def run_voice_task(self):
        self.assertTrue(self.system.trigger_vqa(None))
        self.finish()

    def finish(self):
        self.system._on_demand_thread.join(timeout=3)
        self.assertFalse(self.system._on_demand_thread.is_alive())

    def texts_after_chime(self):
        index = self.posted.index(("", CHIME))
        return [text for text, _ in self.posted[index + 1:]]

    def all_texts(self):
        return [text for text, _ in self.posted]

    def test_voice_question_is_sent_to_vqa_with_frame_after_recording(self):
        after = make_packet(2)
        self.recorder.on_record = lambda: setattr(self.camera.get_latest_frame, "return_value", after)
        self.build()
        self.run_voice_task()
        request = self.vqa.answer.call_args.args[0]
        self.assertEqual(request.question, "Cái áo này màu gì?")
        self.assertIs(request.image, after.image)
        self.assertEqual(self.texts_after_chime(), ["Bạn hỏi: Cái áo này màu gì?", "Áo màu đỏ."])
        self.engine.play_sound.assert_any_call(CHIME)

    def test_question_without_final_punctuation_gets_period(self):
        self.transcriber.transcribe.return_value = "  Đây là tiền gì  "
        self.build()
        self.run_voice_task()
        self.assertIn("Bạn hỏi: Đây là tiền gì.", self.all_texts())
        self.assertEqual(self.vqa.answer.call_args.args[0].question, "Đây là tiền gì")

    def test_recording_waits_until_audio_idle(self):
        speaking_until = time.monotonic() + 0.5
        self.engine.is_speaking.side_effect = lambda: time.monotonic() < speaking_until
        self.build()
        self.run_voice_task()
        self.assertGreaterEqual(self.recorder.call_time, speaking_until)
        # Bíp phải được phát (đã rời hàng đợi) trước khi mở micro.
        self.assertIn(("", CHIME), self.posted)

    def test_no_speech_skips_transcriber(self):
        noise = np.full(1600, 0.001, dtype=np.float32)
        self.recorder.result = RecordingResult(noise, "no_speech", False)
        self.build()
        self.run_voice_task()
        self.transcriber.transcribe.assert_not_called()
        self.vqa.answer.assert_not_called()
        self.assertEqual(self.texts_after_chime(), [NO_SPEECH_TEXT])

    def test_empty_transcript_says_no_speech(self):
        self.transcriber.transcribe.return_value = "  "
        self.build()
        self.run_voice_task()
        self.vqa.answer.assert_not_called()
        self.assertEqual(self.texts_after_chime(), [NO_SPEECH_TEXT])

    def test_microphone_error_falls_back_to_default_question(self):
        self.recorder.error = MicrophoneError("không có micro")
        self.build()
        self.run_voice_task()
        self.transcriber.transcribe.assert_not_called()
        self.assertEqual(self.vqa.answer.call_args.args[0].question, DEFAULT_VQA_QUESTION)
        self.assertEqual(self.texts_after_chime(), [VOICE_UNAVAILABLE_TEXT, "Áo màu đỏ."])

    def test_all_zero_recording_is_treated_as_microphone_error(self):
        # macOS trả im lặng số tuyệt đối khi bị từ chối quyền micro.
        self.recorder.result = RecordingResult(np.zeros(1600, np.float32), "no_speech", False)
        self.build()
        self.run_voice_task()
        self.transcriber.transcribe.assert_not_called()
        self.assertEqual(self.vqa.answer.call_args.args[0].question, DEFAULT_VQA_QUESTION)
        self.assertIn(VOICE_UNAVAILABLE_TEXT, self.all_texts())
        self.assertNotIn(NO_SPEECH_TEXT, self.all_texts())

    def test_transcriber_load_error_falls_back(self):
        self.transcriber.transcribe.side_effect = RuntimeError("thiếu model PhoWhisper")
        self.build()
        self.run_voice_task()
        self.assertEqual(self.vqa.answer.call_args.args[0].question, DEFAULT_VQA_QUESTION)
        self.assertIn(VOICE_UNAVAILABLE_TEXT, self.all_texts())

    def test_transcriber_timeout_falls_back(self):
        self.transcriber.transcribe.side_effect = lambda *a, **k: self.release.wait(timeout=5) and ""
        self.build()
        self.system.STT_TIMEOUT_SEC = 0.2
        t0 = time.monotonic()
        self.run_voice_task()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(self.vqa.answer.call_args.args[0].question, DEFAULT_VQA_QUESTION)
        self.assertIn(VOICE_UNAVAILABLE_TEXT, self.all_texts())

    def test_stop_listening_ends_recording(self):
        self.recorder.block = True
        self.build()
        self.assertFalse(self.system.stop_listening())
        self.assertTrue(self.system.trigger_vqa(None))
        self.assertTrue(self.recorder.started.wait(timeout=2))
        self.assertTrue(self.system.is_listening())
        self.assertTrue(self.system.stop_listening())
        self.assertTrue(self.recorder.stop_event.is_set())
        self.assertTrue(self.recorder.returned.wait(timeout=0.2))
        self.finish()
        self.assertFalse(self.system.is_listening())
        # Dừng bằng Q vẫn nhận dạng phần đã nói.
        self.assertEqual(self.vqa.answer.call_args.args[0].question, "Cái áo này màu gì?")

    def test_enter_warning_mode_cancels_recording_without_late_output(self):
        self.recorder.block = True
        self.build()
        self.assertTrue(self.system.trigger_vqa(None))
        self.assertTrue(self.recorder.started.wait(timeout=2))
        self.system.enter_warning_mode()
        self.assertTrue(self.recorder.stop_event.wait(timeout=0.2))
        self.assertTrue(self.recorder.returned.wait(timeout=0.2))
        self.finish()
        self.assertFalse(self.system.is_listening())
        self.transcriber.transcribe.assert_not_called()
        self.vqa.answer.assert_not_called()
        texts = self.all_texts()
        self.assertFalse(any(t.startswith("Bạn hỏi") for t in texts))
        self.assertNotIn("Áo màu đỏ.", texts)
        self.assertEqual(self.system.get_runtime_snapshot()["selected_mode"], "WARNING")

    def test_stop_speech_during_transcription_releases_worker_without_late_output(self):
        entered = threading.Event()

        def slow_transcribe(*args, **kwargs):
            entered.set()
            self.release.wait(timeout=5)
            return "Cái áo này màu gì?"

        self.transcriber.transcribe.side_effect = slow_transcribe
        self.build()
        self.assertTrue(self.system.trigger_vqa(None))
        self.assertTrue(entered.wait(timeout=2))
        self.system.stop_speech()
        self.system._on_demand_thread.join(timeout=0.5)
        self.assertFalse(self.system._on_demand_thread.is_alive())
        self.release.set()
        time.sleep(0.05)
        self.vqa.answer.assert_not_called()
        self.assertFalse(any(t.startswith("Bạn hỏi") for t in self.all_texts()))

    def test_shutdown_during_recording_does_not_hang(self):
        self.recorder.block = True
        self.build()
        self.assertTrue(self.system.trigger_vqa(None))
        self.assertTrue(self.recorder.started.wait(timeout=2))
        t0 = time.monotonic()
        self.system.stop()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertTrue(self.recorder.stop_event.is_set())
        self.assertFalse(self.system._on_demand_thread.is_alive())
        self.vqa.answer.assert_not_called()

    def test_dangerous_voice_question_is_refused(self):
        self.transcriber.transcribe.return_value = "Đường trống không?"
        backend = MagicMock()
        self.build(vqa=VQAService(backend=backend))
        self.run_voice_task()
        backend.answer.assert_not_called()
        self.assertEqual(self.texts_after_chime(),
                         ["Bạn hỏi: Đường trống không?", SafetyGuardrail.REFUSAL])

    def test_without_speech_components_uses_default_question(self):
        self.build(recorder=None, transcriber=None)
        self.run_voice_task()
        self.assertEqual(self.vqa.answer.call_args.args[0].question, DEFAULT_VQA_QUESTION)
        self.assertNotIn(("", CHIME), self.posted)
        self.assertNotIn(VOICE_UNAVAILABLE_TEXT, self.all_texts())

    def test_text_question_keeps_old_flow(self):
        self.build()
        self.assertTrue(self.system.trigger_vqa("Có chữ gì?"))
        self.finish()
        self.assertFalse(self.recorder.started.is_set())
        self.assertEqual(self.vqa.answer.call_args.args[0].question, "Có chữ gì?")
        self.assertEqual(self.system.get_runtime_snapshot()["vqa_question"], "Có chữ gì?")

    def test_snapshot_has_listening_and_question(self):
        self.recorder.block = True
        self.build()
        snapshot = self.system.get_runtime_snapshot()
        self.assertFalse(snapshot["listening"])
        self.assertIsNone(snapshot["vqa_question"])
        self.assertTrue(self.system.trigger_vqa(None))
        self.assertTrue(self.recorder.started.wait(timeout=2))
        self.assertTrue(self.system.get_runtime_snapshot()["listening"])
        self.system.stop_listening()
        self.finish()
        snapshot = self.system.get_runtime_snapshot()
        self.assertFalse(snapshot["listening"])
        self.assertEqual(snapshot["vqa_question"], "Cái áo này màu gì?")
        self.system.enter_warning_mode()
        self.assertIsNone(self.system.get_runtime_snapshot()["vqa_question"])


class TestAudioChimeOnly(unittest.TestCase):
    def test_task_with_only_sound_file_plays_sound_without_speaking(self):
        engine = MagicMock(spec=TTSEngine)
        engine.is_speaking.return_value = False
        audio = AudioCoordinator(engine)
        audio.start()
        try:
            self.assertTrue(audio.post_task(AudioTask(AudioPriority.ON_DEMAND, "", sound_file=CHIME)))
            self.assertTrue(audio.wait_until_idle(timeout=1))
        finally:
            audio.stop()
        engine.play_sound.assert_called_once_with(CHIME)
        engine.speak.assert_not_called()


class TestVoiceKeyAndHUD(unittest.TestCase):
    def setUp(self):
        silence_prints(self)

    def run_keys(self, coordinator, keys):
        runner = AppRunner(coordinator, hud_renderer=MagicMock(), enable_gui=True)
        keys = iter(keys)
        with patch("src.ui.app_runner.cv2") as cv2:
            cv2.waitKey.side_effect = lambda _: next(keys)
            runner.run()

    def test_q_starts_voice_question(self):
        coordinator = MagicMock()
        coordinator.is_listening.return_value = False
        self.run_keys(coordinator, [ord('q'), 27])
        coordinator.trigger_vqa.assert_called_once_with(None)
        coordinator.stop_listening.assert_not_called()

    def test_q_while_listening_stops_recording(self):
        coordinator = MagicMock()
        coordinator.is_listening.return_value = True
        self.run_keys(coordinator, [ord('Q'), 27])
        coordinator.stop_listening.assert_called_once()
        coordinator.trigger_vqa.assert_not_called()

    def test_hud_shows_listening_and_question(self):
        renderer = HUDRenderer()
        snapshot = {"packet": make_packet(1), "mode": "VQA", "selected_mode": "VQA",
                    "on_demand_active": True, "safety_alerts_enabled": False,
                    "assessments": [], "listening": True, "vqa_question": "Cái áo này màu gì?"}
        with patch("src.ui.hud_renderer.cv2.putText") as text:
            renderer.render(snapshot)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertIn("DANG NGHE...", labels)
        self.assertTrue(any("Cai ao nay mau gi?" in label for label in labels))


if __name__ == "__main__":
    unittest.main()
