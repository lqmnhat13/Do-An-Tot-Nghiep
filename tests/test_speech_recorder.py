import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.speech.endpointing import SilenceEndpointer
from src.speech.recorder import MicRecorder, MicrophoneError, RecordingResult

SR = 16000
BLOCK = int(SR * 0.03)


def _blocks(seconds: float, amplitude: float):
    """Sinh các block 0,03 s; amplitude 0 là im lặng, >0 là tín hiệu có RMS = amplitude."""
    count = int(round(seconds / 0.03))
    for _ in range(count):
        if amplitude == 0:
            yield np.zeros(BLOCK, dtype=np.float32)
        else:
            # Sóng vuông ±amplitude có RMS đúng bằng amplitude.
            signs = np.where(np.arange(BLOCK) % 2 == 0, 1.0, -1.0)
            yield (signs * amplitude).astype(np.float32)


def _ambient(seconds: float, rms: float = 0.0136, peak: float = 0.031, seed: int = 0):
    """Tiếng nền phòng yên tĩnh đo trên micro MacBook: Gaussian RMS ~0,0136, đỉnh ~0,031."""
    rng = np.random.default_rng(seed)
    count = int(round(seconds / 0.03))
    for _ in range(count):
        yield np.clip(rng.normal(0.0, rms, BLOCK), -peak, peak).astype(np.float32)


def _feed_all(endpointer, chunks):
    """Feed lần lượt, trả (lý do đầu tiên khác None, số block đã feed)."""
    fed = 0
    for chunk in chunks:
        fed += 1
        reason = endpointer.feed(chunk)
        if reason is not None:
            return reason, fed
    return None, fed


class TestSilenceEndpointer(unittest.TestCase):
    def test_end_of_speech_after_silence(self):
        ep = SilenceEndpointer()
        chunks = list(_blocks(0.5, 0)) + list(_blocks(1.0, 0.1)) + list(_blocks(1.3, 0))
        reason, _ = _feed_all(ep, chunks)
        self.assertEqual(reason, "end_of_speech")
        self.assertTrue(ep.heard_speech)

    def test_no_speech_after_5s(self):
        ep = SilenceEndpointer()
        reason, _ = _feed_all(ep, _blocks(5.1, 0))
        self.assertEqual(reason, "no_speech")
        self.assertFalse(ep.heard_speech)

    def test_max_duration_8s_in_noisy_room(self):
        ep = SilenceEndpointer()
        reason, _ = _feed_all(ep, _blocks(8.1, 0.1))
        self.assertEqual(reason, "max_duration")
        self.assertTrue(ep.heard_speech)

    def test_short_pause_does_not_end(self):
        ep = SilenceEndpointer()
        head = list(_blocks(1.0, 0.1)) + list(_blocks(0.8, 0)) + list(_blocks(1.0, 0.1))
        for chunk in head:
            self.assertIsNone(ep.feed(chunk))
        tail = list(_blocks(1.2, 0))
        for chunk in tail[:-1]:
            self.assertIsNone(ep.feed(chunk))
        self.assertEqual(ep.feed(tail[-1]), "end_of_speech")

    def test_real_ambient_noise_is_not_speech(self):
        # Ngưỡng cố định 0,01 coi tiếng nền này là tiếng nói và thu đủ 8 s.
        ep = SilenceEndpointer()
        reason, fed = _feed_all(ep, _ambient(5.1))
        self.assertEqual(reason, "no_speech")
        self.assertFalse(ep.heard_speech)
        self.assertAlmostEqual(fed * 0.03, 5.0, delta=0.035)

    def test_speech_over_ambient_ends_after_silence(self):
        ep = SilenceEndpointer()
        speech_end = len(list(_blocks(0.5, 0))) + len(list(_blocks(1.0, 0.13)))
        chunks = (list(_ambient(0.5, seed=1)) + list(_blocks(1.0, 0.13))
                  + list(_ambient(2.0, seed=2)))
        reason, fed = _feed_all(ep, chunks)
        self.assertEqual(reason, "end_of_speech")
        self.assertTrue(ep.heard_speech)
        self.assertAlmostEqual((fed - speech_end) * 0.03, 1.2, delta=0.035)
        self.assertEqual(ep.last_speech_end_samples, speech_end * BLOCK)

    def test_single_click_is_not_speech(self):
        ep = SilenceEndpointer()
        chunks = list(_ambient(1.0, seed=3)) + list(_blocks(0.03, 0.3)) + list(_ambient(4.2, seed=4))
        reason, _ = _feed_all(ep, chunks)
        self.assertEqual(reason, "no_speech")
        self.assertFalse(ep.heard_speech)

    def test_two_loud_blocks_are_not_speech(self):
        ep = SilenceEndpointer()
        chunks = list(_blocks(0.5, 0)) + list(_blocks(0.06, 0.3)) + list(_blocks(4.6, 0))
        reason, _ = _feed_all(ep, chunks)
        self.assertEqual(reason, "no_speech")
        self.assertFalse(ep.heard_speech)

    def test_speech_starting_during_calibration_is_detected(self):
        # Nói đè lên bíp: nền đo được rất cao nhưng ngưỡng bị chặn trên 0,06.
        ep = SilenceEndpointer()
        reason, fed = _feed_all(ep, list(_blocks(1.0, 0.13)) + list(_blocks(1.3, 0)))
        self.assertEqual(reason, "end_of_speech")
        self.assertTrue(ep.heard_speech)
        self.assertEqual(fed, 33 + 40)

    def test_calibration_blocks_are_not_speech(self):
        ep = SilenceEndpointer()
        for chunk in _blocks(0.3, 0.13):
            self.assertIsNone(ep.feed(chunk))
            self.assertFalse(ep.heard_speech)
        self.assertIsNone(ep.last_speech_end_samples)

    def test_threshold_is_at_least_rms_threshold(self):
        # Nền im tuyệt đối: ngưỡng là rms_threshold, tín hiệu 0,005 không phải tiếng nói.
        ep = SilenceEndpointer(rms_threshold=0.01)
        reason, _ = _feed_all(ep, list(_blocks(0.3, 0)) + list(_blocks(4.8, 0.005)))
        self.assertEqual(reason, "no_speech")
        self.assertFalse(ep.heard_speech)


class _FakeStream:
    """Giả lập sounddevice.InputStream: trả các block định sẵn, sau đó im lặng."""

    def __init__(self, script=None, delay=0.0):
        self.script = list(script or [])
        self.delay = delay
        self.read_count = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def read(self, n):
        if self.delay:
            time.sleep(self.delay)
        self.read_count += 1
        if self.script:
            block = self.script.pop(0)
        else:
            block = np.zeros(n, dtype=np.float32)
        return block.reshape(n, 1).astype(np.float32), False


class _PortAudioError(Exception):
    pass


def _fake_sounddevice(stream=None, error=None):
    module = types.ModuleType("sounddevice")
    module.PortAudioError = _PortAudioError
    module.calls = []

    def input_stream(**kwargs):
        module.calls.append(kwargs)
        if error is not None:
            raise error
        return stream

    module.InputStream = input_stream
    return module


class TestMicRecorder(unittest.TestCase):
    def test_record_stops_on_event(self):
        stream = _FakeStream(delay=0.03)
        fake = _fake_sounddevice(stream)
        stop = threading.Event()
        timer = threading.Timer(0.1, stop.set)
        with patch.dict(sys.modules, {"sounddevice": fake}):
            start = time.monotonic()
            timer.start()
            result = MicRecorder().record(stop)
            elapsed = time.monotonic() - start
        timer.join()
        self.assertIsInstance(result, RecordingResult)
        self.assertEqual(result.reason, "stopped")
        self.assertLess(elapsed, 0.5)
        self.assertTrue(stream.closed)
        self.assertEqual(fake.calls[0]["samplerate"], SR)
        self.assertEqual(fake.calls[0]["channels"], 1)

    def test_record_device_error_raises_microphone_error(self):
        fake = _fake_sounddevice(error=_PortAudioError("Error querying device -1"))
        with patch.dict(sys.modules, {"sounddevice": fake}):
            with self.assertRaises(MicrophoneError):
                MicRecorder().record(threading.Event())

    def test_record_os_error_raises_microphone_error(self):
        fake = _fake_sounddevice(error=OSError("PortAudio library not found"))
        with patch.dict(sys.modules, {"sounddevice": fake}):
            with self.assertRaises(MicrophoneError):
                MicRecorder().record(threading.Event())

    def test_record_import_failure_raises_microphone_error(self):
        # None trong sys.modules khiến `import sounddevice` ném ImportError.
        with patch.dict(sys.modules, {"sounddevice": None}):
            with self.assertRaises(MicrophoneError):
                MicRecorder().record(threading.Event())

    def test_record_read_error_closes_stream(self):
        stream = _FakeStream()

        def broken_read(n):
            raise _PortAudioError("Input overflowed hard")

        stream.read = broken_read
        fake = _fake_sounddevice(stream)
        with patch.dict(sys.modules, {"sounddevice": fake}):
            with self.assertRaises(MicrophoneError):
                MicRecorder().record(threading.Event())
        self.assertTrue(stream.closed)

    def test_record_trims_trailing_silence(self):
        # 0,3 s nền (hiệu chuẩn) + 0,6 s nói, sau đó im lặng tới khi endpointer dừng.
        script = list(_blocks(0.3, 0)) + list(_blocks(0.6, 0.1))
        first_block = script[0].copy()
        stream = _FakeStream(script=script)
        fake = _fake_sounddevice(stream)
        with patch.dict(sys.modules, {"sounddevice": fake}):
            result = MicRecorder().record(threading.Event())
        self.assertEqual(result.reason, "end_of_speech")
        self.assertTrue(result.heard_speech)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(stream.read_count, 30 + 40)
        # Giữ phần nói + 0,3 s đuôi, bỏ 0,9 s im lặng còn lại.
        self.assertEqual(len(result.audio), 30 * BLOCK + int(0.3 * SR))
        np.testing.assert_allclose(result.audio[:BLOCK], first_block)
        np.testing.assert_allclose(result.audio[29 * BLOCK:30 * BLOCK], script[-1])
        self.assertFalse(np.any(result.audio[30 * BLOCK:]))

    def test_record_without_speech_is_not_trimmed(self):
        stream = _FakeStream(script=list(_ambient(0.3)))
        fake = _fake_sounddevice(stream)
        with patch.dict(sys.modules, {"sounddevice": fake}):
            result = MicRecorder().record(threading.Event())
        self.assertEqual(result.reason, "no_speech")
        self.assertEqual(len(result.audio), stream.read_count * BLOCK)

    def test_record_value_error_raises_microphone_error(self):
        # sounddevice ném ValueError khi không tìm thấy thiết bị/không hỗ trợ cấu hình.
        fake = _fake_sounddevice(error=ValueError("No input device matching"))
        with patch.dict(sys.modules, {"sounddevice": fake}):
            with self.assertRaises(MicrophoneError):
                MicRecorder().record(threading.Event())

    def test_record_uses_fresh_endpointer(self):
        fake = _fake_sounddevice(_FakeStream(script=list(_blocks(0.6, 0.1))))
        recorder = MicRecorder()
        with patch.dict(sys.modules, {"sounddevice": fake}):
            first = recorder.record(threading.Event())
            fake.InputStream = lambda **kw: _FakeStream()
            second = recorder.record(threading.Event())
        self.assertTrue(first.heard_speech)
        self.assertEqual(second.reason, "no_speech")
        self.assertFalse(second.heard_speech)


if __name__ == "__main__":
    unittest.main()
