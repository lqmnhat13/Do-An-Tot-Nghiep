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

    def test_record_returns_concatenated_audio(self):
        script = list(_blocks(0.6, 0.1))
        first_block = script[0].copy()
        stream = _FakeStream(script=script)
        fake = _fake_sounddevice(stream)
        with patch.dict(sys.modules, {"sounddevice": fake}):
            result = MicRecorder().record(threading.Event())
        self.assertEqual(result.reason, "end_of_speech")
        self.assertTrue(result.heard_speech)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(len(result.audio), stream.read_count * BLOCK)
        np.testing.assert_allclose(result.audio[:BLOCK], first_block)

    def test_record_uses_fresh_endpointer(self):
        fake = _fake_sounddevice(_FakeStream(script=list(_blocks(0.3, 0.1))))
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
