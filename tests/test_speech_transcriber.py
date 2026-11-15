import unicodedata
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.runtime.model_loading import pretrained_kwargs
from src.speech.factory import build_speech_components
from src.speech.recorder import MicRecorder
from src.speech.transcriber import PhoWhisperTranscriber

AUDIO = np.zeros(16000, dtype=np.float32)  # 1 s, đủ dài để gọi model


def _fake_classes(decoded):
    """Processor/model giả: processor(...) trả input_features, model.generate trả ids."""
    processor = MagicMock()
    processor.return_value.input_features = MagicMock()
    processor.batch_decode.return_value = decoded
    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = MagicMock()
    processor_cls = MagicMock()
    processor_cls.from_pretrained.return_value = processor
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model
    return processor_cls, model_cls, processor, model


def _patch_transformers(processor_cls, model_cls):
    return (
        patch("transformers.WhisperProcessor", processor_cls),
        patch("transformers.WhisperForConditionalGeneration", model_cls),
    )


class _QuietCase(unittest.TestCase):
    def setUp(self):
        # Giữ output test sạch: chặn print của module transcriber.
        quiet = patch("src.speech.transcriber.print", create=True)
        quiet.start()
        self.addCleanup(quiet.stop)


class TestPhoWhisperTranscriber(_QuietCase):
    def test_lazy_load_not_at_init(self):
        processor_cls, model_cls, _, _ = _fake_classes([""])
        p1, p2 = _patch_transformers(processor_cls, model_cls)
        with p1, p2:
            transcriber = PhoWhisperTranscriber(device="cpu")
        processor_cls.from_pretrained.assert_not_called()
        model_cls.from_pretrained.assert_not_called()
        self.assertIsNone(transcriber.load_error)

    def test_transcribe_returns_nfc_stripped(self):
        nfd = unicodedata.normalize("NFD", "Đường trống không")
        processor_cls, model_cls, processor, model = _fake_classes(["  " + nfd + " "])
        p1, p2 = _patch_transformers(processor_cls, model_cls)
        with p1, p2:
            text = PhoWhisperTranscriber(device="cpu").transcribe(AUDIO)
        self.assertEqual(text, unicodedata.normalize("NFC", "Đường trống không"))
        self.assertTrue(unicodedata.is_normalized("NFC", text))
        kwargs = model.generate.call_args.kwargs
        self.assertEqual(kwargs.get("language"), "vi")
        self.assertEqual(kwargs.get("task"), "transcribe")
        self.assertEqual(processor.call_args.kwargs.get("sampling_rate"), 16000)

    def test_short_audio_skips_model(self):
        processor_cls, model_cls, _, model = _fake_classes(["không được gọi"])
        p1, p2 = _patch_transformers(processor_cls, model_cls)
        with p1, p2:
            transcriber = PhoWhisperTranscriber(device="cpu")
            self.assertEqual(transcriber.transcribe(np.zeros(800, dtype=np.float32)), "")
            self.assertEqual(transcriber.transcribe(np.zeros(0, dtype=np.float32)), "")
        model.generate.assert_not_called()

    def test_load_failure_raises_and_is_cached(self):
        processor_cls, model_cls, _, _ = _fake_classes([""])
        processor_cls.from_pretrained.side_effect = OSError("không có cache")
        p1, p2 = _patch_transformers(processor_cls, model_cls)
        with p1, p2:
            transcriber = PhoWhisperTranscriber(device="cpu")
            with self.assertRaises(RuntimeError):
                transcriber.transcribe(AUDIO)
            self.assertIsNotNone(transcriber.load_error)
            with self.assertRaises(RuntimeError):
                transcriber.transcribe(AUDIO)
        self.assertEqual(processor_cls.from_pretrained.call_count, 1)

    def test_uses_offline_kwargs(self):
        processor_cls, model_cls, _, _ = _fake_classes(["xin chào"])
        p1, p2 = _patch_transformers(processor_cls, model_cls)
        with p1, p2:
            PhoWhisperTranscriber(device="cpu").transcribe(AUDIO)
        expected = pretrained_kwargs()
        for cls in (processor_cls, model_cls):
            kwargs = cls.from_pretrained.call_args.kwargs
            for key, value in expected.items():
                self.assertEqual(kwargs.get(key), value)
        self.assertEqual(processor_cls.from_pretrained.call_args.args[0], "vinai/PhoWhisper-small")

    def test_mps_unavailable_falls_back_to_cpu(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(PhoWhisperTranscriber(device="mps").device, "cpu")


class TestSpeechFactory(_QuietCase):
    def test_factory_disabled_returns_none(self):
        self.assertEqual(build_speech_components({"speech": {"enabled": False}}, "cpu"), (None, None))

    def test_factory_reads_config(self):
        cfg = {
            "speech": {
                "enabled": True,
                "model_name": "vinai/PhoWhisper-small",
                "sample_rate": 16000,
                "rms_threshold": 0.02,
                "silence_sec": 0.9,
                "max_sec": 6.0,
                "no_speech_sec": 4.0,
            }
        }
        recorder, transcriber = build_speech_components(cfg, "cpu")
        self.assertIsInstance(recorder, MicRecorder)
        self.assertIsInstance(transcriber, PhoWhisperTranscriber)
        self.assertEqual(recorder.sample_rate, 16000)
        self.assertEqual(recorder.endpointer_kwargs["silence_sec"], 0.9)
        self.assertEqual(recorder.endpointer_kwargs["max_sec"], 6.0)
        self.assertEqual(recorder.endpointer_kwargs["rms_threshold"], 0.02)
        self.assertEqual(recorder.endpointer_kwargs["no_speech_sec"], 4.0)
        self.assertEqual(transcriber.model_name, "vinai/PhoWhisper-small")
        self.assertEqual(transcriber.device, "cpu")


if __name__ == "__main__":
    unittest.main()
