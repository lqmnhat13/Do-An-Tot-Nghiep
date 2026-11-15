"""Dựng micro + bộ nhận dạng giọng nói từ khối ``speech`` trong model_config.yaml."""

from typing import Any, Dict, Optional, Tuple

from src.speech.recorder import MicRecorder
from src.speech.transcriber import PhoWhisperTranscriber

DEFAULT_SPEECH_MODEL = "vinai/PhoWhisper-small"
_ENDPOINTER_KEYS = {
    "rms_threshold": 0.01,
    "silence_sec": 1.2,
    "max_sec": 8.0,
    "no_speech_sec": 5.0,
}


def build_speech_components(
    model_cfg: Dict[str, Any], device: str
) -> Tuple[Optional[MicRecorder], Optional[PhoWhisperTranscriber]]:
    """``speech.enabled: false`` -> ``(None, None)``. Model PhoWhisper chưa được nạp ở đây."""
    speech_cfg = model_cfg.get("speech", {}) or {}
    if not speech_cfg.get("enabled", True):
        return None, None
    sample_rate = int(speech_cfg.get("sample_rate", 16000))
    endpointer_kwargs = {
        key: float(speech_cfg.get(key, default)) for key, default in _ENDPOINTER_KEYS.items()
    }
    recorder = MicRecorder(sample_rate=sample_rate, endpointer_kwargs=endpointer_kwargs)
    transcriber = PhoWhisperTranscriber(
        model_name=speech_cfg.get("model_name", DEFAULT_SPEECH_MODEL), device=device
    )
    return recorder, transcriber
