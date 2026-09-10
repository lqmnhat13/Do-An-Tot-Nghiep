"""Chính sách tải model dùng chung cho runtime offline."""

import os
from typing import Dict


OFFLINE_ENV_VAR = "SECOND_EYE_OFFLINE"
HF_HUB_OFFLINE_ENV_VAR = "HF_HUB_OFFLINE"
TRANSFORMERS_OFFLINE_ENV_VAR = "TRANSFORMERS_OFFLINE"
_FALSE_VALUES = {"0", "false", "no", "off"}


def is_offline_mode() -> bool:
    """Runtime mặc định offline; chỉ bật mạng khi người dùng chủ động opt-in."""
    value = os.getenv(OFFLINE_ENV_VAR, "1")
    return value.strip().lower() not in _FALSE_VALUES


def pretrained_kwargs(*, allow_download: bool = False) -> Dict[str, bool]:
    """Các tham số thống nhất cho mọi lời gọi Hugging Face from_pretrained."""
    return {"local_files_only": is_offline_mode() and not allow_download}


def enable_model_downloads() -> None:
    """Chỉ script chuẩn bị gọi hàm này, trước khi import Hugging Face."""
    os.environ[OFFLINE_ENV_VAR] = "0"
    os.environ[HF_HUB_OFFLINE_ENV_VAR] = "0"
    os.environ[TRANSFORMERS_OFFLINE_ENV_VAR] = "0"


def offline_load_error(component: str, model_name: str, exc: Exception) -> str:
    if is_offline_mode():
        return (
            f"{component} chưa có trong cache local ({model_name}). "
            "Hãy chạy scripts/download_models.py khi có mạng để chuẩn bị model. "
            f"Chi tiết: {exc}"
        )
    return f"Không thể nạp {component} ({model_name}): {exc}"
