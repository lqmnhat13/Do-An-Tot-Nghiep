"""Chính sách tải model dùng chung cho runtime offline."""

import os
from pathlib import Path
from typing import Dict, Union

# Mọi model của dự án nằm dưới models/weights/ (đã được .gitignore bỏ qua) thay vì cache chung
# của máy (~/.cache/huggingface, ~/.EasyOCR), để dễ kiểm soát và mang sang máy khác.
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "weights"
HF_CACHE_DIR = str(MODEL_DIR / "huggingface")
EASYOCR_DIR = str(MODEL_DIR / "easyocr")


OFFLINE_ENV_VAR = "SECOND_EYE_OFFLINE"
HF_HUB_OFFLINE_ENV_VAR = "HF_HUB_OFFLINE"
TRANSFORMERS_OFFLINE_ENV_VAR = "TRANSFORMERS_OFFLINE"
_FALSE_VALUES = {"0", "false", "no", "off"}


def is_offline_mode() -> bool:
    """Runtime mặc định offline; chỉ bật mạng khi người dùng chủ động opt-in."""
    value = os.getenv(OFFLINE_ENV_VAR, "1")
    return value.strip().lower() not in _FALSE_VALUES


def pretrained_kwargs(*, allow_download: bool = False) -> Dict[str, Union[bool, str]]:
    """Các tham số thống nhất cho mọi lời gọi Hugging Face from_pretrained."""
    return {"local_files_only": is_offline_mode() and not allow_download,
            "cache_dir": HF_CACHE_DIR}


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
