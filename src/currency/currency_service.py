"""Nhận diện mệnh giá tiền Việt Nam từ chữ in trên tờ tiền (OCR Apple Vision + luật).

Tờ tiền in số mệnh giá ("100000"/"100.000") và chữ ("MỘT TRĂM NGHÌN ĐỒNG"). Chỉ trả lời khi
bằng chứng đủ và không mâu thuẫn; còn lại nói "chưa đọc rõ". Đọc sai mệnh giá làm người dùng
mất tiền thật, nên không bao giờ đoán và không bao giờ nói tiền thật/giả.

Các bẫy đo được trên ảnh thật (Vision đọc): sai dấu ("ĐÓNG", "NGHIN", "MỘI TR"), chữ in ngược
("A00005"), số dính chữ ("5000m"), số seri ("WD 20123707"), cụm "VIỆT NAM NGHÌN ĐỒNG".
"""
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.contracts.request import OCRRequest

# Các mệnh giá tiền giấy/polymer đang lưu hành.
DENOMINATIONS: Dict[int, str] = {
    1000: "Một nghìn đồng",
    2000: "Hai nghìn đồng",
    5000: "Năm nghìn đồng",
    10000: "Mười nghìn đồng",
    20000: "Hai mươi nghìn đồng",
    50000: "Năm mươi nghìn đồng",
    100000: "Một trăm nghìn đồng",
    200000: "Hai trăm nghìn đồng",
    500000: "Năm trăm nghìn đồng",
}
_WORD_VALUES = {
    "MOT": 1000, "HAI": 2000, "NAM": 5000, "MUOI": 10000, "HAI MUOI": 20000,
    "NAM MUOI": 50000, "MOT TRAM": 100000, "HAI TRAM": 200000, "NAM TRAM": 500000,
}
# Số mệnh giá nhỏ dễ trùng năm in/ký hiệu: chỉ có số (không có chữ) thì phải thấy >= 2 lần.
_SMALL = {1000, 2000, 5000}

UNCLEAR_TEXT = "Chưa đọc rõ mệnh giá. Hãy để tờ tiền phẳng, gần camera, đủ sáng rồi bấm M lại."
MIXED_TEXT = "Thấy nhiều mệnh giá khác nhau. Hãy đưa từng tờ một trước camera."

# Số đứng riêng (không dính chữ/số/ký hiệu khác): 100000, 100.000, 100,000, 100 000.
# "5000~", "5000-" là mảnh của "500000" bị che (đo trên ảnh xấp tiền), nên cũng bị loại.
_NUMBER = re.compile(r"(?<![\w.,~\-/])(\d{1,3}(?:[.,]\d{3})+|\d{1,3}(?: \d{3})+|\d{4,6})"
                     r"(?![\w,~\-/]|\.\d)")
# Chữ mệnh giá: phương án dài đứng trước để "HAI MUOI" không bị đọc thành "MUOI".
_WORDS = re.compile(r"\b(HAI MUOI|NAM MUOI|MOT TRAM|HAI TRAM|NAM TRAM|MUOI|MOT|HAI|NAM)\s+(?:NGHIN|NGAN)\b")
# Cụm chứa "NAM"/"NGAN" không phải mệnh giá; bỏ trước khi tìm chữ mệnh giá.
_NOT_WORDS = re.compile(r"\bVIET\s*NAM\b|\bNGAN\s+HANG\b")
# Dấu hiệu chữ in trên tờ tiền: số mà không có dấu hiệu nào thì có thể là bảng giá.
_BANKNOTE_MARK = re.compile(r"NGAN\s+HANG\s+NHA\s+NUOC|CONG\s+HOA\s+XA\s+HOI|\bDONG\b|\bNGHIN\b|\bNGAN\b")


@dataclass(frozen=True)
class BanknoteReading:
    status: str  # "OK" | "MIXED" | "UNCLEAR"
    denomination: Optional[int] = None
    denominations: Tuple[int, ...] = ()


def _fold(text: str) -> str:
    """Bỏ dấu và viết hoa: Vision hay đọc sai dấu ("ĐÓNG", "NGHIN")."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.replace("đ", "d").replace("Đ", "D").upper().split())


def parse_banknote_text(text: str) -> BanknoteReading:
    folded = _fold(text or "")
    numbers: Dict[int, int] = {}
    for match in _NUMBER.finditer(folded):
        value = int(re.sub(r"[., ]", "", match.group(1)))
        if value in DENOMINATIONS:
            numbers[value] = numbers.get(value, 0) + 1
    words = {_WORD_VALUES[m.group(1)] for m in _WORDS.finditer(_NOT_WORDS.sub(" | ", folded))}
    marked = bool(_BANKNOTE_MARK.search(folded))

    found = set(words)
    for value, count in numbers.items():
        if value in words or (marked and (value not in _SMALL or count >= 2)):
            found.add(value)
        elif words:
            found.add(value)  # số mâu thuẫn với chữ mệnh giá -> MIXED, không bỏ qua
    if not found:
        return BanknoteReading("UNCLEAR")
    if len(found) > 1:
        return BanknoteReading("MIXED", denominations=tuple(sorted(found)))
    value = found.pop()
    return BanknoteReading("OK", denomination=value, denominations=(value,))


def spoken_text(reading: BanknoteReading) -> str:
    if reading.status == "OK":
        return f"{DENOMINATIONS[reading.denomination]}."
    return MIXED_TEXT if reading.status == "MIXED" else UNCLEAR_TEXT


@dataclass(frozen=True)
class CurrencyResult:
    text: str
    reading: BanknoteReading
    latency_sec: float
    cancelled: bool = False


class CurrencyService:
    """Nhận diện mệnh giá qua OCRService có sẵn (chung tiến trình Vision, huỷ và dự phòng EasyOCR)."""

    def __init__(self, ocr_service):
        self.ocr_service = ocr_service

    @staticmethod
    def spoken_texts() -> List[str]:
        """Mọi câu có thể đọc, để TTSEngine dựng sẵn clip."""
        return [f"{words}." for words in DENOMINATIONS.values()] + [UNCLEAR_TEXT, MIXED_TEXT]

    def recognize(self, image: np.ndarray, request_id: str = "money") -> CurrencyResult:
        t0 = time.monotonic()
        res = self.ocr_service.process(OCRRequest(request_id=request_id, image=image))
        if res.error_message == "cancelled":
            return CurrencyResult("", BanknoteReading("UNCLEAR"), time.monotonic() - t0, cancelled=True)
        # OCR lỗi/ảnh kém: text là câu hướng dẫn của OCR, không phải chữ trên tờ tiền.
        reading = parse_banknote_text(res.text if res.success else "")
        return CurrencyResult(spoken_text(reading), reading, time.monotonic() - t0)
