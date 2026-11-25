#!/usr/bin/env python3
"""Đánh giá nhận diện mệnh giá tiền (OCR Apple Vision + luật) trên ảnh có nhãn.

Nhãn lấy từ tiền tố tên tệp: `100000_mat_truoc.jpg` (mệnh giá) hoặc `none_tui_xach.jpg` (không có
tờ tiền đọc được). Chỉ số quan trọng nhất là `wrong` (đọc ra sai mệnh giá): phải bằng 0; trả lời
"chưa đọc rõ" (`unclear`) thì chấp nhận được.

Chạy: python evaluation/benchmarks/eval_currency.py --images thu_muc_anh --size 640x480 --out kq.json
"""
import argparse
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2

from src.currency.currency_service import DENOMINATIONS, CurrencyService
from src.ocr.ocr_service import OCRService


def label_of(path: str):
    prefix = os.path.basename(path).split("_")[0].lower()
    return int(prefix) if prefix.isdigit() and int(prefix) in DENOMINATIONS else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", required=True, help="Thư mục ảnh có nhãn ở tiền tố tên tệp")
    parser.add_argument("--size", default="640x480", help="Đổi kích thước như camera của app; 'gốc' để giữ nguyên")
    parser.add_argument("--engine", default="vision", choices=["vision", "easyocr"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ocr = OCRService(engine=args.engine, languages=["vi", "en"], read_timeout_sec=10.0)
    service = CurrencyService(ocr)
    rows, counts = [], {"correct": 0, "unclear": 0, "wrong": 0, "false_alarm": 0, "correct_none": 0}
    try:
        for path in sorted(glob.glob(os.path.join(args.images, "*"))):
            image = cv2.imread(path)
            if image is None:
                continue
            if args.size != "gốc":
                w, h = (int(v) for v in args.size.lower().split("x"))
                image = cv2.resize(image, (w, h))
            label = label_of(path)
            result = service.recognize(image)
            got = result.reading.denomination
            if label is None:
                outcome = "false_alarm" if got is not None else "correct_none"
            elif got is None:
                outcome = "unclear"
            else:
                outcome = "correct" if got == label else "wrong"
            counts[outcome] += 1
            rows.append({"file": os.path.basename(path), "label": label, "got": got,
                         "status": result.reading.status, "spoken": result.text,
                         "outcome": outcome, "latency_sec": round(result.latency_sec, 3)})
            print(f"{outcome:13s} {os.path.basename(path)}: nhãn={label} đọc={got} ({result.reading.status})")
    finally:
        ocr.close()
    print("Tổng:", counts)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"size": args.size, "engine": args.engine, "counts": counts, "rows": rows},
                      f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
