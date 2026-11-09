#!/usr/bin/env python3
"""MC-OCR 2021 (hoá đơn bán lẻ tiếng Việt chụp thật, RIVF 2021) qua OCRService của dự án.

Nguồn: Hugging Face `DThai/mcocr-test` (split validation, 289 ảnh, GT dạng trường).

GT chỉ có các trường (tên cửa hàng, địa chỉ, ngày, tổng tiền...), nên đo
"tỉ lệ trường đọc được": giá trị trường (bỏ ký tự không chữ/số) có nằm trong
văn bản OCR (đã chuẩn hoá tương tự) hay không. Có dấu và bỏ dấu.

Chạy: python eval_mcocr.py --parquet $W/data/mcocr/val.parquet --n 120 --engines vision
"""
import argparse
import json
import statistics
import time
import unicodedata

import common  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.contracts.request import OCRRequest  # noqa: E402
from src.ocr.image_quality import ImageQualityChecker  # noqa: E402
from src.ocr.ocr_service import OCRService  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--engines", default="vision", help="vision,easyocr")
ap.add_argument("--out", default="mcocr.json")
args = ap.parse_args()
N = args.n
FIELDS = {"Tên cửa hàng": "store", "Địa chỉ": "address", "Ngày": "date",
          "Tổng số tiền": "total", "Tổng tiền": "total", "Tổng cộng": "total",
          "Sản phẩm": "items"}


def norm(s, strip_accents=False):
    s = unicodedata.normalize("NFC", str(s)).lower().replace("đ", "d" if strip_accents else "đ")
    if strip_accents:
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return "".join(c for c in s if c.isalnum())


cfg = common.load_configs()["model"]["ocr"]
engines = args.engines.split(",")
t = pq.read_table(args.parquet).to_pylist()[:N]
report = {}
for engine in engines:
    svc = OCRService(languages=cfg["languages"], engine=engine, min_confidence=cfg["min_confidence"],
                     quality_checker=ImageQualityChecker(blur_threshold=cfg["blur_laplacian_threshold"],
                                                         min_brightness=cfg["min_brightness"],
                                                         max_brightness=cfg["max_brightness"]))
    per_field = {}
    lat, bad, rows = [], 0, []
    for i, r in enumerate(t):
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        gt = json.loads(r["ground_truth"])["gt_parse"]
        t0 = time.monotonic()
        res = svc.process(OCRRequest(request_id=str(i), image=img))
        lat.append(time.monotonic() - t0)
        bad += res.quality_status == "BAD_QUALITY"
        text, text_na = norm(res.text), norm(res.text, True)
        hits = {}
        for k, key in FIELDS.items():
            if k not in gt:
                continue
            vals = gt[k] if isinstance(gt[k], list) else [gt[k]]
            for val in vals:
                nv, nva = norm(val), norm(val, True)
                if len(nva) < 2:
                    continue
                d = per_field.setdefault(key, [0, 0, 0])
                d[0] += nv in text
                d[1] += nva in text_na
                d[2] += 1
                hits.setdefault(key, []).append(nv in text)
        rows.append({"i": i, "shape": img.shape[:2], "quality": res.quality_status,
                     "latency": round(lat[-1], 2), "hits": hits, "text": res.text[:400],
                     "gt": {k: gt[k] for k in gt if k in FIELDS}})
    report[engine] = {
        "n": len(t), "bad_quality": bad,
        "latency_median_s": round(statistics.median(lat), 3),
        "latency_p90_s": round(float(np.percentile(lat, 90)), 3),
        "field_recall": {k: {"exact": round(v[0] / v[2], 3), "no_accent": round(v[1] / v[2], 3), "n": v[2]}
                         for k, v in per_field.items()},
        "rows": rows,
    }
    print(engine, json.dumps({k: v for k, v in report[engine].items() if k != "rows"}, ensure_ascii=False))
report["dataset"] = "MC-OCR 2021 validation (DThai/mcocr-test), {} ảnh đầu".format(N)
print("saved", common.save_json(args.out, report))
