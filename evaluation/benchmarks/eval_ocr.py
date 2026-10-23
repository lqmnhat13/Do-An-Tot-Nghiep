#!/usr/bin/env python3
"""Đánh giá OCR (OCRService của dự án) trên ảnh chụp thật tiếng Việt và tiếng Anh.

Nguồn dữ liệu:
- VinText (VinAI, CVPR 2021): chữ tiếng Việt trong ảnh cảnh thật (biển hiệu, đường phố).
- ICDAR 2015 Incidental Scene Text (test): chữ tiếng Anh chụp bằng Google Glass.
- SROIE (ICDAR 2019, test): hoá đơn tiếng Anh chụp/scan — tình huống "chụp văn bản".
- synthetic_vi_document: trang văn bản tiếng Việt TỔNG HỢP (make_vi_documents.py), chỉ để bổ sung.

Chỉ số (bag-of-words, không phân biệt hoa thường, bỏ dấu câu):
- word_f1: khớp chính xác cả dấu tiếng Việt.
- word_f1_no_accent: bỏ dấu trước khi so — chênh lệch với word_f1 = lỗi do dấu thanh.
"""

import argparse
import glob
import json
import os
import random
import time

import cv2
import numpy as np

from common import load_configs, normalize_text, save_json, summarize, word_prf


def load_vintext(root, n, rng):
    imgs = sorted(glob.glob(os.path.join(root, "vietnamese", "test_image", "*.jpg")))
    imgs = rng.sample(imgs, min(n, len(imgs))) if n else imgs
    for p in imgs:
        num = int(os.path.basename(p)[2:-4])
        words = []
        with open(os.path.join(root, "vietnamese", "labels", f"gt_{num}.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(",", 8)
                if len(parts) == 9 and parts[8] != "###":
                    words.append(parts[8])
        yield os.path.basename(p), cv2.imread(p), words


def load_icdar15(root, n, rng):
    imgs = sorted(glob.glob(os.path.join(root, "test_images", "*.jpg")))
    imgs = rng.sample(imgs, min(n, len(imgs))) if n else imgs
    for p in imgs:
        stem = os.path.basename(p)[:-4]
        words = []
        with open(os.path.join(root, "test_gt", f"gt_{stem}.txt"), encoding="utf-8-sig") as f:
            for line in f:
                parts = line.rstrip("\n").split(",", 8)
                if len(parts) == 9 and parts[8] != "###":
                    words.append(parts[8])
        yield os.path.basename(p), cv2.imread(p), words


def load_sroie(root, n, rng):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(root, "test.parquet")).to_pylist()
    rows = rng.sample(rows, min(n, len(rows))) if n else rows
    for r in rows:
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        yield r["key"], img, list(r["words"])


def load_vi_docs(root, n, rng):
    """Ảnh văn bản tiếng Việt tổng hợp từ make_vi_documents.py (không phải ảnh chụp thật)."""
    meta = json.load(open(os.path.join(root, "labels.json"), encoding="utf-8"))
    meta = rng.sample(meta, min(n, len(meta))) if n else meta
    for m in meta:
        yield m["image"], cv2.imread(os.path.join(root, m["image"])), m["text"].split()


DATASETS = {
    "vintext_vi_scene": ("vi", load_vintext, "vintext"),
    "icdar2015_en_scene": ("en", load_icdar15, "icdar2015"),
    "sroie_en_receipt": ("en", load_sroie, "sroie"),
    "synthetic_vi_document": ("vi", load_vi_docs, "vi_docs"),
}


def build_service(cfg, engine, device):
    from src.ocr.image_quality import ImageQualityChecker
    from src.ocr.ocr_service import OCRService
    o = cfg["model"].get("ocr", {})
    return OCRService(
        languages=o.get("languages", ["vi", "en"]),
        engine=engine,
        use_gpu=(device == "mps"),
        min_confidence=float(o.get("min_confidence", 0.35)),
        quality_checker=ImageQualityChecker(
            blur_threshold=float(o.get("blur_laplacian_threshold", 50.0)),
            min_brightness=float(o.get("min_brightness", 40.0)),
            max_brightness=float(o.get("max_brightness", 235.0)),
        ),
    )


def evaluate(service, samples, keep_examples=8):
    from src.contracts.request import OCRRequest
    rows, examples = [], []
    agg = {"strict": [0, 0, 0], "noacc": [0, 0, 0]}  # tp, n_pred, n_gt
    status = {"ok": 0, "rejected_quality": 0, "error": 0, "no_text": 0}
    for name, img, gt_words in samples:
        res = service.process(OCRRequest(request_id=name, image=img))
        if res.quality_status == "BAD_QUALITY":
            status["rejected_quality"] += 1
            pred = ""
        elif not res.success:
            status["error"] += 1
            pred = ""
        elif res.text.startswith("Không phát hiện"):
            status["no_text"] += 1
            pred = ""
        else:
            status["ok"] += 1
            pred = res.text
        gt_text = " ".join(gt_words)
        s = word_prf(normalize_text(pred).split(), normalize_text(gt_text).split())
        a = word_prf(normalize_text(pred, True).split(), normalize_text(gt_text, True).split())
        for k, m in (("strict", s), ("noacc", a)):
            agg[k][0] += m["tp"]
            agg[k][1] += m["n_pred"]
            agg[k][2] += m["n_gt"]
        rows.append({"image": name, "latency_sec": res.latency_sec, "f1": s["f1"], "f1_no_accent": a["f1"],
                     "quality_status": res.quality_status, "error": res.error_message})
        if len(examples) < keep_examples and gt_words:
            examples.append({"image": name, "gt": gt_text[:300], "pred": pred[:300],
                             "f1": round(s["f1"], 3), "f1_no_accent": round(a["f1"], 3)})

    def micro(t):
        tp, npred, ngt = t
        p = tp / npred if npred else 0.0
        r = tp / ngt if ngt else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}

    return {
        "num_images": len(rows), "status": status,
        "word_micro_strict": micro(agg["strict"]),
        "word_micro_no_accent": micro(agg["noacc"]),
        "per_image_f1": summarize([r["f1"] for r in rows]),
        "latency_sec": summarize([r["latency_sec"] for r in rows if r["quality_status"] != "BAD_QUALITY"]),
        "error_messages": sorted({r["error"] for r in rows if r["error"]})[:5],
        "examples": examples,
        "per_image": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Thư mục chứa vintext/, icdar2015/, sroie/")
    ap.add_argument("--engines", default="vision_unpatched,vision,easyocr")
    ap.add_argument("--n", type=int, default=150, help="Số ảnh mỗi bộ (0 = toàn bộ)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    cfg = load_configs()
    report = {"config_engine": cfg["model"]["ocr"].get("engine"), "results": {}}
    for engine in args.engines.split(","):
        if engine == "vision":
            from common import patch_vision_options
            patch_vision_options()
        service = build_service(cfg, "vision" if engine.startswith("vision") else engine, args.device)
        report["results"][engine] = {}
        for ds in args.datasets.split(","):
            lang, loader, sub = DATASETS[ds]
            rng = random.Random(0)
            t0 = time.time()
            r = evaluate(service, loader(os.path.join(args.data_dir, sub), args.n, rng))
            r["language"] = lang
            r["wall_sec"] = time.time() - t0
            report["results"][engine][ds] = r
            print(f"[{engine}] {ds}: F1={r['word_micro_strict']['f1']:.3f} "
                  f"F1_noacc={r['word_micro_no_accent']['f1']:.3f} status={r['status']} "
                  f"lat_med={r['latency_sec'].get('median', 0):.2f}s")
            if engine == "vision_unpatched" and r["status"]["error"] == r["num_images"]:
                print("  -> Vision chưa vá lỗi hoàn toàn; bỏ qua các bộ còn lại cho engine này.")
                break
    path = save_json("ocr.json", report)
    print("saved", path)


if __name__ == "__main__":
    main()
