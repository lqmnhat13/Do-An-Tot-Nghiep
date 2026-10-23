#!/usr/bin/env python3
"""Đánh giá nhận diện vật thể (YOLOv8n + ClassFilter của dự án) trên COCO val2017.

Nguồn dữ liệu: COCO 2017 (cocodataset.org), tập val mà YOLOv8n không huấn luyện.
Chỉ đánh giá 24 lớp enabled_classes trong configs/model_config.yaml.

Ví dụ:
    python evaluation/benchmarks/eval_detection_coco.py --coco-dir /path/coco --num-images 500
"""

import argparse
import io
import json
import os
import random
import time
import urllib.request
import zipfile
from contextlib import redirect_stdout

import cv2
import numpy as np

from common import build_class_filter, build_detector, load_configs, make_packet, save_json, summarize


def load_annotations(coco_dir):
    path = os.path.join(coco_dir, "instances_val2017.json")
    if not os.path.exists(path):
        with zipfile.ZipFile(os.path.join(coco_dir, "annotations_trainval2017.zip")) as zf:
            with zf.open("annotations/instances_val2017.json") as src, open(path, "wb") as dst:
                dst.write(src.read())
    return path


def ensure_images(coco_dir, file_names):
    img_dir = os.path.join(coco_dir, "val2017")
    os.makedirs(img_dir, exist_ok=True)
    for name in file_names:
        out = os.path.join(img_dir, name)
        if not os.path.exists(out):
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{name}", out)
    return img_dir


def degrade(image, kind, rng):
    if kind == "clean":
        return image
    if kind == "low_light":
        # Ánh sáng yếu: giảm độ sáng mạnh + nhiễu cảm biến (mô phỏng phòng tối).
        dark = (np.power(image.astype(np.float32) / 255.0, 2.2) * 255.0 * 0.45)
        noise = rng.normal(0, 6, image.shape)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)
    if kind == "motion_blur":
        k = np.zeros((15, 15), np.float32)
        k[7, :] = 1.0 / 15
        return cv2.filter2D(image, -1, k)
    if kind == "jpeg_q15":
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 15])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(kind)


def run_coco_eval(coco_gt, dets, img_ids, cat_ids):
    from pycocotools.cocoeval import COCOeval
    if not dets:
        return {"mAP50_95": 0.0, "mAP50": 0.0}
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    ev.params.catIds = cat_ids
    with redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    per_class = {}
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    # COCOeval sắp xếp lại catIds (np.unique), nên phải dùng đúng thứ tự của nó.
    for k, cid in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        p50 = precision[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        per_class[coco_gt.cats[cid]["name"]] = {
            "AP50_95": float(p.mean()) if p.size else None,
            "AP50": float(p50.mean()) if p50.size else None,
        }
    return {
        "mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
        "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
        "AR100": float(s[8]), "per_class": per_class,
    }


def operating_point(coco_gt, dets, img_ids, cat_ids, conf, iou_thr=0.5):
    """Precision/recall tại ngưỡng confidence thực tế của app (greedy matching IoU>=0.5)."""
    from pycocotools import mask as mask_utils
    tp = fp = n_gt = 0
    per_class = {cid: [0, 0, 0] for cid in cat_ids}  # tp, fp, gt
    by_img = {}
    for d in dets:
        if d["score"] >= conf:
            by_img.setdefault((d["image_id"], d["category_id"]), []).append(d)
    for img_id in img_ids:
        for cid in cat_ids:
            gts = [a for a in coco_gt.imgToAnns[img_id] if a["category_id"] == cid and not a.get("iscrowd", 0)]
            preds = sorted(by_img.get((img_id, cid), []), key=lambda d: -d["score"])
            n_gt += len(gts)
            per_class[cid][2] += len(gts)
            if not preds:
                continue
            if not gts:
                fp += len(preds)
                per_class[cid][1] += len(preds)
                continue
            ious = mask_utils.iou([p["bbox"] for p in preds], [g["bbox"] for g in gts], [0] * len(gts))
            used = set()
            for i in range(len(preds)):
                cand = [(ious[i][jj], jj) for jj in range(len(gts)) if jj not in used]
                best = max(cand) if cand else (0, -1)
                if best[0] >= iou_thr:
                    used.add(best[1])
                    tp += 1
                    per_class[cid][0] += 1
                else:
                    fp += 1
                    per_class[cid][1] += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_gt if n_gt else 0.0
    return {
        "confidence": conf, "tp": tp, "fp": fp, "gt": n_gt,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "per_class": {
            coco_gt.cats[c]["name"]: {
                "precision": v[0] / (v[0] + v[1]) if v[0] + v[1] else None,
                "recall": v[0] / v[2] if v[2] else None, "gt": v[2]}
            for c, v in per_class.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", required=True)
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--robust-images", type=int, default=150)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=None, help="Ghi đè detection.weights_path để so sánh model")
    ap.add_argument("--out", default="detection_coco.json")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    cfg = load_configs()
    if args.weights:
        cfg["model"]["detection"]["weights_path"] = args.weights
    class_filter = build_class_filter(cfg)
    with redirect_stdout(io.StringIO()):
        coco = COCO(load_annotations(args.coco_dir))
    name_to_id = {c["name"]: c["id"] for c in coco.cats.values()}
    enabled = sorted(class_filter.enabled_classes)
    cat_ids = [name_to_id[n] for n in enabled]
    alert_ids = [name_to_id[n] for n in sorted(class_filter.alert_classes)]

    candidates = sorted({a["image_id"] for a in coco.dataset["annotations"] if a["category_id"] in cat_ids})
    rng = random.Random(args.seed)
    img_ids = sorted(rng.sample(candidates, min(args.num_images, len(candidates))))
    files = {i: coco.imgs[i]["file_name"] for i in img_ids}
    img_dir = ensure_images(args.coco_dir, files.values())

    # Chạy ở conf thấp để tính mAP chuẩn; ngưỡng thực tế của app được áp lại sau.
    detector = build_detector(cfg, device=args.device, confidence=0.001)
    app_conf = float(cfg["model"]["detection"].get("confidence_threshold", 0.45))

    def detect_all(ids, kind="clean"):
        nrng = np.random.default_rng(args.seed)
        out, lat = [], []
        for idx, img_id in enumerate(ids):
            img = cv2.imread(os.path.join(img_dir, files[img_id]))
            img = degrade(img, kind, nrng)
            res = detector.detect(make_packet(img, idx))
            lat.append(res.latency_ms)
            for d in res.detections:
                b = d.bbox
                out.append({"image_id": img_id, "category_id": name_to_id[d.class_name],
                            "bbox": [b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin],
                            "score": d.confidence})
        return out, lat

    detect_all(img_ids[:5])  # warm-up MPS
    t0 = time.time()
    dets, latencies = detect_all(img_ids)
    report = {
        "dataset": "COCO val2017 (cocodataset.org)",
        "num_images": len(img_ids),
        "classes_evaluated": enabled,
        "num_gt_instances": sum(1 for a in coco.dataset["annotations"]
                                if a["image_id"] in set(img_ids) and a["category_id"] in cat_ids and not a.get("iscrowd")),
        "all_enabled_classes": run_coco_eval(coco, dets, img_ids, cat_ids),
        "alert_classes_only": {k: v for k, v in run_coco_eval(coco, dets, img_ids, alert_ids).items() if k != "per_class"},
        "operating_point_app_conf": operating_point(coco, dets, img_ids, cat_ids, app_conf),
        "operating_point_conf_0.45": operating_point(coco, dets, img_ids, cat_ids, 0.45),
        "latency_ms_detect_call": summarize(latencies),
        "wall_time_sec": time.time() - t0,
    }

    robust_ids = img_ids[: args.robust_images]
    report["robustness"] = {}
    for kind in ("clean", "low_light", "motion_blur", "jpeg_q15"):
        d, _ = detect_all(robust_ids, kind)
        r = run_coco_eval(coco, d, robust_ids, cat_ids)
        op = operating_point(coco, d, robust_ids, cat_ids, app_conf)
        report["robustness"][kind] = {"mAP50_95": r["mAP50_95"], "mAP50": r["mAP50"],
                                      "precision@app_conf": op["precision"], "recall@app_conf": op["recall"]}
        print(f"[robust] {kind}: mAP50={r['mAP50']:.3f} R={op['recall']:.3f}")

    report["weights"] = cfg["model"]["detection"].get("weights_path")
    path = save_json(args.out, report)
    a = report["all_enabled_classes"]
    o = report["operating_point_app_conf"]
    print(json.dumps({"mAP50_95": a["mAP50_95"], "mAP50": a["mAP50"], "P": o["precision"], "R": o["recall"],
                      "lat_median_ms": report["latency_ms_detect_call"]["median"]}, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
