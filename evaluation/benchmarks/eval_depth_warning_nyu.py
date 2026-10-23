#!/usr/bin/env python3
"""Đánh giá độ sâu và quyết định cảnh báo HIGH trên NYU Depth V2 (tập test 654 ảnh).

Nguồn: NYU Depth V2 (Silberman et al., ECCV 2012), bản đóng gói
huggingface.co/datasets/sayakpaul/nyu_depth_v2 (val = tập test chuẩn 654 ảnh).

Hai câu hỏi được trả lời:
1. Độ sâu: sai số của Depth Anything V2 Small (cấu hình app 256x256) theo chuẩn
   AbsRel/δ1 sau khi căn scale+shift (vì model chỉ cho độ sâu tương đối).
2. Cảnh báo: với mỗi vật thuộc alert_classes nằm trong hành lang, điểm "độ gần"
   của hệ thống (chuẩn hoá min-max theo frame) có tương ứng với khoảng cách thật
   (mét) không? HIGH có đúng là vật gần không? So sánh với phương án dùng model
   metric (Depth-Anything-V2-Metric-Indoor-Small) và ngưỡng theo mét.
"""

import argparse
import dataclasses
import io
import json
import os
import tarfile
import time

import cv2
import numpy as np

from common import (build_depth, build_detector, build_risk_stack, load_configs, make_packet,
                    save_json, summarize)

EIGEN_CROP = (slice(45, 471), slice(41, 601))
MIN_D, MAX_D = 1e-3, 10.0


def iter_nyu(tar_paths, limit):
    import h5py
    n = 0
    for tp in tar_paths:
        with tarfile.open(tp) as tf:
            for m in tf:
                if not m.name.endswith(".h5"):
                    continue
                with h5py.File(io.BytesIO(tf.extractfile(m).read()), "r") as f:
                    rgb = np.transpose(np.array(f["rgb"]), (1, 2, 0))
                    depth = np.array(f["depth"]).astype(np.float32)
                yield m.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth
                n += 1
                if limit and n >= limit:
                    return


def depth_metrics(pred, gt):
    thresh = np.maximum(gt / pred, pred / gt)
    return {
        "abs_rel": float(np.mean(np.abs(gt - pred) / gt)),
        "rmse": float(np.sqrt(np.mean((gt - pred) ** 2))),
        "delta1": float(np.mean(thresh < 1.25)),
    }


def align_disparity(pred_disp, gt_depth, mask):
    """Căn scale+shift trong không gian disparity (chuẩn cho model độ sâu tương đối)."""
    x = pred_disp[mask].astype(np.float64)
    y = 1.0 / gt_depth[mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], 1)
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    disp = s * pred_disp + t
    return 1.0 / np.clip(disp, 1.0 / MAX_D, 1.0 / MIN_D)


def eroded_roi(bbox, ratio, w, h):
    pad_x, pad_y = bbox.width * ratio, bbox.height * ratio
    x1, y1 = int(max(0, bbox.xmin + pad_x)), int(max(0, bbox.ymin + pad_y))
    x2, y2 = int(min(w, bbox.xmax - pad_x)), int(min(h, bbox.ymax - pad_y))
    return (slice(y1, y2), slice(x1, x2)) if x2 > x1 and y2 > y1 else None


def classification(rows, pred_key, gt_limit):
    tp = sum(1 for r in rows if r[pred_key] and r["gt_dist_m"] <= gt_limit)
    fp = sum(1 for r in rows if r[pred_key] and r["gt_dist_m"] > gt_limit)
    fn = sum(1 for r in rows if not r[pred_key] and r["gt_dist_m"] <= gt_limit)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * rc / (p + rc) if p and rc else 0.0
    return {"gt_near_limit_m": gt_limit, "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": rc, "f1": f1}


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation) if len(a) > 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyu-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from src.contracts.risk import DataQuality, RiskLevel
    from src.fusion.synchronizer import SynchronizedPair

    from src.contracts.depth_map import DepthRepresentation

    cfg = load_configs()
    detector = build_detector(cfg, device=args.device)
    depth_sys = build_depth(cfg, device=args.device)  # đúng cấu hình app hiện tại
    depth_rel = build_depth(cfg, device=args.device, representation="relative_depth",
                            model_name="depth-anything/Depth-Anything-V2-Small-hf")
    for est in (depth_sys, depth_rel):
        if est.load_error:
            raise SystemExit(f"Không nạp được depth model: {est.load_error}")
    erosion = float(cfg["model"]["depth"].get("roi_erosion_ratio", 0.15))
    sys_is_metric = depth_sys.representation == DepthRepresentation.METRIC_DEPTH

    tars = sorted(os.path.join(args.nyu_dir, f) for f in os.listdir(args.nyu_dir) if f.endswith(".tar"))
    depth_rows = {"system": [], "relative_small_256_aligned": []}
    obj_rows, generic_rows = [], []
    lat = {"depth_system_ms": [], "depth_relative_ms": [], "detect_ms": []}

    for idx, (name, bgr, gt) in enumerate(iter_nyu(tars, args.limit)):
        h, w = gt.shape
        packet = make_packet(bgr, idx)
        crop = np.zeros_like(gt, dtype=bool)
        crop[EIGEN_CROP] = True
        mask = crop & (gt > MIN_D) & (gt < MAX_D)

        outs = {}
        for key, est, lkey in (("system", depth_sys, "depth_system_ms"),
                               ("relative_small_256_aligned", depth_rel, "depth_relative_ms")):
            t0 = time.monotonic()
            dm = est.estimate(packet)
            lat[lkey].append((time.monotonic() - t0) * 1000)
            outs[key] = dm
            if dm.representation == DepthRepresentation.METRIC_DEPTH:
                pred = np.clip(dm.values, MIN_D, MAX_D)  # mét (đã nhân metric_scale), không căn theo GT
            else:
                pred = align_disparity(dm.values, gt, mask)
            depth_rows[key].append(depth_metrics(pred[mask], gt[mask]))

        # --- Quyết định cảnh báo theo đúng code RiskFSM của dự án ---
        det = detector.detect(packet)
        lat["detect_ms"].append(det.latency_ms)
        dets = [dataclasses.replace(d, track_id=i + 1) for i, d in enumerate(det.detections)]
        det = dataclasses.replace(det, detections=dets)
        fsm, roi, corridor = build_risk_stack(cfg)
        dm_sys = dataclasses.replace(outs["system"], frame_id=idx)
        pair = SynchronizedPair(detection_result=det, depth_map=dm_sys, data_quality=DataQuality.VALID,
                                pair_skew_ms=0.0, effective_timestamp=packet.timestamp_mono)
        assessments = fsm.update(pair, current_mono=packet.timestamp_mono)
        metric_vals = outs["system"].values if sys_is_metric else None
        for a in assessments[len(dets):]:  # cảnh báo vật cản chung (không gắn detection)
            b = fsm.generic_obstacle_detector.find(dm_sys, [d.bbox for d in dets]).bbox
            region = gt[int(b.ymin):int(b.ymax), int(b.xmin):int(b.xmax)]
            region = region[(region > MIN_D) & (region < MAX_D)]
            generic_rows.append({"image": name, "reason": a.reason,
                                 "gt_dist_m": float(np.percentile(region, 20)) if region.size else None})
        for d, a in zip(dets, assessments):
            if not fsm.class_filter.is_alert_candidate(d.class_name):
                continue
            overlap = corridor.overlap_ratio(d.bbox, w, h)
            if overlap is None or overlap < corridor.min_overlap_ratio:
                continue
            sl = eroded_roi(d.bbox, erosion, w, h)
            if sl is None:
                continue
            g = gt[sl][(gt[sl] > MIN_D) & (gt[sl] < MAX_D)]
            if g.size < 15:
                continue
            obj_rows.append({
                "image": name, "class": d.class_name, "conf": d.confidence,
                "system_proximity": a.relative_proximity,
                "system_risk": a.risk_level.value,
                "system_high": a.risk_level == RiskLevel.HIGH,
                # Khoảng cách thật tới phần gần của vật (phân vị 20 của GT trong ROI).
                "gt_dist_m": float(np.percentile(g, 20)),
                "metric_pred_m": float(np.percentile(metric_vals[sl], 20)) if sys_is_metric else None,
            })
        if idx % 50 == 0:
            print(f"[{idx}] objects_in_corridor={len(obj_rows)}")

    report = {"dataset": "NYU Depth V2 test (654 ảnh, Eigen crop, GT Kinect)", "num_images": len(depth_rows["system"]),
              "system_depth": {"model": depth_sys.model_name, "representation": depth_sys.representation.value,
                               "metric_scale": depth_sys.metric_scale if sys_is_metric else None}}
    report["depth_accuracy"] = {
        k: {m: float(np.mean([r[m] for r in v])) for m in ("abs_rel", "rmse", "delta1")}
        for k, v in depth_rows.items()
    }
    report["depth_accuracy"]["_note"] = ("relative: đã căn scale+shift theo GT từng ảnh (cận trên, hệ thống thật "
                                         "không có GT); metric: mét sau metric_scale, không căn theo GT.")
    prox = [r["system_proximity"] for r in obj_rows]
    dist = [r["gt_dist_m"] for r in obj_rows]
    high_d = [r["gt_dist_m"] for r in obj_rows if r["system_high"]]
    report["warning_objects"] = {
        "num_alert_objects_in_corridor": len(obj_rows),
        "num_system_high": sum(r["system_high"] for r in obj_rows),
        "spearman_proximity_vs_neg_distance": spearman(prox, [-d for d in dist]),
        "gt_distance_of_HIGH_objects_m": summarize(high_d),
        "gt_distance_of_non_HIGH_objects_m": summarize([r["gt_dist_m"] for r in obj_rows if not r["system_high"]]),
    }
    if sys_is_metric:
        report["warning_objects"]["metric_pred_abs_err_m"] = summarize(
            [abs(r["metric_pred_m"] - r["gt_dist_m"]) for r in obj_rows])
    # Nửa ảnh "giữ lại" (không dùng để fit metric_scale=0.60): ảnh ở vị trí lẻ trong danh sách tên đã sắp xếp.
    imgs = sorted({r["image"] for r in obj_rows})
    heldout = set(imgs[1::2])
    comps = {}
    for limit in (1.0, 1.5, 2.0):
        comps[f"gt<= {limit}m"] = {
            "system_HIGH_all": classification(obj_rows, "system_high", limit),
            "system_HIGH_heldout": classification([r for r in obj_rows if r["image"] in heldout], "system_high", limit),
        }
    report["warning_vs_ground_truth"] = comps
    gd = [r["gt_dist_m"] for r in generic_rows if r["gt_dist_m"] is not None]
    report["generic_obstacle"] = {
        "images_with_alert": len(generic_rows),
        "gt_distance_m": summarize(gd),
        "precision_gt<=1.0m": float(np.mean([d <= 1.0 for d in gd])) if gd else None,
        "precision_gt<=1.5m": float(np.mean([d <= 1.5 for d in gd])) if gd else None,
        "samples": generic_rows[:20],
    }
    report["latency_ms"] = {k: summarize(v) for k, v in lat.items()}
    report["objects"] = obj_rows
    path = save_json("depth_warning_nyu.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("objects",)}, ensure_ascii=False, indent=1)[:4000])
    print("saved", path)


if __name__ == "__main__":
    main()
