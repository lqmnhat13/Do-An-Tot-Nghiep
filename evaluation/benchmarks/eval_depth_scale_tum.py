#!/usr/bin/env python3
"""Đo hệ số tỉ lệ tối ưu (GT/dự đoán thô) của Depth-Anything-V2-Metric-Indoor trên các chuỗi TUM.

So với metric_scale = 0.60 (fit trên NYU) để thấy sai số khoảng cách khi đổi camera/cảnh.
Lấy 1/15 frame mỗi chuỗi; hệ số mỗi frame = trung vị(GT / dự đoán thô) trên pixel GT 0.3–6 m.

Chạy: python eval_depth_scale_tum.py --tum-dir $W/data/tum --out depth_scale_tum.json
"""
import argparse
import json
import os

import common  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tum-dir", required=True, help="Thư mục chứa các rgbd_dataset_*")
ap.add_argument("--out", default="depth_scale_tum.json")
args = ap.parse_args()
cfg = common.load_configs()
depth = common.build_depth(cfg, metric_scale=1.0)
out = {}
for seq in sorted(os.listdir(args.tum_dir)):
    d = os.path.join(args.tum_dir, seq)
    if not os.path.isdir(d):
        continue
    rgb = [l.split() for l in open(os.path.join(d, "rgb.txt")) if not l.startswith("#")]
    dep = [l.split() for l in open(os.path.join(d, "depth.txt")) if not l.startswith("#")]
    dts = np.array([float(t) for t, _ in dep])
    ratios, abs_err_060, abs_err_best, near = [], [], [], []
    for k in range(0, len(rgb), 15):
        t, p = float(rgb[k][0]), rgb[k][1]
        j = int(np.argmin(np.abs(dts - t)))
        if abs(dts[j] - t) > 0.02:
            continue
        img = cv2.imread(os.path.join(d, p))
        gt = cv2.imread(os.path.join(d, dep[j][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        pred = depth.estimate(common.make_packet(img, k)).values
        m = (gt > 0.3) & (gt < 6) & (pred > 0)
        if m.sum() < 1000:
            continue
        ratios.append(float(np.median(gt[m] / pred[m])))
        near.append((gt[m], pred[m]))
    s = float(np.median(ratios))
    e060 = np.concatenate([np.abs(0.60 * p - g) for g, p in near])
    ebest = np.concatenate([np.abs(s * p - g) for g, p in near])
    g_all = np.concatenate([g for g, _ in near])
    p_all = np.concatenate([p for _, p in near])
    band = (g_all > 1.0) & (g_all < 2.0)
    out[seq] = {
        "frames": len(ratios), "best_scale_median": round(s, 3),
        "scale_p10_p90": [round(float(np.percentile(ratios, 10)), 3), round(float(np.percentile(ratios, 90)), 3)],
        "abs_err_m_scale_0.60_median": round(float(np.median(e060)), 3),
        "abs_err_m_best_scale_median": round(float(np.median(ebest)), 3),
        "gt_1_2m_pred_with_0.60_median": round(float(np.median(0.60 * p_all[band])), 3) if band.any() else None,
    }
    print(seq, json.dumps(out[seq]))
print("saved", common.save_json(args.out, {"configured_metric_scale": 0.60, "sequences": out}))
