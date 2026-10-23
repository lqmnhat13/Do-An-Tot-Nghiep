#!/usr/bin/env python3
"""Đánh giá chế độ Cảnh báo theo thời gian trên chuỗi video thật TUM RGB-D.

Nguồn: TUM RGB-D benchmark (Sturm et al., IROS 2012), cvg.cit.tum.de/data/datasets/rgbd-dataset.
Ảnh 640x480 @30Hz kèm độ sâu Kinect (png 16-bit, chia 5000 = mét).

Pipeline chạy theo đúng nhịp runtime: detection + ByteTrack mỗi frame, depth ~8 Hz
(mỗi 4 frame), Synchronizer kiểm tra lệch/tuổi dữ liệu theo timestamp thật,
RiskFSM + AlertAggregator quyết định câu cảnh báo được phát.

Ground truth (độc lập với model):
- Mức vật: khoảng cách thật (phân vị 20 GT trong ROI) của mỗi vật alert_classes trong hành lang.
- Mức frame: "vật cản gần" = phân vị 10 GT trong phần trên 75% của hành lang < ngưỡng
  (bỏ 25% đáy để không tính sàn nhà), bất kể vật có thuộc lớp COCO nào.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from common import (build_depth, build_detector, build_risk_stack, load_configs, make_packet,
                    save_json, summarize)


def read_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            t, p = line.split()[:2]
            rows.append((float(t), p))
    return rows


def associate(rgb, depth, max_dt=0.02):
    dts = np.array([t for t, _ in depth])
    out = []
    for t, p in rgb:
        j = int(np.argmin(np.abs(dts - t)))
        if abs(dts[j] - t) <= max_dt:
            out.append((t, p, depth[j][1]))
    return out


def corridor_mask(corridor, w, h, skip_bottom=0.25):
    poly = np.array(corridor.polygon(w, h), dtype=np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    m[int(h * (1 - skip_bottom)):, :] = 0
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="Thư mục rgbd_dataset_freiburg*_*")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--depth-every", type=int, default=4, help="30fps/4 ≈ 7.5 Hz như depth worker")
    ap.add_argument("--near-m", type=float, default=1.0)
    args = ap.parse_args()

    from src.fusion.alert_aggregator import AlertAggregator
    from src.fusion.synchronizer import Synchronizer
    from src.contracts.risk import RiskLevel
    from src.tracking.byte_tracker import ByteTrackerAdapter

    cfg = load_configs()
    detector = build_detector(cfg, device=args.device)
    depth_est = build_depth(cfg, device=args.device)
    tracker = ByteTrackerAdapter(iou_threshold=0.3)
    sc = cfg["fusion"].get("synchronizer", {})
    sync = Synchronizer(max_pair_skew_ms=float(sc.get("max_pair_skew_ms", 600)),
                        max_detection_age_ms=float(sc.get("max_detection_age_ms", 1000)),
                        max_depth_age_ms=float(sc.get("max_depth_age_ms", 1200)),
                        max_alert_age_ms=float(sc.get("max_alert_age_ms", 1500)))
    fsm, roi, corridor = build_risk_stack(cfg)
    ac = cfg["fusion"].get("alert_aggregator", {})
    aggregator = AlertAggregator(
        risk_fsm=fsm,
        alert_lifetime_sec=float(ac.get("alert_lifetime_sec", 3.0)),
        global_alert_interval_sec=float(ac.get("global_alert_interval_sec", 3.5)),
        max_detection_age_ms=float(sc.get("max_detection_age_ms", 1000)),
        max_depth_age_ms=float(sc.get("max_depth_age_ms", 1200)),
        max_alert_age_ms=float(sc.get("max_alert_age_ms", 1500)),
    )
    erosion = float(cfg["model"]["depth"].get("roi_erosion_ratio", 0.15))

    frames = associate(read_list(os.path.join(args.seq_dir, "rgb.txt")),
                       read_list(os.path.join(args.seq_dir, "depth.txt")))
    t_base = frames[0][0]
    obj_rows, frame_rows, spoken = [], [], []
    lat_det, lat_depth = [], []
    cmask = None

    for idx, (t, rgb_p, d_p) in enumerate(frames):
        ts = 1000.0 + (t - t_base)  # timestamp giả lập kiểu monotonic
        bgr = cv2.imread(os.path.join(args.seq_dir, rgb_p))
        gt = cv2.imread(os.path.join(args.seq_dir, d_p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        h, w = gt.shape
        if cmask is None:
            cmask = corridor_mask(corridor, w, h)
        packet = make_packet(bgr, idx, timestamp=ts)

        if idx % args.depth_every == 0:
            t0 = time.monotonic()
            dm = depth_est.estimate(packet)
            lat_depth.append((time.monotonic() - t0) * 1000.0)
            sync.register_depth(dm)
        det = detector.detect(packet)
        lat_det.append(det.latency_ms)
        tracked = tracker.update(det)
        pair = sync.synchronize(tracked, current_mono=ts)
        assessments = fsm.update(pair, current_mono=ts)
        task = aggregator.aggregate(assessments, current_mono=ts)
        if task is not None:
            spoken.append({"t": t - t_base, "text": task.text})

        any_high = any(a.risk_level == RiskLevel.HIGH for a in assessments)
        g = gt[cmask & (gt > 0.1)]
        near_any = bool(g.size > 500 and np.percentile(g, 10) < args.near_m)
        frame_rows.append({"t": t - t_base, "near_any_gt": near_any, "system_high": any_high,
                           "generic_high": any(a.track_id == -100 for a in assessments),
                           "quality": pair.data_quality.value})

        for d, a in zip(tracked.detections, assessments):
            if not fsm.class_filter.is_alert_candidate(d.class_name):
                continue
            ov = corridor.overlap_ratio(d.bbox, w, h)
            if ov is None or ov < corridor.min_overlap_ratio:
                continue
            b = d.bbox
            x1, y1 = int(b.xmin + b.width * erosion), int(b.ymin + b.height * erosion)
            x2, y2 = int(b.xmax - b.width * erosion), int(b.ymax - b.height * erosion)
            roi_gt = gt[y1:y2, x1:x2]
            roi_gt = roi_gt[roi_gt > 0.1]
            if roi_gt.size < 15:
                continue
            obj_rows.append({"t": t - t_base, "class": d.class_name, "track": d.track_id,
                             "risk": a.risk_level.value, "prox": a.relative_proximity,
                             "gt_dist_m": float(np.percentile(roi_gt, 20))})

    def cls(rows, limit):
        valid = [r for r in rows if r["risk"] not in ("UNDETERMINED",)]
        tp = sum(1 for r in valid if r["risk"] == "HIGH" and r["gt_dist_m"] <= limit)
        fp = sum(1 for r in valid if r["risk"] == "HIGH" and r["gt_dist_m"] > limit)
        fn = sum(1 for r in valid if r["risk"] != "HIGH" and r["gt_dist_m"] <= limit)
        return {"limit_m": limit, "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None}

    near_frames = [f for f in frame_rows if f["near_any_gt"]]
    duration = frame_rows[-1]["t"] if frame_rows else 0
    from collections import Counter
    report = {
        "sequence": os.path.basename(args.seq_dir.rstrip("/")),
        "dataset": "TUM RGB-D (cvg.cit.tum.de)",
        "frames": len(frame_rows), "duration_sec": duration,
        "data_quality_counts": dict(Counter(f["quality"] for f in frame_rows)),
        "frame_level": {
            "frames_with_near_obstacle_gt(<%.1fm, any object)" % args.near_m: len(near_frames),
            "of_which_system_HIGH": sum(f["system_high"] for f in near_frames),
            "frames_without_near_obstacle": len(frame_rows) - len(near_frames),
            "of_which_system_HIGH(false_alarm_frames)": sum(f["system_high"] for f in frame_rows if not f["near_any_gt"]),
            "generic_obstacle_frames": sum(f["generic_high"] for f in frame_rows),
            "generic_obstacle_frames_with_near_gt": sum(f["generic_high"] for f in near_frames),
        },
        "object_level": {f"<= {L}m": cls(obj_rows, L) for L in (1.0, 1.5)},
        "object_gt_distance_by_risk": {
            r: summarize([o["gt_dist_m"] for o in obj_rows if o["risk"] == r])
            for r in sorted({o["risk"] for o in obj_rows})
        },
        "alert_classes_seen": dict(Counter(o["class"] for o in obj_rows)),
        "spoken_alerts": {"count": len(spoken),
                          "per_minute": len(spoken) / (duration / 60) if duration else None,
                          "samples": spoken[:15]},
        "latency_ms": {"detect": summarize(lat_det), "depth": summarize(lat_depth)},
    }
    path = save_json(f"warning_tum_{report['sequence']}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:3500])
    print("saved", path)


if __name__ == "__main__":
    main()
