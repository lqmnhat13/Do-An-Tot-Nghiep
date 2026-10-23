#!/usr/bin/env python3
"""Hiệu chỉnh depth.metric_scale cho camera đang dùng.

Model Metric-Indoor ước lượng mét lệch theo camera và độ phân giải (trên NYU ở 256px:
xa hơn thực tế ~1.66 lần). Đặt camera đối diện một mặt phẳng (tường, cửa tủ) ở khoảng
cách đã đo bằng thước, chạy script ở 2-3 khoảng cách khác nhau (ví dụ 1.0, 1.5, 2.0 m).

Ví dụ:
    python scripts/calibrate_depth.py --distance 1.5            # webcam 0
    python scripts/calibrate_depth.py --distance 1.0 --image anh.jpg
    python scripts/calibrate_depth.py --distance 1.0 --distance 2.0 --image a.jpg --image b.jpg --write
"""

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.contracts.frame_packet import FramePacket
from src.depth.factory import build_depth_estimator

MODEL_CONFIG = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def center_median(values: np.ndarray, valid: np.ndarray, ratio: float) -> float:
    h, w = values.shape
    dy, dx = int(h * ratio / 2), int(w * ratio / 2)
    roi = values[h // 2 - dy:h // 2 + dy, w // 2 - dx:w // 2 + dx]
    mask = valid[h // 2 - dy:h // 2 + dy, w // 2 - dx:w // 2 + dx]
    return float(np.median(roi[mask])) if mask.any() else float("nan")


def grab_frames(source: int, count: int, warmup_sec: float) -> List[np.ndarray]:
    capture = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được camera {source}. Kiểm tra quyền Camera.")
    try:
        print(f"Giữ yên camera, chụp sau {warmup_sec:.0f} giây...")
        deadline = time.monotonic() + warmup_sec
        while time.monotonic() < deadline:
            capture.read()
        frames = []
        while len(frames) < count:
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def write_scale(scale: float) -> None:
    """Chỉ thay giá trị metric_scale, giữ nguyên chú thích trong YAML."""
    with open(MODEL_CONFIG, "r", encoding="utf-8") as handle:
        text = handle.read()
    new_text, n = re.subn(r"(\n\s+metric_scale:\s*)[0-9.]+", rf"\g<1>{scale:.3f}", text, count=1)
    if n != 1:
        raise RuntimeError("Không tìm thấy khoá depth.metric_scale trong model_config.yaml")
    with open(MODEL_CONFIG, "w", encoding="utf-8") as handle:
        handle.write(new_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hiệu chỉnh depth.metric_scale theo khoảng cách đo tay")
    parser.add_argument("--distance", type=float, action="append", required=True,
                        help="Khoảng cách thật (mét) tới mặt phẳng ở giữa khung hình; lặp lại cho nhiều lần đo")
    parser.add_argument("--image", action="append", default=[],
                        help="Ảnh tương ứng từng --distance; bỏ qua để chụp từ webcam")
    parser.add_argument("--source", type=int, default=0, help="Chỉ số webcam")
    parser.add_argument("--frames", type=int, default=15, help="Số frame trung bình cho mỗi lần đo webcam")
    parser.add_argument("--center-ratio", type=float, default=0.2, help="Kích thước vùng giữa dùng để đo")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--write", action="store_true", help="Ghi hệ số vào configs/model_config.yaml")
    args = parser.parse_args()

    if args.image and len(args.image) != len(args.distance):
        parser.error("Số --image phải bằng số --distance")

    model_cfg = load_yaml(MODEL_CONFIG)
    depth_cfg = dict(model_cfg.get("depth", {}))
    if depth_cfg.get("representation") != "metric_depth":
        parser.error("depth.representation trong model_config.yaml không phải metric_depth")
    depth_cfg["metric_scale"] = 1.0  # đo giá trị thô của model
    estimator = build_depth_estimator({"depth": depth_cfg}, args.device)
    if estimator.load_error:
        print(estimator.load_error)
        return 1

    ratios = []
    for i, distance in enumerate(args.distance):
        if args.image:
            image = cv2.imread(args.image[i])
            if image is None:
                print(f"Không đọc được ảnh {args.image[i]}")
                return 1
            frames = [image]
        else:
            input(f"Đặt camera cách mặt phẳng {distance:.2f} m rồi nhấn Enter...")
            frames = grab_frames(args.source, args.frames, warmup_sec=2.0)
        raw = []
        for j, frame in enumerate(frames):
            h, w = frame.shape[:2]
            packet = FramePacket(frame_id=j, source_id="calibrate", image=frame,
                                 original_size=(w, h), timestamp_mono=time.monotonic())
            dm = estimator.estimate(packet)
            raw.append(center_median(dm.values, dm.valid_mask, args.center_ratio))
        predicted = float(np.nanmedian(raw))
        ratios.append(distance / predicted)
        print(f"{distance:.2f} m thật -> model {predicted:.2f} m -> hệ số {distance / predicted:.3f}")

    scale = float(np.median(ratios))
    spread = (max(ratios) - min(ratios)) / scale if len(ratios) > 1 else 0.0
    print(f"metric_scale đề xuất: {scale:.3f} (hiện tại {model_cfg['depth'].get('metric_scale')})")
    if spread > 0.25:
        print("Cảnh báo: các lần đo lệch nhau >25%; kiểm tra lại khoảng cách hoặc đo thêm.")
    if args.write:
        write_scale(scale)
        print(f"Đã ghi metric_scale={scale:.3f} vào {MODEL_CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
