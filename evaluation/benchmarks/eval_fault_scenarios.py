#!/usr/bin/env python3
"""Kịch bản sự cố trên SystemCoordinator thật: người dùng nghe được gì?

Không cần model hay camera: detector/depth giả, TTS ghi lại câu nói thay vì phát loa.
1. depth_hang: depth treo ở giây 8 trong khi một người đứng cách 0.9 m phía trước.
2. camera_unplug: rút webcam ở giây 8 (không mở lại được), cắm lại ở giây 16.
Cả hai xảy ra sau thời gian khởi động 5 s nên đo được độ trễ phát hiện khi đang chạy.

Chạy: PYTHONPATH=. python evaluation/benchmarks/eval_fault_scenarios.py [--out file.json]
"""
import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.audio.audio_coordinator import AudioCoordinator  # noqa: E402
from src.camera import camera_manager as cm  # noqa: E402
from src.contracts.depth_map import DepthMap, DepthRepresentation  # noqa: E402
from src.contracts.detection import BoundingBox, Detection, DetectionResult  # noqa: E402
from src.depth.roi_extractor import ROIExtractor  # noqa: E402
from src.detection.class_filter import ClassFilter  # noqa: E402
from src.fusion.alert_aggregator import AlertAggregator  # noqa: E402
from src.fusion.risk_fsm import RiskFSM  # noqa: E402
from src.fusion.spatial_zones import SpatialZones  # noqa: E402
from src.fusion.synchronizer import Synchronizer  # noqa: E402
from src.runtime.system_coordinator import SystemCoordinator  # noqa: E402
from src.tracking.byte_tracker import ByteTrackerAdapter  # noqa: E402


class LogTTS:
    def __init__(self, t0):
        self.t0, self.spoken = t0, []

    def speak(self, text):
        self.spoken.append({"t": round(time.monotonic() - self.t0, 1), "text": text})
        return True

    def play_sound(self, f): return True
    def stop(self): pass
    def is_speaking(self): return False
    def wait_until_done(self, timeout=None): return True


class PersonAhead:
    def detect(self, p):
        return DetectionResult([Detection("person", 0.9, BoundingBox(220, 100, 420, 470))],
                               p.frame_id, p.timestamp_mono, 5,
                               preprocess_transform={"original_size": p.original_size})


class Depth:
    """Mọi điểm cách 0.9 m; treo sau hang_after giây nếu được đặt."""

    def __init__(self, t0, hang_after=None):
        self.t0, self.hang_after, self.release = t0, hang_after, threading.Event()
        self.load_error = None

    def estimate(self, p):
        if self.hang_after is not None and time.monotonic() - self.t0 > self.hang_after:
            self.release.wait()
        v = np.full((480, 640), 0.9, np.float32)
        return DepthMap(DepthRepresentation.METRIC_DEPTH, "meter", False, v, np.ones_like(v, bool),
                        p.frame_id, p.timestamp_mono)


class UnpluggableCapture:
    """Webcam giả: trả ảnh đến khi bị rút, mở lại được khi cắm lại."""
    plugged = threading.Event()

    def __init__(self, *a, **k):
        self.open = self.plugged.is_set()

    def isOpened(self): return self.open

    def read(self):
        if not self.plugged.is_set():
            self.open = False
            return False, None
        return True, np.full((480, 640, 3), 120, np.uint8)

    def set(self, *a): pass
    def release(self): pass


def run(name, duration, depth_hang_after=None, unplug_at=None, replug_at=None):
    t0 = time.monotonic()
    tts = LogTTS(t0)
    if unplug_at is not None:
        UnpluggableCapture.plugged.set()
        cm.cv2.VideoCapture = UnpluggableCapture
        camera = cm.CameraManager(source=0, reconnect_delay_sec=0.5, max_reconnect_attempts=5)
    else:
        camera = cm.CameraManager(source="dummy")
    cf = ClassFilter(enabled_classes=["person"], alert_classes=["person"])
    fsm = RiskFSM(class_filter=cf, spatial_zones=SpatialZones(), roi_extractor=ROIExtractor())
    depth = Depth(t0, depth_hang_after)
    coord = SystemCoordinator(camera, PersonAhead(), ByteTrackerAdapter(), depth, Synchronizer(), fsm,
                              AlertAggregator(fsm, global_alert_interval_sec=3.5),
                              AudioCoordinator(tts_engine=tts))
    coord.start()
    modes = []
    while time.monotonic() - t0 < duration:
        el = time.monotonic() - t0
        if unplug_at is not None and el >= unplug_at and UnpluggableCapture.plugged.is_set() and \
                (replug_at is None or el < replug_at):
            UnpluggableCapture.plugged.clear()
        if replug_at is not None and el >= replug_at:
            UnpluggableCapture.plugged.set()
        s = coord.get_runtime_snapshot()
        modes.append({"t": round(el, 1), "mode": s["mode"], "health": s["health_problem"]})
        time.sleep(0.5)
    coord.stop()
    depth.release.set()
    return {"scenario": name, "spoken": tts.spoken, "timeline": modes[::2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = [
        run("depth_hang (người đứng 0.9 m, depth treo từ giây 8)", 20.0, depth_hang_after=8.0),
        run("camera_unplug (rút webcam giây 8, cắm lại giây 16)", 22.0, unplug_at=8.0, replug_at=16.0),
    ]
    for r in results:
        print("\n==", r["scenario"])
        for s in r["spoken"]:
            print(f"  t={s['t']:>5}s  {s['text']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    sys.stdout.flush()
    os._exit(0)  # luồng depth giả bị treo có chủ đích


if __name__ == "__main__":
    main()
