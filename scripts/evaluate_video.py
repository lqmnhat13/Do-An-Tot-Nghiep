#!/usr/bin/env python3
"""
Đánh giá pipeline trên video thực nghiệm hoặc video clip kiểm thử.
Tuân thủ mục 9.3 của Kế hoạch:
- Hỗ trợ replay thời gian thực và offline quality.
- Báo cáo số đối tượng phát hiện, số cảnh báo kích hoạt, FPS thực tế và latencies.
"""

import argparse
import os
import time
import json
import numpy as np

from src.camera.camera_manager import CameraManager
from src.detection.class_filter import ClassFilter
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTrackerAdapter
from src.depth.depth_estimator import DepthEstimator
from src.depth.roi_extractor import ROIExtractor
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import Synchronizer
from src.fusion.risk_fsm import RiskFSM
from src.fusion.alert_aggregator import AlertAggregator
from src.audio.tts_engine import TTSEngine
from src.audio.audio_coordinator import AudioCoordinator
from src.runtime.system_coordinator import SystemCoordinator

def main():
    parser = argparse.ArgumentParser(description="Đánh giá pipeline trên video")
    parser.add_argument("--video", type=str, default="dummy", help="Đường dẫn tệp video hoặc 'dummy'")
    parser.add_argument("--duration", type=float, default=5.0, help="Thời gian đánh giá (giây)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out", type=str, default="evaluation/results/eval_report.json")
    args = parser.parse_args()

    print(f"[EVALUATION] Bắt đầu chạy đánh giá trên nguồn '{args.video}' trong {args.duration}s...")

    camera = CameraManager(source=args.video, target_fps=30.0)
    class_filter = ClassFilter()
    detector = YoloDetector(device=args.device, class_filter=class_filter)
    tracker = ByteTrackerAdapter()
    depth_estimator = DepthEstimator(device=args.device)
    roi_extractor = ROIExtractor()
    spatial_zones = SpatialZones()
    synchronizer = Synchronizer()
    risk_fsm = RiskFSM(class_filter, spatial_zones, roi_extractor)
    alert_aggregator = AlertAggregator(risk_fsm)
    tts_engine = TTSEngine()
    audio_coordinator = AudioCoordinator(tts_engine=tts_engine)

    coordinator = SystemCoordinator(
        camera_manager=camera,
        detector=detector,
        tracker=tracker,
        depth_estimator=depth_estimator,
        synchronizer=synchronizer,
        risk_fsm=risk_fsm,
        alert_aggregator=alert_aggregator,
        audio_coordinator=audio_coordinator
    )

    coordinator.start()
    t_start = time.monotonic()
    total_frames_processed = 0

    try:
        while (time.monotonic() - t_start) < args.duration:
            time.sleep(0.1)
            snap = coordinator.get_runtime_snapshot()
            if snap.get("packet"):
                total_frames_processed = snap["packet"].frame_id
    finally:
        coordinator.stop()

    summary = coordinator.metrics.get_summary()
    report = {
        "video_source": args.video,
        "evaluation_duration_sec": args.duration,
        "total_camera_frames": camera.frame_count,
        "total_frames_processed": total_frames_processed,
        "performance_metrics": summary
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[EVALUATION] Hoàn tất đánh giá. Báo cáo đã lưu tại: {args.out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
