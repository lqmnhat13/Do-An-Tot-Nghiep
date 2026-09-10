#!/usr/bin/env python3
"""
Entry point chính của Hệ Thống AI Đa Phương Thức Hỗ Trợ Người Khiếm Thị.
Khởi tạo các thành phần theo đúng kế hoạch kiến trúc và chạy ứng dụng.
"""

import argparse
import os
import sys
import yaml

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
from src.ocr.ocr_service import OCRService
from src.vqa.vqa_service import VQAService
from src.runtime.system_coordinator import SystemCoordinator
from src.ui.hud_renderer import HUDRenderer
from src.ui.app_runner import AppRunner

def load_yaml(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def main():
    parser = argparse.ArgumentParser(description="Hệ thống AI đa phương thức hỗ trợ người khiếm thị")
    parser.add_argument("--source", type=str, default=None, help="Nguồn camera: 0 (webcam), đường dẫn tệp video, hoặc 'dummy'")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn trọng số YOLO (mặc định: models/weights/yolov8n.pt)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán: 'mps' hoặc 'cpu'")
    parser.add_argument("--no-gui", action="store_true", help="Chạy ở chế độ không mở cửa sổ giao diện")
    parser.add_argument("--duration", type=float, default=None, help="Thời lượng chạy tối đa (giây)")
    args = parser.parse_args()

    # Nạp cấu hình từ configs/
    app_cfg = load_yaml("configs/app_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")
    fusion_cfg = load_yaml("configs/fusion_rules.yaml")

    cam_source = args.source if args.source is not None else app_cfg.get("camera", {}).get("source", 0)
    device = args.device if args.device is not None else app_cfg.get("app", {}).get("device", "mps")
    weights_path = args.weights if args.weights is not None else model_cfg.get("detection", {}).get("weights_path", "models/weights/yolov8n.pt")

    print("=" * 60)
    print("KHỞI ĐỘNG HỆ THỐNG AI ĐA PHƯƠNG THỨC HỖ TRỢ NGƯỜI KHIẾM THỊ")
    print(f"- Nguồn camera: {cam_source}")
    print(f"- Thiết bị tính toán: {device}")
    print(f"- YOLO weights: {weights_path}")
    print(f"- Giao diện GUI: {'Tắt (--no-gui)' if args.no_gui else 'Bật'}")
    print("=" * 60)

    # 1. Camera Manager
    camera = CameraManager(
        source=cam_source,
        width=app_cfg.get("camera", {}).get("width", 640),
        height=app_cfg.get("camera", {}).get("height", 480),
        target_fps=app_cfg.get("camera", {}).get("fps", 30.0)
    )

    # 2. Detection & Tracking
    class_filter = ClassFilter(
        enabled_classes=model_cfg.get("detection", {}).get("enabled_classes"),
        alert_classes=model_cfg.get("detection", {}).get("alert_classes"),
        custom_vi_names=fusion_cfg.get("labels_vi")
    )
    detector = YoloDetector(
        weights_path=weights_path,
        class_filter=class_filter,
        confidence_threshold=model_cfg.get("detection", {}).get("confidence_threshold", 0.45),
        device=device
    )
    tracker = ByteTrackerAdapter(iou_threshold=0.3)

    # 3. Depth Estimation & ROI
    depth_input_size = (
        int(model_cfg.get("depth", {}).get("input_height", 256)),
        int(model_cfg.get("depth", {}).get("input_width", 256))
    )
    depth_estimator = DepthEstimator(
        model_name=model_cfg.get("depth", {}).get("hf_repo_id", "depth-anything/Depth-Anything-V2-Small-hf"),
        device=device,
        input_size=depth_input_size
    )
    roi_extractor = ROIExtractor(
        erosion_ratio=model_cfg.get("depth", {}).get("roi_erosion_ratio", 0.15),
        percentile=float(model_cfg.get("depth", {}).get("percentile_roi", 80.0))
    )

    # 4. Synchronizer & Fusion
    spatial_cfg = fusion_cfg.get("spatial", {})
    spatial_zones = SpatialZones(
        left_ratio=spatial_cfg.get("left_ratio", 0.35),
        center_ratio=spatial_cfg.get("center_ratio", 0.30),
        right_ratio=spatial_cfg.get("right_ratio", 0.35)
    )
    sync_cfg = fusion_cfg.get("synchronizer", {})
    synchronizer = Synchronizer(
        max_pair_skew_ms=sync_cfg.get("max_pair_skew_ms", 120.0),
        max_detection_age_ms=sync_cfg.get("max_detection_age_ms", 300.0),
        max_depth_age_ms=sync_cfg.get("max_depth_age_ms", 350.0)
    )
    fsm_cfg = fusion_cfg.get("risk_fsm", {})
    risk_fsm = RiskFSM(
        class_filter=class_filter,
        spatial_zones=spatial_zones,
        roi_extractor=roi_extractor,
        confirmations_to_escalate=fsm_cfg.get("confirmations_to_escalate", 2),
        confirmations_to_deescalate=fsm_cfg.get("confirmations_to_deescalate", 3),
        instant_high_risk_in_center=fsm_cfg.get("instant_high_risk_in_center", True),
        cooldown_sec=fsm_cfg.get("track_alert_cooldown_sec", 3.0),
        allow_escalation_override=fsm_cfg.get("allow_escalation_override", True)
    )
    alert_aggregator = AlertAggregator(risk_fsm=risk_fsm)

    # 5. Audio Coordinator
    audio_cfg = app_cfg.get("audio", {})
    tts_engine = TTSEngine(
        voice=audio_cfg.get("voice", "Linh"),
        speech_rate_wpm=audio_cfg.get("speech_rate_wpm", 200)
    )
    audio_coordinator = AudioCoordinator(tts_engine=tts_engine)

    # 6. On-Demand OCR & VQA
    ocr_service = OCRService(use_gpu=(device == "mps"))
    vqa_cfg = model_cfg.get("vqa", {})
    vqa_service = VQAService(
        device=device,
        use_vlm=vqa_cfg.get("use_vlm", True),
        backend_name=vqa_cfg.get("backend", "legacy_caption"),
        caption_model_name=vqa_cfg.get("caption_model_name", "Salesforce/blip-image-captioning-base"),
        translation_model_name=vqa_cfg.get("translation_model_name", "Helsinki-NLP/opus-mt-en-vi"),
        lazy_load=True
    )

    # 7. System Coordinator
    coordinator = SystemCoordinator(
        camera_manager=camera,
        detector=detector,
        tracker=tracker,
        depth_estimator=depth_estimator,
        synchronizer=synchronizer,
        risk_fsm=risk_fsm,
        alert_aggregator=alert_aggregator,
        audio_coordinator=audio_coordinator,
        ocr_service=ocr_service,
        vqa_service=vqa_service
    )

    # 8. App Runner
    preview_w = app_cfg.get("runtime", {}).get("preview_width", 800)
    preview_h = app_cfg.get("runtime", {}).get("preview_height", 600)
    hud_renderer = HUDRenderer(target_width=preview_w, target_height=preview_h)

    runner = AppRunner(
        coordinator=coordinator,
        hud_renderer=hud_renderer,
        enable_gui=(not args.no_gui),
        duration_sec=args.duration
    )

    runner.run()

if __name__ == "__main__":
    main()
