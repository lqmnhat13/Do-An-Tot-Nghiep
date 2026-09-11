#!/usr/bin/env python3
"""
Sinh ảnh demo giao diện HUD camera (HUDRenderer) cho các trạng thái hoạt động:
1. Chế độ Quan sát bình thường (Observation - Low Risk)
2. Cảnh báo mức độ Chú ý (Medium Risk)
3. Cảnh báo Nguy cơ cao khi đang phát âm thanh (High Risk + Speaking)
4. Màn hình chờ kết nối camera (Loading / No Camera)
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.frame_packet import FramePacket
from src.contracts.detection import Detection, DetectionResult, BoundingBox
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.risk import RiskAssessment, RiskLevel, Direction, DataQuality
from src.ui.hud_renderer import HUDRenderer


def create_synthetic_indoor_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Tạo khung hình giả lập không gian phòng khách trong nhà."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Tường phòng (nửa trên, gradient xám ấm)
    for y in range(int(height * 0.6)):
        factor = y / (height * 0.6)
        c = int(140 + factor * 40)
        img[y, :] = (c - 10, c, c + 5)

    # Sàn nhà gỗ (nửa dưới, gradient nâu gỗ)
    for y in range(int(height * 0.6), height):
        factor = (y - height * 0.6) / (height * 0.4)
        r = int(120 - factor * 40)
        g = int(85 - factor * 30)
        b = int(55 - factor * 20)
        img[y, :] = (b, g, r)

    # Đường chân tường
    wall_y = int(height * 0.6)
    cv2.line(img, (0, wall_y), (width, wall_y), (40, 50, 60), 2)

    # Mô phỏng chiếc ghế sofa bên trái
    cv2.rectangle(img, (40, 240), (190, 380), (70, 70, 110), -1)
    cv2.rectangle(img, (35, 230), (195, 280), (60, 60, 95), -1)

    # Mô phỏng bàn ăn ở giữa
    cv2.rectangle(img, (240, 270), (410, 390), (45, 65, 85), -1)
    cv2.rectangle(img, (230, 260), (420, 290), (35, 55, 75), -1)

    # Mô phỏng người đứng phía trước
    cv2.circle(img, (480, 200), 26, (140, 160, 190), -1)
    cv2.rectangle(img, (455, 226), (505, 360), (110, 80, 70), -1)

    return img


def create_synthetic_depth_map(width: int = 640, height: int = 480, frame_id: int = 1) -> DepthMap:
    """Tạo bản đồ độ sâu relative tương ứng với không gian phòng."""
    vals = np.zeros((height, width), dtype=np.float32)

    # Sàn nhà dốc từ xa lại gần (lớn hơn = gần hơn)
    for y in range(height):
        factor = y / float(height)
        vals[y, :] = factor * 0.55

    # Vùng người đứng (độ gần cao hơn ~0.78)
    vals[170:370, 450:510] = 0.78

    # Vùng bàn (độ gần vừa phải ~0.50)
    vals[260:390, 230:420] = 0.50

    # Vùng ghế sofa (độ gần ~0.35)
    vals[230:380, 35:195] = 0.35

    mask = np.ones((height, width), dtype=bool)
    return DepthMap(
        representation=DepthRepresentation.RELATIVE_DEPTH,
        unit="relative",
        near_is_larger=True,
        values=vals,
        valid_mask=mask,
        frame_id=frame_id,
        capture_timestamp=time.monotonic()
    )


def main():
    out_dir = PROJECT_ROOT / "artifacts" / "hud_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = HUDRenderer(target_width=800, target_height=600)
    cam_frame = create_synthetic_indoor_frame(640, 480)
    now = time.monotonic()

    packet = FramePacket(
        frame_id=101,
        source_id="webcam_live",
        image=cam_frame,
        original_size=(640, 480),
        timestamp_mono=now
    )

    depth_map = create_synthetic_depth_map(640, 480, frame_id=101)

    # 1. State 1: Chế độ quan sát bình thường (Observation - Low Risk)
    det_obs = DetectionResult(
        detections=[
            Detection("couch", 0.85, BoundingBox(35, 230, 195, 380), track_id=1),
            Detection("dining table", 0.82, BoundingBox(230, 260, 420, 390), track_id=2),
            Detection("person", 0.91, BoundingBox(450, 170, 510, 370), track_id=3)
        ],
        frame_id=101,
        timestamp_mono=now,
        latency_ms=13.4
    )
    assess_obs = [
        RiskAssessment(1, "ghế sofa", Direction.LEFT, RiskLevel.LOW, DataQuality.VALID, 0.35, "xa", "xa", now, now + 1.0),
        RiskAssessment(2, "bàn ăn", Direction.CENTER, RiskLevel.LOW, DataQuality.VALID, 0.40, "vừa phải", "vừa phải", now, now + 1.0),
        RiskAssessment(3, "người", Direction.RIGHT, RiskLevel.LOW, DataQuality.VALID, 0.30, "xa", "xa", now, now + 1.0)
    ]
    snap_obs = {
        "packet": packet,
        "detection": det_obs,
        "depth": depth_map,
        "assessments": assess_obs,
        "mode": "OBSERVATION",
        "metrics": {"fps": 28.5, "detection": {"p50": 13.8, "p95": 14.5}, "depth": {"p50": 63.2, "p95": 82.0}},
        "is_speaking": False
    }
    img_obs = renderer.render(snap_obs)
    cv2.imwrite(str(out_dir / "01_observation.png"), img_obs)
    print(f"[OK] Đã lưu: {out_dir / '01_observation.png'}")

    # 2. State 2: Cảnh báo mức độ chú ý (Medium Risk)
    det_med = DetectionResult(
        detections=[
            Detection("chair", 0.89, BoundingBox(220, 240, 410, 410), track_id=4)
        ],
        frame_id=102,
        timestamp_mono=now,
        latency_ms=13.7
    )
    assess_med = [
        RiskAssessment(4, "ghế", Direction.CENTER, RiskLevel.MEDIUM, DataQuality.VALID, 0.58, "gần", "vùng giữa", now, now + 1.0)
    ]
    snap_med = {
        "packet": packet,
        "detection": det_med,
        "depth": depth_map,
        "assessments": assess_med,
        "mode": "OBSERVATION",
        "metrics": {"fps": 27.2, "detection": {"p50": 13.9, "p95": 14.8}, "depth": {"p50": 64.0, "p95": 85.0}},
        "is_speaking": False
    }
    img_med = renderer.render(snap_med)
    cv2.imwrite(str(out_dir / "02_medium_risk.png"), img_med)
    print(f"[OK] Đã lưu: {out_dir / '02_medium_risk.png'}")

    # 3. State 3: Cảnh báo Nguy cơ cao + Trạng thái Đang phát âm thanh
    det_high = DetectionResult(
        detections=[
            Detection("person", 0.94, BoundingBox(200, 120, 440, 440), track_id=5)
        ],
        frame_id=103,
        timestamp_mono=now,
        latency_ms=14.1
    )
    assess_high = [
        RiskAssessment(5, "người", Direction.CENTER, RiskLevel.HIGH, DataQuality.VALID, 0.82, "rất gần", "khoảng cách nguy cấp", now, now + 1.0)
    ]
    snap_high = {
        "packet": packet,
        "detection": det_high,
        "depth": depth_map,
        "assessments": assess_high,
        "mode": "OBSERVATION",
        "metrics": {"fps": 26.8, "detection": {"p50": 14.1, "p95": 15.2}, "depth": {"p50": 64.5, "p95": 86.0}},
        "is_speaking": True
    }
    img_high = renderer.render(snap_high)
    cv2.imwrite(str(out_dir / "03_high_risk_speaking.png"), img_high)
    print(f"[OK] Đã lưu: {out_dir / '03_high_risk_speaking.png'}")

    # 4. State 4: Màn hình chờ camera
    snap_loading = {
        "packet": None,
        "detection": None,
        "depth": None,
        "assessments": [],
        "mode": "OBSERVATION",
        "metrics": {"fps": 0.0},
        "is_speaking": False
    }
    img_loading = renderer.render(snap_loading)
    cv2.imwrite(str(out_dir / "04_no_camera_loading.png"), img_loading)
    print(f"[OK] Đã lưu: {out_dir / '04_no_camera_loading.png'}")

    # 5. Đo hiệu năng Render độc lập (Benchmark 150 frames)
    print("\nĐang đo hiệu năng kết xuất đồ họa (150 runs)...")
    latencies = []
    for _ in range(150):
        t0 = time.monotonic()
        _ = renderer.render(snap_high)
        latencies.append((time.monotonic() - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps_render = 1000.0 / p50 if p50 > 0 else 0
    print(f"Kết quả Benchmark HUDRenderer:")
    print(f"  - Latency P50 : {p50:.2f} ms")
    print(f"  - Latency P95 : {p95:.2f} ms")
    print(f"  - Render rate : {fps_render:.1f} FPS")


if __name__ == "__main__":
    main()
