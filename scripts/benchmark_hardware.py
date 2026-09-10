#!/usr/bin/env python3
"""
Đo kiểm (Benchmark) hiệu năng phần cứng độc lập cho từng thành phần AI trên Apple Silicon M1.
Tuân thủ mục 2.1 và 9.1 của Kế hoạch:
- Đo P50 và P95 latency thực tế, không dùng số liệu giả định.
- Đánh giá YOLOv8n, Depth Anything V2 Small, EasyOCR và macOS TTS.
"""

import time
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.runtime.model_loading import pretrained_kwargs

def benchmark_yolo(device="mps", num_runs=50):
    print("\n--- BENCHMARK: YOLOv8n Detection ---")
    from ultralytics import YOLO
    weights = "models/weights/yolov8n.pt"
    if not os.path.exists(weights):
        print(f"[SKIP] Không tìm thấy {weights}")
        return

    model = YOLO(weights)
    model.to(device)

    # Khung hình giả lập 640x480
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup 5 lần
    for _ in range(5):
        model.predict(dummy_img, verbose=False, device=device)

    latencies = []
    for _ in range(num_runs):
        t0 = time.monotonic()
        model.predict(dummy_img, verbose=False, device=device)
        latencies.append((time.monotonic() - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / p50 if p50 > 0 else 0
    print(f"YOLOv8n [{device}] ({num_runs} runs):")
    print(f"  - Latency P50: {p50:.2f} ms")
    print(f"  - Latency P95: {p95:.2f} ms")
    print(f"  - Tốc độ tương đương: {fps:.1f} FPS")

def benchmark_depth(device="mps", num_runs=20):
    print("\n--- BENCHMARK: Depth Anything V2 Small ---")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image

    repo_id = "depth-anything/Depth-Anything-V2-Small-hf"
    load_kwargs = pretrained_kwargs()
    processor = AutoImageProcessor.from_pretrained(repo_id, **load_kwargs)
    model = AutoModelForDepthEstimation.from_pretrained(repo_id, **load_kwargs)
    model.to(device)
    model.eval()

    dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    inputs = processor(images=dummy_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup 3 lần
    with torch.no_grad():
        for _ in range(3):
            model(**inputs)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.monotonic()
            outputs = model(**inputs)
            # Sync nếu là MPS
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
            latencies.append((time.monotonic() - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / p50 if p50 > 0 else 0
    print(f"Depth Anything V2 Small [{device}] ({num_runs} runs):")
    print(f"  - Latency P50: {p50:.2f} ms")
    print(f"  - Latency P95: {p95:.2f} ms")
    print(f"  - Tốc độ tương đương: {fps:.1f} FPS")

def benchmark_tts():
    print("\n--- BENCHMARK: macOS TTS 'say -v Linh' ---")
    import subprocess
    t0 = time.monotonic()
    # Đo thời gian khởi động tiến trình say
    proc = subprocess.Popen(["say", "-v", "Linh", "Kiểm tra độ trễ."],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    launch_time = (time.monotonic() - t0) * 1000.0
    proc.wait()
    total_time = (time.monotonic() - t0) * 1000.0
    print(f"macOS TTS (Linh):")
    print(f"  - Thời gian khởi phát tiến trình: {launch_time:.2f} ms")
    print(f"  - Tổng thời gian hoàn tất: {total_time:.2f} ms")

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 60)
    print(f"BẮT ĐẦU ĐO KIỂM HIỆU NĂNG PHẦN CỨNG (Thiết bị: {device})")
    print("=" * 60)
    benchmark_yolo(device=device)
    benchmark_depth(device=device)
    benchmark_tts()
    print("\n" + "=" * 60)
    print("HOÀN TẤT ĐO KIỂM HIỆU NĂNG.")
    print("=" * 60)
