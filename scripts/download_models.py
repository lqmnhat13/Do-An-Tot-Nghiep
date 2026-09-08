#!/usr/bin/env python3
"""
Tải hoặc liên kết các trọng số mô hình vào thư mục models/weights/.
Tạo các file âm thanh WAV mẫu để phục vụ phát cảnh báo cố định không độ trễ.
"""

import os
import sys
import shutil
import wave
import struct
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "models", "weights")

def setup_yolo_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    target_yolo = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    
    if os.path.exists(target_yolo):
        print(f"[OK] YOLOv8n đã có tại: {target_yolo}")
        return

    # Kiểm tra các thư mục backup
    backup_candidates = [
        os.path.expanduser("~/Second-Eye-backup/models/yolov8n.pt"),
        os.path.expanduser("~/Documents/SecondEye-Project/weights/yolov8n.pt")
    ]
    for src in backup_candidates:
        if os.path.exists(src):
            print(f"[COPY] Sao chép YOLOv8n từ {src} -> {target_yolo}...")
            shutil.copyfile(src, target_yolo)
            print(f"[OK] Đã sao chép thành công: {os.path.getsize(target_yolo)/(1024*1024):.2f} MB")
            return

    # Nếu không có sẵn, tải tự động bằng Ultralytics
    print("[DOWNLOAD] Đang tải yolov8n.pt từ Ultralytics...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        # Di chuyển về target
        if os.path.exists("yolov8n.pt") and not os.path.exists(target_yolo):
            shutil.move("yolov8n.pt", target_yolo)
        print(f"[OK] Đã tải và lưu YOLOv8n tại: {target_yolo}")
    except Exception as e:
        print(f"[FAIL] Không thể tải yolov8n.pt: {e}")

def setup_depth_weights():
    print("[CHECK] Kiểm tra Depth Anything V2 Small...")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        repo_id = "depth-anything/Depth-Anything-V2-Small-hf"
        print(f"        Đang nạp processor & model từ repo: {repo_id}...")
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModelForDepthEstimation.from_pretrained(repo_id)
        print(f"[OK] Depth Anything V2 Small đã sẵn sàng (đã lưu cache local).")
    except Exception as e:
        print(f"[FAIL] Không thể kiểm tra Depth Anything V2: {e}")

def create_sine_wav(filepath, freq=880.0, duration=0.2, volume=0.5, sample_rate=44100):
    """Tạo file WAV sóng sin đơn giản làm âm beep/chime cảnh báo cố định."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    num_samples = int(duration * sample_rate)
    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(1) # mono
        wav_file.setsampwidth(2) # 16 bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            # Thêm envelope decay nhẹ
            decay = 1.0 - (i / num_samples)
            value = int(volume * decay * 32767.0 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            frames.extend(struct.pack("<h", max(-32768, min(32767, value))))
        wav_file.writeframes(frames)
    print(f"[AUDIO] Đã tạo file âm thanh mẫu: {filepath}")

def setup_audio_assets():
    audio_dir = os.path.join(PROJECT_ROOT, "assets", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    # Beep cảnh báo nguy cơ cao (tần số cao 1000 Hz, 2 tiếng bip)
    create_sine_wav(os.path.join(audio_dir, "alert_high.wav"), freq=1000.0, duration=0.15)
    # Chime thông báo thường (tần số 587 Hz)
    create_sine_wav(os.path.join(audio_dir, "chime.wav"), freq=587.33, duration=0.25)
    # Beep xác nhận OCR/VQA (tần số 440 Hz)
    create_sine_wav(os.path.join(audio_dir, "ready.wav"), freq=440.0, duration=0.1)

if __name__ == "__main__":
    setup_yolo_weights()
    setup_depth_weights()
    setup_audio_assets()
    print("[ALL DONE] Hoàn tất chuẩn bị weights và tài nguyên âm thanh.")
