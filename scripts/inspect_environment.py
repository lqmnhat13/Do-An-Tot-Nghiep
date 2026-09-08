#!/usr/bin/env python3
"""
Kiểm tra toàn diện môi trường phần cứng, thư viện, MPS, Camera, TTS và mô hình.
Theo yêu cầu mục 2.1 và 8 của Kế hoạch triển khai.
"""

import os
import sys
import platform
import subprocess
import shutil

def check_system():
    print("=" * 60)
    print("1. THÔNG TIN HỆ THỐNG & PHẦN CỨNG")
    print("=" * 60)
    print(f"Hệ điều hành: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"macOS version: {platform.mac_ver()[0]}")
    
    # RAM
    try:
        mem_bytes = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        mem_gb = int(mem_bytes) / (1024 ** 3)
        print(f"RAM hợp nhất: {mem_gb:.1f} GB")
    except Exception as e:
        print(f"RAM: Không đọc được ({e})")
        
    # CPU
    try:
        cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        print(f"Bộ vi xử lý: {cpu_brand}")
    except Exception as e:
        print(f"CPU: Không đọc được ({e})")
        
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

def check_libraries():
    print("\n" + "=" * 60)
    print("2. KIỂM TRA THƯ VIỆN & TĂNG TỐC PHẦN CỨNG (MPS)")
    print("=" * 60)
    
    # PyTorch & MPS
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        mps_avail = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        print(f"     MPS Built: {mps_built}, MPS Available: {mps_avail}")
        if mps_avail:
            x = torch.ones(2, 2, device="mps")
            print("     [OK] MPS Tensor Compute Smoke Test PASSED.")
        else:
            print("     [WARNING] MPS không khả dụng, sẽ dùng CPU.")
    except Exception as e:
        print(f"[FAIL] PyTorch: {e}")

    # OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"[FAIL] OpenCV: {e}")

    # Ultralytics
    try:
        import ultralytics
        print(f"[OK] Ultralytics (YOLO): {ultralytics.__version__}")
    except Exception as e:
        print(f"[FAIL] Ultralytics: {e}")

    # Transformers
    try:
        import transformers
        print(f"[OK] HuggingFace Transformers: {transformers.__version__}")
    except Exception as e:
        print(f"[FAIL] Transformers: {e}")

    # EasyOCR
    try:
        import easyocr
        print(f"[OK] EasyOCR: {easyocr.__version__}")
    except Exception as e:
        print(f"[FAIL] EasyOCR: {e}")

def check_tts():
    print("\n" + "=" * 60)
    print("3. KIỂM TRA GIỌNG ĐỌC TIẾNG VIỆT (MACOS TTS)")
    print("=" * 60)
    try:
        output = subprocess.check_output(["say", "-v", "?"], text=True)
        vi_voices = [line.strip() for line in output.splitlines() if "vi_VN" in line or "Linh" in line]
        if vi_voices:
            for v in vi_voices:
                print(f"[OK] Tìm thấy giọng: {v}")
            # Test say
            res = subprocess.run(["say", "-v", "Linh", "Kiểm tra hệ thống âm thanh."], capture_output=True)
            if res.returncode == 0:
                print("[OK] Lệnh 'say -v Linh' phát âm thanh thành công.")
            else:
                print(f"[WARNING] 'say -v Linh' trả về mã lỗi: {res.returncode}")
        else:
            print("[WARNING] Không tìm thấy giọng 'Linh' hoặc 'vi_VN'. Hãy kiểm tra Cài đặt Hệ thống -> Trợ năng -> Đọc nội dung.")
    except Exception as e:
        print(f"[FAIL] Kiểm tra TTS: {e}")

def check_models():
    print("\n" + "=" * 60)
    print("4. KIỂM TRA MÔ HÌNH VÀ TRỌNG SỐ (WEIGHTS)")
    print("=" * 60)
    yolo_local = os.path.expanduser("~/Do-An-Tot-Nghiep/models/weights/yolov8n.pt")
    yolo_backup = os.path.expanduser("~/Second-Eye-backup/models/yolov8n.pt")
    
    if os.path.exists(yolo_local):
        size_mb = os.path.getsize(yolo_local) / (1024 * 1024)
        print(f"[OK] YOLOv8n weights tại dự án: {yolo_local} ({size_mb:.2f} MB)")
    elif os.path.exists(yolo_backup):
        size_mb = os.path.getsize(yolo_backup) / (1024 * 1024)
        print(f"[FOUND] Tìm thấy YOLOv8n trong backup: {yolo_backup} ({size_mb:.2f} MB)")
        print("        -> Có thể copy vào models/weights bằng scripts/download_models.py")
    else:
        print("[WARNING] Chưa có file yolov8n.pt tại local.")

    # Depth Anything V2 cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    da_v2_small = os.path.join(hf_cache, "models--depth-anything--Depth-Anything-V2-Small-hf")
    if os.path.exists(da_v2_small):
        print(f"[OK] Đã cache Depth-Anything-V2-Small-hf tại: {da_v2_small}")
    else:
        print(f"[INFO] Chưa cache Depth-Anything-V2-Small-hf tại {da_v2_small}")

def check_camera():
    print("\n" + "=" * 60)
    print("5. KIỂM TRA CAMERA PHẦN CỨNG")
    print("=" * 60)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, c = frame.shape
                print(f"[OK] Camera ID 0 mở thành công: độ phân giải {w}x{h}, {c} channels.")
            else:
                print("[WARNING] Mở được Camera ID 0 nhưng chưa đọc được khung hình.")
            cap.release()
        else:
            print("[INFO] Camera ID 0 không mở được (có thể chưa cấp quyền hoặc không có webcam).")
            print("       Hệ thống hỗ trợ chế độ 'dummy' hoặc video file thay thế.")
    except Exception as e:
        print(f"[FAIL] Lỗi kiểm tra camera: {e}")

if __name__ == "__main__":
    check_system()
    check_libraries()
    check_tts()
    check_models()
    check_camera()
    print("\n" + "=" * 60)
    print("HOÀN TẤT BÁO CÁO KIỂM TRA MÔI TRƯỜNG.")
    print("=" * 60)
