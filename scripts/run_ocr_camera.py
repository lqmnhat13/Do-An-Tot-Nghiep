#!/usr/bin/env python3
"""Chạy OCR kiểm thử trực tiếp với webcam hoặc file ảnh mà không khởi tạo pipeline cảnh báo."""

import argparse
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

import cv2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.tts_engine import TTSEngine
from src.contracts.request import OCRRequest, OCRResult
from src.ocr.image_quality import ImageQualityChecker
from src.ocr.ocr_service import OCRService


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiểm thử OCR độc lập với webcam hoặc ảnh tĩnh, không chạy detection/depth/cảnh báo."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (mặc định: 0)")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh tĩnh để nhận diện trực tiếp")
    parser.add_argument("--width", type=int, default=None, help="Chiều rộng camera")
    parser.add_argument("--height", type=int, default=None, help="Chiều cao camera")
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--min-confidence", type=float, default=None, help="Ngưỡng độ tin cậy tối thiểu cho EasyOCR")
    parser.add_argument("--no-speech", action="store_true", help="Chỉ in kết quả ra terminal, không đọc TTS")
    return parser


class OCRCameraSession:
    """Quản lý một request OCR nền và loại bỏ kết quả đã hủy."""

    def __init__(
        self,
        service: OCRService,
        tts_engine: Optional[TTSEngine] = None,
    ):
        self.service = service
        self.tts_engine = tts_engine
        self.status = "READY"
        self.last_text = ""
        self._generation = 0
        self._closed = False
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def trigger(self, frame) -> bool:
        """Chụp bản sao frame và chạy OCR nếu chưa có request đang xử lý."""
        with self._lock:
            if self._closed or (self._worker is not None and self._worker.is_alive()):
                return False
            self._generation += 1
            token = self._generation
            self.status = "PROCESSING"
            snapshot = frame.copy()
            worker = threading.Thread(
                target=self._recognize,
                args=(token, snapshot),
                name="OCRCameraWorker",
                daemon=True,
            )
            self._worker = worker

        try:
            worker.start()
            return True
        except Exception:
            with self._lock:
                if self._worker is worker:
                    self._worker = None
                if self._generation == token:
                    self.status = "ERROR"
            raise

    def _recognize(self, token: int, frame) -> None:
        request = OCRRequest(
            request_id=f"camera_ocr_{token}_{time.monotonic_ns()}",
            image=frame,
        )
        try:
            result = self.service.process(request)
            text = result.text
            latency = result.latency_sec
            quality = result.quality_status
            success = result.success
        except Exception as exc:
            text = f"Lỗi OCR: {exc}"
            latency = 0.0
            quality = "ERROR"
            success = False

        with self._lock:
            if self._closed or token != self._generation:
                return
            self.last_text = text
            self.status = "READY"
            print(f"\n[OCR] Kết quả: {text}")
            print(f"[OCR] Trạng thái chất lượng: {quality} | Thành công: {success}")
            print(f"[OCR] Latency: {latency:.3f}s", flush=True)
            if self.tts_engine is not None and text:
                self.tts_engine.speak(text)

    def cancel(self) -> None:
        """Ngừng TTS và vô hiệu hóa kết quả inference đang chạy."""
        with self._lock:
            self._generation += 1
            self.status = "CANCELLED"
            if self.tts_engine is not None:
                self.tts_engine.stop()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
            worker = self._worker
            if self.tts_engine is not None:
                self.tts_engine.stop()
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.5)


def open_camera(index: int, width: int, height: int):
    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        if index != 0:
            print(f"[THÔNG BÁO] Camera index {index} không tồn tại (hệ thống chỉ có 1 camera). Tự động chuyển về camera index 0...")
            capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Không thể mở camera index {index} (hoặc camera 0). Hãy kiểm tra kết nối camera và quyền truy cập trong System Settings.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def run_on_image(image_path: str, service: OCRService, tts_engine: Optional[TTSEngine]) -> int:
    if not os.path.exists(image_path):
        print(f"[LỖI] Không tìm thấy file ảnh: {image_path}")
        return 1

    img = cv2.imread(image_path)
    if img is None:
        print(f"[LỖI] Không thể đọc file ảnh: {image_path}")
        return 1

    print(f"[OCR Image] Đang xử lý file ảnh: {image_path} ({img.shape[1]}x{img.shape[0]})...")
    req = OCRRequest(request_id=f"image_ocr_{time.monotonic_ns()}", image=img)
    t0 = time.monotonic()
    result = service.process(req)
    total_time = time.monotonic() - t0

    print(f"\n[OCR] Kết quả: {result.text}")
    print(f"[OCR] Trạng thái chất lượng: {result.quality_status}")
    print(f"[OCR] Thành công: {result.success}")
    print(f"[OCR] Latency: {result.latency_sec:.3f}s (Tổng thời gian: {total_time:.3f}s)")

    if tts_engine is not None and result.text:
        print("[OCR] Đang phát âm thanh kết quả qua TTS...")
        tts_engine.speak(result.text)
        # Chờ phát âm thanh hoàn tất
        while tts_engine.is_speaking():
            time.sleep(0.1)

    return 0


def main() -> int:
    args = build_parser().parse_args()
    app_config = load_yaml(os.path.join(PROJECT_ROOT, "configs", "app_config.yaml"))
    model_config = load_yaml(os.path.join(PROJECT_ROOT, "configs", "model_config.yaml"))

    camera_config = app_config.get("camera", {})
    audio_config = app_config.get("audio", {})
    ocr_config = model_config.get("ocr", {})

    width = args.width or int(camera_config.get("width", 640))
    height = args.height or int(camera_config.get("height", 480))
    device = args.device or ocr_config.get("device") or app_config.get("app", {}).get("device", "mps")
    use_gpu = (device == "mps")
    min_confidence = args.min_confidence if args.min_confidence is not None else float(ocr_config.get("min_confidence", 0.3))

    quality_checker = ImageQualityChecker(
        blur_threshold=float(ocr_config.get("blur_laplacian_threshold", 50.0)),
        min_brightness=float(ocr_config.get("min_brightness", 40.0)),
        max_brightness=float(ocr_config.get("max_brightness", 235.0))
    )
    service = OCRService(
        languages=ocr_config.get("languages", ["vi", "en"]),
        engine=ocr_config.get("engine", "easyocr"),
        use_gpu=use_gpu,
        min_confidence=min_confidence,
        quality_checker=quality_checker,
    )
    backend_desc = "Vision (EasyOCR dự phòng)" if service.engine == "vision" else f"EasyOCR ({'GPU' if use_gpu else 'CPU'})"

    tts_engine = None if args.no_speech else TTSEngine(
        voice=audio_config.get("voice", "Linh"),
        speech_rate_wpm=audio_config.get("speech_rate_wpm", 200)
    )

    # 1. Chế độ nhận diện trên file ảnh tĩnh
    if args.image:
        return run_on_image(args.image, service, tts_engine)

    # 2. Chế độ tương tác thời gian thực với Webcam
    print("=" * 60)
    print("[OCR Camera] Độc lập: Chỉ OCR được kích hoạt, pipeline an toàn tắt.")
    print(f"[OCR Camera] Camera={args.camera} | Backend: {backend_desc}")
    print("[OCR Camera] Phím SPACE: Chụp & Đọc chữ | S: Dừng đọc/hủy | ESC: Thoát")
    print("=" * 60)

    try:
        capture = open_camera(args.camera, width, height)
    except Exception as exc:
        print(f"[LỖI] {exc}")
        return 1

    session = OCRCameraSession(service=service, tts_engine=tts_engine)
    window_name = "Kiểm thử OCR Camera (Bấm SPACE để đọc, S để dừng, ESC thoát)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # Header hướng dẫn
            cv2.rectangle(display, (0, 0), (w, 36), (20, 20, 20), -1)
            cv2.putText(
                display,
                "SPACE: Doc chu | S: Dung am thanh | ESC: Thoat",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Footer trạng thái
            cv2.rectangle(display, (0, h - 30), (w, h), (20, 20, 20), -1)
            status_text = f"Trang thai: {session.status}"
            cv2.putText(
                display,
                status_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if session.status == "READY" else (0, 165, 255),
                1,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord(" "):  # SPACE
                if not session.trigger(frame):
                    print("[OCR Camera] Request trước vẫn đang xử lý; bấm S để hủy.")
            elif key in (ord("s"), ord("S")):
                session.cancel()
                print("[OCR Camera] Đã hủy / dừng giọng nói.")
    finally:
        session.close()
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
