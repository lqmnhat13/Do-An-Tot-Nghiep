#!/usr/bin/env python3
"""Chạy VQA trực tiếp với webcam mà không khởi tạo pipeline cảnh báo."""

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
from src.contracts.request import VQARequest
from src.vqa.vqa_service import VQAService


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiểm thử VQA với webcam, không chạy detection/depth/OCR/cảnh báo."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (mặc định: 0)")
    parser.add_argument("--width", type=int, default=None, help="Chiều rộng camera")
    parser.add_argument("--height", type=int, default=None, help="Chiều cao camera")
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--backend", default=None,
                        help="VQA backend: legacy_caption | disabled | mlx_vlm")
    parser.add_argument(
        "--question",
        default="Mô tả khung cảnh phía trước.",
        help="Câu hỏi được gửi khi bấm Q",
    )
    parser.add_argument("--no-speech", action="store_true", help="Chỉ in kết quả, không đọc TTS")
    return parser


class VQACameraSession:
    """Quản lý đúng một request VQA nền và loại bỏ kết quả đã hủy."""

    def __init__(
        self,
        service: VQAService,
        question: str,
        tts_engine: Optional[TTSEngine] = None,
    ):
        self.service = service
        self.question = question
        self.tts_engine = tts_engine
        self.status = "READY"
        self.last_answer = ""
        self._generation = 0
        self._closed = False
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def trigger(self, frame) -> bool:
        """Chụp bản sao frame và chạy VQA nếu chưa có request đang xử lý."""
        with self._lock:
            if self._closed or (self._worker is not None and self._worker.is_alive()):
                return False
            self._generation += 1
            token = self._generation
            self.status = "PROCESSING"
            snapshot = frame.copy()
            worker = threading.Thread(
                target=self._answer,
                args=(token, snapshot),
                name="VQACameraWorker",
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

    def _answer(self, token: int, frame) -> None:
        request = VQARequest(
            request_id=f"camera_vqa_{token}_{time.monotonic_ns()}",
            image=frame,
            question=self.question,
        )
        try:
            result = self.service.answer(request)
            answer = result.answer
            latency = result.latency_sec
        except Exception as exc:
            answer = f"VQA lỗi: {exc}"
            latency = 0.0

        with self._lock:
            if self._closed or token != self._generation:
                return
            self.last_answer = answer
            self.status = "READY"
            print(f"\n[VQA] {answer}")
            print(f"[VQA] Latency: {latency:.3f}s", flush=True)
            if self.tts_engine is not None:
                self.tts_engine.speak(answer)

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
        raise RuntimeError(f"Không thể mở camera index {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main() -> int:
    args = build_parser().parse_args()
    app_config = load_yaml(os.path.join(PROJECT_ROOT, "configs", "app_config.yaml"))
    model_config = load_yaml(os.path.join(PROJECT_ROOT, "configs", "model_config.yaml"))

    camera_config = app_config.get("camera", {})
    audio_config = app_config.get("audio", {})
    vqa_config = model_config.get("vqa", {})
    width = args.width or int(camera_config.get("width", 640))
    height = args.height or int(camera_config.get("height", 480))
    device = args.device or vqa_config.get("device") or app_config.get("app", {}).get("device", "mps")
    backend = args.backend or vqa_config.get("backend", "legacy_caption")

    service = VQAService(
        device=device,
        use_vlm=vqa_config.get("use_vlm", True),
        backend_name=backend,
        caption_model_name=vqa_config.get(
            "caption_model_name", "Salesforce/blip-image-captioning-base"
        ),
        translation_model_name=vqa_config.get(
            "translation_model_name", "Helsinki-NLP/opus-mt-en-vi"
        ),
        lazy_load=True,
        mlx_vlm_config=vqa_config.get("mlx_vlm", {}),
    )
    tts_engine = None if args.no_speech else TTSEngine(
        voice=audio_config.get("voice", "Linh"),
        speech_rate_wpm=int(audio_config.get("speech_rate_wpm", 200)),
    )
    session = VQACameraSession(service, args.question, tts_engine)
    capture = open_camera(args.camera, width, height)
    window_name = "VQA Camera Only"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[VQA Camera] Chỉ VQA được khởi tạo; detection/depth/OCR/cảnh báo đều tắt.")
    print(f"[VQA Camera] Backend={backend}, device={device}, question={args.question!r}")
    print("[VQA Camera] Q: hỏi | S: hủy/dừng đọc | ESC: thoát")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[VQA Camera] Không đọc được frame từ camera.")
                return 1

            preview = frame.copy()
            cv2.putText(
                preview,
                f"VQA ONLY | {session.status}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if session.status == "READY" else (0, 200, 255),
                2,
            )
            cv2.putText(
                preview,
                "Q: Ask  S: Cancel speech  ESC: Exit",
                (20, preview.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("q"), ord("Q")):
                if not session.trigger(frame):
                    print("[VQA Camera] Request trước vẫn đang xử lý; bấm S để hủy kết quả.")
            elif key in (ord("s"), ord("S")):
                session.cancel()
                print("[VQA Camera] Đã dừng giọng đọc và hủy kết quả đang chờ.")
    except KeyboardInterrupt:
        print("\n[VQA Camera] Đang thoát...")
    finally:
        session.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
