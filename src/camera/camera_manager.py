import cv2
import time
import threading
import numpy as np
from typing import List, Optional, Union, Tuple
from src.contracts.frame_packet import FramePacket
from src.camera.bounded_buffer import BoundedBuffer
from src.camera.video_playback import VideoPlayback

class CameraManager:
    """
    Quản lý luồng Camera duy nhất (Camera Owner).
    Thu nhận khung hình, tạo FramePacket bất biến và phát tới các BoundedBuffer độc lập.
    Hỗ trợ auto-reconnect, quản lý buffer drop-oldest, không gây tắc nghẽn luồng.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        target_fps: float = 30.0,
        reconnect_delay_sec: float = 2.0,
        max_reconnect_attempts: int = 5
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_attempts = max_reconnect_attempts

        self._subscribers: List[BoundedBuffer[FramePacket]] = []
        self._subscribers_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_playback: Optional[VideoPlayback] = None
        self._is_dummy = (str(self.source).lower() == "dummy")

        self._frame_counter = 0
        self._reconnect_counter = 0
        self._last_frame_time = 0.0
        self._is_alive = False
        self._latest_packet: Optional[FramePacket] = None
        self._frame_lock = threading.Lock()

    def get_latest_frame(self) -> Optional[FramePacket]:
        """Lấy khung hình mới nhất trực tiếp từ camera (phục vụ render mượt mà 30 FPS)."""
        with self._frame_lock:
            return self._latest_packet

    def subscribe(self, buffer: BoundedBuffer[FramePacket]) -> None:
        """Đăng ký một BoundedBuffer nhận khung hình mới từ camera."""
        with self._subscribers_lock:
            if buffer not in self._subscribers:
                self._subscribers.append(buffer)

    def unsubscribe(self, buffer: BoundedBuffer[FramePacket]) -> None:
        """Hủy đăng ký BoundedBuffer."""
        with self._subscribers_lock:
            if buffer in self._subscribers:
                self._subscribers.remove(buffer)

    def start(self) -> None:
        """Bắt đầu luồng thu nhận camera."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="CameraWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Dừng luồng camera an toàn."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()

    def is_alive(self) -> bool:
        return self._is_alive and self._running

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_counter

    def _open_capture(self) -> bool:
        if self._is_dummy:
            return True

        if isinstance(self.source, str) and not self.source.isdigit():
            # Nguồn tệp video
            try:
                self._video_playback = VideoPlayback(self.source, loop=True, target_fps=self.target_fps)
                return True
            except Exception as e:
                print(f"[CameraManager] Lỗi mở tệp video: {e}")
                return False

        # Nguồn thiết bị camera (webcam)
        cam_idx = int(self.source)
        try:
            # Ưu tiên AVFOUNDATION trên macOS
            self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(cam_idx)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                return True
        except Exception as e:
            print(f"[CameraManager] Lỗi mở webcam ID {cam_idx}: {e}")
        return False

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._video_playback is not None:
            self._video_playback.release()
            self._video_playback = None

    def _read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._is_dummy:
            # Tạo khung hình mẫu giả lập có đồ họa
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Vẽ một hình chữ nhật giả lập vật cản di chuyển
            t = time.monotonic()
            offset_x = int(100 + 150 * np.sin(t))
            cv2.rectangle(frame, (offset_x, 150), (offset_x + 120, 380), (0, 200, 255), -1)
            cv2.putText(frame, f"Dummy Cam - Frame {self._frame_counter}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Chair Simulation", (offset_x + 5, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return True, frame

        if self._video_playback is not None:
            return self._video_playback.read()

        if self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
            return ret, frame

        return False, None

    def _capture_loop(self) -> None:
        interval = 1.0 / max(1.0, self.target_fps)
        reconnect_attempts = 0

        while self._running:
            if not self._is_dummy and (self._cap is None and self._video_playback is None):
                success = self._open_capture()
                if not success:
                    reconnect_attempts += 1
                    self._reconnect_counter += 1
                    self._is_alive = False
                    if reconnect_attempts > self.max_reconnect_attempts:
                        print(f"[CameraManager] Vượt quá số lần reconnect ({self.max_reconnect_attempts}), chuyển sang chế độ dummy.")
                        self._is_dummy = True
                    time.sleep(self.reconnect_delay_sec)
                    continue
                else:
                    reconnect_attempts = 0

            loop_start = time.monotonic()
            ret, raw_frame = self._read_frame()

            if not ret or raw_frame is None:
                # Frame drop hoặc camera bị mất tín hiệu
                self._is_alive = False
                self._release_capture()
                time.sleep(self.reconnect_delay_sec)
                continue

            self._is_alive = True
            self._frame_counter += 1
            now_mono = time.monotonic()
            self._last_frame_time = now_mono

            # Đóng gói FramePacket bất biến
            h, w = raw_frame.shape[:2]
            packet = FramePacket(
                frame_id=self._frame_counter,
                source_id=str(self.source),
                image=raw_frame.copy(), # Đảm bảo frame độc lập
                original_size=(w, h),
                timestamp_mono=now_mono,
                timestamp_sensor=now_mono
            )

            # Lưu frame mới nhất phục vụ live preview mượt mà
            with self._frame_lock:
                self._latest_packet = packet

            # Phân phối tới các subscribers qua BoundedBuffer (drop-oldest)
            with self._subscribers_lock:
                for sub in self._subscribers:
                    sub.put(packet)

            # Điều hòa nhịp khung hình
            elapsed = time.monotonic() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

        self._release_capture()
        self._is_alive = False
