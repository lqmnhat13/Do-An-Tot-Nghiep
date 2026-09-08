import cv2
import time
from typing import Optional

from src.runtime.system_coordinator import SystemCoordinator
from src.ui.hud_renderer import HUDRenderer

class AppRunner:
    """
    Vòng lặp tương tác giao diện và bắt sự kiện phím bấm của ứng dụng.
    """

    def __init__(
        self,
        coordinator: SystemCoordinator,
        hud_renderer: Optional[HUDRenderer] = None,
        enable_gui: bool = True,
        duration_sec: Optional[float] = None
    ):
        self.coordinator = coordinator
        self.hud_renderer = hud_renderer or HUDRenderer()
        self.enable_gui = enable_gui
        self.duration_sec = duration_sec

    def run(self) -> None:
        print("[AppRunner] Khởi động hệ thống...")
        self.coordinator.start()
        t_start = time.monotonic()

        window_name = "Do-An-Tot-Nghiep - Multimodal Assistive System"
        if self.enable_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                # Kiểm tra giới hạn thời gian chạy nếu có
                if self.duration_sec and (time.monotonic() - t_start) >= self.duration_sec:
                    print(f"[AppRunner] Đã đạt thời lượng chỉ định ({self.duration_sec}s), dừng ứng dụng.")
                    break

                snapshot = self.coordinator.get_runtime_snapshot()

                if self.enable_gui:
                    rendered_frame = self.hud_renderer.render(snapshot)
                    cv2.imshow(window_name, rendered_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: # ESC
                        print("[AppRunner] Người dùng bấm ESC, đang thoát...")
                        break
                    elif key == 32: # SPACE
                        print("[AppRunner] Phím SPACE: Kích hoạt đọc chữ OCR.")
                        self.coordinator.trigger_ocr()
                    elif key in (ord('q'), ord('Q')):
                        print("[AppRunner] Phím Q: Kích hoạt mô tả toàn cảnh VQA.")
                        self.coordinator.trigger_vqa("Mô tả khung cảnh phía trước")
                    elif key in (ord('s'), ord('S')):
                        print("[AppRunner] Phím S: Dừng âm thanh.")
                        self.coordinator.stop_speech()
                else:
                    # Chế độ headless (không có GUI)
                    time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[AppRunner] Nhận tín hiệu KeyboardInterrupt, đang dừng...")
        finally:
            self.coordinator.stop()
            if self.enable_gui:
                cv2.destroyAllWindows()
            print("[AppRunner] Đã dừng toàn bộ hệ thống.")
