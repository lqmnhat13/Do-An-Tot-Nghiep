import cv2
import time
from typing import Optional

# DEFAULT_VQA_QUESTION được re-export cho các script evaluation/benchmarks.
from src.runtime.system_coordinator import DEFAULT_VQA_QUESTION, SystemCoordinator  # noqa: F401
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
                        # Q: hỏi VQA bằng giọng nói (bíp rồi nói câu hỏi); Q lần nữa khi đang
                        # nghe thì dừng thu. Không có micro/nhận dạng thì coordinator dùng câu
                        # DEFAULT_VQA_QUESTION ("Phía trước có gì?") đã được đánh giá trên
                        # OpenViVQA/KTVIC/NYU ("Mô tả khung cảnh phía trước" gây ảo giác).
                        if self.coordinator.is_listening():
                            print("[AppRunner] Phím Q: Dừng thu câu hỏi.")
                            self.coordinator.stop_listening()
                        else:
                            print("[AppRunner] Phím Q: Hỏi VQA bằng giọng nói.")
                            self.coordinator.trigger_vqa(None)
                    elif key in (ord('m'), ord('M')):
                        print("[AppRunner] Phím M: Nhận diện mệnh giá tiền.")
                        self.coordinator.trigger_currency()
                    elif key in (ord('c'), ord('C')):
                        print("[AppRunner] Phím C: Chuyển sang chế độ Cảnh báo.")
                        self.coordinator.enter_warning_mode()
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
