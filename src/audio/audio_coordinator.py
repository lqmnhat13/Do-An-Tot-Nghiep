import queue
import threading
import time
from typing import Optional, Dict

from src.contracts.audio import AudioTask, AudioPriority
from src.audio.tts_engine import TTSEngine

class AudioCoordinator:
    """
    Điều phối viên âm thanh duy nhất (Audio Owner).
    Tuân thủ mục 6 của Kế hoạch:
    - Quản lý hàng đợi Priority Queue với 4 mức ưu tiên.
    - Cảnh báo HIGH_RISK được quyền ngắt tiếng tức thì (preempt) tác vụ ưu tiên thấp hơn.
    - Loại bỏ triệt để các thông báo đã hết hạn (expires_at).
    - Chống lặp (deduplication) cho các câu thông báo giống hệt nhau trong thời gian ngắn.
    """

    def __init__(self, tts_engine: Optional[TTSEngine] = None, dedup_window_sec: float = 1.5):
        self.tts_engine = tts_engine or TTSEngine()
        self.dedup_window_sec = dedup_window_sec

        self._queue: queue.PriorityQueue[AudioTask] = queue.PriorityQueue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._current_task: Optional[AudioTask] = None
        self._current_lock = threading.Lock()

        self._recent_texts: Dict[str, float] = {}
        self._dropped_expired_count = 0
        self._interrupted_count = 0
        self._pending_tasks_count = 0

    def start(self) -> None:
        """Khởi động luồng điều phối âm thanh."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, name="AudioCoordinatorWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Dừng luồng âm thanh an toàn."""
        self._running = False
        self.interrupt()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def post_task(self, task: AudioTask) -> bool:
        """
        Đẩy một AudioTask vào hàng đợi điều phối.
        Xử lý cơ chế Preemption: nếu task mới có độ ưu tiên cao hơn task đang phát, ngắt task cũ ngay!
        """
        now = time.monotonic()

        # 1. Bỏ qua nếu task đã hết hạn trước khi vào queue
        if task.is_expired(now):
            self._dropped_expired_count += 1
            return False

        # 2. Chống lặp cho cùng một câu nói trong khoảng thời gian ngắn
        last_time = self._recent_texts.get(task.text, 0.0)
        if now - last_time < self.dedup_window_sec:
            # Nếu không phải HIGH_RISK thì bỏ qua thông báo trùng lặp
            if task.priority != AudioPriority.HIGH_RISK:
                return False

        self._recent_texts[task.text] = now

        # 3. Preemption: Cảnh báo HIGH_RISK được quyền ngắt task đang nói nếu task đó có ưu tiên thấp hơn
        with self._current_lock:
            if self._current_task is not None:
                if task.priority < self._current_task.priority and self._current_task.interruptible:
                    # Ngắt ngay lập tức!
                    self.tts_engine.stop()
                    self._interrupted_count += 1
                    self._current_task = None
            self._pending_tasks_count += 1

        self._queue.put(task)
        return True

    def interrupt(self) -> None:
        """Ngắt ngay lập tức âm thanh hiện tại và xóa hàng đợi."""
        self.tts_engine.stop()
        with self._current_lock:
            self._current_task = None
            self._pending_tasks_count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_speaking(self) -> bool:
        return self.tts_engine.is_speaking()

    @property
    def is_busy(self) -> bool:
        with self._current_lock:
            has_task = (self._current_task is not None) or (self._pending_tasks_count > 0)
        return has_task or (not self._queue.empty()) or self.tts_engine.is_speaking()

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        """
        Đợi cho đến khi AudioCoordinator hoàn thành toàn bộ tác vụ âm thanh
        (bao gồm cả việc chờ trong queue và phát xong qua TTS/afplay).
        """
        t_start = time.monotonic()
        time.sleep(0.02)
        while True:
            if not self.is_busy:
                return True
            if timeout is not None and (time.monotonic() - t_start) >= timeout:
                return False
            time.sleep(0.02)

    @property
    def dropped_expired_count(self) -> int:
        return self._dropped_expired_count

    @property
    def interrupted_count(self) -> int:
        return self._interrupted_count

    def _audio_loop(self) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.monotonic()
            # Kiểm tra thời hạn hiệu lực trước khi phát
            if task.is_expired(now):
                self._dropped_expired_count += 1
                with self._current_lock:
                    self._pending_tasks_count = max(0, self._pending_tasks_count - 1)
                continue

            with self._current_lock:
                self._current_task = task

            try:
                # Phát âm thanh chime/beep trước nếu có
                if task.sound_file:
                    self.tts_engine.play_sound(task.sound_file)
                    self.tts_engine.wait_until_done(timeout=0.5)

                # Phát câu nói bằng TTS
                if task.text:
                    self.tts_engine.speak(task.text)

                # Đợi cho tới khi nói xong hoặc bị ngắt
                while self._running and self.tts_engine.is_speaking():
                    time.sleep(0.02)
            finally:
                with self._current_lock:
                    self._current_task = None
                    self._pending_tasks_count = max(0, self._pending_tasks_count - 1)
