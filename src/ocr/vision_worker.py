"""Apple Vision chạy trong một tiến trình phụ, có thể tắt hẳn khi treo hoặc khi người dùng huỷ.

Vision thỉnh thoảng không trả về ở lần gọi đầu của một tiến trình (đo: 2/45 lần, chỉ ở lần
gọi đầu; bình thường 0,1-0,6 s). Luồng Python không thể bị dừng cưỡng bức, nên trước đây một lần
treo bắt người dùng chờ hết hạn, giữ luồng OCR sống (bấm Space bị từ chối) và chuyển hẳn sang
EasyOCR tới hết phiên. Ở tiến trình phụ thì treo -> chụp stack (`sample`) -> kill -> khởi động lại.

Chạy trực tiếp (`python -m src.ocr.vision_worker <fd> <ngôn ngữ> <module:Lớp>`) là phía tiến
trình con; module này không import Vision ở mức module để dùng được trên mọi nền tảng.
"""
import atexit
import importlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Connection
from typing import Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_READER_SPEC = "src.ocr.vision_engine:VisionOCRReader"

# Mọi tiến trình Vision còn sống: kill khi app thoát (một tiến trình đang treo trong Vision sẽ
# không tự thoát; tiến trình con còn có luồng canh tiến trình cha làm lớp bảo vệ thứ hai).
_LIVE_PROCS = set()
_LIVE_LOCK = threading.Lock()


@atexit.register
def _kill_live_workers() -> None:
    with _LIVE_LOCK:
        procs = list(_LIVE_PROCS)
    for proc in procs:
        if proc.poll() is None:
            proc.kill()


class OCRCancelled(Exception):
    """Lần đọc bị huỷ (người dùng bấm C/S hoặc tắt hệ thống)."""


class VisionUnavailable(RuntimeError):
    """Vision không dùng được trên máy này hoặc đã bị tắt sau nhiều lần treo liên tiếp."""


def _warmup_image() -> np.ndarray:
    import cv2
    image = np.full((120, 360, 3), 255, np.uint8)
    cv2.putText(image, "OK 123", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    return image


class VisionWorkerClient:
    """Phía app: một yêu cầu một lúc, có hạn chờ, huỷ được và tự khởi động lại tiến trình."""

    def __init__(self, languages: Sequence[str], read_timeout_sec: float = 3.0,
                 warmup_timeout_sec: float = 30.0, hang_log_dir: Optional[str] = None,
                 max_consecutive_failures: int = 3, reader_spec: str = DEFAULT_READER_SPEC,
                 extra_sys_path: Sequence[str] = ()):
        self.languages = list(languages)
        self.read_timeout_sec = read_timeout_sec
        self.warmup_timeout_sec = warmup_timeout_sec
        self.hang_log_dir = hang_log_dir
        self.max_consecutive_failures = max_consecutive_failures
        self.reader_spec = reader_spec
        self.extra_sys_path = list(extra_sys_path)
        self._request_lock = threading.Lock()  # một yêu cầu đọc tại một thời điểm
        self._state_lock = threading.Lock()    # bảo vệ _proc/_conn/_ready/_busy
        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[Connection] = None
        self._ready = False
        self._busy = False
        self._cancel = threading.Event()
        self._failures = 0
        self.disabled_reason: Optional[str] = None
        self.last_hang_log: Optional[str] = None
        self._closed = False

    @property
    def disabled(self) -> bool:
        return self.disabled_reason is not None

    @property
    def pid(self) -> Optional[int]:
        proc = self._proc
        return proc.pid if proc is not None else None

    # ------------------------------------------------------------------ API
    def warm_up(self) -> None:
        """Khởi động tiến trình và gọi thử một lần ở nền (lần nạp chậm xảy ra lúc mở app)."""
        if not self.disabled and not self._closed:
            threading.Thread(target=self._warm, name="VisionWarmup", daemon=True).start()

    def readtext(self, image: np.ndarray):
        if self.disabled:
            raise VisionUnavailable(self.disabled_reason)
        if not self._request_lock.acquire(timeout=self.read_timeout_sec):
            raise TimeoutError("Vision đang khởi động hoặc bận")
        try:
            self._cancel.clear()
            return self._request(image, self.read_timeout_sec, cancellable=True)
        finally:
            self._request_lock.release()

    def cancel(self) -> None:
        """Huỷ lần đọc đang chạy (trả về trong ~50 ms). Không có lần đọc nào thì bỏ qua."""
        with self._state_lock:
            if self._busy:
                self._cancel.set()

    def close(self) -> None:
        """Tắt tiến trình Vision; sau đó không khởi động lại nữa."""
        with self._state_lock:
            self._closed = True
            self._cancel.set()
            proc, conn = self._proc, self._conn
            self._proc, self._conn, self._ready = None, None, False
        self._terminate(proc, conn, hung=False, background=False)

    # ------------------------------------------------------------ nội bộ
    def _warm(self) -> None:
        if not self._request_lock.acquire(timeout=5.0):
            return
        try:
            t0 = time.monotonic()
            self._request(_warmup_image(), self.warmup_timeout_sec, cancellable=False)
            print(f"[VisionWorker] Vision sẵn sàng sau {time.monotonic() - t0:.2f} s.")
        except OCRCancelled:
            pass  # đang tắt hệ thống
        except Exception as exc:
            print(f"[VisionWorker] Khởi động Vision thất bại: {exc}")
        finally:
            self._request_lock.release()

    def _spawn(self) -> None:
        if self._closed:
            raise OCRCancelled()
        parent_sock, child_sock = socket.socketpair()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [ROOT, *self.extra_sys_path] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.ocr.vision_worker", str(child_sock.fileno()),
                 ",".join(self.languages), self.reader_spec],
                pass_fds=(child_sock.fileno(),), cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        finally:
            child_sock.close()
        with _LIVE_LOCK:
            _LIVE_PROCS.add(proc)
        with self._state_lock:
            self._proc, self._conn, self._ready = proc, Connection(parent_sock.detach()), False

    def _request(self, image: np.ndarray, timeout: float, cancellable: bool):
        deadline = time.monotonic() + timeout
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
        proc, conn = self._proc, self._conn
        with self._state_lock:
            self._busy = cancellable
        try:
            if not self._ready:
                kind, payload = self._recv(proc, conn, deadline, cancellable)
                if kind == "init_error":
                    self._discard(proc, conn, hung=False)
                    self.disabled_reason = f"Vision không khởi tạo được: {payload}"
                    raise VisionUnavailable(self.disabled_reason)
                self._ready = True
            conn.send(image)
            kind, payload = self._recv(proc, conn, deadline, cancellable)
        finally:
            with self._state_lock:
                self._busy = False
        if kind == "ok":
            self._failures = 0
            return payload
        raise RuntimeError(f"Vision lỗi khi đọc: {payload}")  # tiến trình vẫn dùng tiếp được

    def _recv(self, proc, conn, deadline: float, cancellable: bool):
        while True:
            if cancellable and self._cancel.is_set():
                self._discard(proc, conn, hung=False)
                self.warm_up()  # có sẵn tiến trình mới cho lần bấm Space sau
                raise OCRCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._discard(proc, conn, hung=True)
                self._count_failure("treo")
                raise TimeoutError("Vision không phản hồi đúng hạn")
            try:
                if conn.poll(min(0.05, remaining)):
                    return conn.recv()
            except (EOFError, OSError):
                self._discard(proc, conn, hung=False)
                self._count_failure("thoát bất thường")
                raise RuntimeError("Tiến trình Vision đã thoát")

    def _count_failure(self, what: str) -> None:
        self._failures += 1
        if self._closed:
            return
        if self._failures >= self.max_consecutive_failures:
            self.disabled_reason = f"Vision {what} {self._failures} lần liên tiếp"
            print(f"[VisionWorker] {self.disabled_reason}: tắt Vision, chỉ dùng EasyOCR.")
        else:
            self.warm_up()

    def _discard(self, proc, conn, hung: bool) -> None:
        with self._state_lock:
            if self._proc is proc:
                self._proc, self._conn, self._ready = None, None, False
        self._terminate(proc, conn, hung=hung, background=True)

    def _terminate(self, proc, conn, hung: bool, background: bool) -> None:
        def run():
            if hung and proc is not None and proc.poll() is None:
                self._save_hang_stack(proc.pid)
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                with _LIVE_LOCK:
                    _LIVE_PROCS.discard(proc)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        if background:
            threading.Thread(target=run, name="VisionWorkerKill", daemon=True).start()
        else:
            run()

    def _save_hang_stack(self, pid: int) -> None:
        """Bằng chứng để tìm nguyên nhân gốc: stack native của tiến trình Vision lúc treo."""
        if not self.hang_log_dir or not shutil.which("sample"):
            return
        try:
            os.makedirs(self.hang_log_dir, exist_ok=True)
            path = os.path.join(self.hang_log_dir, time.strftime("vision_hang_%Y%m%d_%H%M%S.txt"))
            subprocess.run(["sample", str(pid), "1", "-file", path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            if os.path.exists(path):
                self.last_hang_log = path
                print(f"[VisionWorker] Vision treo; đã lưu stack: {path}")
        except Exception as exc:
            print(f"[VisionWorker] Không chụp được stack Vision: {exc}")


def _exit_when_parent_dies(parent_pid: int) -> None:
    """Chạy ở luồng riêng: vẫn thoát được khi luồng chính đang kẹt trong Vision."""
    while os.getppid() == parent_pid:
        time.sleep(1.0)
    os._exit(1)


def _worker_main(fd: int, languages: str, reader_spec: str) -> None:
    threading.Thread(target=_exit_when_parent_dies, args=(os.getppid(),), daemon=True).start()
    conn = Connection(fd)
    try:
        module_name, class_name = reader_spec.split(":")
        reader = getattr(importlib.import_module(module_name), class_name)(languages.split(","))
    except Exception as exc:
        conn.send(("init_error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", None))
    while True:
        try:
            image = conn.recv()
        except (EOFError, OSError):
            return  # app đã đóng kết nối hoặc đã thoát
        try:
            reply = ("ok", [(box, str(text), float(conf)) for box, text, conf in reader.readtext(image)])
        except Exception as exc:
            reply = ("error", f"{type(exc).__name__}: {exc}")
        try:
            conn.send(reply)
        except OSError:
            return  # app đã huỷ yêu cầu này và đóng kết nối


if __name__ == "__main__":
    _worker_main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
