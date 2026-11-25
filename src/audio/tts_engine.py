import hashlib
import os
import subprocess
import threading
import time
import wave
from typing import Dict, Iterable, Optional

import numpy as np

CLIP_RATE = 44100


def load_wav_mono(path: str, rate: int = CLIP_RATE) -> np.ndarray:
    """Đọc WAV PCM 16-bit thành mảng int16 mono ở tần số `rate`."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("Chỉ hỗ trợ WAV PCM 16-bit")
        src_rate, channels = w.getframerate(), w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels).mean(axis=1)
    if src_rate != rate and len(data) > 1:
        n = int(round(len(data) * rate / src_rate))
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return data.astype(np.int16)


class _ClipOutput:
    """Một output stream sounddevice mở suốt thời gian chạy; phát clip = thay bộ đệm.

    Đo trên M1: mỗi lần chạy `afplay`/`say` tốn ~0,75 s khởi động, mở+đóng một stream mới tốn
    ~120 ms (close/abort ~110 ms); với stream mở sẵn, phát và dừng gần như tức thì.
    """

    END_MARGIN_SEC = 0.5  # stream lỗi/mất thiết bị: coi như phát xong để luồng âm thanh không treo

    def __init__(self, rate: int = CLIP_RATE):
        self.rate = rate
        self._lock = threading.Lock()
        self._data: Optional[np.ndarray] = None
        self._pos = 0
        self._deadline = 0.0
        self._done = threading.Event()
        self._done.set()
        self._stream = None

    def _ensure_stream(self) -> None:
        if self._stream is not None and self._stream.active:
            return
        self.close()
        import sounddevice as sd  # import trễ: CI/Ubuntu không có thiết bị âm thanh
        stream = sd.OutputStream(samplerate=self.rate, channels=1, dtype="int16", callback=self._callback)
        stream.start()
        self._stream = stream

    def play(self, data: np.ndarray) -> None:
        self._ensure_stream()
        with self._lock:
            self._data, self._pos = data, 0
            self._deadline = time.monotonic() + len(data) / self.rate + self.END_MARGIN_SEC
            self._done.clear()

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            if self._data is None:
                outdata.fill(0)
                return
            chunk = self._data[self._pos:self._pos + frames]
            n = len(chunk)
            outdata[:n, 0] = chunk
            outdata[n:] = 0
            self._pos += n
            if self._pos >= len(self._data):
                self._data = None
                self._done.set()

    def stop(self) -> None:
        with self._lock:
            self._data = None
            self._done.set()

    def is_active(self) -> bool:
        if self._done.is_set():
            return False
        if time.monotonic() > self._deadline:
            self.stop()
            return False
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        remaining = max(0.0, self._deadline - time.monotonic())
        finished = self._done.wait(remaining if timeout is None else min(timeout, remaining))
        return finished or not self.is_active()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


class TTSEngine:
    """
    Trình phát âm thanh và tổng hợp giọng nói bản địa trên macOS.
    - Câu nằm trong bộ clip dựng sẵn (`prerender`) và các tệp WAV được phát ngay trong tiến trình.
    - Câu khác dùng lệnh 'say' với giọng tiếng Việt 'Linh' offline; WAV dùng 'afplay' khi không
      phát được trong tiến trình (thiếu sounddevice hoặc thiết bị âm thanh).
    - Cho phép ngắt tiếng tức thì (< 50ms).
    """

    def __init__(self, voice: str = "Linh", speech_rate_wpm: int = 200, clip_dir: Optional[str] = None):
        self.voice = voice
        self.speech_rate_wpm = speech_rate_wpm
        self.clip_dir = clip_dir
        self._current_process: Optional[subprocess.Popen] = None
        self._output: Optional[_ClipOutput] = None
        self._clips: Dict[str, np.ndarray] = {}

    def clip_path(self, text: str) -> Optional[str]:
        """Đường dẫn clip dựng sẵn của câu; đổi giọng hoặc tốc độ sẽ dùng clip khác."""
        if not self.clip_dir:
            return None
        key = hashlib.sha1(f"{self.voice}|{self.speech_rate_wpm}|{text}".encode("utf-8")).hexdigest()[:20]
        return os.path.join(self.clip_dir, f"{key}.wav")

    def prerender(self, texts: Iterable[str]) -> int:
        """Dựng sẵn clip WAV cho các câu chưa có (chạy nền lúc khởi động). Trả về số clip mới."""
        if not self.clip_dir:
            return 0
        os.makedirs(self.clip_dir, exist_ok=True)
        created = 0
        for text in texts:
            path = self.clip_path(text)
            if os.path.exists(path):
                continue
            tmp = f"{path}.part"
            try:
                subprocess.run(["say", "-v", self.voice, "-r", str(self.speech_rate_wpm),
                                "--file-format=WAVE", "--data-format=LEI16@22050", "-o", tmp, text],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=15, check=True)
                os.replace(tmp, path)  # đổi tên nguyên tử: không bao giờ phát clip ghi dở
                created += 1
            except Exception as e:
                print(f"[TTSEngine] Không dựng được clip '{text}': {e}")
                if os.path.exists(tmp):
                    os.remove(tmp)
        return created

    def _play_in_process(self, path: str) -> bool:
        try:
            data = self._clips.get(path)
            if data is None:
                data = self._clips[path] = load_wav_mono(path)
            if self._output is None:
                self._output = _ClipOutput()
            self._output.play(data)
        except Exception as e:
            print(f"[TTSEngine] Không phát được '{path}' trong tiến trình: {e}")
            return False
        return True

    def warm_up(self) -> bool:
        """Mở sẵn output stream (~120 ms) để cảnh báo đầu tiên không phải chờ."""
        try:
            if self._output is None:
                self._output = _ClipOutput()
            self._output._ensure_stream()
            return True
        except Exception as e:
            print(f"[TTSEngine] Không mở được thiết bị âm thanh, dùng say/afplay: {e}")
            return False

    def close(self) -> None:
        """Đóng output stream (khi dừng hệ thống); lần phát sau sẽ tự mở lại."""
        if self._output is not None:
            self._output.stop()
            self._output.close()

    def speak(self, text: str) -> bool:
        """Phát âm một chuỗi văn bản bằng giọng tiếng Việt macOS."""
        self.stop() # Ngắt âm thanh trước đó nếu đang nói

        clip = self.clip_path(text)
        if clip and os.path.exists(clip) and self._play_in_process(clip):
            return True

        cmd = ["say", "-v", self.voice, "-r", str(self.speech_rate_wpm), text]
        try:
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception as e:
            print(f"[TTSEngine] Lỗi khởi chạy lệnh say: {e}")
            return False

    def play_sound(self, sound_file: str) -> bool:
        """Phát một tệp âm thanh WAV (trong tiến trình, dự phòng bằng 'afplay')."""
        if not os.path.exists(sound_file):
            return False

        self.stop()
        if self._play_in_process(sound_file):
            return True
        try:
            self._current_process = subprocess.Popen(
                ["afplay", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception as e:
            print(f"[TTSEngine] Lỗi khởi chạy afplay: {e}")
            return False

    def stop(self) -> None:
        """Ngắt ngay lập tức âm thanh đang phát."""
        if self._output is not None:
            self._output.stop()
        proc = self._current_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.04)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._current_process = None

    def is_speaking(self) -> bool:
        """Kiểm tra xem âm thanh có đang được phát hay không."""
        output = self._output
        if output is not None and output.is_active():
            return True
        return self._current_process is not None and self._current_process.poll() is None

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        """Đợi cho đến khi phát âm thanh xong."""
        output = self._output
        if output is not None and output.is_active():
            return output.wait(timeout)
        proc = self._current_process
        if proc is None:
            return True
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
