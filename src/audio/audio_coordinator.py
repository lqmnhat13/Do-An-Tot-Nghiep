import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict

from src.contracts.audio import AudioTask, AudioPriority
from src.audio.tts_engine import TTSEngine


@dataclass(eq=False)
class _QueuedAudio:
    task: AudioTask
    generation: int
    safety_generation: int

    def __lt__(self, other: "_QueuedAudio") -> bool:
        return self.task < other.task


class AudioCoordinator:
    """Single audio owner with priority, expiry and atomic cancellation/mode gating."""

    def __init__(self, tts_engine: Optional[TTSEngine] = None, dedup_window_sec: float = 1.5):
        self.tts_engine = tts_engine or TTSEngine()
        self.dedup_window_sec = dedup_window_sec
        self._queue: queue.PriorityQueue[_QueuedAudio] = queue.PriorityQueue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_task: Optional[AudioTask] = None
        self._current_entry: Optional[_QueuedAudio] = None
        self._current_lock = threading.Lock()
        self._generation = 0
        self._safety_generation = 0
        self._safety_alerts_enabled = True
        self._recent_texts: Dict[str, float] = {}
        self._dropped_expired_count = 0
        self._interrupted_count = 0
        self._pending_tasks_count = 0

    def start(self) -> None:
        with self._current_lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._audio_loop, name="AudioCoordinatorWorker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._current_lock:
            self._running = False
            self._interrupt_locked()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def set_safety_alerts_enabled(self, enabled: bool) -> None:
        """Gate HIGH_RISK; disabling also invalidates previously dequeued safety tasks."""
        with self._current_lock:
            self._safety_alerts_enabled = enabled
            if enabled:
                return
            self._safety_generation += 1
            if self._current_task and self._current_task.priority == AudioPriority.HIGH_RISK:
                self.tts_engine.stop()
                self._current_task = None
                self._current_entry = None
                self._interrupted_count += 1
            retained = []
            while True:
                try:
                    entry = self._queue.get_nowait()
                except queue.Empty:
                    break
                if entry.task.priority == AudioPriority.HIGH_RISK:
                    self._pending_tasks_count = max(0, self._pending_tasks_count - 1)
                else:
                    retained.append(entry)
            for entry in retained:
                self._queue.put(entry)

    def post_task(self, task: AudioTask) -> bool:
        with self._current_lock:
            if task.priority == AudioPriority.HIGH_RISK and not self._safety_alerts_enabled:
                return False
            now = time.monotonic()
            if task.is_expired(now):
                self._dropped_expired_count += 1
                return False
            last_time = self._recent_texts.get(task.text, 0.0)
            if now - last_time < self.dedup_window_sec and task.priority != AudioPriority.HIGH_RISK:
                return False
            self._recent_texts[task.text] = now
            if (self._current_task is not None
                    and task.priority < self._current_task.priority
                    and self._current_task.interruptible):
                self.tts_engine.stop()
                self._interrupted_count += 1
                self._current_task = None
                self._current_entry = None
            self._pending_tasks_count += 1
            self._queue.put(_QueuedAudio(task, self._generation, self._safety_generation))
            return True

    def _interrupt_locked(self) -> None:
        self._generation += 1
        self.tts_engine.stop()
        self._current_task = None
        self._current_entry = None
        self._pending_tasks_count = 0
        self._recent_texts.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def interrupt(self) -> None:
        """Stop playback and invalidate queued AND dequeued tasks atomically."""
        with self._current_lock:
            self._interrupt_locked()

    @property
    def is_speaking(self) -> bool:
        with self._current_lock:
            return self.tts_engine.is_speaking()

    @property
    def is_busy(self) -> bool:
        with self._current_lock:
            return (self._current_task is not None or self._pending_tasks_count > 0
                    or not self._queue.empty() or self.tts_engine.is_speaking())

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        t_start = time.monotonic()
        while self.is_busy:
            if timeout is not None and time.monotonic() - t_start >= timeout:
                return False
            time.sleep(0.02)
        return True

    @property
    def dropped_expired_count(self) -> int:
        with self._current_lock:
            return self._dropped_expired_count

    @property
    def interrupted_count(self) -> int:
        with self._current_lock:
            return self._interrupted_count

    def _eligible_locked(self, entry: _QueuedAudio) -> bool:
        if not self._running or entry.generation != self._generation:
            return False
        if entry.task.priority == AudioPriority.HIGH_RISK:
            if not self._safety_alerts_enabled or entry.safety_generation != self._safety_generation:
                return False
        if entry.task.is_expired(time.monotonic()):
            self._dropped_expired_count += 1
            return False
        return True

    def _play_entry(self, entry: _QueuedAudio) -> None:
        """All playback starts hold the same lock as mode changes; waits never do."""
        task = entry.task
        with self._current_lock:
            if not self._eligible_locked(entry):
                return
            self._current_entry = entry
            self._current_task = task
            if task.sound_file:
                self.tts_engine.play_sound(task.sound_file)
        if task.sound_file:
            self.tts_engine.wait_until_done(timeout=0.5)
        with self._current_lock:
            if self._current_entry is not entry or not self._eligible_locked(entry):
                return
            if task.text:
                self.tts_engine.speak(task.text)
        while True:
            with self._current_lock:
                if (not self._running or self._current_entry is not entry
                        or not self.tts_engine.is_speaking()):
                    return
            time.sleep(0.02)

    def _audio_loop(self) -> None:
        while self._running:
            try:
                entry = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._play_entry(entry)
            finally:
                with self._current_lock:
                    if self._current_entry is entry:
                        self._current_task = None
                        self._current_entry = None
                    # A cancelled generation must not decrement counts for new tasks.
                    if entry.generation == self._generation:
                        self._pending_tasks_count = max(0, self._pending_tasks_count - 1)
