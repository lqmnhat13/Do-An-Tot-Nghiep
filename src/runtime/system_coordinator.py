import threading
import time
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple
import numpy as np

from src.contracts.frame_packet import FramePacket
from src.contracts.detection import DetectionResult
from src.contracts.depth_map import DepthMap
from src.contracts.risk import RiskAssessment, RiskLevel, DataQuality
from src.contracts.audio import AudioTask, AudioPriority
from src.contracts.modes import ON_DEMAND_MODES
from src.contracts.request import OCRRequest, OCRResult, VQARequest, VQAResult

from src.camera.bounded_buffer import BoundedBuffer
from src.camera.camera_manager import CameraManager
from src.detection.class_filter import ClassFilter
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTrackerAdapter
from src.depth.depth_estimator import DepthEstimator
from src.depth.roi_extractor import ROIExtractor
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import Synchronizer
from src.fusion.risk_fsm import RiskFSM
from src.fusion.alert_aggregator import AlertAggregator
from src.ocr.ocr_service import OCRService
from src.currency.currency_service import CurrencyService
from src.vqa.vqa_service import VQAService
from src.audio.audio_coordinator import AudioCoordinator
from src.runtime.metrics import PerformanceMetrics
from src.runtime.watchdog import Watchdog

if TYPE_CHECKING:  # chỉ cho type hint: transcriber import torch, không nạp khi không cần
    from src.speech.recorder import MicRecorder
    from src.speech.transcriber import PhoWhisperTranscriber

# Câu hỏi phím Q khi không có/không dùng được giọng nói. "Mô tả khung cảnh phía trước"
# khiến Qwen2.5-VL-3B-4bit sinh câu ảo giác cố định trên ảnh phòng khách (NYU) và chạm
# giới hạn token 33% số lần; câu dưới đây là câu đã được đánh giá trên OpenViVQA/KTVIC/NYU.
DEFAULT_VQA_QUESTION = "Phía trước có gì?"
NO_SPEECH_TEXT = "Không nghe rõ câu hỏi. Bấm Q và hỏi lại."
BUSY_TEXT = "Đang xử lý, vui lòng chờ hoặc bấm C."
# Bấm C chỉ huỷ phần đọc kết quả; luồng OCR/VQA đang treo (vd. Apple Vision) vẫn chạy tới hạn.
STOPPING_TEXT = "Tác vụ trước chưa dừng hẳn, vui lòng thử lại sau giây lát."
# Câu thông báo chế độ: ngắn (dựng sẵn clip, ~1 s) và không bị cảnh báo vật cản cắt ngang.
MODE_ANNOUNCEMENTS = {
    "WARNING": "Chế độ cảnh báo",
    "OCR": "Chế độ đọc chữ, tắt cảnh báo",
    "VQA": "Chế độ hỏi đáp, tắt cảnh báo",
    "MONEY": "Chế độ đọc tiền, tắt cảnh báo",
}
VOICE_UNAVAILABLE_TEXT = ("Không dùng được micro hoặc nhận dạng giọng nói. "
                          "Hệ thống sẽ mô tả phía trước.")
LISTEN_CHIME_FILE = "assets/audio/chime.wav"


class _OnDemandCancelled(Exception):
    """Token on-demand không còn hiện hành (C/S/Esc) trong lúc chờ một bước chạy lâu."""


class SystemCoordinator:
    """
    Điều phối viên tối cao của toàn bộ hệ thống (Runtime Coordinator).
    Tuân thủ mục 3.1 và mục 5.1 của Kế hoạch:
    - Quản lý các worker threads độc lập (Camera, Detection, Depth, Audio).
    - Bộ đệm riêng biệt cho từng luồng (BoundedBuffer).
    - Điều phối chế độ hoạt động (Quan sát liên tục vs On-Demand OCR/VQA).
    - Đo đạc metrics P50/P95 và giám sát watchdog.
    """

    MAX_ON_DEMAND_FRAME_AGE_MS = 1000.0
    # Giới hạn thời gian một tác vụ on-demand: Apple Vision từng treo > 40 s trong app
    # (1/5 lần chạy), khoá luôn VQA. Hết hạn thì báo người dùng và giải phóng worker;
    # luồng suy luận bị treo (daemon) không thể bị dừng cưỡng bức.
    OCR_TIMEOUT_SEC = 30.0
    VQA_TIMEOUT_SEC = 60.0  # gồm cả lần nạp model MLX đầu tiên (~11 s)
    # Nhận dạng giọng nói quá hạn này thì coi như model lỗi (đo thật: ~0,7 s nóng, ~4,5 s lần đầu).
    STT_TIMEOUT_SEC = 15.0
    DEFAULT_RECORD_MAX_SEC = 8.0
    VOICE_AUDIO_IDLE_TIMEOUT_SEC = 4.0
    # Câu nhận dạng dài hơn thế này gần như chắc là PhoWhisper "ảo giác" trên tiếng ồn.
    MAX_VOICE_QUESTION_CHARS = 150
    DEFAULT_VQA_QUESTION = DEFAULT_VQA_QUESTION
    NO_SPEECH_TEXT = NO_SPEECH_TEXT
    BUSY_TEXT = BUSY_TEXT
    STOPPING_TEXT = STOPPING_TEXT
    # Cảnh báo vật cản chờ câu "Chế độ cảnh báo" đọc xong, nhưng không lâu hơn mức này
    # (luồng âm thanh kẹt thì cảnh báo vẫn phải chạy).
    MODE_ANNOUNCE_MAX_SEC = 2.5
    VOICE_UNAVAILABLE_TEXT = VOICE_UNAVAILABLE_TEXT

    # Giám sát sức khỏe: người khiếm thị không thấy HUD, nên mất camera/depth phải được nói ra.
    HEALTH_CHECK_INTERVAL_SEC = 0.5
    CAMERA_TIMEOUT_SEC = 2.0
    STATUS_REPEAT_SEC = 10.0
    STARTUP_GRACE_SEC = 5.0
    STATUS_MESSAGES = {
        "camera": "Mất tín hiệu camera. Cảnh báo vật cản đang tạm ngừng, "
                  "hãy dừng lại và dùng gậy dò đường.",
        "depth": "Dữ liệu độ sâu bị gián đoạn. Cảnh báo vật cản đang tạm ngừng, "
                 "hãy dừng lại và dùng gậy dò đường.",
        "detection": "Bộ nhận diện vật cản bị gián đoạn. Cảnh báo vật cản đang tạm ngừng, "
                     "hãy dừng lại và dùng gậy dò đường.",
    }
    RECOVERED_WARNING = "Cảnh báo vật cản đã hoạt động trở lại."
    RECOVERED_OTHER = "Hệ thống đã hoạt động trở lại."

    def __init__(
        self,
        camera_manager: CameraManager,
        detector: YoloDetector,
        tracker: ByteTrackerAdapter,
        depth_estimator: DepthEstimator,
        synchronizer: Synchronizer,
        risk_fsm: RiskFSM,
        alert_aggregator: AlertAggregator,
        audio_coordinator: AudioCoordinator,
        ocr_service: Optional[OCRService] = None,
        vqa_service: Optional[VQAService] = None,
        watchdog_timeout_sec: float = 3.0,
        voice_recorder: Optional["MicRecorder"] = None,
        transcriber: Optional["PhoWhisperTranscriber"] = None,
        currency_service: Optional[CurrencyService] = None
    ):
        self.camera_manager = camera_manager
        self.detector = detector
        self.tracker = tracker
        self.depth_estimator = depth_estimator
        self.synchronizer = synchronizer
        self.risk_fsm = risk_fsm
        self.alert_aggregator = alert_aggregator
        self.audio_coordinator = audio_coordinator
        self.ocr_service = ocr_service or OCRService()
        self.vqa_service = vqa_service or VQAService()
        # Nhận diện tiền dùng chung OCRService (chung tiến trình Vision, huỷ và dự phòng EasyOCR).
        self.currency_service = currency_service or CurrencyService(self.ocr_service)
        # Hỏi bằng giọng nói chỉ bật khi có đủ cả micro lẫn bộ nhận dạng.
        self.voice_recorder = voice_recorder
        self.transcriber = transcriber

        self.metrics = PerformanceMetrics()
        self.watchdog = Watchdog(timeout_sec=watchdog_timeout_sec)

        # Bộ đệm riêng cho detection và depth
        self.buf_detection: BoundedBuffer[FramePacket] = BoundedBuffer(maxsize=2)
        self.buf_depth: BoundedBuffer[FramePacket] = BoundedBuffer(maxsize=2)
        self.camera_manager.subscribe(self.buf_detection)
        self.camera_manager.subscribe(self.buf_depth)

        # Trạng thái runtime
        self._running = False
        self._det_thread: Optional[threading.Thread] = None
        self._depth_thread: Optional[threading.Thread] = None
        self._on_demand_thread: Optional[threading.Thread] = None
        self._on_demand_starting = False
        self._shutting_down = False

        # Lưu trữ trạng thái mới nhất phục vụ hiển thị HUD
        self._latest_packet: Optional[FramePacket] = None
        self._latest_detection: Optional[DetectionResult] = None
        self._latest_depth: Optional[DepthMap] = None
        self._latest_assessments: List[RiskAssessment] = []
        self._current_mode = "OBSERVATION" # "OBSERVATION" | "OCR" | "VQA" | "DEGRADED"
        self._selected_mode = "WARNING"
        self._warning_since = float("-inf")
        self._mode_announcement: Optional[AudioTask] = None
        self._mode_announced_at = float("-inf")
        self._on_demand_active = False
        self._on_demand_generation = 0
        self._active_on_demand_token: Optional[int] = None
        self._state_lock = threading.Lock()
        # Thu âm câu hỏi VQA (đọc/ghi dưới _state_lock)
        self._listening = False
        self._listen_stop_event: Optional[threading.Event] = None
        self._vqa_question: Optional[str] = None

        # Giám sát sức khỏe (đọc/ghi dưới _state_lock)
        self._health_thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._last_valid_pair_time: Optional[float] = None
        self._health_problem: Optional[str] = None
        self._announced_problem: Optional[str] = None
        self._last_status_time = float("-inf")

    def is_on_demand_active(self) -> bool:
        """Kiểm tra xem hệ thống có đang trong tác vụ On-Demand (VQA/OCR) hay không."""
        with self._state_lock:
            return self._on_demand_active

    def _select_mode_locked(self, mode: str) -> None:
        changed = self._selected_mode != mode
        self._selected_mode = mode
        self._current_mode = "OBSERVATION" if mode == "WARNING" else mode
        self.audio_coordinator.interrupt()
        self.audio_coordinator.set_safety_alerts_enabled(mode == "WARNING")
        if changed:
            now = time.monotonic()
            if mode == "WARNING":
                self._warning_since = now
            # Không cho cảnh báo HIGH ngắt: vừa bấm C luôn có vật HIGH, câu bị cắt sau ~0,2 s.
            task = AudioTask(priority=AudioPriority.ON_DEMAND, text=MODE_ANNOUNCEMENTS[mode],
                             interruptible=False)
            if self.audio_coordinator.post_task(task):
                self._mode_announcement = task
                self._mode_announced_at = now

    def _cancel_on_demand_locked(self) -> None:
        self._on_demand_generation += 1
        self._on_demand_active = False
        self._active_on_demand_token = None
        # Đang thu micro thì dừng ngay (recorder kiểm tra sự kiện giữa mỗi block ~30 ms).
        if self._listen_stop_event is not None:
            self._listen_stop_event.set()
        # Vision ở tiến trình phụ dừng được ngay, để luồng OCR kết thúc và Space dùng lại được.
        cancel_ocr = getattr(self.ocr_service, "cancel", None)
        if callable(cancel_ocr):
            cancel_ocr()

    def enter_warning_mode(self) -> None:
        """C: cancel on-demand output and explicitly restore fresh safety alerts."""
        with self._state_lock:
            if self._shutting_down:
                return
            self._cancel_on_demand_locked()
            self._vqa_question = None
            self._select_mode_locked("WARNING")

    def set_on_demand_active(self, active: bool, mode: str = "OBSERVATION") -> None:
        """Compatibility setter: ending a task no longer changes the selected mode."""
        if active and mode not in ON_DEMAND_MODES:
            raise ValueError(f"On-demand mode must be one of {ON_DEMAND_MODES}")
        with self._state_lock:
            if self._shutting_down and active:
                return
            self._cancel_on_demand_locked()
            if active:
                self._select_mode_locked(mode)
                self._on_demand_active = True
                self._active_on_demand_token = self._on_demand_generation
            else:
                self.audio_coordinator.interrupt()

    def _is_current_on_demand(self, token: int) -> bool:
        with self._state_lock:
            return self._on_demand_active and self._active_on_demand_token == token

    def _post_on_demand_audio(self, token: int, task: AudioTask) -> bool:
        """Chỉ token hiện hành mới được tạo side effect âm thanh."""
        with self._state_lock:
            if not self._on_demand_active or self._active_on_demand_token != token:
                return False
            return self.audio_coordinator.post_task(task)

    def _wait_on_demand_audio(self, token: int, timeout: float) -> bool:
        """Bounded polling lets C/S/shutdown release a cancelled worker promptly."""
        deadline = time.monotonic() + timeout
        while self._is_current_on_demand(token):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self.audio_coordinator.wait_until_idle(timeout=min(0.1, remaining)):
                return True
        return False

    def _publish_on_demand_result(
        self,
        token: int,
        metric_name: str,
        latency: float,
        task: AudioTask
    ) -> bool:
        """Ghi metrics và phát kết quả như một thao tác có kiểm tra generation."""
        with self._state_lock:
            if not self._on_demand_active or self._active_on_demand_token != token:
                return False
            self.metrics.record_latency(metric_name, latency)
            return self.audio_coordinator.post_task(task)

    def _finish_on_demand(self, token: int) -> None:
        """Only finish the current task; retain the selected OCR/VQA mode."""
        with self._state_lock:
            if self._active_on_demand_token != token:
                return
            self._on_demand_active = False
            self._active_on_demand_token = None

    def _start_on_demand(
        self,
        mode: str,
        target,
        args: tuple = ()
    ) -> bool:
        """Dành token và khởi động đúng một worker inference on-demand."""
        with self._state_lock:
            if self._shutting_down or self._on_demand_starting or (
                self._on_demand_thread is not None and self._on_demand_thread.is_alive()
            ):
                rejection = "worker on-demand trước chưa kết thúc"
                thread = None
                token = -1
            elif self._on_demand_active:
                rejection = "tác vụ on-demand khác đang hoạt động"
                thread = None
                token = -1
            else:
                rejection = ""
                self._on_demand_generation += 1
                token = self._on_demand_generation
                self._on_demand_active = True
                self._active_on_demand_token = token
                self._vqa_question = None
                self._select_mode_locked(mode)
                self._on_demand_starting = True
                thread = threading.Thread(
                    target=target,
                    args=(token, *args),
                    name=f"{mode}TaskThread",
                    daemon=True
                )
                self._on_demand_thread = thread

        if thread is None:
            print(f"[SystemCoordinator] Từ chối {mode}: {rejection}.")
            return False

        try:
            # Không start thread trong lúc giữ _state_lock.
            thread.start()
            return True
        except Exception as exc:
            with self._state_lock:
                if self._on_demand_thread is thread:
                    self._on_demand_thread = None
                if self._active_on_demand_token == token:
                    self._on_demand_generation += 1
                    self._active_on_demand_token = None
                    self._on_demand_active = False
            print(f"[SystemCoordinator] Không thể khởi động {mode}: {exc}")
            return False
        finally:
            with self._state_lock:
                self._on_demand_starting = False

    def start(self) -> None:
        """Khởi động toàn bộ các luồng của hệ thống."""
        if self._running:
            return
        with self._state_lock:
            self._shutting_down = False
        self._running = True

        self.audio_coordinator.start()
        self.camera_manager.start()

        self._det_thread = threading.Thread(target=self._detection_worker, name="DetectionWorker", daemon=True)
        self._depth_thread = threading.Thread(target=self._depth_worker, name="DepthWorker", daemon=True)

        self._det_thread.start()
        self._depth_thread.start()

        with self._state_lock:
            self._started_at = time.monotonic()
        self._health_thread = threading.Thread(target=self._health_worker, name="HealthMonitor", daemon=True)
        self._health_thread.start()

        # Phát âm thanh chào mừng hệ thống khởi động
        self.audio_coordinator.post_task(AudioTask(
            priority=AudioPriority.INFO,
            text="Hệ thống đã sẵn sàng hỗ trợ bạn.",
            sound_file="assets/audio/ready.wav"
        ))

    def stop(self) -> None:
        """Dừng toàn bộ hệ thống an toàn."""
        with self._state_lock:
            self._shutting_down = True
            self._cancel_on_demand_locked()
            self.audio_coordinator.interrupt()
        self._running = False
        self.camera_manager.stop()
        if self._det_thread and self._det_thread.is_alive():
            self._det_thread.join(timeout=1.5)
        if self._depth_thread and self._depth_thread.is_alive():
            self._depth_thread.join(timeout=1.5)
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=1.5)
        if (
            self._on_demand_thread
            and self._on_demand_thread.is_alive()
            and self._on_demand_thread is not threading.current_thread()
        ):
            self._on_demand_thread.join(timeout=1.5)
        close_ocr = getattr(self.ocr_service, "close", None)
        if callable(close_ocr):
            close_ocr()
        self.audio_coordinator.stop()

    # ================= Giám sát sức khỏe =================

    def _health_worker(self) -> None:
        while self._running:
            time.sleep(self.HEALTH_CHECK_INTERVAL_SEC)
            if not self._running:
                break
            try:
                self._check_health(time.monotonic())
            except Exception as e:
                print(f"[SystemCoordinator] Lỗi giám sát sức khỏe: {e}")

    def _diagnose(self, now: float) -> Optional[str]:
        """Trả về 'camera' | 'detection' | 'depth' khi cảnh báo vật cản không còn tin cậy."""
        with self._state_lock:
            started_at = self._started_at
            last_valid = self._last_valid_pair_time
        if started_at is None or now - started_at < self.STARTUP_GRACE_SEC:
            return None
        packet = self.camera_manager.get_latest_frame()
        if (not self.camera_manager.is_alive() or packet is None
                or now - packet.timestamp_mono > self.CAMERA_TIMEOUT_SEC):
            return "camera"
        load_error = getattr(self.depth_estimator, "load_error", None)
        if isinstance(load_error, str) and load_error:
            return "depth"
        health = self.watchdog.check_health()
        if not health.get("detection_worker", False):
            return "detection"
        if not health.get("depth_worker", False):
            return "depth"
        # Depth còn chạy nhưng cặp detection–depth không còn VALID (stale/degraded).
        if now - (last_valid if last_valid is not None else started_at) > self.watchdog.timeout_sec:
            return "depth"
        return None

    def _check_health(self, now: float) -> Optional[str]:
        """Nói ra khi hỏng (nhắc lại mỗi STATUS_REPEAT_SEC) và khi phục hồi.

        Mất camera được báo ở mọi chế độ; lỗi depth/detection chỉ báo ở chế độ Cảnh báo,
        vì ở OCR/VQA cảnh báo vật cản vốn đã tắt.
        """
        problem = self._diagnose(now)
        with self._state_lock:
            if self._shutting_down:
                return problem
            warning = self._selected_mode == "WARNING"
            self._health_problem = problem
            if problem is not None and warning:
                self._current_mode = "DEGRADED"
            audible = problem if (problem == "camera" or warning) else None
            text = None
            if audible is not None:
                if (audible != self._announced_problem
                        or now - self._last_status_time >= self.STATUS_REPEAT_SEC):
                    text = self.STATUS_MESSAGES[audible]
                    self._announced_problem = audible
                    self._last_status_time = now
            elif self._announced_problem is not None:
                if problem is None:
                    text = self.RECOVERED_WARNING if warning else self.RECOVERED_OTHER
                self._announced_problem = None
            if text is not None:
                self.audio_coordinator.post_task(AudioTask(
                    priority=AudioPriority.SYSTEM_STATUS, text=text,
                    expires_at=time.monotonic() + 5.0, interruptible=True))
        return problem

    def _call_with_deadline(self, fn, timeout_sec: float, token: Optional[int] = None):
        """Chạy fn ở luồng daemon riêng; TimeoutError nếu quá hạn (luồng treo bị bỏ lại).

        Có ``token``: kiểm tra mỗi 50 ms và ném ``_OnDemandCancelled`` ngay khi tác vụ bị hủy,
        để C/S/Esc giải phóng worker on-demand mà không chờ hết hạn.
        """
        box: Dict[str, Any] = {}

        def run():
            try:
                box["result"] = fn()
            except BaseException as exc:  # chuyển lỗi về worker on-demand
                box["error"] = exc

        worker = threading.Thread(target=run, name="OnDemandInference", daemon=True)
        worker.start()
        if token is None:
            worker.join(timeout_sec)
        else:
            deadline = time.monotonic() + timeout_sec
            while worker.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                worker.join(min(0.05, remaining))
                if worker.is_alive() and not self._is_current_on_demand(token):
                    raise _OnDemandCancelled()
        if worker.is_alive():
            raise TimeoutError(f"quá {timeout_sec:.0f} s")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _depth_worker(self) -> None:
        """Luồng chuyên chạy ước lượng độ sâu (6 - 8 Hz), nhường GPU cho luồng detection."""
        depth_interval = 0.125 # 8 FPS mục tiêu theo kiến trúc
        while self._running:
            packet = self.buf_depth.get(timeout=0.2)
            if packet is None:
                continue

            self.watchdog.beat("depth_worker")
            t0 = time.monotonic()
            try:
                depth_map = self.depth_estimator.estimate(packet)
                self.synchronizer.register_depth(depth_map)
                dt_ms = (time.monotonic() - t0) * 1000.0
                self.metrics.record_latency("depth_ms", dt_ms)

                with self._state_lock:
                    self._latest_depth = depth_map
            except Exception as e:
                print(f"[DepthWorker] Lỗi suy luận độ sâu: {e}")

            # Điều hòa nhịp độ sâu để tránh nghẽn Metal/GPU trên Apple Silicon
            elapsed = time.monotonic() - t0
            rem = depth_interval - elapsed
            if rem > 0.005:
                time.sleep(rem)

    def _detection_worker(self) -> None:
        """Luồng chuyên chạy Detection, Tracking và Risk Fusion (10 - 20 Hz)."""
        while self._running:
            packet = self.buf_detection.get(timeout=0.2)
            if packet is None:
                continue

            self.watchdog.beat("detection_worker")
            self.metrics.record_frame()
            t0 = time.monotonic()

            try:
                # 1. Detection qua YOLO
                raw_det = self.detector.detect(packet)
                self.metrics.record_latency("detection_ms", raw_det.latency_ms)

                # 2. Tracking qua ByteTrack
                tracked_det = self.tracker.update(raw_det)

                # 3. Synchronizer ghép cặp với Depth map tương ứng
                sync_pair = self.synchronizer.synchronize(tracked_det)

                # 4. Risk Fusion FSM đánh giá rủi ro
                assessments = self.risk_fsm.update(sync_pair)

                # 5. Chỉ phát cảnh báo khi người dùng chọn chế độ Cảnh báo.
                self._post_safety_alert(assessments, sync_pair.data_quality)

                e2e_ms = (time.monotonic() - t0) * 1000.0
                self.metrics.record_latency("end_to_end_ms", e2e_ms)

                with self._state_lock:
                    self._latest_packet = packet
                    self._latest_detection = tracked_det
                    self._latest_assessments = assessments
                    if sync_pair.data_quality == DataQuality.VALID:
                        self._last_valid_pair_time = time.monotonic()
                    if self._selected_mode == "WARNING":
                        if sync_pair.data_quality in (DataQuality.STALE, DataQuality.DEGRADED):
                            self._current_mode = "DEGRADED"
                        else:
                            self._current_mode = "OBSERVATION"

            except Exception as e:
                print(f"[DetectionWorker] Lỗi suy luận và fusion: {e}")

    def _post_safety_alert(self, assessments: List[RiskAssessment],
                           data_quality: Optional[DataQuality] = None) -> None:
        """Gate before aggregation so muted modes do not consume cooldown.

        data_quality là chất lượng của cặp detection/depth vừa đánh giá; khi có, câu cảnh báo về
        vật không còn HIGH trên dữ liệu VALID sẽ bị dừng (người dùng đã đi qua vật).
        """
        with self._state_lock:
            if self._selected_mode != "WARNING" or self._shutting_down:
                return
            if self._mode_announcement is not None:
                # Chưa tổng hợp (không tốn cooldown) khi "Chế độ cảnh báo" còn chờ/đang đọc.
                if (time.monotonic() - self._mode_announced_at < self.MODE_ANNOUNCE_MAX_SEC
                        and self.audio_coordinator.is_pending_or_playing(self._mode_announcement)):
                    return
                self._mode_announcement = None
            candidates = [a for a in assessments if a.source_timestamp > self._warning_since]
            if data_quality is not None:
                keys = self.alert_aggregator.supported_alert_keys(candidates, data_quality)
                if keys is not None and self.audio_coordinator.retain_safety_alerts(keys):
                    self.alert_aggregator.release_global_interval()
            if not candidates:
                return
            task = self.alert_aggregator.aggregate(candidates)
            if task is not None and not self.audio_coordinator.post_task(task):
                self.alert_aggregator.release_global_interval()

    # ================= On-Demand OCR & VQA =================

    def trigger_ocr(self) -> bool:
        """Kích hoạt tác vụ đọc chữ theo yêu cầu."""
        if self._start_on_demand("OCR", self._run_ocr_task):
            return True
        self._say_rejected()
        return False

    def _say_rejected(self) -> None:
        """Phím Space/Q bị từ chối: nói lý do để người dùng không tưởng phím hỏng.

        Im lặng khi đang thu micro (tiếng máy lẫn vào câu hỏi) hoặc đang tắt hệ thống.
        """
        with self._state_lock:
            if self._shutting_down or self._listening:
                return
            if self._on_demand_active:
                text = BUSY_TEXT
            elif self._on_demand_starting or (
                    self._on_demand_thread is not None and self._on_demand_thread.is_alive()):
                text = STOPPING_TEXT
            else:
                return
            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND, text=text, interruptible=True))

    @classmethod
    def _is_fresh_on_demand_frame(cls, packet: FramePacket, now: float) -> bool:
        age_ms = packet.age_ms(now)
        return 0.0 <= age_ms <= cls.MAX_ON_DEMAND_FRAME_AGE_MS

    def _run_ocr_task(self, token: int) -> None:
        def process(packet: FramePacket):
            res = self.ocr_service.process(
                OCRRequest(request_id=f"ocr_{token}_{packet.frame_id}", image=packet.image))
            return (None if res.error_message == "cancelled" else res.text), res.latency_sec
        self._run_frame_task(token, "OCR", "ocr_sec", process,
                             no_frame_text="Chưa có ảnh camera mới để đọc chữ. Vui lòng thử lại.",
                             processing_text="Đang xử lý đọc chữ...",
                             timeout_text="Đọc chữ quá thời gian. Vui lòng thử lại.")

    def trigger_currency(self) -> bool:
        """Phím M: nhận diện mệnh giá tờ tiền trước camera."""
        if self._start_on_demand("MONEY", self._run_currency_task):
            return True
        self._say_rejected()
        return False

    def _run_currency_task(self, token: int) -> None:
        def process(packet: FramePacket):
            res = self.currency_service.recognize(packet.image,
                                                  request_id=f"money_{token}_{packet.frame_id}")
            return (None if res.cancelled else res.text), res.latency_sec
        self._run_frame_task(token, "MONEY", "money_sec", process,
                             no_frame_text="Chưa có ảnh camera mới để nhận diện tiền. Vui lòng thử lại.",
                             processing_text="Đang nhận diện tiền...",
                             timeout_text="Nhận diện tiền quá thời gian. Vui lòng thử lại.")

    def _run_frame_task(self, token: int, label: str, metric_name: str, process,
                        no_frame_text: str, processing_text: str, timeout_text: str) -> None:
        """Khung chung cho tác vụ một ảnh (OCR, nhận diện tiền): ảnh mới -> bíp + "đang xử lý"
        -> ``process(packet)`` có hạn OCR_TIMEOUT_SEC -> đọc kết quả. ``process`` trả
        ``(câu nói, latency)``; câu nói ``None`` nghĩa là đã bị huỷ, không đọc gì."""
        try:
            if not self._is_current_on_demand(token):
                return
            packet = self.get_latest_frame()
            if packet is None or not self._is_fresh_on_demand_frame(
                packet, time.monotonic()
            ):
                if not self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text=no_frame_text,
                    interruptible=True
                )):
                    return
                if self._is_current_on_demand(token):
                    self._wait_on_demand_audio(token, timeout=4.0)
                return

            if not self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text=processing_text,
                sound_file="assets/audio/chime.wav",
                interruptible=True
            )):
                return

            try:
                text, latency = self._call_with_deadline(lambda: process(packet), self.OCR_TIMEOUT_SEC)
            except TimeoutError:
                print(f"[SystemCoordinator] {label} quá {self.OCR_TIMEOUT_SEC:.0f} s, hủy tác vụ.")
                self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text=timeout_text,
                    interruptible=True))
                return
            if text is None:
                return

            if not self._publish_on_demand_result(
                token,
                metric_name,
                latency,
                AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text=text,
                    interruptible=True
                )
            ):
                return

            if self._is_current_on_demand(token):
                self._wait_on_demand_audio(token, timeout=45.0)
        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý {label}: {e}")
        finally:
            self._finish_on_demand(token)

    def trigger_vqa(self, question: Optional[str] = DEFAULT_VQA_QUESTION) -> bool:
        """Kích hoạt tác vụ hỏi đáp thị giác (VQA).

        ``question=None``: hỏi bằng giọng nói nếu có đủ micro và bộ nhận dạng; không có thì
        dùng ``DEFAULT_VQA_QUESTION`` như trước, không báo gì thêm.
        """
        if self._start_on_demand("VQA", self._run_vqa_task, (question,)):
            return True
        self._say_rejected()
        return False

    def is_listening(self) -> bool:
        with self._state_lock:
            return self._listening

    def stop_listening(self) -> bool:
        """Q lần nữa: dừng thu và nhận dạng phần đã nói. False nếu không đang thu."""
        with self._state_lock:
            if not self._listening or self._listen_stop_event is None:
                return False
            self._listen_stop_event.set()
            return True

    def _voice_enabled(self) -> bool:
        return self.voice_recorder is not None and self.transcriber is not None

    def _say_on_demand(self, token: int, text: str, wait_sec: float = 10.0) -> None:
        if self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND, text=text, interruptible=True)):
            self._wait_on_demand_audio(token, timeout=wait_sec)

    def _capture_fresh_frame(self) -> Optional[FramePacket]:
        packet = self.get_latest_frame()
        if packet is None or not self._is_fresh_on_demand_frame(packet, time.monotonic()):
            return None
        return packet

    def _capture_vqa_scene(self) -> Tuple[Optional[FramePacket], List[RiskAssessment]]:
        """Chụp ảnh và đánh giá nguy cơ khớp ảnh cùng lúc: nhận dạng giọng nói mất ~1-5 s,
        đợi tới lúc đó thì kết quả detection lúc chụp đã hết hạn/bị thay."""
        packet = self._capture_fresh_frame()
        if packet is None:
            return None, []
        with self._state_lock:
            assessments = list(self._latest_assessments)
        return packet, self._fresh_vqa_assessments(packet, assessments, time.monotonic())

    def _voice_fallback(self, token: int):
        """Không hỏi bằng giọng nói được: báo rồi dùng câu hỏi mặc định với ảnh chụp lúc này."""
        if not self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND, text=VOICE_UNAVAILABLE_TEXT,
                interruptible=True)):
            return None
        return (DEFAULT_VQA_QUESTION, *self._capture_vqa_scene(), False)

    def _record_question(self, token: int):
        """Thu micro dưới hạn ``max_sec + 2``; hủy/quá hạn đều set stop_event để đóng stream."""
        stop_event = threading.Event()
        with self._state_lock:
            if not self._on_demand_active or self._active_on_demand_token != token:
                raise _OnDemandCancelled()
            self._listening = True
            self._listen_stop_event = stop_event
        kwargs = getattr(self.voice_recorder, "endpointer_kwargs", None) or {}
        max_sec = float(kwargs.get("max_sec", self.DEFAULT_RECORD_MAX_SEC))
        try:
            return self._call_with_deadline(
                lambda: self.voice_recorder.record(stop_event), max_sec + 2.0, token)
        finally:
            stop_event.set()
            with self._state_lock:
                if self._listen_stop_event is stop_event:
                    self._listening = False
                    self._listen_stop_event = None

    def _ask_by_voice(self, token: int):
        """Bíp -> thu -> nhận dạng. Trả ``(câu hỏi, ảnh lúc dừng thu, đánh giá nguy cơ khớp ảnh,
        là_giọng_nói)``; ảnh ``None`` nghĩa là không có ảnh mới. ``None`` nếu tác vụ đã kết
        thúc/bị hủy."""
        # Chờ câu "Chế độ hỏi đáp..." phát xong rồi mới bíp, và chờ bíp xong mới mở micro,
        # để micro không thu lẫn tiếng máy. Loa vẫn bận quá hạn thì không mở micro đè lên.
        if not self._wait_on_demand_audio(token, self.VOICE_AUDIO_IDLE_TIMEOUT_SEC):
            if not self._is_current_on_demand(token):
                return None
            print("[SystemCoordinator] Loa vẫn bận, không mở micro.")
            return self._voice_fallback(token)
        self._post_on_demand_audio(token, AudioTask(
            priority=AudioPriority.ON_DEMAND, text="", sound_file=LISTEN_CHIME_FILE,
            interruptible=True))
        if not self._wait_on_demand_audio(token, self.VOICE_AUDIO_IDLE_TIMEOUT_SEC):
            if not self._is_current_on_demand(token):
                return None
            print("[SystemCoordinator] Bíp chưa phát xong, không mở micro.")
            return self._voice_fallback(token)

        try:
            recording = self._record_question(token)
            if not self._is_current_on_demand(token):
                return None
            audio = np.asarray(recording.audio, dtype=np.float32).reshape(-1)
            if audio.size > 0 and not np.any(audio):
                # macOS trả im lặng tuyệt đối khi app chưa được cấp quyền micro.
                raise RuntimeError("micro chỉ trả về im lặng tuyệt đối (chưa cấp quyền micro?)")
            kwargs = getattr(self.voice_recorder, "endpointer_kwargs", None) or {}
            min_rms = float(kwargs.get("rms_threshold", 0.01))
            quiet = audio.size == 0 or float(
                np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < min_rms
            if not recording.heard_speech or quiet:
                # Không gọi PhoWhisper: trên im lặng/tiếng ồn nó "ảo giác" ra cả câu.
                self._say_on_demand(token, NO_SPEECH_TEXT)
                return None
            # Ảnh và kết quả detection cho VQA là của lúc dừng thu (trước khi nhận dạng).
            packet, assessments = self._capture_vqa_scene()
            if packet is None:
                return None, None, [], True
            sample_rate = int(getattr(self.voice_recorder, "sample_rate", 16000))
            text = self._call_with_deadline(
                lambda: self.transcriber.transcribe(audio, sample_rate=sample_rate),
                self.STT_TIMEOUT_SEC, token)
        except _OnDemandCancelled:
            return None
        except Exception as exc:  # MicrophoneError, lỗi nạp model, TimeoutError...
            print(f"[SystemCoordinator] Không hỏi được bằng giọng nói: {exc}")
            return self._voice_fallback(token)

        question = (text or "").strip()
        if not question or len(question) > self.MAX_VOICE_QUESTION_CHARS:
            self._say_on_demand(token, NO_SPEECH_TEXT)
            return None
        return question, packet, assessments, True

    @staticmethod
    def _echo_question_text(question: str) -> str:
        if question[-1] not in ".?!…":
            question += "."
        return f"Bạn hỏi: {question}"

    @staticmethod
    def _fresh_vqa_assessments(packet: FramePacket, assessments: List[RiskAssessment],
                               now: float) -> List[RiskAssessment]:
        return [
            assessment for assessment in assessments
            if assessment.data_quality == DataQuality.VALID
            and not assessment.is_expired(now)
            and abs(packet.timestamp_mono - assessment.source_timestamp) <= 0.75
        ]

    def _run_vqa_task(self, token: int, question: Optional[str]) -> None:
        # Detection/depth và đánh giá nguy cơ vẫn chạy; âm thanh cảnh báo đã tắt.
        try:
            if not self._is_current_on_demand(token):
                return
            if question is None and self._voice_enabled():
                # 1. Hỏi bằng giọng nói; ảnh chụp lúc dừng thu.
                outcome = self._ask_by_voice(token)
                if outcome is None:
                    return
                question, packet, fresh_assessments, from_voice = outcome
                announce_observing = False
            else:
                # 1. Câu hỏi có sẵn (hoặc mặc định khi không có micro/nhận dạng).
                if question is None:
                    question = DEFAULT_VQA_QUESTION
                packet, fresh_assessments = self._capture_vqa_scene()
                from_voice = False
                announce_observing = True

            # 2. Không có ảnh camera mới thì không trả lời.
            if packet is None:
                if not self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Chưa có ảnh camera mới để trả lời. Vui lòng thử lại.",
                    interruptible=True
                )):
                    return
                if self._is_current_on_demand(token):
                    self._wait_on_demand_audio(token, timeout=4.0)
                return

            with self._state_lock:
                if self._active_on_demand_token == token:
                    self._vqa_question = question

            # 3. Nhắc lại câu hỏi đã nghe, hoặc thông báo đang quan sát
            if from_voice:
                if not self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text=self._echo_question_text(question),
                    interruptible=True
                )):
                    return
            elif announce_observing and not self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Đang quan sát để trả lời...",
                sound_file="assets/audio/chime.wav",
                interruptible=True
            )):
                return

            detected_names = [a.class_name for a in fresh_assessments]
            spatial_objects = [
                {
                    "name": a.class_name,
                    "direction": a.direction.value,
                    "proximity": a.proximity_desc,
                    "risk_level": a.risk_level.value
                }
                for a in fresh_assessments
            ]

            # 4. Thực thi mô hình VQA (BLIP Captioning + MarianMT + Spatial Grounding)
            if not self._is_current_on_demand(token):
                return

            req = VQARequest(request_id=f"vqa_{token}_{packet.frame_id}", image=packet.image, question=question)
            context = {"detected_classes": detected_names, "spatial_objects": spatial_objects}
            try:
                res = self._call_with_deadline(
                    lambda: self.vqa_service.answer(req, visual_context=context),
                    self.VQA_TIMEOUT_SEC)
            except TimeoutError:
                print(f"[SystemCoordinator] VQA quá {self.VQA_TIMEOUT_SEC:.0f} s, hủy tác vụ.")
                self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Trả lời quá thời gian. Vui lòng thử lại.",
                    interruptible=True))
                return

            if not self._publish_on_demand_result(
                token,
                "vqa_sec",
                res.latency_sec,
                AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text=res.answer,
                    interruptible=True
                )
            ):
                return

            # 6. Đợi câu trả lời phát xong, không đổi chế độ đã chọn.
            if self._is_current_on_demand(token):
                self._wait_on_demand_audio(token, timeout=45.0)

        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý VQA: {e}")
        finally:
            # 7. Kết thúc tác vụ; tiếp tục giữ chế độ VQA.
            self._finish_on_demand(token)

    def stop_speech(self) -> None:
        """S: cancel speech/results while keeping the selected mode."""
        with self._state_lock:
            self._cancel_on_demand_locked()
            self.audio_coordinator.interrupt()

    # ================= State Getters for HUD =================

    def get_latest_frame(self) -> Optional[FramePacket]:
        live_packet = self.camera_manager.get_latest_frame()
        if live_packet is not None:
            return live_packet
        with self._state_lock:
            return self._latest_packet

    def get_runtime_snapshot(self) -> Dict[str, Any]:
        live_packet = self.camera_manager.get_latest_frame()
        with self._state_lock:
            return {
                "packet": live_packet if live_packet is not None else self._latest_packet,
                "detection": self._latest_detection,
                "depth": self._latest_depth,
                "assessments": list(self._latest_assessments),
                "mode": self._current_mode,
                "selected_mode": self._selected_mode,
                "on_demand_active": self._on_demand_active,
                "safety_alerts_enabled": self._selected_mode == "WARNING" and not self._shutting_down,
                "health_problem": self._health_problem,
                "metrics": self.metrics.get_summary(),
                "is_speaking": self.audio_coordinator.is_speaking,
                "listening": self._listening,
                "vqa_question": self._vqa_question
            }
