import threading
import time
from typing import Optional, Dict, Any, List
import numpy as np

from src.contracts.frame_packet import FramePacket
from src.contracts.detection import DetectionResult
from src.contracts.depth_map import DepthMap
from src.contracts.risk import RiskAssessment, RiskLevel, DataQuality
from src.contracts.audio import AudioTask, AudioPriority
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
from src.vqa.vqa_service import VQAService
from src.audio.audio_coordinator import AudioCoordinator
from src.runtime.metrics import PerformanceMetrics
from src.runtime.watchdog import Watchdog

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
        vqa_service: Optional[VQAService] = None
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

        self.metrics = PerformanceMetrics()
        self.watchdog = Watchdog(timeout_sec=3.0)

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
        self._on_demand_active = False
        self._on_demand_generation = 0
        self._active_on_demand_token: Optional[int] = None
        self._state_lock = threading.Lock()

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
            if mode == "WARNING":
                self._warning_since = time.monotonic()
                text = "Đã chuyển sang chế độ cảnh báo."
            else:
                text = f"Đã chuyển sang {mode}. Cảnh báo vật cản đã tắt."
            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND, text=text, interruptible=True))

    def _cancel_on_demand_locked(self) -> None:
        self._on_demand_generation += 1
        self._on_demand_active = False
        self._active_on_demand_token = None

    def enter_warning_mode(self) -> None:
        """C: cancel on-demand output and explicitly restore fresh safety alerts."""
        with self._state_lock:
            if self._shutting_down:
                return
            self._cancel_on_demand_locked()
            self._select_mode_locked("WARNING")

    def set_on_demand_active(self, active: bool, mode: str = "OBSERVATION") -> None:
        """Compatibility setter: ending a task no longer changes the selected mode."""
        if active and mode not in ("OCR", "VQA"):
            raise ValueError("On-demand mode must be OCR or VQA")
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
        if (
            self._on_demand_thread
            and self._on_demand_thread.is_alive()
            and self._on_demand_thread is not threading.current_thread()
        ):
            self._on_demand_thread.join(timeout=1.5)
        self.audio_coordinator.stop()

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
                self._post_safety_alert(assessments)

                e2e_ms = (time.monotonic() - t0) * 1000.0
                self.metrics.record_latency("end_to_end_ms", e2e_ms)

                with self._state_lock:
                    self._latest_packet = packet
                    self._latest_detection = tracked_det
                    self._latest_assessments = assessments
                    if self._selected_mode == "WARNING":
                        if sync_pair.data_quality in (DataQuality.STALE, DataQuality.DEGRADED):
                            self._current_mode = "DEGRADED"
                        else:
                            self._current_mode = "OBSERVATION"

            except Exception as e:
                print(f"[DetectionWorker] Lỗi suy luận và fusion: {e}")

    def _post_safety_alert(self, assessments: List[RiskAssessment]) -> None:
        """Gate before aggregation so muted modes do not consume cooldown."""
        with self._state_lock:
            if self._selected_mode != "WARNING" or self._shutting_down:
                return
            candidates = [a for a in assessments if a.source_timestamp > self._warning_since]
            if not candidates:
                return
            task = self.alert_aggregator.aggregate(candidates)
            if task is not None:
                self.audio_coordinator.post_task(task)

    # ================= On-Demand OCR & VQA =================

    def trigger_ocr(self) -> bool:
        """Kích hoạt tác vụ đọc chữ theo yêu cầu."""
        return self._start_on_demand("OCR", self._run_ocr_task)

    @classmethod
    def _is_fresh_on_demand_frame(cls, packet: FramePacket, now: float) -> bool:
        age_ms = packet.age_ms(now)
        return 0.0 <= age_ms <= cls.MAX_ON_DEMAND_FRAME_AGE_MS

    def _run_ocr_task(self, token: int) -> None:
        try:
            if not self._is_current_on_demand(token):
                return
            packet = self.get_latest_frame()
            if packet is None or not self._is_fresh_on_demand_frame(
                packet, time.monotonic()
            ):
                if not self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Chưa có ảnh camera mới để đọc chữ. Vui lòng thử lại.",
                    interruptible=True
                )):
                    return
                if self._is_current_on_demand(token):
                    self._wait_on_demand_audio(token, timeout=4.0)
                return

            if not self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Đang xử lý đọc chữ...",
                sound_file="assets/audio/chime.wav",
                interruptible=True
            )):
                return

            req = OCRRequest(request_id=f"ocr_{token}_{packet.frame_id}", image=packet.image)
            res = self.ocr_service.process(req)

            if not self._publish_on_demand_result(
                token,
                "ocr_sec",
                res.latency_sec,
                AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text=res.text,
                    interruptible=True
                )
            ):
                return

            if self._is_current_on_demand(token):
                self._wait_on_demand_audio(token, timeout=45.0)
        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý OCR: {e}")
        finally:
            self._finish_on_demand(token)

    def trigger_vqa(self, question: str = "Phía trước có gì?") -> bool:
        """Kích hoạt tác vụ hỏi đáp thị giác (VQA)."""
        return self._start_on_demand("VQA", self._run_vqa_task, (question,))

    @staticmethod
    def _fresh_vqa_assessments(packet: FramePacket, assessments: List[RiskAssessment],
                               now: float) -> List[RiskAssessment]:
        return [
            assessment for assessment in assessments
            if assessment.data_quality == DataQuality.VALID
            and not assessment.is_expired(now)
            and abs(packet.timestamp_mono - assessment.source_timestamp) <= 0.75
        ]

    def _run_vqa_task(self, token: int, question: str) -> None:
        # Detection/depth và đánh giá nguy cơ vẫn chạy; âm thanh cảnh báo đã tắt.
        try:
            # 2. Ngắt cảnh báo cũ và làm sạch hàng đợi âm thanh
            if not self._is_current_on_demand(token):
                return
            packet = self.get_latest_frame()
            if packet is None or not self._is_fresh_on_demand_frame(
                packet, time.monotonic()
            ):
                if not self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Chưa có ảnh camera mới để trả lời. Vui lòng thử lại.",
                    interruptible=True
                )):
                    return
                if self._is_current_on_demand(token):
                    self._wait_on_demand_audio(token, timeout=4.0)
                return

            # 3. Thông báo đang quan sát
            if not self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Đang quan sát để trả lời...",
                sound_file="assets/audio/chime.wav",
                interruptible=True
            )):
                return

            with self._state_lock:
                fresh_assessments = self._fresh_vqa_assessments(
                    packet, self._latest_assessments, time.monotonic()
                )
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
            res = self.vqa_service.answer(req, visual_context={
                "detected_classes": detected_names,
                "spatial_objects": spatial_objects
            })

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
                "metrics": self.metrics.get_summary(),
                "is_speaking": self.audio_coordinator.is_speaking
            }
