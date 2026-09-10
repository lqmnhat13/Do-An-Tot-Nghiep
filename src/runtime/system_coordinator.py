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

        # Lưu trữ trạng thái mới nhất phục vụ hiển thị HUD
        self._latest_packet: Optional[FramePacket] = None
        self._latest_detection: Optional[DetectionResult] = None
        self._latest_depth: Optional[DepthMap] = None
        self._latest_assessments: List[RiskAssessment] = []
        self._current_mode = "OBSERVATION" # "OBSERVATION" | "OCR" | "VQA" | "DEGRADED"
        self._on_demand_active = False
        self._on_demand_generation = 0
        self._active_on_demand_token: Optional[int] = None
        self._state_lock = threading.Lock()

    def is_on_demand_active(self) -> bool:
        """Kiểm tra xem hệ thống có đang trong tác vụ On-Demand (VQA/OCR) hay không."""
        with self._state_lock:
            return self._on_demand_active

    def set_on_demand_active(self, active: bool, mode: str = "OBSERVATION") -> None:
        """Thiết lập trạng thái On-Demand mà không dừng pipeline an toàn."""
        with self._state_lock:
            self._on_demand_generation += 1
            self._on_demand_active = active
            self._active_on_demand_token = self._on_demand_generation if active else None
            self._current_mode = mode if active else "OBSERVATION"

    def _is_current_on_demand(self, token: int) -> bool:
        with self._state_lock:
            return self._on_demand_active and self._active_on_demand_token == token

    def _post_on_demand_audio(self, token: int, task: AudioTask) -> bool:
        """Chỉ token hiện hành mới được tạo side effect âm thanh."""
        with self._state_lock:
            if not self._on_demand_active or self._active_on_demand_token != token:
                return False
            return self.audio_coordinator.post_task(task)

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
        """Chỉ owner của generation hiện hành mới được kết thúc mode on-demand."""
        with self._state_lock:
            if self._active_on_demand_token != token:
                return
            self._on_demand_active = False
            self._active_on_demand_token = None
            self._current_mode = "OBSERVATION"

    def _start_on_demand(
        self,
        mode: str,
        target,
        args: tuple = ()
    ) -> bool:
        """Dành token và khởi động đúng một worker inference on-demand."""
        with self._state_lock:
            if self._on_demand_thread is not None and self._on_demand_thread.is_alive():
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
                self._current_mode = mode
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
                    self._current_mode = "OBSERVATION"
            print(f"[SystemCoordinator] Không thể khởi động {mode}: {exc}")
            return False

    def start(self) -> None:
        """Khởi động toàn bộ các luồng của hệ thống."""
        if self._running:
            return
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
        self._running = False
        self.set_on_demand_active(False)
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

                # 5. Alert Aggregator luôn giữ đường cảnh báo HIGH_RISK hoạt động.
                self._post_safety_alert(assessments)

                e2e_ms = (time.monotonic() - t0) * 1000.0
                self.metrics.record_latency("end_to_end_ms", e2e_ms)

                with self._state_lock:
                    self._latest_packet = packet
                    self._latest_detection = tracked_det
                    self._latest_assessments = assessments
                    if not self._on_demand_active:
                        if sync_pair.data_quality in (DataQuality.STALE, DataQuality.DEGRADED):
                            self._current_mode = "DEGRADED"
                        else:
                            self._current_mode = "OBSERVATION"

            except Exception as e:
                print(f"[DetectionWorker] Lỗi suy luận và fusion: {e}")

    def _post_safety_alert(self, assessments: List[RiskAssessment]) -> None:
        """Trong on-demand chỉ cho HIGH_RISK đi qua để không ngắt bởi LOW/MEDIUM."""
        alert_candidates = assessments
        if self.is_on_demand_active():
            alert_candidates = [a for a in assessments if a.risk_level == RiskLevel.HIGH]

        if not alert_candidates:
            return

        alert_task = self.alert_aggregator.aggregate(alert_candidates)
        if alert_task is not None:
            self.audio_coordinator.post_task(alert_task)

    # ================= On-Demand OCR & VQA =================

    def trigger_ocr(self) -> bool:
        """Kích hoạt tác vụ đọc chữ theo yêu cầu."""
        return self._start_on_demand("OCR", self._run_ocr_task)

    def _run_ocr_task(self, token: int) -> None:
        try:
            if not self._is_current_on_demand(token):
                return
            self.audio_coordinator.interrupt()

            packet = self.get_latest_frame()
            if packet is None:
                if not self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Chưa nhận được khung hình từ camera để đọc.",
                    interruptible=True
                )):
                    return
                if self._is_current_on_demand(token):
                    self.audio_coordinator.wait_until_idle(timeout=4.0)
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
                self.audio_coordinator.wait_until_idle(timeout=45.0)
        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý OCR: {e}")
        finally:
            self._finish_on_demand(token)

    def trigger_vqa(self, question: str = "Phía trước có gì?") -> bool:
        """Kích hoạt tác vụ hỏi đáp thị giác (VQA)."""
        return self._start_on_demand("VQA", self._run_vqa_task, (question,))

    def _run_vqa_task(self, token: int, question: str) -> None:
        # Pipeline an toàn và HIGH_RISK vẫn hoạt động trong toàn bộ tác vụ.
        try:
            # 2. Ngắt cảnh báo cũ và làm sạch hàng đợi âm thanh
            if not self._is_current_on_demand(token):
                return
            self.audio_coordinator.interrupt()

            packet = self.get_latest_frame()
            if packet is None:
                if not self._post_on_demand_audio(token, AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Chưa có hình ảnh để trả lời.",
                    interruptible=True
                )):
                    return
                if self._is_current_on_demand(token):
                    self.audio_coordinator.wait_until_idle(timeout=4.0)
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
                detected_names = [a.class_name for a in self._latest_assessments]
                spatial_objects = [
                    {
                        "name": a.class_name,
                        "direction": a.direction.value,
                        "proximity": a.proximity_desc,
                        "risk_level": a.risk_level.value
                    }
                    for a in self._latest_assessments
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

            # 6. Đợi cả câu trả lời hoặc cảnh báo HIGH_RISK chen ngang phát xong.
            if self._is_current_on_demand(token):
                self.audio_coordinator.wait_until_idle(timeout=45.0)

        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý VQA: {e}")
        finally:
            # 7. Phục hồi trạng thái quan sát sau khi tác vụ on-demand hoàn tất.
            self._finish_on_demand(token)

    def stop_speech(self) -> None:
        """Ngắt tiếng khẩn cấp và kết thúc chế độ on-demand."""
        self.set_on_demand_active(False)
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
                "metrics": self.metrics.get_summary(),
                "is_speaking": self.audio_coordinator.is_speaking
            }
