import threading
import time
from typing import Optional, Dict, Any, List
import numpy as np

from src.contracts.frame_packet import FramePacket
from src.contracts.detection import DetectionResult
from src.contracts.depth_map import DepthMap
from src.contracts.risk import RiskAssessment, DataQuality
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

        # Lưu trữ trạng thái mới nhất phục vụ hiển thị HUD
        self._latest_packet: Optional[FramePacket] = None
        self._latest_detection: Optional[DetectionResult] = None
        self._latest_depth: Optional[DepthMap] = None
        self._latest_assessments: List[RiskAssessment] = []
        self._current_mode = "OBSERVATION" # "OBSERVATION" | "OCR" | "VQA" | "DEGRADED"
        self._on_demand_active = False
        self._state_lock = threading.Lock()

    def is_on_demand_active(self) -> bool:
        """Kiểm tra xem hệ thống có đang trong tác vụ On-Demand (VQA/OCR) hay không."""
        with self._state_lock:
            return self._on_demand_active

    def set_on_demand_active(self, active: bool, mode: str = "OBSERVATION") -> None:
        """Thiết lập trạng thái On-Demand (bật/tắt chế độ tạm dừng cảnh báo)."""
        with self._state_lock:
            self._on_demand_active = active
            self._current_mode = mode if active else "OBSERVATION"

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
        self.camera_manager.stop()
        if self._det_thread and self._det_thread.is_alive():
            self._det_thread.join(timeout=1.5)
        if self._depth_thread and self._depth_thread.is_alive():
            self._depth_thread.join(timeout=1.5)
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

                # 5. Alert Aggregator tổng hợp và phát cảnh báo
                # TẠM DỪNG CẢNH BÁO LIÊN TỤC KHI ĐANG Ở CHẾ ĐỘ ON-DEMAND (VQA / OCR)
                if not self.is_on_demand_active():
                    alert_task = self.alert_aggregator.aggregate(assessments)
                    if alert_task is not None:
                        self.audio_coordinator.post_task(alert_task)

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

    # ================= On-Demand OCR & VQA =================

    def trigger_ocr(self) -> None:
        """Kích hoạt tác vụ đọc chữ theo yêu cầu."""
        if self.is_on_demand_active():
            print("[SystemCoordinator] Tác vụ theo yêu cầu đang thực thi, bỏ qua yêu cầu OCR.")
            return
        threading.Thread(target=self._run_ocr_task, name="OCRTaskThread", daemon=True).start()

    def _run_ocr_task(self) -> None:
        self.set_on_demand_active(True, mode="OCR")
        try:
            self.audio_coordinator.interrupt()

            packet = self.get_latest_frame()
            if packet is None:
                self.audio_coordinator.post_task(AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Chưa nhận được khung hình từ camera để đọc.",
                    interruptible=False
                ))
                self.audio_coordinator.wait_until_idle(timeout=4.0)
                return

            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Đang xử lý đọc chữ...",
                sound_file="assets/audio/chime.wav",
                interruptible=False
            ))

            req = OCRRequest(request_id=f"ocr_{packet.frame_id}", image=packet.image)
            res = self.ocr_service.process(req)

            if not self.is_on_demand_active():
                return

            self.metrics.record_latency("ocr_sec", res.latency_sec)
            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text=res.text,
                interruptible=False
            ))

            self.audio_coordinator.wait_until_idle(timeout=45.0)
        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý OCR: {e}")
        finally:
            self.set_on_demand_active(False)

    def trigger_vqa(self, question: str = "Phía trước có gì?") -> None:
        """Kích hoạt tác vụ hỏi đáp thị giác (VQA)."""
        if self.is_on_demand_active():
            print("[SystemCoordinator] Tác vụ theo yêu cầu đang thực thi, bỏ qua yêu cầu VQA.")
            return
        threading.Thread(target=self._run_vqa_task, args=(question,), name="VQATaskThread", daemon=True).start()

    def _run_vqa_task(self, question: str) -> None:
        # 1. Bật chế độ On-Demand: Tạm dừng cảnh báo liên tục
        self.set_on_demand_active(True, mode="VQA")
        try:
            # 2. Ngắt cảnh báo cũ và làm sạch hàng đợi âm thanh
            self.audio_coordinator.interrupt()

            packet = self.get_latest_frame()
            if packet is None:
                self.audio_coordinator.post_task(AudioTask(
                    priority=AudioPriority.ON_DEMAND,
                    text="Chưa có hình ảnh để trả lời.",
                    interruptible=False
                ))
                self.audio_coordinator.wait_until_idle(timeout=4.0)
                return

            # 3. Thông báo đang quan sát
            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Đang quan sát để trả lời...",
                sound_file="assets/audio/chime.wav",
                interruptible=False
            ))

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
            req = VQARequest(request_id=f"vqa_{packet.frame_id}", image=packet.image, question=question)
            res = self.vqa_service.answer(req, visual_context={
                "detected_classes": detected_names,
                "spatial_objects": spatial_objects
            })

            # Kiểm tra nếu người dùng đã hủy (bấm phím S)
            if not self.is_on_demand_active():
                return

            self.metrics.record_latency("vqa_sec", res.latency_sec)

            # 5. Phát âm toàn văn câu trả lời VQA
            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text=res.answer,
                interruptible=False
            ))

            # 6. ĐỢI ĐỌC XONG VQA HOÀN TOÀN MỚI CHO PHÉP CẢNH BÁO TIẾP TỤC
            self.audio_coordinator.wait_until_idle(timeout=45.0)

        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý VQA: {e}")
        finally:
            # 7. Phục hồi trạng thái bình thường để tiếp tục cảnh báo va chạm
            self.set_on_demand_active(False)

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
