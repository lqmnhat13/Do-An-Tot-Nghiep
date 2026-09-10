import unittest
import threading
import time
from unittest.mock import MagicMock, patch
import numpy as np

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
from src.audio.tts_engine import TTSEngine
from src.audio.audio_coordinator import AudioCoordinator
from src.contracts.audio import AudioTask, AudioPriority
from src.contracts.frame_packet import FramePacket
from src.contracts.risk import RiskAssessment, RiskLevel, Direction, DataQuality
from src.runtime.system_coordinator import SystemCoordinator

class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_dummy(self):
        # Thiết lập camera dummy
        camera = CameraManager(source="dummy", width=320, height=240, target_fps=20.0)
        class_filter = ClassFilter()
        detector = YoloDetector(device="cpu") # Dùng CPU trong unit test để tránh tranh chấp MPS
        tracker = ByteTrackerAdapter()
        depth_estimator = DepthEstimator(device="cpu")
        roi_extractor = ROIExtractor()
        spatial_zones = SpatialZones()
        synchronizer = Synchronizer()
        risk_fsm = RiskFSM(class_filter, spatial_zones, roi_extractor, cooldown_sec=1.0)
        alert_aggregator = AlertAggregator(risk_fsm)

        # Mock TTS để test trong môi trường test runner không phát loa thật
        mock_tts = MagicMock(spec=TTSEngine)
        mock_tts.is_speaking.return_value = False
        audio_coordinator = AudioCoordinator(tts_engine=mock_tts)

        coordinator = SystemCoordinator(
            camera_manager=camera,
            detector=detector,
            tracker=tracker,
            depth_estimator=depth_estimator,
            synchronizer=synchronizer,
            risk_fsm=risk_fsm,
            alert_aggregator=alert_aggregator,
            audio_coordinator=audio_coordinator
        )

        coordinator.start()
        time.sleep(1.0) # Cho pipeline chạy 1 giây

        snapshot = coordinator.get_runtime_snapshot()
        self.assertIsNotNone(snapshot["packet"])
        self.assertGreater(snapshot["packet"].frame_id, 0)

        # Test trigger on-demand VQA
        coordinator.trigger_vqa("Phía trước có gì?")
        time.sleep(0.2)

        coordinator.stop()
        self.assertFalse(coordinator.camera_manager.is_alive())

    def test_vqa_high_risk_preempts_audio_without_cancelling_vqa(self):
        """HIGH_RISK ngắt tiếng VQA; MEDIUM không ngắt và VQA vẫn giữ trạng thái."""
        camera = MagicMock()
        camera.get_latest_frame.return_value = FramePacket(
            frame_id=1,
            source_id="test",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            original_size=(8, 8),
            timestamp_mono=time.monotonic()
        )
        risk_fsm = MagicMock()
        risk_fsm.can_trigger_alert.return_value = True
        alert_aggregator = AlertAggregator(risk_fsm)

        mock_tts = MagicMock(spec=TTSEngine)
        mock_tts.is_speaking.return_value = False
        audio_coordinator = AudioCoordinator(tts_engine=mock_tts)

        mock_vqa = MagicMock()
        def slow_answer(*args, **kwargs):
            time.sleep(0.3)
            return MagicMock(
                request_id="vqa_test",
                success=True,
                answer="Khung cảnh phía trước có bàn ghế.",
                latency_sec=0.3
            )
        mock_vqa.answer.side_effect = slow_answer

        coordinator = SystemCoordinator(
            camera_manager=camera,
            detector=MagicMock(),
            tracker=MagicMock(),
            depth_estimator=MagicMock(),
            synchronizer=MagicMock(),
            risk_fsm=risk_fsm,
            alert_aggregator=alert_aggregator,
            audio_coordinator=audio_coordinator,
            vqa_service=mock_vqa
        )

        coordinator.start()
        time.sleep(0.5)

        self.assertFalse(coordinator.is_on_demand_active())
        self.assertEqual(coordinator.get_runtime_snapshot()["mode"], "OBSERVATION")

        # Kích hoạt VQA
        coordinator.trigger_vqa("Phía trước có gì?")
        time.sleep(0.1)

        # Trong khi VQA đang chạy, on_demand_active phải là True và mode là "VQA"
        self.assertTrue(coordinator.is_on_demand_active())
        self.assertEqual(coordinator.get_runtime_snapshot()["mode"], "VQA")

        now = time.monotonic()
        medium_risk = RiskAssessment(
            track_id=1,
            class_name="người",
            direction=Direction.CENTER,
            risk_level=RiskLevel.MEDIUM,
            data_quality=DataQuality.VALID,
            relative_proximity=0.4,
            proximity_desc="gần",
            reason="test",
            source_timestamp=now,
            expires_at=now + 1.0
        )
        high_risk = RiskAssessment(
            track_id=2,
            class_name="xe",
            direction=Direction.CENTER,
            risk_level=RiskLevel.HIGH,
            data_quality=DataQuality.VALID,
            relative_proximity=0.9,
            proximity_desc="rất gần",
            reason="test",
            source_timestamp=now,
            expires_at=now + 1.0
        )

        audio_coordinator._current_task = AudioTask(
            priority=AudioPriority.ON_DEMAND,
            text="Đang trả lời VQA",
            interruptible=True
        )
        mock_tts.is_speaking.return_value = True
        mock_tts.stop.reset_mock()

        coordinator._post_safety_alert([medium_risk])
        mock_tts.stop.assert_not_called()

        coordinator._post_safety_alert([medium_risk, high_risk])
        mock_tts.stop.assert_called_once()
        self.assertEqual(audio_coordinator.interrupted_count, 1)
        self.assertTrue(coordinator.is_on_demand_active())
        self.assertEqual(coordinator.get_runtime_snapshot()["mode"], "VQA")

        mock_tts.is_speaking.return_value = False

        # Đợi VQA chạy và hàng đợi âm thanh hoàn tất.
        time.sleep(0.8)

        # Sau khi hoàn thành và đọc xong, on_demand_active phải tự động chuyển về False
        self.assertFalse(coordinator.is_on_demand_active())

        # Kiểm tra phím dừng khẩn cấp stop_speech()
        coordinator.set_on_demand_active(True, mode="VQA")
        self.assertTrue(coordinator.is_on_demand_active())
        coordinator.stop_speech()
        self.assertFalse(coordinator.is_on_demand_active())

        coordinator.stop()

    def test_shutdown_waits_for_on_demand_worker_without_deadlock(self):
        """Shutdown hủy trạng thái VQA và chờ worker thoát mà không đăng kết quả muộn."""
        camera = MagicMock()
        packet = FramePacket(
            frame_id=1,
            source_id="test",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            original_size=(8, 8),
            timestamp_mono=time.monotonic()
        )
        camera.get_latest_frame.return_value = packet

        entered = threading.Event()
        release = threading.Event()
        mock_vqa = MagicMock()

        def blocking_answer(*args, **kwargs):
            entered.set()
            release.wait(timeout=1.0)
            return MagicMock(answer="Kết quả không được phát sau shutdown", latency_sec=0.1)

        mock_vqa.answer.side_effect = blocking_answer
        mock_tts = MagicMock(spec=TTSEngine)
        mock_tts.is_speaking.return_value = False
        audio_coordinator = AudioCoordinator(tts_engine=mock_tts)
        coordinator = SystemCoordinator(
            camera_manager=camera,
            detector=MagicMock(),
            tracker=MagicMock(),
            depth_estimator=MagicMock(),
            synchronizer=MagicMock(),
            risk_fsm=MagicMock(),
            alert_aggregator=MagicMock(),
            audio_coordinator=audio_coordinator,
            vqa_service=mock_vqa
        )

        coordinator.trigger_vqa("Phía trước có gì?")
        self.assertTrue(entered.wait(timeout=0.5))

        stop_thread = threading.Thread(target=coordinator.stop)
        stop_thread.start()
        deadline = time.monotonic() + 0.5
        while coordinator.is_on_demand_active() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(coordinator.is_on_demand_active())

        release.set()
        stop_thread.join(timeout=1.0)

        self.assertFalse(stop_thread.is_alive())
        self.assertIsNotNone(coordinator._on_demand_thread)
        self.assertFalse(coordinator._on_demand_thread.is_alive())
        self.assertTrue(audio_coordinator._queue.empty())

    def test_cancelled_generation_cannot_publish_or_clear_next_vqa(self):
        """A bị hủy không được publish; B bị từ chối đến khi A thoát rồi nhận token mới."""
        camera = MagicMock()
        camera.get_latest_frame.return_value = FramePacket(
            frame_id=7,
            source_id="test",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            original_size=(8, 8),
            timestamp_mono=time.monotonic()
        )

        a_entered = threading.Event()
        a_release = threading.Event()
        b_entered = threading.Event()
        b_release = threading.Event()
        mock_vqa = MagicMock()

        def controlled_answer(request, **kwargs):
            if request.question == "A":
                a_entered.set()
                a_release.wait(timeout=1.0)
                return MagicMock(answer="Kết quả cũ A", latency_sec=0.11)
            b_entered.set()
            b_release.wait(timeout=1.0)
            return MagicMock(answer="Kết quả mới B", latency_sec=0.22)

        mock_vqa.answer.side_effect = controlled_answer
        audio_coordinator = MagicMock(spec=AudioCoordinator)
        audio_coordinator.post_task.return_value = True
        audio_coordinator.wait_until_idle.return_value = True
        coordinator = SystemCoordinator(
            camera_manager=camera,
            detector=MagicMock(),
            tracker=MagicMock(),
            depth_estimator=MagicMock(),
            synchronizer=MagicMock(),
            risk_fsm=MagicMock(),
            alert_aggregator=MagicMock(),
            audio_coordinator=audio_coordinator,
            vqa_service=mock_vqa
        )

        self.assertTrue(coordinator.trigger_vqa("A"))
        self.assertTrue(a_entered.wait(timeout=0.5))
        token_a = coordinator._active_on_demand_token
        thread_a = coordinator._on_demand_thread

        coordinator.stop_speech()
        self.assertFalse(coordinator.trigger_vqa("B"))
        self.assertEqual(coordinator.get_runtime_snapshot()["mode"], "OBSERVATION")

        a_release.set()
        thread_a.join(timeout=0.5)
        self.assertFalse(thread_a.is_alive())
        spoken_texts = [call.args[0].text for call in audio_coordinator.post_task.call_args_list]
        self.assertNotIn("Kết quả cũ A", spoken_texts)
        self.assertEqual(coordinator.metrics.get_percentiles("vqa_sec")["mean"], 0.0)

        self.assertTrue(coordinator.trigger_vqa("B"))
        self.assertTrue(b_entered.wait(timeout=0.5))
        self.assertNotEqual(token_a, coordinator._active_on_demand_token)
        self.assertEqual(coordinator.get_runtime_snapshot()["mode"], "VQA")

        # Một finalizer muộn của A không được quyền xóa state thuộc generation B.
        coordinator._finish_on_demand(token_a)
        self.assertTrue(coordinator.is_on_demand_active())
        self.assertEqual(coordinator.get_runtime_snapshot()["mode"], "VQA")

        b_release.set()
        coordinator._on_demand_thread.join(timeout=0.5)
        self.assertFalse(coordinator._on_demand_thread.is_alive())
        self.assertFalse(coordinator.is_on_demand_active())
        spoken_texts = [call.args[0].text for call in audio_coordinator.post_task.call_args_list]
        self.assertIn("Kết quả mới B", spoken_texts)
        self.assertAlmostEqual(
            coordinator.metrics.get_percentiles("vqa_sec")["mean"], 0.22
        )

    def test_thread_start_failure_rolls_back_on_demand_state(self):
        camera = MagicMock()
        coordinator = SystemCoordinator(
            camera_manager=camera,
            detector=MagicMock(),
            tracker=MagicMock(),
            depth_estimator=MagicMock(),
            synchronizer=MagicMock(),
            risk_fsm=MagicMock(),
            alert_aggregator=MagicMock(),
            audio_coordinator=MagicMock(spec=AudioCoordinator),
            vqa_service=MagicMock()
        )

        with patch.object(threading.Thread, "start", side_effect=RuntimeError("start failed")):
            self.assertFalse(coordinator.trigger_vqa("B"))

        self.assertFalse(coordinator.is_on_demand_active())
        self.assertIsNone(coordinator._active_on_demand_token)
        self.assertIsNone(coordinator._on_demand_thread)
        self.assertEqual(coordinator.get_runtime_snapshot()["mode"], "OBSERVATION")

if __name__ == "__main__":
    unittest.main()
