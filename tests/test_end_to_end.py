import unittest
import time
from unittest.mock import MagicMock
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

    def test_vqa_alert_gating_logic(self):
        """Kiểm tra logic: khi VQA kích hoạt, cảnh báo va chạm bị tạm dừng cho đến khi VQA hoàn thành."""
        camera = CameraManager(source="dummy", width=320, height=240, target_fps=20.0)
        class_filter = ClassFilter()
        detector = YoloDetector(device="cpu")
        tracker = ByteTrackerAdapter()
        depth_estimator = DepthEstimator(device="cpu")
        roi_extractor = ROIExtractor()
        spatial_zones = SpatialZones()
        synchronizer = Synchronizer()
        risk_fsm = RiskFSM(class_filter, spatial_zones, roi_extractor, cooldown_sec=1.0)
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
            detector=detector,
            tracker=tracker,
            depth_estimator=depth_estimator,
            synchronizer=synchronizer,
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

        # Đợi VQA chạy xong (mock VQA chạy tức thì)
        time.sleep(0.8)

        # Sau khi hoàn thành và đọc xong, on_demand_active phải tự động chuyển về False
        self.assertFalse(coordinator.is_on_demand_active())

        # Kiểm tra phím dừng khẩn cấp stop_speech()
        coordinator.set_on_demand_active(True, mode="VQA")
        self.assertTrue(coordinator.is_on_demand_active())
        coordinator.stop_speech()
        self.assertFalse(coordinator.is_on_demand_active())

        coordinator.stop()

if __name__ == "__main__":
    unittest.main()
