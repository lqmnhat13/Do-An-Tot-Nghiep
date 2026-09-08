import unittest
import numpy as np
import time

from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.risk import DataQuality, RiskLevel, Direction
from src.detection.class_filter import ClassFilter
from src.depth.roi_extractor import ROIExtractor
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import Synchronizer
from src.fusion.risk_fsm import RiskFSM
from src.fusion.alert_aggregator import AlertAggregator

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.class_filter = ClassFilter()
        self.spatial = SpatialZones(left_ratio=0.35, center_ratio=0.30, right_ratio=0.35)
        self.roi = ROIExtractor()
        self.fsm = RiskFSM(
            class_filter=self.class_filter,
            spatial_zones=self.spatial,
            roi_extractor=self.roi,
            confirmations_to_escalate=2,
            confirmations_to_deescalate=3,
            instant_high_risk_in_center=True,
            cooldown_sec=2.0
        )
        self.aggregator = AlertAggregator(risk_fsm=self.fsm)

    def test_spatial_zones(self):
        # Frame width 640. Left < 224, Center 224..416, Right > 416
        b_left = BoundingBox(10, 100, 100, 200)
        b_center = BoundingBox(280, 100, 360, 200)
        b_right = BoundingBox(500, 100, 600, 200)

        self.assertEqual(self.spatial.determine_direction(b_left, 640), Direction.LEFT)
        self.assertEqual(self.spatial.determine_direction(b_center, 640), Direction.CENTER)
        self.assertEqual(self.spatial.determine_direction(b_right, 640), Direction.RIGHT)

    def test_synchronizer_stale_detection(self):
        sync = Synchronizer(max_detection_age_ms=300.0)
        t_old = time.monotonic() - 2.5 # 2.5s trước
        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=np.ones((10, 10), dtype=np.float32),
            valid_mask=np.ones((10, 10), dtype=bool),
            frame_id=1,
            capture_timestamp=t_old
        )
        sync.register_depth(dm)
        res = DetectionResult(detections=[], frame_id=2, timestamp_mono=t_old, latency_ms=10.0)
        pair = sync.synchronize(res, current_mono=time.monotonic())
        self.assertEqual(pair.data_quality, DataQuality.STALE)

    def test_synchronizer_frame_id_match(self):
        sync = Synchronizer()
        now = time.monotonic()
        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=np.ones((10, 10), dtype=np.float32),
            valid_mask=np.ones((10, 10), dtype=bool),
            frame_id=100,
            capture_timestamp=now
        )
        sync.register_depth(dm)

        res = DetectionResult(detections=[], frame_id=100, timestamp_mono=now, latency_ms=10.0)
        pair = sync.synchronize(res, current_mono=now)
        self.assertEqual(pair.data_quality, DataQuality.VALID)
        self.assertIsNotNone(pair.depth_map)
        self.assertEqual(pair.depth_map.frame_id, 100)

    def test_risk_instant_escalation(self):
        # Vật cản rất gần ở chính giữa (instant route)
        now = time.monotonic()
        vals = np.ones((480, 640), dtype=np.float32) * 1.0 # nền ở xa
        vals[100:300, 280:360] = 10.0 # vật thể rất gần ở giữa
        mask = np.ones((480, 640), dtype=bool)
        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals,
            valid_mask=mask,
            frame_id=1,
            capture_timestamp=now
        )
        det = Detection(class_name="chair", confidence=0.9, bbox=BoundingBox(280, 100, 360, 300), track_id=1)
        res = DetectionResult(detections=[det], frame_id=1, timestamp_mono=now, latency_ms=10.0)

        sync = Synchronizer()
        sync.register_depth(dm)
        pair = sync.synchronize(res, current_mono=now)

        assessments = self.fsm.update(pair, current_mono=now)
        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].risk_level, RiskLevel.HIGH)

        # Test Aggregator generates Vietnamese alert
        alert = self.aggregator.aggregate(assessments, current_mono=now)
        self.assertIsNotNone(alert)
        self.assertIn("ghế", alert.text)
        self.assertIn("phía trước", alert.text)

if __name__ == "__main__":
    unittest.main()
