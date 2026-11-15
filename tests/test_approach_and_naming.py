import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.risk import DataQuality, RiskLevel
from src.depth.roi_extractor import ROIExtractor
from src.detection.class_filter import ClassFilter
from src.fusion.risk_fsm import GENERIC_OBSTACLE_NAME, RiskFSM
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import SynchronizedPair

W, H = 640, 480


def make_pair(t, bbox, distance_m=2.0, quality=DataQuality.VALID, confidence=0.9, frame_id=1):
    """Ghế giữa hành lang; depth metric đồng nhất (2.0 m -> MEDIUM)."""
    depth = DepthMap(representation=DepthRepresentation.METRIC_DEPTH, unit="meter",
                     near_is_larger=False, values=np.full((H, W), distance_m, np.float32),
                     valid_mask=np.ones((H, W), bool), frame_id=frame_id, capture_timestamp=t)
    det = Detection(class_name="chair", confidence=confidence, bbox=bbox, track_id=1)
    result = DetectionResult(detections=[det], frame_id=frame_id, timestamp_mono=t, latency_ms=1.0,
                             preprocess_transform={"original_size": (W, H)})
    return SynchronizedPair(detection_result=result, depth_map=depth, data_quality=quality,
                            pair_skew_ms=0.0, effective_timestamp=t)


def box(height):
    return BoundingBox(270, 300 - height / 2, 370, 300 + height / 2)


class TestApproachEscalation(unittest.TestCase):
    def make_fsm(self, ttc=1.5):
        return RiskFSM(class_filter=ClassFilter(), spatial_zones=SpatialZones(),
                       roi_extractor=ROIExtractor(), approach_ttc_sec=ttc)

    def run_heights(self, fsm, heights, dt=0.1, **kwargs):
        out = None
        for i, h in enumerate(heights):
            out = fsm.update(make_pair(100.0 + i * dt, box(h), frame_id=i + 1, **kwargs),
                             current_mono=100.0 + i * dt)
        return out[0]

    def test_fast_growing_bbox_escalates_medium_to_high(self):
        # Chiều cao x2 trong 1 s -> TTC = 1 / (2 - 1) = 1 s < 1.5 s
        a = self.run_heights(self.make_fsm(), np.linspace(100, 200, 11))
        self.assertEqual(a.risk_level, RiskLevel.HIGH)
        self.assertIn("tiến lại gần", a.reason)

    def test_static_or_jittering_bbox_stays_medium(self):
        a = self.run_heights(self.make_fsm(), [100, 104, 98, 103, 99, 102, 100, 105, 101, 103, 104])
        self.assertEqual(a.risk_level, RiskLevel.MEDIUM)

    def test_slow_approach_stays_medium(self):
        # +20% trong 1 s -> TTC = 5 s
        a = self.run_heights(self.make_fsm(), np.linspace(100, 120, 11))
        self.assertEqual(a.risk_level, RiskLevel.MEDIUM)

    def test_disabled_by_default(self):
        a = self.run_heights(self.make_fsm(ttc=0.0), np.linspace(100, 200, 11))
        self.assertEqual(a.risk_level, RiskLevel.MEDIUM)

    def test_too_short_history_does_not_escalate(self):
        a = self.run_heights(self.make_fsm(), [100, 150, 200])  # chỉ 0.2 s
        self.assertEqual(a.risk_level, RiskLevel.MEDIUM)

    def test_far_object_is_not_escalated(self):
        a = self.run_heights(self.make_fsm(), np.linspace(100, 200, 11), distance_m=4.0)
        self.assertEqual(a.risk_level, RiskLevel.NO_ALERT)

    def test_stale_depth_never_reports_high_or_safe(self):
        a = self.run_heights(self.make_fsm(), np.linspace(100, 200, 11), quality=DataQuality.STALE)
        self.assertEqual(a.risk_level, RiskLevel.UNDETERMINED)

    def test_recovers_after_stale_period(self):
        fsm = self.make_fsm()
        for i, h in enumerate(np.linspace(100, 150, 6)):
            fsm.update(make_pair(100.0 + i * 0.1, box(h), frame_id=i + 1, quality=DataQuality.STALE),
                       current_mono=100.0 + i * 0.1)
        a = None
        for i, h in enumerate(np.linspace(160, 220, 6), start=6):
            a = fsm.update(make_pair(100.0 + i * 0.1, box(h), frame_id=i + 1), current_mono=100.0 + i * 0.1)[0]
        self.assertEqual(a.risk_level, RiskLevel.HIGH)


class TestLowConfidenceNaming(unittest.TestCase):
    def name_for(self, confidence, threshold=0.45):
        fsm = RiskFSM(class_filter=ClassFilter(), spatial_zones=SpatialZones(),
                      roi_extractor=ROIExtractor(), named_min_confidence=threshold)
        return fsm.update(make_pair(1.0, box(150), distance_m=1.0, confidence=confidence),
                          current_mono=1.0)[0]

    def test_confident_detection_keeps_class_name(self):
        self.assertEqual(self.name_for(0.8).class_name, "ghế")

    def test_low_confidence_is_called_generic_obstacle_but_still_high(self):
        a = self.name_for(0.3)
        self.assertEqual(a.class_name, GENERIC_OBSTACLE_NAME)
        self.assertEqual(a.risk_level, RiskLevel.HIGH)

    def test_disabled_threshold_keeps_name(self):
        self.assertEqual(self.name_for(0.3, threshold=0.0).class_name, "ghế")


class TestYoloWeightsFallback(unittest.TestCase):
    def test_missing_weights_fall_back_to_yolov8n(self):
        from src.detection.yolo_detector import YoloDetector

        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "yolov8n.pt"), "wb").close()
            with patch("ultralytics.YOLO") as yolo:
                detector = YoloDetector(weights_path=os.path.join(tmp, "yolov8s.pt"), device="cpu")
        self.assertEqual(detector.weights_path, os.path.join(tmp, "yolov8n.pt"))
        yolo.assert_called_once_with(os.path.join(tmp, "yolov8n.pt"))

    def test_missing_without_fallback_raises(self):
        from src.detection.yolo_detector import YoloDetector

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                YoloDetector(weights_path=os.path.join(tmp, "yolov8s.pt"), device="cpu")


if __name__ == "__main__":
    unittest.main()
