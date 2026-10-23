import time
import unittest

import numpy as np

from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.risk import DataQuality, RiskLevel
from src.depth.roi_extractor import ROIExtractor
from src.detection.class_filter import ClassFilter
from src.fusion.alert_aggregator import AlertAggregator
from src.fusion.generic_obstacle import GenericObstacleDetector, build_generic_obstacle_detector
from src.fusion.risk_fsm import GENERIC_OBSTACLE_TRACK_ID, RiskFSM
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import SynchronizedPair
from src.fusion.walking_corridor import WalkingCorridor

W, H = 160, 120


def depth(values, representation=DepthRepresentation.METRIC_DEPTH, frame_id=1):
    metric = representation == DepthRepresentation.METRIC_DEPTH
    return DepthMap(representation=representation, unit="meter" if metric else "relative",
                    near_is_larger=not metric, values=values.astype(np.float32),
                    valid_mask=np.ones(values.shape, bool), frame_id=frame_id,
                    capture_timestamp=time.monotonic())


def wall_at(distance):
    values = np.full((H, W), 5.0)
    values[40:80, :] = distance  # mặt phẳng chắn giữa khung hình
    return values


class TestGenericObstacleDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GenericObstacleDetector(WalkingCorridor())

    def test_near_wall_in_corridor_is_found(self):
        found = self.detector.find(depth(wall_at(0.7)), [])
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found.distance_m, 0.7, places=3)

    def test_far_wall_is_ignored(self):
        self.assertIsNone(self.detector.find(depth(wall_at(1.4)), []))

    def test_floor_band_at_bottom_is_ignored(self):
        values = np.full((H, W), 5.0)
        values[int(H * 0.8):, :] = 0.4  # sàn sát chân, nằm trong dải đáy bị bỏ qua
        self.assertIsNone(self.detector.find(depth(values), []))

    def test_relative_depth_is_never_used(self):
        self.assertIsNone(self.detector.find(
            depth(1.0 / wall_at(0.7), DepthRepresentation.RELATIVE_DEPTH), []))

    def test_region_covered_by_detection_is_not_duplicated(self):
        self.assertIsNone(self.detector.find(depth(wall_at(0.7)), [BoundingBox(0, 30, W, 90)]))

    def test_disabled_by_config(self):
        self.assertIsNone(build_generic_obstacle_detector({"generic_obstacle": {"enabled": False}},
                                                          WalkingCorridor()))
        self.assertIsNotNone(build_generic_obstacle_detector({"generic_obstacle": {"enabled": True}},
                                                             WalkingCorridor()))


class TestGenericObstacleInRiskFSM(unittest.TestCase):
    def make_fsm(self):
        return RiskFSM(class_filter=ClassFilter(), spatial_zones=SpatialZones(),
                       roi_extractor=ROIExtractor(),
                       generic_obstacle_detector=GenericObstacleDetector(WalkingCorridor()))

    def pair(self, values, quality=DataQuality.VALID, detections=()):
        now = time.monotonic()
        det = DetectionResult(detections=list(detections), frame_id=1, timestamp_mono=now, latency_ms=1.0,
                              preprocess_transform={"original_size": (W, H)})
        return SynchronizedPair(detection_result=det, depth_map=depth(values), data_quality=quality,
                                pair_skew_ms=0.0, effective_timestamp=now)

    def test_generic_high_is_emitted_and_spoken_once_per_cooldown(self):
        fsm = self.make_fsm()
        aggregator = AlertAggregator(risk_fsm=fsm, global_alert_interval_sec=0.0)
        now = time.monotonic()
        assessments = fsm.update(self.pair(wall_at(0.7)), current_mono=now)
        generic = [a for a in assessments if a.track_id == GENERIC_OBSTACLE_TRACK_ID]
        self.assertEqual(len(generic), 1)
        self.assertEqual(generic[0].risk_level, RiskLevel.HIGH)
        task = aggregator.aggregate(assessments, current_mono=now)
        self.assertIsNotNone(task)
        self.assertIn("vật cản", task.text)
        again = fsm.update(self.pair(wall_at(0.7)), current_mono=now + 0.5)
        self.assertIsNone(aggregator.aggregate(again, current_mono=now + 0.5))

    def test_no_generic_alert_on_stale_data(self):
        fsm = self.make_fsm()
        assessments = fsm.update(self.pair(wall_at(0.7), quality=DataQuality.STALE))
        self.assertFalse(any(a.track_id == GENERIC_OBSTACLE_TRACK_ID for a in assessments))

    def test_detected_object_is_assessed_by_class_not_generic(self):
        fsm = self.make_fsm()
        chair = Detection(class_name="chair", confidence=0.9, bbox=BoundingBox(0, 30, W, 90), track_id=1)
        assessments = fsm.update(self.pair(wall_at(0.7), detections=[chair]))
        self.assertEqual([a.track_id for a in assessments], [1])
        self.assertEqual(assessments[0].risk_level, RiskLevel.HIGH)


if __name__ == "__main__":
    unittest.main()
