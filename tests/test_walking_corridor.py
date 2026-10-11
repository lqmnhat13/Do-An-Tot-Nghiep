import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.risk import DataQuality, Direction, RiskLevel
from src.depth.roi_extractor import ROIExtractor
from src.detection.class_filter import ClassFilter
from src.fusion.alert_aggregator import AlertAggregator
from src.fusion.risk_fsm import RiskFSM
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import SynchronizedPair
from src.fusion.walking_corridor import WalkingCorridor
from src.ui.hud_renderer import HUDRenderer


class TestWalkingCorridor(unittest.TestCase):
    def setUp(self):
        self.corridor = WalkingCorridor()

    def test_exact_geometry_and_overlap(self):
        self.assertEqual(self.corridor.polygon(100, 100),
                         ((40, 35), (60, 35), (80, 100), (20, 100)))
        for box in (BoundingBox(49, 60, 51, 65), BoundingBox(0, 0, 100, 100),
                    BoundingBox(-100, -100, 200, 200)):
            self.assertAlmostEqual(self.corridor.overlap_ratio(box, 100, 100), 1)
        # Wide box whose center is left of the old middle zone still crosses the corridor.
        box = BoundingBox(0, 60, 55, 95)
        self.assertLess(box.center_x, 35)
        self.assertGreater(self.corridor.overlap_ratio(box, 100, 100), .10)
        self.assertEqual(self.corridor.overlap_ratio(BoundingBox(0, 35, 20, 100), 100, 100), 0)
        self.assertEqual(self.corridor.overlap_ratio(BoundingBox(40, 10, 60, 35), 100, 100), 0)
        self.assertLess(self.corridor.overlap_ratio(BoundingBox(0, 35, 22, 100), 100, 100), .10)

    def test_known_partial_area_and_resolution_invariance(self):
        # Bottom half of the corridor: trapezoid area (40+60)/2 * 32.5 = 1625.
        box = BoundingBox(0, 67.5, 100, 100)
        self.assertAlmostEqual(self.corridor.overlap_ratio(box, 100, 100), 1625 / 2600)
        for w, h in ((640, 480), (1280, 720), (480, 640)):
            scaled = BoundingBox(0, .675 * h, w, h)
            self.assertAlmostEqual(self.corridor.overlap_ratio(scaled, w, h), 1625 / 2600)

    def test_overlap_threshold_boundary(self):
        corridor = WalkingCorridor(top_y=0, top_width=.4, bottom_width=.4)
        ratio = corridor.overlap_ratio(BoundingBox(21, 40, 31, 50), 100, 100)
        self.assertEqual(ratio, corridor.min_overlap_ratio)

    def test_invalid_geometry(self):
        for box in (BoundingBox(1, 1, 1, 2), BoundingBox(10, 10, 5, 5),
                    BoundingBox(math.nan, 1, 5, 5), BoundingBox(1, 1, math.inf, 5),
                    BoundingBox(101, 101, 110, 110)):
            self.assertIsNone(self.corridor.overlap_ratio(box, 100, 100))
        for size in ((0, 100), (-1, 100), (100, math.nan)):
            self.assertIsNone(self.corridor.overlap_ratio(BoundingBox(40, 40, 60, 60), *size))
        for config in ({'top_y': 1}, {'top_width': 0}, {'center_x': .1},
                       {'min_overlap_ratio': 0}, {'bottom_width': math.nan}):
            with self.assertRaises(ValueError):
                WalkingCorridor(**config)

    def test_hud_uses_shared_geometry_with_letterbox(self):
        renderer = HUDRenderer(walking_corridor=self.corridor)
        self.assertIs(renderer.walking_corridor, self.corridor)
        for w, h, px, py in ((800, 450, 0, 75), (450, 600, 175, 0)):
            canvas = np.zeros((600, 800, 3), dtype=np.uint8)
            with patch('src.ui.hud_renderer.cv2.polylines') as draw:
                renderer._draw_spatial_guides(canvas, px, py, w, h)
            points = draw.call_args.args[1][0]
            expected = np.array([(round(x + px), round(y + py))
                                 for x, y in self.corridor.polygon(w, h)])
            expected[:, 0] = np.clip(expected[:, 0], px, px + w - 1)
            expected[:, 1] = np.clip(expected[:, 1], py, py + h - 1)
            np.testing.assert_array_equal(points, expected)


class TestCorridorRisk(unittest.TestCase):
    inside = BoundingBox(280, 200, 360, 400)
    outside = BoundingBox(10, 200, 80, 400)

    def setUp(self):
        self.roi = Mock(spec=ROIExtractor)
        self.roi.extract_proximity.return_value = (.8, 'độ gần tương đối cao', 'test')
        self.fsm = RiskFSM(ClassFilter(), SpatialZones(), self.roi)
        self.aggregator = AlertAggregator(self.fsm, global_alert_interval_sec=3.5)

    def assess(self, box=None, frame=1, quality=DataQuality.VALID, size=(640, 480), depth=True, class_name="chair"):
        now = 100 + frame * .1
        detection = Detection(class_name, .9, box or self.inside, track_id=0)
        result = DetectionResult([detection], frame, now, 0, {'original_size': size})
        dm = DepthMap(DepthRepresentation.RELATIVE_DEPTH, 'relative', True,
                      np.ones((480, 640), dtype=np.float32),
                      np.ones((480, 640), dtype=bool), frame, now) if depth else None
        return self.fsm.update(SynchronizedPair(result, dm, quality, 0, now), now)[0]

    def test_near_outside_does_not_alert_or_extract_depth(self):
        a = self.assess(self.outside)
        self.assertEqual(a.risk_level, RiskLevel.NO_ALERT)
        self.assertIn('ngoài hành lang', a.reason)
        self.roi.extract_proximity.assert_not_called()
        self.assertIsNone(self.aggregator.aggregate([a], 100.1))

    def test_thresholds_and_immediate_deescalation(self):
        for frame, (score, expected) in enumerate(((.75, RiskLevel.HIGH), (.45, RiskLevel.MEDIUM),
                                                  (.25, RiskLevel.LOW), (.20, RiskLevel.NO_ALERT),
                                                  (0, RiskLevel.NO_ALERT)), 1):
            self.roi.extract_proximity.return_value = (score, 'ở xa', 'test')
            a = self.assess(frame=frame)
            self.assertEqual(a.risk_level, expected)

    def test_exit_reentry_and_cooldown_escalation(self):
        self.roi.extract_proximity.return_value = (.6, 'gần', 'test')
        medium = self.assess()
        self.assertIsNone(self.aggregator.aggregate([medium], 100.1))
        self.assertEqual(self.fsm._tracks[0].last_alert_time, 0.0)
        self.assertEqual(self.aggregator._last_global_alert_time, 0.0)
        self.assertEqual(self.assess(self.outside, frame=2).risk_level, RiskLevel.NO_ALERT)
        self.assertIsNone(self.aggregator.aggregate([self.assess(frame=3)], 100.3))
        self.roi.extract_proximity.return_value = (.8, 'gần', 'test')
        high = self.assess(frame=4)
        self.assertIsNotNone(self.aggregator.aggregate([high], 100.4))
        self.assertIsNone(self.aggregator.aggregate([self.assess(frame=5)], 100.5))
        self.roi.extract_proximity.return_value = (.6, 'trung bình', 'test')
        self.assertIsNone(self.aggregator.aggregate([self.assess(frame=6)], 100.6))
        self.assertEqual(self.fsm._tracks[0].last_alert_time, 100.4)
        self.roi.extract_proximity.return_value = (.8, 'gần', 'test')
        self.assertIsNone(self.aggregator.aggregate([self.assess(frame=7)], 100.7))
        self.assertIsNotNone(self.aggregator.aggregate([self.assess(frame=45)], 104.5))
        self.assertIn(0, self.fsm._tracks)

    def test_only_near_person_and_obstacles_trigger_audio(self):
        for class_name in ('person', 'chair', 'dining table'):
            for score in (.20, .25, .45, .749, .75, .9):
                with self.subTest(class_name=class_name, score=score):
                    self.setUp()
                    self.roi.extract_proximity.return_value = (score, 'test', 'test')
                    a = self.assess(class_name=class_name)
                    task = self.aggregator.aggregate([a], 100.1)
                    self.assertEqual(task is not None, score >= .75)
                    if score < .75:
                        self.assertEqual(self.fsm._tracks[0].last_alert_time, 0.0)
                        self.assertEqual(self.aggregator._last_global_alert_time, 0.0)
                        self.assertIsNone(self.fsm._tracks[0].last_alert_risk)

    def test_side_center_can_block_corridor_and_has_same_depth_threshold(self):
        # center_x = 200 < 224, and old central overlap < 40%.
        box = BoundingBox(100, 200, 300, 450)
        a = self.assess(box)
        self.assertEqual(a.direction, Direction.LEFT)
        self.assertEqual(a.risk_level, RiskLevel.HIGH)
        self.roi.extract_proximity.return_value = (.45, 'gần', 'test')
        self.assertEqual(self.assess(box, frame=2).risk_level, RiskLevel.MEDIUM)

    def test_reused_depth_cannot_confirm_escalation_or_reentry(self):
        self.fsm.instant_high_risk_in_center = False
        self.fsm.confirmations_to_escalate = 2
        self.assertEqual(self.assess().risk_level, RiskLevel.MEDIUM)
        self.assertEqual(self.assess().risk_level, RiskLevel.MEDIUM)
        self.assertEqual(self.fsm._tracks[0].consecutive_high_count, 1)
        self.assess(self.outside)
        self.assess()
        self.assertEqual(self.fsm._tracks[0].consecutive_high_count, 0)
        self.assertEqual(self.assess(frame=2).risk_level, RiskLevel.MEDIUM)
        self.assertEqual(self.assess(frame=3).risk_level, RiskLevel.HIGH)

    def test_invalid_data_resets_high(self):
        for quality in (DataQuality.STALE, DataQuality.DEGRADED, DataQuality.UNAVAILABLE):
            self.assess()
            self.roi.reset_mock()
            a = self.assess(frame=2, quality=quality)
            self.assertEqual(a.risk_level, RiskLevel.UNDETERMINED)
            self.roi.extract_proximity.assert_not_called()
            self.assertIsNone(self.aggregator.aggregate([a], 100.2))
            self.assertEqual(self.fsm._tracks[0].consecutive_high_count, 0)
        self.assertEqual(self.assess(depth=False).risk_level, RiskLevel.UNDETERMINED)
        for size in ((0, 480), (640, -1), (1280, 720)):
            self.assertEqual(self.assess(size=size).risk_level, RiskLevel.UNDETERMINED)
        self.assertEqual(self.assess(BoundingBox(math.nan, 0, 50, 50)).risk_level,
                         RiskLevel.UNDETERMINED)

    def test_invalid_roi_and_insufficient_pixels(self):
        self.assess()
        self.roi.extract_proximity.return_value = (0, 'chưa rõ', 'Không đủ pixel hợp lệ')
        a = self.assess(frame=2)
        self.assertEqual(a.risk_level, RiskLevel.UNDETERMINED)
        self.assertIn('Không đủ pixel', a.reason)
        self.roi.extract_proximity.return_value = (math.nan, 'gần', 'test')
        self.assertEqual(self.assess(frame=3).risk_level, RiskLevel.UNDETERMINED)
        self.fsm.roi_extractor = ROIExtractor(min_valid_pixels=100000)
        self.assertEqual(self.assess(frame=4).risk_level, RiskLevel.UNDETERMINED)
