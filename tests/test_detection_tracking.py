import unittest
import numpy as np
import time
from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.detection.class_filter import ClassFilter
from src.tracking.byte_tracker import ByteTrackerAdapter

class TestDetectionTracking(unittest.TestCase):
    def test_class_filter(self):
        cf = ClassFilter()
        # 21 classes test
        self.assertTrue(cf.is_enabled("chair"))
        self.assertTrue(cf.is_enabled("person"))
        self.assertTrue(cf.is_enabled("cat"))
        self.assertTrue(cf.is_enabled("laptop"))
        self.assertFalse(cf.is_enabled("airplane"))
        self.assertFalse(cf.is_enabled("car"))

        # Alert classes test
        self.assertTrue(cf.is_alert_candidate("chair"))
        self.assertTrue(cf.is_alert_candidate("person"))
        self.assertFalse(cf.is_alert_candidate("mouse"))
        self.assertFalse(cf.is_alert_candidate("cup"))

        # Vietnamese translation test
        self.assertEqual(cf.get_vietnamese_name("chair"), "ghế")
        self.assertEqual(cf.get_vietnamese_name("dining table"), "bàn ăn")

    def test_tracker_consistency(self):
        tracker = ByteTrackerAdapter(iou_threshold=0.3)
        
        # Frame 1: 1 chair
        box1 = BoundingBox(100, 100, 200, 200)
        res1 = DetectionResult(
            detections=[Detection(class_name="chair", confidence=0.8, bbox=box1)],
            frame_id=1,
            timestamp_mono=1.0,
            latency_ms=10.0
        )
        tracked1 = tracker.update(res1)
        self.assertEqual(len(tracked1.detections), 1)
        track_id = tracked1.detections[0].track_id
        self.assertIsNotNone(track_id)

        # Frame 2: chair moves slightly
        box2 = BoundingBox(105, 102, 205, 202)
        res2 = DetectionResult(
            detections=[Detection(class_name="chair", confidence=0.82, bbox=box2)],
            frame_id=2,
            timestamp_mono=1.05,
            latency_ms=10.0
        )
        tracked2 = tracker.update(res2)
        self.assertEqual(len(tracked2.detections), 1)
        self.assertEqual(tracked2.detections[0].track_id, track_id)

if __name__ == "__main__":
    unittest.main()
