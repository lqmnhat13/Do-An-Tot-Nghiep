import unittest
import numpy as np
import time
from src.contracts.detection import BoundingBox
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.depth.roi_extractor import ROIExtractor
from src.depth.depth_representation import DepthRepresentationHelper

class TestDepthROI(unittest.TestCase):
    def test_normalize_relative_depth(self):
        # 10x10 array with values from 1.0 to 10.0
        # near_is_larger=True: 10.0 is nearest (norm = 1.0), 1.0 is farthest (norm = 0.0)
        vals = np.linspace(1.0, 10.0, 100).reshape((10, 10)).astype(np.float32)
        mask = np.ones((10, 10), dtype=bool)

        norm = DepthRepresentationHelper.normalize_relative_depth(vals, mask, near_is_larger=True)
        self.assertAlmostEqual(float(norm[0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(norm[9, 9]), 1.0, places=4)

    def test_roi_extractor_relative_proximity_levels_no_fake_meters(self):
        """Kiểm tra các mức độ gần tương đối và bảo đảm không chứa số mét giả định."""
        extractor = ROIExtractor(erosion_ratio=0.1, percentile=80.0)

        # 1. Vật thể độ gần tương đối cao (score >= 0.75)
        vals_high = np.ones((100, 100), dtype=np.float32) * 1.0
        vals_high[40:80, 40:80] = 10.0
        dm_high = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_high,
            valid_mask=np.ones((100, 100), dtype=bool),
            frame_id=1,
            capture_timestamp=time.monotonic()
        )
        score_high, desc_high, reason_high = extractor.extract_proximity(BoundingBox(40, 40, 80, 80), dm_high)
        self.assertGreaterEqual(score_high, 0.75)
        self.assertEqual(desc_high, "độ gần tương đối cao")
        self.assertNotIn("~", desc_high)
        self.assertNotIn("m", desc_high)
        self.assertNotIn("~", reason_high)
        self.assertNotIn("m", reason_high.split("px")[0].replace("độ gần tương đối", ""))

        # 2. Vật thể độ gần tương đối trung bình (0.50 <= score < 0.75)
        vals_med = np.ones((100, 100), dtype=np.float32) * 1.0
        vals_med[40:80, 40:80] = 6.0
        vals_med[0:10, 0:10] = 10.0 # Để max toàn khung hình là 10.0
        dm_med = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_med,
            valid_mask=np.ones((100, 100), dtype=bool),
            frame_id=2,
            capture_timestamp=time.monotonic()
        )
        score_med, desc_med, reason_med = extractor.extract_proximity(BoundingBox(40, 40, 80, 80), dm_med)
        self.assertGreaterEqual(score_med, 0.50)
        self.assertLess(score_med, 0.75)
        self.assertEqual(desc_med, "độ gần tương đối trung bình")
        self.assertNotIn("~", desc_med)
        self.assertNotIn("m", desc_med)

        # 3. Vật thể độ gần tương đối thấp (0.30 <= score < 0.50)
        vals_low = np.ones((100, 100), dtype=np.float32) * 1.0
        vals_low[40:80, 40:80] = 4.0
        vals_low[0:10, 0:10] = 10.0
        dm_low = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_low,
            valid_mask=np.ones((100, 100), dtype=bool),
            frame_id=3,
            capture_timestamp=time.monotonic()
        )
        score_low, desc_low, reason_low = extractor.extract_proximity(BoundingBox(40, 40, 80, 80), dm_low)
        self.assertGreaterEqual(score_low, 0.30)
        self.assertLess(score_low, 0.50)
        self.assertEqual(desc_low, "độ gần tương đối thấp")
        self.assertNotIn("~", desc_low)
        self.assertNotIn("m", desc_low)

        # 4. Vật thể ở xa (score < 0.30)
        vals_far = np.ones((100, 100), dtype=np.float32) * 1.0
        vals_far[40:80, 40:80] = 2.0
        vals_far[0:10, 0:10] = 10.0
        dm_far = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_far,
            valid_mask=np.ones((100, 100), dtype=bool),
            frame_id=4,
            capture_timestamp=time.monotonic()
        )
        score_far, desc_far, reason_far = extractor.extract_proximity(BoundingBox(40, 40, 80, 80), dm_far)
        self.assertLess(score_far, 0.30)
        self.assertEqual(desc_far, "ở xa")
        self.assertNotIn("~", desc_far)
        self.assertNotIn("m", desc_far)

        # Bảo toàn chiều near_is_larger
        self.assertGreater(score_high, score_med)
        self.assertGreater(score_med, score_low)
        self.assertGreater(score_low, score_far)

    def test_roi_extractor_invalid_mask(self):
        vals = np.ones((100, 100), dtype=np.float32)
        mask = np.zeros((100, 100), dtype=bool) # Không có pixel nào hợp lệ

        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals,
            valid_mask=mask,
            frame_id=5,
            capture_timestamp=time.monotonic()
        )

        extractor = ROIExtractor()
        bbox = BoundingBox(20, 20, 60, 60)
        score, desc, reason = extractor.extract_proximity(bbox, dm)
        self.assertEqual(score, 0.0)
        self.assertEqual(desc, "chưa rõ")
        self.assertIn("Không đủ pixel", reason)


def _metric_map(values, frame_id=1):
    return DepthMap(
        representation=DepthRepresentation.METRIC_DEPTH,
        unit="meter",
        near_is_larger=False,
        values=values.astype(np.float32),
        valid_mask=values > 0,
        frame_id=frame_id,
        capture_timestamp=time.monotonic()
    )


class TestMetricDepthROI(unittest.TestCase):
    def test_near_object_is_high_even_when_floor_is_closer(self):
        # Sàn sát camera (0.5 m) gần hơn vật; chuẩn hoá theo frame sẽ hạ điểm của vật.
        values = np.full((100, 100), 4.0)
        values[90:, :] = 0.5
        values[30:70, 30:70] = 1.2
        extractor = ROIExtractor(erosion_ratio=0.1, percentile=80.0)
        score, desc, reason = extractor.extract_proximity(BoundingBox(30, 30, 70, 70), _metric_map(values))
        self.assertGreaterEqual(score, 0.75)
        self.assertEqual(desc, "ở rất gần")
        self.assertNotIn("m", desc)  # câu nói không đọc số mét
        self.assertIn("1.2 m", reason)

        relative = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH, unit="relative", near_is_larger=True,
            values=(1.0 / values).astype(np.float32), valid_mask=np.ones_like(values, bool),
            frame_id=2, capture_timestamp=time.monotonic())
        rel_score, _, _ = ROIExtractor(erosion_ratio=0.1).extract_proximity(BoundingBox(30, 30, 70, 70), relative)
        self.assertLess(rel_score, 0.75)

    def test_far_object_is_not_high_even_when_it_is_nearest_in_frame(self):
        values = np.full((100, 100), 8.0)
        values[30:70, 30:70] = 3.0
        score, _, _ = ROIExtractor(erosion_ratio=0.1).extract_proximity(
            BoundingBox(30, 30, 70, 70), _metric_map(values))
        self.assertLess(score, 0.45)

    def test_breakpoints_align_meter_and_score_thresholds(self):
        from src.depth.factory import build_roi_extractor
        extractor = build_roi_extractor(
            {"depth": {}},
            {"risk_fsm": {"distance_threshold_high_m": 1.2, "distance_threshold_medium_m": 2.0,
                          "distance_threshold_low_m": 3.0, "depth_threshold_high": 0.8,
                          "depth_threshold_medium": 0.5, "depth_threshold_low": 0.3}})
        self.assertAlmostEqual(extractor.score_from_distance(1.2), 0.8)
        self.assertAlmostEqual(extractor.score_from_distance(2.0), 0.5)
        self.assertAlmostEqual(extractor.score_from_distance(3.0), 0.3)
        self.assertGreater(extractor.score_from_distance(1.19), 0.8)
        self.assertLess(extractor.score_from_distance(1.21), 0.8)

    def test_invalid_breakpoints_are_rejected(self):
        from src.depth.roi_extractor import metric_breakpoints
        with self.assertRaises(ValueError):
            metric_breakpoints(2.0, 1.5, 3.0, 0.75, 0.45, 0.25)
        with self.assertRaises(ValueError):
            ROIExtractor(metric_breakpoints=((0.0, 0.2), (1.0, 0.9)))


class TestMetricDepthEstimator(unittest.TestCase):
    def test_metric_output_is_scaled_meters(self):
        import torch
        from unittest.mock import patch
        from src.contracts.frame_packet import FramePacket
        from src.depth.depth_estimator import DepthEstimator

        class Processor:
            def __call__(self, images, return_tensors, size):
                return {"pixel_values": torch.zeros(1, 3, 14, 14)}

        class Model:
            def __call__(self, **inputs):
                class Out:
                    predicted_depth = torch.full((1, 14, 14), 2.0)
                return Out()

        with patch.object(DepthEstimator, "_load_model", lambda self: None):
            est = DepthEstimator(device="cpu", representation="metric_depth", metric_scale=0.6)
        est._processor, est._model = Processor(), Model()
        packet = FramePacket(frame_id=3, source_id="t", image=np.zeros((20, 30, 3), np.uint8),
                             original_size=(30, 20), timestamp_mono=time.monotonic())
        dm = est.estimate(packet)
        self.assertEqual(dm.representation, DepthRepresentation.METRIC_DEPTH)
        self.assertEqual(dm.unit, "meter")
        self.assertFalse(dm.near_is_larger)
        self.assertEqual(dm.shape, (20, 30))
        self.assertTrue(np.allclose(dm.values, 1.2))

    def test_inverse_depth_is_rejected(self):
        from src.depth.depth_estimator import DepthEstimator
        with self.assertRaises(ValueError):
            DepthEstimator(device="cpu", representation="inverse_depth")


if __name__ == "__main__":
    unittest.main()
