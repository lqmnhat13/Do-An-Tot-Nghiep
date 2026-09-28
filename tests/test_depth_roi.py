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

if __name__ == "__main__":
    unittest.main()
