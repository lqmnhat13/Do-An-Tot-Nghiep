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

    def test_roi_extractor_valid_object(self):
        # Tạo depth map 100x100 trong đó góc (50..80, 50..80) có giá trị độ sâu lớn (ở gần)
        vals = np.ones((100, 100), dtype=np.float32) * 2.0
        vals[50:80, 50:80] = 9.0 # vật thể rất gần
        mask = np.ones((100, 100), dtype=bool)

        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals,
            valid_mask=mask,
            frame_id=1,
            capture_timestamp=time.monotonic()
        )

        extractor = ROIExtractor(erosion_ratio=0.1, percentile=80.0)
        bbox = BoundingBox(50, 50, 80, 80)
        score, desc, reason = extractor.extract_proximity(bbox, dm)

        self.assertGreaterEqual(score, 0.70)
        self.assertTrue(desc.startswith("rất gần"))

    def test_roi_extractor_invalid_mask(self):
        vals = np.ones((100, 100), dtype=np.float32)
        mask = np.zeros((100, 100), dtype=bool) # Không có pixel nào hợp lệ

        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals,
            valid_mask=mask,
            frame_id=2,
            capture_timestamp=time.monotonic()
        )

        extractor = ROIExtractor()
        bbox = BoundingBox(20, 20, 60, 60)
        score, desc, reason = extractor.extract_proximity(bbox, dm)
        self.assertEqual(desc, "chưa rõ")

if __name__ == "__main__":
    unittest.main()
