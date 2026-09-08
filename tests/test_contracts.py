import unittest
import numpy as np
import time
from src.contracts.frame_packet import FramePacket
from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.risk import DataQuality, RiskLevel, Direction, RiskAssessment
from src.contracts.audio import AudioPriority, AudioTask

class TestContracts(unittest.TestCase):
    def test_frame_packet(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        now = time.monotonic()
        packet = FramePacket(
            frame_id=1,
            source_id="webcam_0",
            image=img,
            original_size=(640, 480),
            timestamp_mono=now
        )
        self.assertEqual(packet.frame_id, 1)
        self.assertEqual(packet.width, 640)
        self.assertEqual(packet.height, 480)
        self.assertGreaterEqual(packet.age_ms(now + 0.05), 49.0)

    def test_bounding_box(self):
        box = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=100.0)
        self.assertEqual(box.width, 40.0)
        self.assertEqual(box.height, 80.0)
        self.assertEqual(box.center_x, 30.0)
        self.assertEqual(box.center_y, 60.0)
        self.assertEqual(box.area, 3200.0)

        clamped = BoundingBox(-10, -5, 700, 500).clamp(640, 480)
        self.assertEqual(clamped.xmin, 0.0)
        self.assertEqual(clamped.ymin, 0.0)
        self.assertEqual(clamped.xmax, 640.0)
        self.assertEqual(clamped.ymax, 480.0)

    def test_depth_map(self):
        vals = np.ones((100, 100), dtype=np.float32) * 5.0
        mask = np.ones((100, 100), dtype=bool)
        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals,
            valid_mask=mask,
            frame_id=10,
            capture_timestamp=time.monotonic()
        )
        self.assertTrue(dm.near_is_larger)
        self.assertEqual(dm.shape, (100, 100))

    def test_risk_assessment(self):
        now = time.monotonic()
        risk = RiskAssessment(
            track_id=1,
            class_name="chair",
            direction=Direction.CENTER,
            risk_level=RiskLevel.HIGH,
            data_quality=DataQuality.VALID,
            relative_proximity=0.85,
            proximity_desc="rất gần",
            reason="Vật thể alert_class ở giữa đường",
            source_timestamp=now,
            expires_at=now + 0.5
        )
        self.assertFalse(risk.is_expired(now + 0.1))
        self.assertTrue(risk.is_expired(now + 0.6))

    def test_audio_priority_order(self):
        t1 = AudioTask(priority=AudioPriority.HIGH_RISK, text="Cảnh báo")
        t2 = AudioTask(priority=AudioPriority.ON_DEMAND, text="Đọc chữ")
        self.assertLess(t1, t2) # HIGH_RISK (1) ưu tiên hơn ON_DEMAND (3)

if __name__ == "__main__":
    unittest.main()
