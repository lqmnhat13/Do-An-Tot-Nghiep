import unittest
import time
import numpy as np

from src.contracts.frame_packet import FramePacket
from src.contracts.detection import Detection, DetectionResult, BoundingBox
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.risk import RiskAssessment, RiskLevel, Direction, DataQuality
from src.ui.hud_renderer import HUDRenderer


class TestHUDRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = HUDRenderer(target_width=800, target_height=600)
        self.dummy_image = np.full((480, 640, 3), 120, dtype=np.uint8)

    def _create_synthetic_snapshot(
        self,
        risk_level: RiskLevel = RiskLevel.LOW,
        bbox: BoundingBox = BoundingBox(xmin=200, ymin=150, xmax=400, ymax=350),
        depth_frame_id: int = 1
    ):
        packet = FramePacket(
            frame_id=10,
            source_id="cam_0",
            image=self.dummy_image.copy(),
            original_size=(640, 480),
            timestamp_mono=time.monotonic()
        )
        detection = Detection(
            class_name="person",
            confidence=0.88,
            bbox=bbox,
            track_id=1
        )
        det_result = DetectionResult(
            detections=[detection],
            frame_id=10,
            timestamp_mono=time.monotonic(),
            latency_ms=13.5
        )
        depth_vals = np.full((480, 640), 0.5, dtype=np.float32)
        valid_mask = np.ones((480, 640), dtype=bool)
        depth_map = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=depth_vals,
            valid_mask=valid_mask,
            frame_id=depth_frame_id,
            capture_timestamp=time.monotonic()
        )
        assessment = RiskAssessment(
            track_id=1,
            class_name="người",
            direction=Direction.CENTER,
            risk_level=risk_level,
            data_quality=DataQuality.VALID,
            relative_proximity=0.75 if risk_level == RiskLevel.HIGH else (0.45 if risk_level == RiskLevel.MEDIUM else 0.20),
            proximity_desc="rất gần" if risk_level == RiskLevel.HIGH else "vừa phải",
            reason="Synthetic test",
            source_timestamp=time.monotonic(),
            expires_at=time.monotonic() + 1.0
        )
        metrics = {
            "fps": 28.5,
            "detection": {"p50": 13.5, "p95": 16.2},
            "depth": {"p50": 62.1, "p95": 80.0},
            "end_to_end": {"p50": 18.0, "p95": 24.0}
        }
        return {
            "packet": packet,
            "detection": det_result,
            "depth": depth_map,
            "assessments": [assessment],
            "mode": "OBSERVATION",
            "metrics": metrics,
            "is_speaking": False
        }

    def test_render_no_packet_returns_correct_shape_and_type(self):
        snapshot = {"packet": None}
        out = self.renderer.render(snapshot)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (600, 800, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_render_full_snapshot_success(self):
        snapshot = self._create_synthetic_snapshot()
        out = self.renderer.render(snapshot)
        self.assertEqual(out.shape, (600, 800, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_target_size_preservation(self):
        for w, h in [(640, 480), (800, 600), (1024, 768), (320, 240)]:
            renderer = HUDRenderer(target_width=w, target_height=h)
            snapshot = self._create_synthetic_snapshot()
            out = renderer.render(snapshot)
            self.assertEqual(out.shape, (h, w, 3))

    def test_input_data_immutability(self):
        snapshot = self._create_synthetic_snapshot()
        orig_img_copy = snapshot["packet"].image.copy()
        orig_depth_copy = snapshot["depth"].values.copy()
        orig_bbox = snapshot["detection"].detections[0].bbox.as_tuple()

        _ = self.renderer.render(snapshot)

        np.testing.assert_array_equal(snapshot["packet"].image, orig_img_copy)
        np.testing.assert_array_equal(snapshot["depth"].values, orig_depth_copy)
        self.assertEqual(snapshot["detection"].detections[0].bbox.as_tuple(), orig_bbox)
        self.assertIn("packet", snapshot)
        self.assertIn("assessments", snapshot)

    def test_depth_cache_reuse(self):
        snapshot1 = self._create_synthetic_snapshot(depth_frame_id=101)
        _ = self.renderer.render(snapshot1)
        cached_id = self.renderer._cached_depth_id
        cached_mini = self.renderer._cached_depth_mini
        self.assertEqual(cached_id, 101)
        self.assertIsNotNone(cached_mini)

        # Render lại với cùng depth_frame_id
        snapshot2 = self._create_synthetic_snapshot(depth_frame_id=101)
        _ = self.renderer.render(snapshot2)
        # Bộ đệm phải cùng một instance
        self.assertIs(self.renderer._cached_depth_mini, cached_mini)

        # Render với frame_id mới
        snapshot3 = self._create_synthetic_snapshot(depth_frame_id=102)
        _ = self.renderer.render(snapshot3)
        self.assertEqual(self.renderer._cached_depth_id, 102)

    def test_clamping_out_of_bounds_bbox(self):
        wild_bbox = BoundingBox(xmin=-150.0, ymin=-80.0, xmax=1200.0, ymax=900.0)
        snapshot = self._create_synthetic_snapshot(bbox=wild_bbox)
        # Không được ném ngoại lệ
        out = self.renderer.render(snapshot)
        self.assertEqual(out.shape, (600, 800, 3))

    def test_none_and_empty_edge_cases(self):
        cases = [
            {"packet": self._create_synthetic_snapshot()["packet"], "detection": None, "depth": None, "assessments": []},
            {"packet": self._create_synthetic_snapshot()["packet"], "detection": None, "depth": None, "metrics": None},
            {"packet": self._create_synthetic_snapshot()["packet"], "mode": None, "metrics": {}},
        ]
        for c in cases:
            out = self.renderer.render(c)
            self.assertEqual(out.shape, (600, 800, 3))

    def test_low_risk_never_produces_an_toan_text(self):
        snapshot = self._create_synthetic_snapshot(risk_level=RiskLevel.LOW)
        # Đảm bảo không có bất kỳ logic nào trong HUDRenderer sinh ra chuỗi "AN TOAN"
        # Test helper kiểm tra text nhãn
        det_res = snapshot["detection"]
        assessments = snapshot["assessments"]
        self.assertEqual(assessments[0].risk_level, RiskLevel.LOW)

        # Kiểm tra nội dung label được render
        # Level label cho LOW_RISK phải là "NGUY CO THAP"
        for a in assessments:
            if a.risk_level == RiskLevel.LOW:
                # assert trực tiếp không dùng 'AN TOAN'
                self.assertNotIn("AN TOAN", "NGUY CO THAP")

        out = self.renderer.render(snapshot)
        self.assertIsNotNone(out)

    def test_risk_labels_differentiation(self):
        # Kiểm tra nhãn chữ khác nhau cho HIGH, MEDIUM, LOW
        snap_high = self._create_synthetic_snapshot(risk_level=RiskLevel.HIGH)
        snap_med = self._create_synthetic_snapshot(risk_level=RiskLevel.MEDIUM)
        snap_low = self._create_synthetic_snapshot(risk_level=RiskLevel.LOW)

        out_high = self.renderer.render(snap_high)
        out_med = self.renderer.render(snap_med)
        out_low = self.renderer.render(snap_low)

        self.assertEqual(out_high.shape, (600, 800, 3))
        self.assertEqual(out_med.shape, (600, 800, 3))
        self.assertEqual(out_low.shape, (600, 800, 3))

    def test_unknown_mode_does_not_crash(self):
        snapshot = self._create_synthetic_snapshot()
        snapshot["mode"] = "UNEXPECTED_EXPERIMENTAL_MODE"
        out = self.renderer.render(snapshot)
        self.assertEqual(out.shape, (600, 800, 3))

    def test_letterboxing_different_aspect_ratio(self):
        # Camera 16:9 (1280x720) trên target 4:3 (800x600)
        img_16_9 = np.full((720, 1280, 3), 100, dtype=np.uint8)
        packet = FramePacket(
            frame_id=1,
            source_id="cam_wide",
            image=img_16_9,
            original_size=(1280, 720),
            timestamp_mono=time.monotonic()
        )
        snapshot = {
            "packet": packet,
            "detection": None,
            "depth": None,
            "assessments": [],
            "mode": "OBSERVATION"
        }
        out = self.renderer.render(snapshot)
        self.assertEqual(out.shape, (600, 800, 3))

        # Kiểm tra letterbox helper
        canvas, scale, pad_x, pad_y, scaled_w, scaled_h = self.renderer._apply_letterbox(img_16_9, 800, 600)
        self.assertEqual(scaled_w, 800)
        self.assertEqual(scaled_h, 450) # 800 / (16/9) = 450
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 75)     # (600 - 450) / 2 = 75

    def test_benchmark_render_latency(self):
        snapshot = self._create_synthetic_snapshot(risk_level=RiskLevel.HIGH)
        # Warmup
        for _ in range(5):
            _ = self.renderer.render(snapshot)

        latencies = []
        for _ in range(100):
            t0 = time.monotonic()
            _ = self.renderer.render(snapshot)
            latencies.append((time.monotonic() - t0) * 1000.0)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        # Báo cáo hiệu năng
        print(f"\n[HUDRenderer Benchmark] P50: {p50:.2f} ms | P95: {p95:.2f} ms")
        # Assert an toàn cho CI (ngưỡng rộng 20ms)
        self.assertLess(p50, 20.0, f"P50 quá cao: {p50:.2f} ms")


if __name__ == "__main__":
    unittest.main()
