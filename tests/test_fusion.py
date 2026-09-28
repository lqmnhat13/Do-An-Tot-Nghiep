import unittest
import numpy as np
import threading
import time

from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.risk import DataQuality, RiskLevel, Direction, RiskAssessment
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
        self.aggregator = AlertAggregator(
            risk_fsm=self.fsm,
            alert_lifetime_sec=3.0,
            max_detection_age_ms=1000.0,
            max_depth_age_ms=1200.0,
            max_alert_age_ms=1500.0
        )

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
        now = time.monotonic()
        t_old = now - 2.5 # 2.5s trước
        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=np.ones((10, 10), dtype=np.float32),
            valid_mask=np.ones((10, 10), dtype=bool),
            frame_id=1,
            capture_timestamp=now
        )
        sync.register_depth(dm)
        res = DetectionResult(detections=[], frame_id=2, timestamp_mono=t_old, latency_ms=10.0)
        pair = sync.synchronize(res, current_mono=now)
        self.assertEqual(pair.data_quality, DataQuality.STALE)

    def test_synchronizer_stale_depth(self):
        sync = Synchronizer(max_depth_age_ms=500.0)
        now = time.monotonic()
        t_old = now - 2.0
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
        res = DetectionResult(detections=[], frame_id=2, timestamp_mono=now, latency_ms=10.0)
        pair = sync.synchronize(res, current_mono=now)
        self.assertEqual(pair.data_quality, DataQuality.STALE)

    def test_synchronizer_same_frame_id_but_stale_timestamp(self):
        sync = Synchronizer(max_detection_age_ms=1000.0, max_depth_age_ms=1200.0)
        now = time.monotonic()
        t_stale = now - 3.0 # Cùng frame_id nhưng cả 2 đều đã quá hạn
        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=np.ones((10, 10), dtype=np.float32),
            valid_mask=np.ones((10, 10), dtype=bool),
            frame_id=100,
            capture_timestamp=t_stale
        )
        sync.register_depth(dm)

        res = DetectionResult(detections=[], frame_id=100, timestamp_mono=t_stale, latency_ms=10.0)
        pair = sync.synchronize(res, current_mono=now)
        self.assertEqual(pair.data_quality, DataQuality.STALE)

    def test_synchronizer_pair_skew_degraded(self):
        sync = Synchronizer(max_pair_skew_ms=300.0, max_detection_age_ms=1000.0, max_depth_age_ms=1200.0)
        now = time.monotonic()
        dm = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=np.ones((10, 10), dtype=np.float32),
            valid_mask=np.ones((10, 10), dtype=bool),
            frame_id=1,
            capture_timestamp=now - 0.450 # 450ms trước (chưa stale nhưng skew = 450ms > 300ms)
        )
        sync.register_depth(dm)

        res = DetectionResult(detections=[], frame_id=2, timestamp_mono=now, latency_ms=10.0)
        pair = sync.synchronize(res, current_mono=now)
        self.assertEqual(pair.data_quality, DataQuality.DEGRADED)

    def test_synchronizer_frame_id_match_valid(self):
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
        now = time.monotonic()
        vals = np.ones((480, 640), dtype=np.float32) * 1.0 # nền ở xa
        vals[100:300, 280:360] = 10.0 # vật thể độ gần cao ở giữa
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
        self.assertEqual(assessments[0].proximity_desc, "độ gần tương đối cao")
        self.assertNotIn("~", assessments[0].reason)
        self.assertNotIn("m", assessments[0].reason.split("px")[0].replace("độ gần tương đối", ""))

        # Test Aggregator tạo câu nói tiếng Việt không chứa số mét
        alert = self.aggregator.aggregate(assessments, current_mono=now)
        self.assertIsNotNone(alert)
        self.assertIn("ghế", alert.text)
        self.assertIn("phía trước", alert.text)
        self.assertNotIn("~", alert.text)
        self.assertNotIn("m", alert.text)
        # Bounded expires_at
        self.assertLessEqual(alert.expires_at, now + self.aggregator.max_source_age_sec + 0.01)

    def test_stale_or_degraded_pair_returns_undetermined_and_no_alert(self):
        now = time.monotonic()
        t_old = now - 2.5
        vals = np.ones((480, 640), dtype=np.float32) * 10.0
        mask = np.ones((480, 640), dtype=bool)
        dm_stale = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals,
            valid_mask=mask,
            frame_id=1,
            capture_timestamp=t_old
        )
        det = Detection(class_name="chair", confidence=0.9, bbox=BoundingBox(280, 100, 360, 300), track_id=2)
        res = DetectionResult(detections=[det], frame_id=2, timestamp_mono=now, latency_ms=10.0)

        sync = Synchronizer(max_depth_age_ms=1000.0)
        sync.register_depth(dm_stale)
        pair = sync.synchronize(res, current_mono=now)
        self.assertEqual(pair.data_quality, DataQuality.STALE)

        # FSM không được trích ROI, trả UNDETERMINED và "chưa rõ"
        assessments = self.fsm.update(pair, current_mono=now)
        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].risk_level, RiskLevel.UNDETERMINED)
        self.assertEqual(assessments[0].proximity_desc, "chưa rõ")
        self.assertEqual(assessments[0].relative_proximity, 0.0)

        # AlertAggregator phải loại bỏ và không tạo cảnh báo
        alert = self.aggregator.aggregate(assessments, current_mono=now)
        self.assertIsNone(alert)

        # Cooldown không được đánh dấu cho track
        self.assertTrue(self.fsm.can_trigger_alert(2, RiskLevel.HIGH, now))

    def test_hysteresis_reset_and_recovery_with_fresh_depth(self):
        """Khi dữ liệu stale đến, bộ đếm hysteresis bị reset; khi depth phục hồi, xác nhận bắt đầu lại từ đầu."""
        now = time.monotonic()
        vals_high = np.ones((480, 640), dtype=np.float32) * 1.0 # Nền ở xa
        vals_high[100:300, 10:100] = 10.0 # Vật thể độ gần tương đối cao (norm = 1.0 >= 0.75)
        mask = np.ones((480, 640), dtype=bool)

        # 1. Vật thể ở bên TRÁI (cần 2 frame xác nhận để lên HIGH theo confirmations_to_escalate=2)
        det_left = Detection(class_name="chair", confidence=0.9, bbox=BoundingBox(10, 100, 100, 300), track_id=10)
        dm_fresh_1 = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_high,
            valid_mask=mask,
            frame_id=1,
            capture_timestamp=now
        )
        sync = Synchronizer()
        sync.register_depth(dm_fresh_1)
        res_1 = DetectionResult(detections=[det_left], frame_id=1, timestamp_mono=now, latency_ms=10.0)
        pair_1 = sync.synchronize(res_1, current_mono=now)

        assessments_1 = self.fsm.update(pair_1, current_mono=now)
        # Frame 1: mới nhận 1 xác nhận (1/2) -> chưa lên HIGH, trả MEDIUM
        self.assertEqual(assessments_1[0].risk_level, RiskLevel.MEDIUM)
        self.assertEqual(self.fsm._tracks[10].consecutive_high_count, 1)

        # 2. Bây giờ depth bị STALE ở frame tiếp theo
        now_2 = now + 2.0
        dm_stale = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_high,
            valid_mask=mask,
            frame_id=2,
            capture_timestamp=now # cũ 2 giây
        )
        sync_stale = Synchronizer(max_depth_age_ms=1000.0)
        sync_stale.register_depth(dm_stale)
        res_2 = DetectionResult(detections=[det_left], frame_id=2, timestamp_mono=now_2, latency_ms=10.0)
        pair_2 = sync_stale.synchronize(res_2, current_mono=now_2)
        self.assertEqual(pair_2.data_quality, DataQuality.STALE)

        assessments_2 = self.fsm.update(pair_2, current_mono=now_2)
        self.assertEqual(assessments_2[0].risk_level, RiskLevel.UNDETERMINED)
        # Bộ đếm consecutive_high_count phải về 0
        self.assertEqual(self.fsm._tracks[10].consecutive_high_count, 0)

        # 3. Phục hồi với depth mới hợp lệ (frame 3)
        now_3 = now_2 + 0.1
        dm_fresh_3 = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_high,
            valid_mask=mask,
            frame_id=3,
            capture_timestamp=now_3
        )
        sync_fresh = Synchronizer()
        sync_fresh.register_depth(dm_fresh_3)
        res_3 = DetectionResult(detections=[det_left], frame_id=3, timestamp_mono=now_3, latency_ms=10.0)
        pair_3 = sync_fresh.synchronize(res_3, current_mono=now_3)
        self.assertEqual(pair_3.data_quality, DataQuality.VALID)

        assessments_3 = self.fsm.update(pair_3, current_mono=now_3)
        # Vì bộ đếm đã reset về 0, frame 3 mới là confirmation 1/2 -> trả về MEDIUM (không nhảy thẳng lên HIGH từ frame cũ)
        self.assertEqual(self.fsm._tracks[10].consecutive_high_count, 1)
        self.assertEqual(assessments_3[0].risk_level, RiskLevel.MEDIUM)

        # 4. Thêm một frame hợp lệ nữa (frame 4) -> đạt 2/2 confirmations -> lên HIGH!
        now_4 = now_3 + 0.1
        dm_fresh_4 = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_high,
            valid_mask=mask,
            frame_id=4,
            capture_timestamp=now_4
        )
        sync_fresh.register_depth(dm_fresh_4)
        res_4 = DetectionResult(detections=[det_left], frame_id=4, timestamp_mono=now_4, latency_ms=10.0)
        pair_4 = sync_fresh.synchronize(res_4, current_mono=now_4)
        assessments_4 = self.fsm.update(pair_4, current_mono=now_4)
        self.assertEqual(self.fsm._tracks[10].consecutive_high_count, 2)
        self.assertEqual(assessments_4[0].risk_level, RiskLevel.HIGH)

    def test_aggregator_ignores_expired_assessment_or_source(self):
        now = time.monotonic()
        # Assessment hết hạn
        a_expired = RiskAssessment(
            track_id=1,
            class_name="ghế",
            direction=Direction.CENTER,
            risk_level=RiskLevel.HIGH,
            data_quality=DataQuality.VALID,
            relative_proximity=0.8,
            proximity_desc="độ gần tương đối cao",
            reason="test",
            source_timestamp=now - 2.0,
            expires_at=now - 0.5 # Đã hết hạn
        )
        task = self.aggregator.aggregate([a_expired], current_mono=now)
        self.assertIsNone(task)
        # Cooldown chưa bị kích hoạt
        self.assertTrue(self.fsm.can_trigger_alert(1, RiskLevel.HIGH, now))

        # Assessment chưa hết hạn nhưng source_timestamp quá hạn (now - source_ts > max_source_age_sec)
        a_source_stale = RiskAssessment(
            track_id=2,
            class_name="bàn",
            direction=Direction.CENTER,
            risk_level=RiskLevel.HIGH,
            data_quality=DataQuality.VALID,
            relative_proximity=0.8,
            proximity_desc="độ gần tương đối cao",
            reason="test",
            source_timestamp=now - 1.5, # 1.5s > 1.0s max_source_age_sec
            expires_at=now + 1.0
        )
        task2 = self.aggregator.aggregate([a_source_stale], current_mono=now)
        self.assertIsNone(task2)
        self.assertTrue(self.fsm.can_trigger_alert(2, RiskLevel.HIGH, now))

    def test_synchronizer_thread_safety_and_late_depth(self):
        """register_depth và synchronize chạy đồng thời; mẫu cũ đến muộn không ghi đè _latest_depth."""
        sync = Synchronizer()
        barrier = threading.Barrier(2)
        errors = []

        now = time.monotonic()
        dm_new = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=np.ones((10, 10), dtype=np.float32),
            valid_mask=np.ones((10, 10), dtype=bool),
            frame_id=20,
            capture_timestamp=now
        )
        dm_old_late = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=np.ones((10, 10), dtype=np.float32),
            valid_mask=np.ones((10, 10), dtype=bool),
            frame_id=10,
            capture_timestamp=now - 1.0 # Mẫu cũ đến muộn
        )

        def worker_register():
            try:
                barrier.wait(timeout=2.0)
                sync.register_depth(dm_new)
                sync.register_depth(dm_old_late) # Mẫu cũ đến muộn
            except Exception as e:
                errors.append(e)

        def worker_sync():
            try:
                barrier.wait(timeout=2.0)
                for i in range(10):
                    det_res = DetectionResult(detections=[], frame_id=20, timestamp_mono=now, latency_ms=5.0)
                    pair = sync.synchronize(det_res, current_mono=now)
                    # pair không bao giờ được bị lỗi partial state
                    self.assertIn(pair.data_quality, [DataQuality.VALID, DataQuality.UNAVAILABLE])
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker_register)
        t2 = threading.Thread(target=worker_sync)
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        self.assertEqual(len(errors), 0)
        # _latest_depth phải là dm_new (frame 20), không bị dm_old_late (frame 10) ghi đè
        self.assertIsNotNone(sync._latest_depth)
        self.assertEqual(sync._latest_depth.frame_id, 20)

    def test_configurable_depth_thresholds(self):
        """Kiểm tra RiskFSM phản ánh chính xác các ngưỡng cấu hình depth_threshold_high/medium/low."""
        fsm_custom = RiskFSM(
            class_filter=self.class_filter,
            spatial_zones=self.spatial,
            roi_extractor=self.roi,
            depth_threshold_high=0.75,
            depth_threshold_medium=0.45,
            depth_threshold_low=0.25,
            confirmations_to_escalate=1,
            confirmations_to_deescalate=1,
            instant_high_risk_in_center=True,
            cooldown_sec=2.0
        )
        now = time.monotonic()
        vals = np.zeros((480, 640), dtype=np.float32)
        vals[0, 0] = 0.0 # Min
        vals[0, 1] = 10.0 # Max -> span = 10.0
        mask = np.ones((480, 640), dtype=bool)

        # Case 1: Đối tượng ở center có giá trị 6.0 -> normalized = 0.60 (0.45 <= 0.60 < 0.75 -> MEDIUM)
        vals_medium = vals.copy()
        vals_medium[100:300, 280:360] = 6.0
        dm_med = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_medium,
            valid_mask=mask,
            frame_id=1,
            capture_timestamp=now
        )
        sync = Synchronizer()
        sync.register_depth(dm_med)
        det_med = Detection(class_name="chair", confidence=0.9, bbox=BoundingBox(280, 100, 360, 300), track_id=1)
        res_med = DetectionResult(detections=[det_med], frame_id=1, timestamp_mono=now, latency_ms=5.0)
        pair_med = sync.synchronize(res_med, current_mono=now)
        assessments_med = fsm_custom.update(pair_med, current_mono=now)
        self.assertEqual(len(assessments_med), 1)
        self.assertEqual(assessments_med[0].risk_level, RiskLevel.MEDIUM)
        self.assertAlmostEqual(assessments_med[0].relative_proximity, 0.60, delta=0.05)

        # Case 2: Đối tượng ở center có giá trị 8.5 -> normalized = 0.85 (>= 0.75 -> HIGH)
        vals_high = vals.copy()
        vals_high[100:300, 280:360] = 8.5
        dm_high = DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True,
            values=vals_high,
            valid_mask=mask,
            frame_id=2,
            capture_timestamp=now + 0.1
        )
        sync.register_depth(dm_high)
        det_high = Detection(class_name="chair", confidence=0.9, bbox=BoundingBox(280, 100, 360, 300), track_id=2)
        res_high = DetectionResult(detections=[det_high], frame_id=2, timestamp_mono=now + 0.1, latency_ms=5.0)
        pair_high = sync.synchronize(res_high, current_mono=now + 0.1)
        assessments_high = fsm_custom.update(pair_high, current_mono=now + 0.1)
        self.assertEqual(len(assessments_high), 1)
        self.assertEqual(assessments_high[0].risk_level, RiskLevel.HIGH)

    def test_global_alert_interval_throttles_interleaved_tracks(self):
        """AlertAggregator áp dụng khoảng lặng toàn cục (global_alert_interval_sec) giữa các track khác nhau nhưng cho phép nguy cơ HIGH vượt hàng đợi."""
        aggregator_spaced = AlertAggregator(
            risk_fsm=self.fsm,
            alert_lifetime_sec=3.0,
            global_alert_interval_sec=3.0,
            max_detection_age_ms=1000.0,
            max_depth_age_ms=1200.0,
            max_alert_age_ms=1500.0
        )
        t0 = 100.0
        # 1. Track 1 phát cảnh báo MEDIUM tại t0 -> thành công
        a1 = RiskAssessment(
            track_id=1,
            class_name="ghế",
            direction=Direction.LEFT,
            risk_level=RiskLevel.MEDIUM,
            data_quality=DataQuality.VALID,
            relative_proximity=0.5,
            proximity_desc="ở cự ly trung bình",
            reason="test",
            source_timestamp=t0,
            expires_at=t0 + 2.0
        )
        task1 = aggregator_spaced.aggregate([a1], current_mono=t0)
        self.assertIsNotNone(task1)
        self.assertEqual(task1.text, "Có ghế bên trái, ở cự ly trung bình")

        # 2. Track 2 (khác track 1) đến lúc t0 + 1.0 (trong khoảng 3.0s interval), cùng mức MEDIUM
        # -> Bị throttle bởi global_alert_interval_sec dù track 2 chưa bao giờ cooldown!
        a2 = RiskAssessment(
            track_id=2,
            class_name="bàn",
            direction=Direction.RIGHT,
            risk_level=RiskLevel.MEDIUM,
            data_quality=DataQuality.VALID,
            relative_proximity=0.55,
            proximity_desc="ở cự ly trung bình",
            reason="test",
            source_timestamp=t0 + 1.0,
            expires_at=t0 + 3.0
        )
        task2 = aggregator_spaced.aggregate([a2], current_mono=t0 + 1.0)
        self.assertIsNone(task2)

        # 3. Track 3 đến lúc t0 + 1.5 với mức HIGH -> Nguy hiểm khẩn cấp (escalation override) được phép ngắt khoảng lặng!
        a3 = RiskAssessment(
            track_id=3,
            class_name="bậc thang",
            direction=Direction.CENTER,
            risk_level=RiskLevel.HIGH,
            data_quality=DataQuality.VALID,
            relative_proximity=0.85,
            proximity_desc="độ gần tương đối cao",
            reason="test",
            source_timestamp=t0 + 1.5,
            expires_at=t0 + 3.5
        )
        task3 = aggregator_spaced.aggregate([a3], current_mono=t0 + 1.5)
        self.assertIsNotNone(task3)
        self.assertEqual(task3.text, "Chú ý, có bậc thang phía trước, độ gần tương đối cao")

        # 4. Track 4 đến lúc t0 + 2.0 với mức HIGH -> Vì cảnh báo trước đó (task3) ĐÃ LÀ HIGH,
        # không còn là escalation từ non-HIGH lên HIGH nữa, nên bị interval chặn để tránh spam tiếng hét liên tục
        a4 = RiskAssessment(
            track_id=4,
            class_name="tường",
            direction=Direction.CENTER,
            risk_level=RiskLevel.HIGH,
            data_quality=DataQuality.VALID,
            relative_proximity=0.90,
            proximity_desc="độ gần tương đối cao",
            reason="test",
            source_timestamp=t0 + 2.0,
            expires_at=t0 + 4.0
        )
        task4 = aggregator_spaced.aggregate([a4], current_mono=t0 + 2.0)
        self.assertIsNone(task4)

        # 5. Track 4 có frame mới sau khi hết khoảng lặng 3.0s (t0 + 1.5 + 3.2 = t0 + 4.7) -> Được phát!
        a4_fresh = RiskAssessment(
            track_id=4,
            class_name="tường",
            direction=Direction.CENTER,
            risk_level=RiskLevel.HIGH,
            data_quality=DataQuality.VALID,
            relative_proximity=0.90,
            proximity_desc="độ gần tương đối cao",
            reason="test",
            source_timestamp=t0 + 4.7,
            expires_at=t0 + 6.7
        )
        task5 = aggregator_spaced.aggregate([a4_fresh], current_mono=t0 + 4.7)
        self.assertIsNotNone(task5)
        self.assertEqual(task5.text, "Chú ý, có tường phía trước, độ gần tương đối cao")

if __name__ == "__main__":
    unittest.main()
