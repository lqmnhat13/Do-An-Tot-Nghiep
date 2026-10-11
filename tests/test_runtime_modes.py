import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine
from src.contracts.audio import AudioPriority, AudioTask
from src.contracts.detection import DetectionResult
from src.contracts.frame_packet import FramePacket
from src.contracts.risk import DataQuality, Direction, RiskAssessment, RiskLevel
from src.fusion.alert_aggregator import AlertAggregator
from src.fusion.synchronizer import SynchronizedPair
from src.runtime.system_coordinator import SystemCoordinator
from src.ui.app_runner import AppRunner
from src.ui.hud_renderer import HUDRenderer


def high_assessment():
    now = time.monotonic()
    return RiskAssessment(1, 'người', Direction.CENTER, RiskLevel.HIGH,
                          DataQuality.VALID, .9, 'gần', 'test', now, now + 1)


class TestRuntimeModes(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)
        self.audio.wait_until_idle = MagicMock(return_value=True)
        self.camera = MagicMock()
        self.packet = FramePacket(1, 'test', np.zeros((32, 32, 3), dtype=np.uint8),
                                  (32, 32), time.monotonic())
        self.camera.get_latest_frame.return_value = self.packet
        self.ocr, self.vqa = MagicMock(), MagicMock()
        self.ocr.process.return_value = MagicMock(text='Đọc xong', latency_sec=.1)
        self.vqa.answer.return_value = MagicMock(answer='Trả lời xong', latency_sec=.1)
        self.fsm = MagicMock()
        self.fsm.can_trigger_alert.return_value = True
        self.aggregator = AlertAggregator(self.fsm)
        self.system = SystemCoordinator(self.camera, MagicMock(), MagicMock(), MagicMock(),
                                        MagicMock(), self.fsm, self.aggregator, self.audio,
                                        self.ocr, self.vqa)

    def tearDown(self):
        self.system.stop()

    def start_task(self, mode):
        return self.system.trigger_ocr() if mode == 'OCR' else self.system.trigger_vqa()

    def finish_task(self):
        self.system._on_demand_thread.join(timeout=1)
        self.assertFalse(self.system._on_demand_thread.is_alive())

    def test_completion_errors_and_missing_images_keep_selected_mode(self):
        for mode in ('OCR', 'VQA'):
            for outcome in ('success', 'failure', 'missing'):
                with self.subTest(mode=mode, outcome=outcome):
                    self.system.enter_warning_mode()
                    self.camera.get_latest_frame.return_value = None if outcome == 'missing' else self.packet
                    service = self.ocr.process if mode == 'OCR' else self.vqa.answer
                    service.side_effect = RuntimeError('inference failed') if outcome == 'failure' else None
                    self.assertTrue(self.start_task(mode))
                    self.finish_task()
                    snapshot = self.system.get_runtime_snapshot()
                    self.assertEqual(snapshot['selected_mode'], mode)
                    self.assertEqual(snapshot['mode'], mode)
                    self.assertFalse(snapshot['on_demand_active'])
                    self.assertFalse(snapshot['safety_alerts_enabled'])
                    self.system._post_safety_alert([high_assessment()])
                    self.fsm.mark_alert_triggered.assert_not_called()
                    self.assertFalse(self.audio.post_task(AudioTask(AudioPriority.HIGH_RISK, 'Vật cản')))

    def test_return_requires_new_sources_and_valid_quality(self):
        self.system.set_on_demand_active(True, 'OCR')
        old = high_assessment()
        self.system.enter_warning_mode()
        self.system._post_safety_alert([old])
        self.fsm.mark_alert_triggered.assert_not_called()
        fresh = high_assessment()
        for quality in (DataQuality.STALE, DataQuality.DEGRADED, DataQuality.UNAVAILABLE):
            self.system._post_safety_alert([replace(fresh, data_quality=quality)])
        self.fsm.mark_alert_triggered.assert_not_called()
        self.system._post_safety_alert([fresh])
        self.fsm.mark_alert_triggered.assert_called_once()

    def test_stop_keeps_mode_and_repeated_requests_are_on_demand(self):
        self.start_task('OCR')
        self.finish_task()
        self.system.stop_speech()
        self.assertEqual(self.system.get_runtime_snapshot()['selected_mode'], 'OCR')
        self.assertFalse(self.system.get_runtime_snapshot()['safety_alerts_enabled'])
        self.assertTrue(self.start_task('OCR'))
        self.finish_task()
        self.assertEqual(self.ocr.process.call_count, 2)

    def test_busy_rejection_c_and_late_results(self):
        entered, release = threading.Event(), threading.Event()
        def process(*args, **kwargs):
            entered.set()
            release.wait(timeout=2)
            return MagicMock(text='Kết quả đã hủy', latency_sec=.1)
        self.ocr.process.side_effect = process
        try:
            self.assertTrue(self.start_task('OCR'))
            self.assertTrue(entered.wait(timeout=1))
            with patch.object(self.audio, 'post_task', wraps=self.audio.post_task) as post:
                self.system._post_safety_alert([high_assessment()])
                self.fsm.mark_alert_triggered.assert_not_called()
                self.assertFalse(self.start_task('VQA'))
                self.assertEqual(self.system.get_runtime_snapshot()['selected_mode'], 'OCR')
                self.system.enter_warning_mode()
                self.assertFalse(self.start_task('VQA'))
                self.assertEqual(self.system.get_runtime_snapshot()['selected_mode'], 'WARNING')
                release.set()
                self.finish_task()
                texts = [call.args[0].text for call in post.call_args_list]
                self.assertNotIn('Kết quả đã hủy', texts)
                self.assertEqual(self.system.get_runtime_snapshot()['selected_mode'], 'WARNING')
                self.assertTrue(self.start_task('VQA'))
                self.finish_task()
        finally:
            release.set()

    def test_start_reservation_survives_cancellation_before_thread_start(self):
        entered, release = threading.Event(), threading.Event()
        original = threading.Thread.start
        def delayed_start(thread):
            if thread.name == 'OCRTaskThread':
                entered.set()
                release.wait(timeout=2)
            return original(thread)
        with patch.object(threading.Thread, 'start', delayed_start):
            caller = threading.Thread(target=self.system.trigger_ocr)
            caller.start()
            try:
                self.assertTrue(entered.wait(timeout=1))
                self.system.enter_warning_mode()
                self.assertFalse(self.system.trigger_vqa())
            finally:
                release.set()
                caller.join(timeout=1)
            self.assertFalse(caller.is_alive())
            self.finish_task()
        self.ocr.process.assert_not_called()
        self.assertEqual(self.system.get_runtime_snapshot()['selected_mode'], 'WARNING')

    def test_c_releases_worker_waiting_for_audio(self):
        waiting = threading.Event()
        def still_busy(timeout):
            waiting.set()
            time.sleep(min(timeout, .02))
            return False
        self.audio.wait_until_idle.side_effect = still_busy
        self.assertTrue(self.start_task('OCR'))
        self.assertTrue(waiting.wait(timeout=1))
        self.system.enter_warning_mode()
        self.finish_task()
        self.assertTrue(self.start_task('VQA'))
        self.assertTrue(self.system.is_on_demand_active())
        self.system.stop_speech()
        self.finish_task()

    def test_detection_and_depth_keep_running_without_overwriting_mode(self):
        self.system.set_on_demand_active(True, 'OCR')
        self.system._finish_on_demand(self.system._active_on_demand_token)
        result = DetectionResult([], 1, time.monotonic(), 0)
        self.system.detector.detect.return_value = result
        self.system.tracker.update.return_value = result
        self.system.synchronizer.synchronize.return_value = SynchronizedPair(result, None, DataQuality.STALE, 0, time.monotonic())
        self.fsm.update.return_value = [high_assessment()]
        self.system._running = True
        detection_done = threading.Event()
        depth_done = threading.Event()
        self.system.buf_detection.put(self.packet)
        self.system.buf_depth.put(self.packet)
        with patch.object(self.system.watchdog, 'beat') as beat:
            def completed(worker):
                (detection_done if worker == 'detection_worker' else depth_done).set()
            beat.side_effect = completed
            det = threading.Thread(target=self.system._detection_worker)
            depth = threading.Thread(target=self.system._depth_worker)
            det.start()
            depth.start()
            try:
                self.assertTrue(detection_done.wait(timeout=1))
                self.assertTrue(depth_done.wait(timeout=1))
            finally:
                self.system._running = False
                det.join(timeout=1)
                depth.join(timeout=1)
        self.assertFalse(det.is_alive())
        self.assertFalse(depth.is_alive())
        self.system.detector.detect.assert_called_once()
        self.system.depth_estimator.estimate.assert_called_once()
        self.fsm.update.assert_called_once()
        self.fsm.mark_alert_triggered.assert_not_called()
        self.assertEqual(self.system.get_runtime_snapshot()['selected_mode'], 'OCR')
        self.assertEqual(self.system.get_runtime_snapshot()['mode'], 'OCR')

    def test_gui_c_key_dispatches_warning_mode(self):
        coordinator = MagicMock()
        runner = AppRunner(coordinator, MagicMock(), enable_gui=True)
        with patch('src.ui.app_runner.cv2.namedWindow'), patch('src.ui.app_runner.cv2.imshow'), \
             patch('src.ui.app_runner.cv2.destroyAllWindows'), \
             patch('src.ui.app_runner.cv2.waitKey', side_effect=[ord('c'), 27]):
            runner.run()
        coordinator.enter_warning_mode.assert_called_once()
        coordinator.stop.assert_called_once()

    def test_hud_status_banner_and_shortcuts(self):
        renderer = HUDRenderer()
        for mode, active in (('OCR', True), ('OCR', False), ('VQA', True), ('VQA', False)):
            snapshot = {'packet': self.packet, 'mode': mode, 'selected_mode': mode,
                        'on_demand_active': active, 'safety_alerts_enabled': False,
                        'assessments': [high_assessment()]}
            with patch.object(renderer, '_draw_alert_banner') as banner, \
                 patch('src.ui.hud_renderer.cv2.putText') as text:
                renderer.render(snapshot)
            banner.assert_not_called()
            labels = [call.args[1] for call in text.call_args_list]
            self.assertTrue(any(mode in label and 'CB TAT' in label for label in labels))
            self.assertTrue(any(('XU LY' if active else 'SAN SANG') in label for label in labels))
        for width in (320, 480, 640, 800):
            with patch('src.ui.hud_renderer.cv2.putText') as text:
                renderer._draw_footer(np.zeros((600, width, 3), dtype=np.uint8), width, 600)
            labels = [call.args[1] for call in text.call_args_list]
            for key in ('C', 'Q', 'SPACE', 'S', 'ESC'):
                self.assertIn(key, labels)
