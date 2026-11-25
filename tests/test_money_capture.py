"""Nhận diện tiền: khung độ phân giải cao (cùng góc nhìn với pipeline) và gộp nhiều khung hình."""
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine
from src.camera.camera_manager import CameraManager
from src.contracts.frame_packet import FramePacket
from src.contracts.request import OCRResult
from src.currency.currency_service import (
    BanknoteReading, CurrencyService, MIXED_TEXT, UNCLEAR_TEXT, combine_readings)
from src.fusion.alert_aggregator import AlertAggregator
from src.runtime.system_coordinator import SystemCoordinator


def hd_frame():
    """1280x720 có dải giữa 960 px (vùng góc nhìn 4:3 của pipeline) màu khác hai bên."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 160:1120] = 200
    return frame


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.props = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        time.sleep(0.01)
        return True, self.frame.copy()

    def release(self):
        pass


class TestCameraFullFrame(unittest.TestCase):
    def test_pipeline_frame_is_center_crop_and_full_frame_is_kept(self):
        cam = CameraManager(source=0, width=640, height=480, capture_width=1280, capture_height=720)
        pipeline, full = cam._split_frame(hd_frame())
        self.assertEqual(pipeline.shape, (480, 640, 3))
        self.assertTrue((pipeline > 150).all())  # chỉ vùng giữa 960x720, không lẫn hai dải bên
        self.assertEqual(full.shape, (720, 1280, 3))

    def test_without_capture_size_old_resize_kept(self):
        cam = CameraManager(source=0, width=640, height=480)
        pipeline, full = cam._split_frame(hd_frame())
        self.assertEqual(pipeline.shape, (480, 640, 3))
        self.assertFalse((pipeline > 150).all())  # co giãn cả khung như trước, không cắt
        self.assertIs(full, None)

    def test_capture_loop_publishes_both_frames(self):
        cam = CameraManager(source=0, width=640, height=480, capture_width=1280, capture_height=720)
        cam._cap = FakeCapture(hd_frame())
        cam.start()
        try:
            deadline = time.monotonic() + 2
            while cam.get_latest_full_frame() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            small, full = cam.get_latest_frame(), cam.get_latest_full_frame()
        finally:
            cam.stop()
        self.assertEqual(small.image.shape, (480, 640, 3))
        self.assertEqual(full.image.shape, (720, 1280, 3))
        self.assertEqual(full.original_size, (1280, 720))
        self.assertLessEqual(abs(full.frame_id - small.frame_id), 1)


class TestCombineReadings(unittest.TestCase):
    OK1 = BanknoteReading("OK", 100000, (100000,))
    OK2 = BanknoteReading("OK", 200000, (200000,))
    UNCLEAR = BanknoteReading("UNCLEAR")

    def test_rules(self):
        self.assertEqual(combine_readings([self.UNCLEAR, self.OK1, self.UNCLEAR]).denomination, 100000)
        self.assertEqual(combine_readings([self.OK1, self.OK1]).status, "OK")
        mixed = combine_readings([self.OK1, self.OK2])
        self.assertEqual((mixed.status, mixed.denominations), ("MIXED", (100000, 200000)))
        self.assertEqual(combine_readings([self.OK1, BanknoteReading("MIXED", None, (100000, 500000))]).status,
                         "MIXED")
        self.assertEqual(combine_readings([self.UNCLEAR, self.UNCLEAR]).status, "UNCLEAR")
        self.assertEqual(combine_readings([]).status, "UNCLEAR")


class TestRecognizeFrames(unittest.TestCase):
    def service(self, texts):
        ocr = MagicMock()
        it = iter(texts)

        def process(req):
            item = next(it)
            if item == "CANCEL":
                return OCRResult(req.request_id, False, "Đã huỷ đọc chữ.", error_message="cancelled")
            return OCRResult(req.request_id, True, item)
        ocr.process.side_effect = process
        return CurrencyService(ocr), ocr

    @staticmethod
    def frames(n=10):
        images = iter([np.zeros((4, 4, 3), np.uint8)] * n)
        return lambda: next(images, None)

    def test_stops_after_two_consistent_reads(self):
        service, ocr = self.service(["MỘT TRĂM NGHÌN ĐỒNG"] * 5)
        result = service.recognize_frames(self.frames())
        self.assertEqual(result.text, "Một trăm nghìn đồng.")
        self.assertEqual(ocr.process.call_count, 2)

    def test_blurry_frames_then_one_clear_frame(self):
        service, ocr = self.service(["", "", "", "HAI TRĂM NGHÌN ĐỒNG"])
        self.assertEqual(service.recognize_frames(self.frames(), max_frames=4).text, "Hai trăm nghìn đồng.")
        self.assertEqual(ocr.process.call_count, 4)

    def test_disagreeing_frames_are_mixed(self):
        service, _ = self.service(["MỘT TRĂM NGHÌN ĐỒNG", "HAI TRĂM NGHÌN ĐỒNG"])
        self.assertEqual(service.recognize_frames(self.frames()).text, MIXED_TEXT)

    def test_nothing_readable_uses_max_frames(self):
        service, ocr = self.service([""] * 10)
        self.assertEqual(service.recognize_frames(self.frames(), max_frames=3).text, UNCLEAR_TEXT)
        self.assertEqual(ocr.process.call_count, 3)

    def test_stops_on_cancel_stop_request_missing_frame_or_budget(self):
        service, ocr = self.service(["", "CANCEL", ""])
        self.assertTrue(service.recognize_frames(self.frames()).cancelled)
        self.assertEqual(ocr.process.call_count, 2)
        service, ocr = self.service([""] * 5)
        self.assertTrue(service.recognize_frames(self.frames(), should_stop=lambda: True).cancelled)
        ocr.process.assert_not_called()
        service, ocr = self.service([""] * 5)
        self.assertEqual(service.recognize_frames(self.frames(n=1)).text, UNCLEAR_TEXT)
        self.assertEqual(ocr.process.call_count, 1)
        service, ocr = self.service([""] * 5)
        service.recognize_frames(self.frames(), budget_sec=0.0)
        self.assertEqual(ocr.process.call_count, 1)  # luôn đọc ít nhất một khung


class TestMoneyModeUsesFullFrame(unittest.TestCase):
    def test_money_reads_full_resolution_frames(self):
        engine = MagicMock(spec=TTSEngine)
        engine.is_speaking.return_value = False
        audio = AudioCoordinator(engine)
        audio.wait_until_idle = MagicMock(return_value=True)
        counter = {"id": 0}

        def packet(size):
            counter["id"] += 1
            w, h = size
            return FramePacket(counter["id"], "cam", np.zeros((h, w, 3), np.uint8), (w, h), time.monotonic())
        camera = MagicMock()
        camera.get_latest_frame.side_effect = lambda: packet((640, 480))
        camera.get_latest_full_frame.side_effect = lambda: packet((1280, 720))
        shapes = []
        ocr = MagicMock()

        def process(req):
            shapes.append(req.image.shape)
            return OCRResult(req.request_id, True, "NĂM MƯƠI NGHÌN ĐỒNG")
        ocr.process.side_effect = process
        system = SystemCoordinator(camera, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                   AlertAggregator(MagicMock()), audio, ocr, MagicMock())
        try:
            self.assertTrue(system.trigger_currency())
            system._on_demand_thread.join(timeout=5)
            spoken = [e.task.text for e in list(audio._queue.queue)]
        finally:
            system.stop()
        self.assertEqual(shapes, [(720, 1280, 3), (720, 1280, 3)])
        self.assertIn("Năm mươi nghìn đồng.", spoken)


if __name__ == "__main__":
    unittest.main()
