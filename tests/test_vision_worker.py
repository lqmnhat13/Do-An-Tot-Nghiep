"""Vision ở tiến trình phụ: hạn chờ, huỷ, khởi động lại, tắt sau nhiều lần treo, lưu stack khi treo."""
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.ocr.vision_worker import OCRCancelled, VisionUnavailable, VisionWorkerClient

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE = np.full((30, 40, 3), 255, np.uint8)


def client(reader, **kwargs):
    kwargs.setdefault("read_timeout_sec", 5.0)
    return VisionWorkerClient(["vi", "en"], reader_spec=f"vision_worker_fakes:{reader}",
                              extra_sys_path=[TESTS_DIR], **kwargs)


class TestVisionWorkerClient(unittest.TestCase):
    def setUp(self):
        self.clients = []

    def tearDown(self):
        for c in self.clients:
            c.close()

    def make(self, reader, **kwargs):
        c = client(reader, **kwargs)
        self.clients.append(c)
        return c

    def test_reads_in_subprocess_and_reuses_it(self):
        c = self.make("EchoReader")
        first = c.readtext(IMAGE)
        box, text, conf = first[0]
        self.assertTrue(text.startswith("40x30 "))
        self.assertNotEqual(int(text.split()[1]), os.getpid())
        self.assertEqual(c.readtext(IMAGE)[0][1], text)  # cùng tiến trình

    def test_hang_times_out_kills_worker_and_next_read_uses_new_process(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"VISION_FAKE_MARKER": os.path.join(tmp, "m")}):
            c = self.make("HangFirstProcessReader", read_timeout_sec=1.5)
            t0 = time.monotonic()
            with self.assertRaises(TimeoutError):
                c.readtext(IMAGE)
            self.assertLess(time.monotonic() - t0, 2.5)
            # Vision không bị tắt sau một lần treo: lần sau chạy ở tiến trình mới.
            deadline = time.monotonic() + 10
            while True:
                try:
                    result = c.readtext(IMAGE)
                    break
                except TimeoutError:  # đang khởi động lại ở nền
                    self.assertLess(time.monotonic(), deadline)
            self.assertFalse(c.disabled)
            self.assertTrue(result[0][1].startswith("40x30 "))

    def test_cancel_interrupts_hung_read_quickly(self):
        c = self.make("HangReader", read_timeout_sec=20)
        threading.Timer(0.8, c.cancel).start()
        t0 = time.monotonic()
        with self.assertRaises(OCRCancelled):
            c.readtext(IMAGE)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertFalse(c.disabled)

    def test_cancel_without_request_is_ignored(self):
        c = self.make("EchoReader")
        c.cancel()
        self.assertTrue(c.readtext(IMAGE))

    def test_init_failure_disables_vision(self):
        c = self.make("BrokenInitReader")
        with self.assertRaises(VisionUnavailable):
            c.readtext(IMAGE)
        self.assertTrue(c.disabled)
        with self.assertRaises(VisionUnavailable):
            c.readtext(IMAGE)

    def test_reader_error_keeps_worker(self):
        c = self.make("ErrorReader")
        with self.assertRaises(RuntimeError):
            c.readtext(IMAGE)
        pid = c.pid
        with self.assertRaises(RuntimeError):
            c.readtext(IMAGE)
        self.assertEqual(c.pid, pid)
        self.assertFalse(c.disabled)

    def test_repeated_hangs_disable_vision(self):
        c = self.make("HangReader", read_timeout_sec=0.8, max_consecutive_failures=2,
                      warmup_timeout_sec=0.8)
        deadline = time.monotonic() + 15
        while not c.disabled:
            self.assertLess(time.monotonic(), deadline)
            try:
                c.readtext(IMAGE)
            except (TimeoutError, VisionUnavailable):
                pass
        with self.assertRaises(VisionUnavailable):
            c.readtext(IMAGE)

    def test_close_kills_worker_and_never_respawns(self):
        c = self.make("HangReader", read_timeout_sec=20)
        threading.Timer(0.8, c.close).start()
        with self.assertRaises((OCRCancelled, RuntimeError)):
            c.readtext(IMAGE)
        time.sleep(0.3)
        self.assertIsNone(c.pid)
        with self.assertRaises(OCRCancelled):
            c.readtext(IMAGE)

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("sample"), "cần lệnh sample của macOS")
    def test_hang_saves_native_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.make("HangReader", read_timeout_sec=1.0, hang_log_dir=tmp)
            with self.assertRaises(TimeoutError):
                c.readtext(IMAGE)
            deadline = time.monotonic() + 10
            while c.last_hang_log is None and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertIsNotNone(c.last_hang_log)
            with open(c.last_hang_log, errors="replace") as f:
                self.assertIn("Call graph", f.read())


if __name__ == "__main__":
    unittest.main()
