import unittest
import time
import threading
from src.camera.bounded_buffer import BoundedBuffer

class TestBoundedBuffer(unittest.TestCase):
    def test_put_get(self):
        buf = BoundedBuffer[int](maxsize=2)
        self.assertTrue(buf.is_empty())
        buf.put(1)
        buf.put(2)
        self.assertEqual(buf.size(), 2)
        self.assertEqual(buf.get(), 1)
        self.assertEqual(buf.get(), 2)
        self.assertTrue(buf.is_empty())

    def test_drop_oldest(self):
        buf = BoundedBuffer[int](maxsize=2)
        dropped1 = buf.put(10)
        self.assertFalse(dropped1)
        dropped2 = buf.put(20)
        self.assertFalse(dropped2)
        # Quá kích thước 2 -> rơi phần tử 10
        dropped3 = buf.put(30)
        self.assertTrue(dropped3)
        self.assertEqual(buf.dropped_count, 1)
        self.assertEqual(buf.total_pushed, 3)
        self.assertEqual(buf.size(), 2)

        # Buffer giờ chứa 20 và 30
        self.assertEqual(buf.get(), 20)
        self.assertEqual(buf.get(), 30)

    def test_get_latest_clear_older(self):
        buf = BoundedBuffer[str](maxsize=5)
        buf.put("a")
        buf.put("b")
        buf.put("c")
        latest = buf.get_latest(clear_older=True)
        self.assertEqual(latest, "c")
        self.assertTrue(buf.is_empty())
        self.assertEqual(buf.dropped_count, 2)

    def test_timeout(self):
        buf = BoundedBuffer[int](maxsize=1)
        t0 = time.monotonic()
        item = buf.get(timeout=0.05)
        dt = time.monotonic() - t0
        self.assertIsNone(item)
        self.assertGreaterEqual(dt, 0.04)

if __name__ == "__main__":
    unittest.main()
