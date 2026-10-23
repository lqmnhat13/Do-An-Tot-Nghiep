import importlib.util
import os
import tempfile
import unittest

import numpy as np

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "calibrate_depth.py")


def load_script():
    spec = importlib.util.spec_from_file_location("calibrate_depth", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCalibrateDepth(unittest.TestCase):
    def test_write_scale_keeps_comments_and_other_keys(self):
        module = load_script()
        content = ("depth:\n  # ghi chú\n  representation: \"metric_depth\"\n"
                   "  metric_scale: 0.60 # hệ số\n  input_height: 256\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model_config.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            module.MODEL_CONFIG = path
            module.write_scale(0.7234)
            with open(path, encoding="utf-8") as handle:
                updated = handle.read()
        self.assertIn("metric_scale: 0.723 # hệ số", updated)
        self.assertIn("# ghi chú", updated)
        self.assertIn("input_height: 256", updated)

    def test_center_median_ignores_invalid_pixels(self):
        module = load_script()
        values = np.full((100, 100), 2.0, dtype=np.float32)
        valid = np.ones_like(values, dtype=bool)
        values[45:55, 45:50] = 50.0
        valid[45:55, 45:50] = False
        self.assertAlmostEqual(module.center_median(values, valid, 0.2), 2.0)


if __name__ == "__main__":
    unittest.main()
