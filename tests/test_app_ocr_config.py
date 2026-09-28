import unittest

from app import build_ocr_service


class TestAppOCRConfig(unittest.TestCase):
    def test_main_ocr_service_uses_model_config(self):
        service = build_ocr_service({"ocr": {
            "engine": "vision",
            "languages": ["vi", "en"],
            "min_confidence": 0.6,
            "blur_laplacian_threshold": 75,
            "min_brightness": 45,
            "max_brightness": 225,
        }}, "cpu")
        self.assertEqual(service.languages, ["vi", "en"])
        self.assertEqual(service.engine, "vision")
        self.assertFalse(service.use_gpu)
        self.assertEqual(service.min_confidence, 0.6)
        self.assertEqual(service.quality_checker.blur_threshold, 75)
        self.assertEqual(service.quality_checker.min_brightness, 45)
        self.assertEqual(service.quality_checker.max_brightness, 225)


if __name__ == "__main__":
    unittest.main()
