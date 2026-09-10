import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.contracts.frame_packet import FramePacket
from src.contracts.request import VQARequest
from src.depth.depth_estimator import DepthEstimator
from src.ocr.ocr_service import OCRService
from src.runtime.model_loading import (
    enable_model_downloads,
    is_offline_mode,
    pretrained_kwargs,
)
from src.vqa.vqa_service import VQAService


class TestOfflineModelLoading(unittest.TestCase):
    def test_offline_is_default_and_download_script_can_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(is_offline_mode())
            self.assertEqual(pretrained_kwargs(), {"local_files_only": True})
            self.assertEqual(
                pretrained_kwargs(allow_download=True),
                {"local_files_only": False}
            )

    def test_download_preparation_explicitly_disables_offline_flags(self):
        with patch.dict(os.environ, {
            "SECOND_EYE_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }):
            enable_model_downloads()
            self.assertFalse(is_offline_mode())
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "0")

    def test_easyocr_disables_downloads_in_offline_runtime(self):
        easyocr = MagicMock()
        easyocr.Reader.return_value = MagicMock()
        with patch.dict(os.environ, {"SECOND_EYE_OFFLINE": "1"}), patch.dict(
            sys.modules, {"easyocr": easyocr}
        ):
            service = OCRService(use_gpu=False)
            service._ensure_reader()

        easyocr.Reader.assert_called_once_with(
            ["vi", "en"], gpu=False, download_enabled=False
        )
        self.assertIsNone(service.load_error)

    @patch("transformers.AutoModelForDepthEstimation.from_pretrained")
    @patch("transformers.AutoImageProcessor.from_pretrained")
    def test_depth_loader_uses_local_cache_only(self, load_processor, load_model):
        load_processor.return_value = MagicMock()
        load_model.return_value = MagicMock()

        with patch.dict(os.environ, {"SECOND_EYE_OFFLINE": "1"}):
            estimator = DepthEstimator(device="cpu")

        load_processor.assert_called_once_with(
            estimator.model_name, local_files_only=True
        )
        load_model.assert_called_once_with(
            estimator.model_name, local_files_only=True
        )
        self.assertIsNone(estimator.load_error)

    @patch("transformers.MarianMTModel.from_pretrained")
    @patch("transformers.MarianTokenizer.from_pretrained")
    @patch("transformers.BlipForConditionalGeneration.from_pretrained")
    @patch("transformers.BlipProcessor.from_pretrained")
    def test_vqa_loaders_use_local_cache_only(
        self,
        load_caption_processor,
        load_caption_model,
        load_translation_tokenizer,
        load_translation_model
    ):
        with patch.dict(os.environ, {"SECOND_EYE_OFFLINE": "1"}):
            service = VQAService(device="cpu", lazy_load=False)

        load_caption_processor.assert_called_once_with(
            service.caption_model_name, local_files_only=True
        )
        load_caption_model.assert_called_once_with(
            service.caption_model_name, local_files_only=True
        )
        load_translation_tokenizer.assert_called_once_with(
            service.translation_model_name, local_files_only=True
        )
        load_translation_model.assert_called_once_with(
            service.translation_model_name, local_files_only=True
        )
        self.assertIsNone(service.load_error)

    @patch(
        "transformers.AutoImageProcessor.from_pretrained",
        side_effect=OSError("not found in local cache")
    )
    def test_missing_depth_cache_returns_immediate_fallback(self, load_processor):
        with patch.dict(os.environ, {"SECOND_EYE_OFFLINE": "1"}):
            estimator = DepthEstimator(device="cpu")

        packet = FramePacket(
            frame_id=1,
            source_id="test",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            original_size=(8, 8),
            timestamp_mono=time.monotonic()
        )
        result = estimator.estimate(packet)

        self.assertEqual(load_processor.call_count, 1)
        self.assertIn("cache local", estimator.load_error)
        self.assertFalse(result.valid_mask.any())

    @patch(
        "transformers.BlipProcessor.from_pretrained",
        side_effect=OSError("not found in local cache")
    )
    def test_missing_vqa_cache_falls_back_without_retrying_network(self, load_processor):
        with patch.dict(os.environ, {"SECOND_EYE_OFFLINE": "1"}):
            service = VQAService(device="cpu", lazy_load=False)
            request = VQARequest(
                request_id="offline",
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                question="Mô tả phía trước"
            )
            first = service.answer(request)
            second = service.answer(request)

        self.assertEqual(load_processor.call_count, 1)
        self.assertIn("cache local", service.load_error)
        self.assertTrue(first.success)
        self.assertTrue(second.success)


if __name__ == "__main__":
    unittest.main()
