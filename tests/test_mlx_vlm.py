import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts import download_models
from scripts.run_vqa_camera import VQACameraSession, build_parser
from src.contracts.request import VQARequest
from src.vqa.backend import MLXVLMBackend, LegacyCaptionBackend
from src.vqa.vqa_service import VQAService


class TestMLXVLM(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "config.json").write_text('{"model_type": "qwen2_vl"}')
        (self.path / "model.safetensors").touch()
        self.load = Mock(return_value=(object(), object()))
        self.generate = Mock(return_value=types.SimpleNamespace(text="Một chiếc ghế"))
        self.template = Mock(side_effect=lambda processor, config, q, **kw: q)
        fake = types.ModuleType("mlx_vlm")
        fake.load, fake.generate = self.load, self.generate
        prompts = types.ModuleType("mlx_vlm.prompt_utils")
        prompts.apply_chat_template = self.template
        self.modules = patch.dict(sys.modules, {
            "mlx_vlm": fake, "mlx_vlm.prompt_utils": prompts,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        # A regression must fail instead of contacting a server or initializing CUDA.
        for target in ("socket.socket.connect", "torch.cuda.init"):
            guard = patch(target, side_effect=AssertionError(target))
            guard.start()
            self.addCleanup(guard.stop)
        self.frame = np.zeros((600, 1200, 3), dtype=np.uint8)
        self.frame[:, :] = [10, 20, 30]
        self.context = {"spatial_objects": [
            {"name": "ghế", "direction": "LEFT", "proximity": "gần"}
        ]}

    def service(self, **config):
        return VQAService(backend_name="mlx_vlm", lazy_load=False,
                          mlx_vlm_config={"model_path": str(self.path), **config})

    def request(self, question="  Ghế màu gì?  "):
        return VQARequest("mlx-test", self.frame, question)

    def assert_fallback(self, service):
        result = service.answer(self.request(), self.context)
        self.assertTrue(result.success)
        self.assertIn("ghế bên trái", result.answer)
        self.assertNotIn("Khung cảnh:", result.answer)

    def test_lazy_question_aware_bgr_conversion_and_hard_limits(self):
        service = self.service(max_image_size=9999, max_tokens=9999)
        self.assertIsInstance(service.backend, MLXVLMBackend)
        self.load.assert_not_called()
        original = self.frame.copy()
        with patch.dict(os.environ, {"SECOND_EYE_OFFLINE": "0"}):
            service.answer(self.request())
            service.answer(self.request("Cửa ở đâu?"))
        self.load.assert_called_once_with(str(self.path.resolve()),
                                          local_files_only=True, trust_remote_code=False)
        self.assertEqual([c.args[2] for c in self.template.call_args_list],
                         ["  Ghế màu gì?  ", "Cửa ở đâu?"])
        self.assertEqual([c.args[2] for c in self.generate.call_args_list],
                         ["  Ghế màu gì?  ", "Cửa ở đâu?"])
        image = self.generate.call_args.kwargs["image"][0]
        self.assertEqual(image.size, (512, 256))
        self.assertEqual(image.getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(self.generate.call_args.kwargs["max_tokens"], 64)
        np.testing.assert_array_equal(self.frame, original)

    def test_configured_smaller_limits_and_string_output(self):
        self.generate.return_value = "  màu đỏ  "
        service = self.service(max_image_size=224, max_tokens=32)
        self.assertIn("màu đỏ", service.answer(self.request()).answer)
        self.assertEqual(self.generate.call_args.kwargs["image"][0].size, (224, 112))
        self.assertEqual(self.generate.call_args.kwargs["max_tokens"], 32)

    def test_guardrail_prevents_even_loading(self):
        result = self.service().answer(self.request("Tôi có đi tiếp an toàn không?"))
        self.assertIn("không thể xác nhận", result.answer)
        self.load.assert_not_called()
        self.generate.assert_not_called()

    def test_missing_model_repo_id_and_incomplete_snapshot(self):
        for path in ("", str(self.path / "missing"), "owner/nonexistent-model"):
            with self.subTest(path=path):
                self.assert_fallback(self.service(model_path=path))
        (self.path / "model.safetensors").unlink()
        self.assert_fallback(self.service())
        self.load.assert_not_called()

    def test_missing_dependency_and_failed_load_are_cached(self):
        with patch.dict(sys.modules, {"mlx_vlm": None}):
            service = self.service()
            self.assert_fallback(service)
        self.assert_fallback(service)
        self.load.assert_not_called()
        self.assertIn("mlx-vlm", service.load_error)
        self.load.side_effect = OSError("broken snapshot")
        service = self.service()
        self.assert_fallback(service)
        self.assert_fallback(service)
        self.load.assert_called_once()

    def test_inference_failure_releases_lock_and_can_recover(self):
        self.generate.side_effect = [RuntimeError("Metal error"), None, "", "recovered"]
        service = self.service()
        for _ in range(3):
            self.assert_fallback(service)
        self.assertIn("recovered", service.answer(self.request()).answer)
        self.load.assert_called_once()

    def test_concurrent_requests_are_serialized_without_deadlock(self):
        entered, release, second_started = (threading.Event() for _ in range(3))
        def generate(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test timed out")
            return "done"
        self.generate.side_effect = generate
        service = self.service()
        results = []
        def run(second=False):
            if second:
                second_started.set()
            results.append(service.answer(self.request()))
        first = threading.Thread(target=run, daemon=True)
        second = threading.Thread(target=run, args=(True,), daemon=True)
        first.start()
        try:
            self.assertTrue(entered.wait(1))
            second.start()
            self.assertTrue(second_started.wait(1))
            self.assertEqual(self.generate.call_count, 1)
        finally:
            release.set()
            first.join(2)
            if second.ident is not None:
                second.join(2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(results), 2)
        self.load.assert_called_once()

    def test_shutdown_suppresses_late_mlx_result(self):
        entered, release = threading.Event(), threading.Event()
        def generate(*args, **kwargs):
            entered.set()
            release.wait(4)
            return "late"
        self.generate.side_effect = generate
        tts = Mock()
        session = VQACameraSession(self.service(), "Có gì?", tts)
        session.trigger(self.frame)
        try:
            self.assertTrue(entered.wait(1))
            session.close()
            self.assertFalse(session.trigger(self.frame))
        finally:
            release.set()
            session._worker.join(2)
        self.assertFalse(session._worker.is_alive())
        self.assertEqual(session.last_answer, "")
        tts.speak.assert_not_called()

    def test_defaults_disabled_and_cli(self):
        self.assertIsInstance(VQAService(device="cpu").backend, LegacyCaptionBackend)
        self.assertIsNone(VQAService(backend_name="disabled").backend)
        self.assertIsNone(VQAService(backend_name="mlx_vlm", use_vlm=False).backend)
        self.assertEqual(build_parser().parse_args(["--backend", "mlx_vlm"]).backend,
                         "mlx_vlm")
        self.load.assert_not_called()

    def test_download_is_explicit_and_uses_yaml(self):
        with patch.object(download_models, "enable_model_downloads"), \
             patch.object(download_models, "setup_mlx_vlm_weights") as mlx, \
             patch.object(download_models, "setup_yolo_weights"), \
             patch.object(download_models, "setup_depth_weights"), \
             patch.object(download_models, "setup_vqa_weights"), \
             patch.object(download_models, "setup_ocr_weights"), \
             patch.object(download_models, "setup_audio_assets"):
            self.assertEqual(download_models.main([]), 0)
            mlx.assert_not_called()
            self.assertEqual(download_models.main(["--mlx-vlm"]), 0)
            mlx.assert_called_once()
            mlx.side_effect = OSError("offline")
            self.assertEqual(download_models.main(["--mlx-vlm"]), 1)
        import yaml
        root = Path(download_models.PROJECT_ROOT)
        with (root / "configs/model_config.yaml").open() as handle:
            config = yaml.safe_load(handle)["vqa"]["mlx_vlm"]
        with patch("huggingface_hub.snapshot_download") as snapshot:
            download_models.setup_mlx_vlm_weights()
        snapshot.assert_called_once_with(repo_id=config["hf_repo_id"],
                                         local_dir=str(root / config["model_path"]),
                                         local_files_only=False)


if __name__ == "__main__":
    unittest.main()
