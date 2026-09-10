import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.benchmark_vqa_backends import MANUAL_CRITERIA, run_benchmark
from src.vqa.backend import VQABackend


class FakeBenchmarkBackend(VQABackend):
    def __init__(self):
        self.calls = []

    def answer(self, image, question):
        self.calls.append(question)
        if question == "fail":
            raise RuntimeError("fake failure")
        return f"fake: {question}"


class TestBenchmarkVQABackends(unittest.TestCase):
    def test_fake_backend_report_warmup_repeats_and_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(images / "sample.png")
            questions = root / "questions.jsonl"
            rows = [
                {
                    "image": "sample.png",
                    "question": "Có gì phía trước?",
                    "expected_answer": "Một căn phòng",
                    "manual_scores": {criterion: 2 for criterion in MANUAL_CRITERIA},
                },
                {"image": "sample.png", "question": "fail"},
            ]
            questions.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8"
            )
            output = root / "report.json"
            backend = FakeBenchmarkBackend()

            report = run_benchmark(
                images,
                questions,
                output,
                backend_name="fake",
                warmup=1,
                repeats=2,
                backend_factory=lambda: backend,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(report["baseline_backend"], "legacy_caption")
            self.assertEqual(report["summary"]["total_runs"], 4)
            self.assertEqual(report["summary"]["error_count"], 2)
            self.assertEqual(report["summary"]["error_rate"], 0.5)
            self.assertGreaterEqual(report["summary"]["p95_latency_ms"], 0.0)
            self.assertGreaterEqual(report["model_load_time_ms"], 0.0)
            self.assertEqual(len(backend.calls), 5)
            self.assertEqual(report["results"][0]["question"], "Có gì phía trước?")
            self.assertEqual(report["results"][0]["manual_scores"]["object_accuracy"], 2)
            self.assertIn("fake failure", report["results"][-1]["error"])

            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"], report["summary"])


if __name__ == "__main__":
    unittest.main()
