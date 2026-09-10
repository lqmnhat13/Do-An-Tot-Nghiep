#!/usr/bin/env python3
"""Benchmark độc lập VQA backend; không khởi tạo pipeline camera/safety."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.model_loading import is_offline_mode
from src.vqa.backend import VQABackend, create_vqa_backend


MANUAL_CRITERIA = (
    "question_adherence",
    "object_accuracy",
    "spatial_accuracy",
    "vietnamese_quality",
)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _validate_manual_scores(raw_scores: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    raw_scores = raw_scores or {}
    scores: Dict[str, Optional[int]] = {}
    for criterion in MANUAL_CRITERIA:
        value = raw_scores.get(criterion)
        if value is None:
            scores[criterion] = None
        elif isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2:
            raise ValueError(f"{criterion} phải là số nguyên từ 0 đến 2")
        else:
            scores[criterion] = value
    return scores


def load_dataset(image_dir: Path, questions_path: Path) -> List[Dict[str, Any]]:
    """Đọc JSONL và ảnh BGR; expected_answer chỉ dùng để người chấm tham khảo."""
    items = []
    with questions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                image_name = row["image"]
                question = row["question"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"JSONL dòng {line_number} không hợp lệ: {exc}") from exc
            if not isinstance(question, str) or not question:
                raise ValueError(f"JSONL dòng {line_number}: question phải là chuỗi không rỗng")

            image_path = image_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh ở dòng {line_number}: {image_path}")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            items.append({
                "image_name": image_name,
                "image": np.ascontiguousarray(rgb[:, :, ::-1]),
                "question": question,
                "expected_answer": row.get("expected_answer"),
                "manual_scores": _validate_manual_scores(row.get("manual_scores")),
            })
    if not items:
        raise ValueError("File JSONL không có request hợp lệ")
    return items


def _prompt_manual_scores(
    question: str,
    expected_answer: Optional[str],
    answer: str,
    input_fn: Callable[[str], str]
) -> Dict[str, Optional[int]]:
    print(f"\nCâu hỏi: {question}")
    print(f"Đáp án mong đợi: {expected_answer or '(không cung cấp)'}")
    print(f"Câu trả lời backend: {answer}")
    scores: Dict[str, Optional[int]] = {}
    for criterion in MANUAL_CRITERIA:
        while True:
            raw = input_fn(f"{criterion} [0-2, Enter để bỏ qua]: ").strip()
            if not raw:
                scores[criterion] = None
                break
            if raw in ("0", "1", "2"):
                scores[criterion] = int(raw)
                break
            print("Điểm phải là 0, 1, 2 hoặc để trống.")
    return scores


def run_benchmark(
    image_dir: Path,
    questions_path: Path,
    output_path: Path,
    backend_name: str = "legacy_caption",
    device: str = "mps",
    warmup: int = 1,
    repeats: int = 1,
    interactive_score: bool = False,
    backend_factory: Optional[Callable[[], VQABackend]] = None,
    input_fn: Callable[[str], str] = input
) -> Dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup phải >= 0")
    if repeats < 1:
        raise ValueError("repeats phải >= 1")

    dataset = load_dataset(image_dir, questions_path)
    factory = backend_factory or (lambda: create_vqa_backend(
        backend_name,
        device=device,
        caption_model_name="Salesforce/blip-image-captioning-base",
        translation_model_name="Helsinki-NLP/opus-mt-en-vi",
        lazy_load=False,
    ))

    load_started = time.perf_counter()
    backend = factory()
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    if backend is None:
        raise ValueError(f"Backend '{backend_name}' không khả dụng cho benchmark")

    warmup_errors = 0
    warmup_item = dataset[0]
    for _ in range(warmup):
        try:
            backend.answer(warmup_item["image"], warmup_item["question"])
        except Exception:
            warmup_errors += 1

    results = []
    latencies = []
    error_count = 0
    for item in dataset:
        for repetition in range(1, repeats + 1):
            started = time.perf_counter()
            answer = ""
            error = None
            try:
                answer = backend.answer(item["image"], item["question"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                error_count += 1
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)

            scores = item["manual_scores"]
            if interactive_score and error is None:
                scores = _prompt_manual_scores(
                    item["question"], item["expected_answer"], answer, input_fn
                )
            results.append({
                "image": item["image_name"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "answer": answer,
                "repetition": repetition,
                "latency_ms": latency_ms,
                "error": error,
                "manual_scores": scores,
            })

    total = len(results)
    report = {
        "backend": backend_name,
        "baseline_backend": "legacy_caption",
        "offline": is_offline_mode(),
        "model_load_time_ms": model_load_ms,
        "warmup_runs": warmup,
        "warmup_errors": warmup_errors,
        "repeats": repeats,
        "summary": {
            "dataset_requests": len(dataset),
            "total_runs": total,
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "error_count": error_count,
            "error_rate": error_count / total if total else 0.0,
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Thư mục ảnh benchmark")
    parser.add_argument("--questions", required=True, type=Path, help="File JSONL câu hỏi")
    parser.add_argument("--backend", default="legacy_caption", help="Tên VQA backend")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/vqa_benchmark.json"))
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--interactive-score", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(
            image_dir=args.images,
            questions_path=args.questions,
            output_path=args.output,
            backend_name=args.backend,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            interactive_score=args.interactive_score,
        )
    except Exception as exc:
        print(f"[FAIL] Benchmark VQA: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Đã ghi report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
