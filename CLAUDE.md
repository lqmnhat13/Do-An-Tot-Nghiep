# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Offline assistive app for visually impaired users, running locally on macOS Apple Silicon (target: M1, 16 GB). A camera feeds continuous obstacle detection + relative-depth estimation that drive Vietnamese audio warnings; OCR and VQA run only on user request. Docs, comments, config comments and all user-facing/spoken strings are in Vietnamese — keep new ones in Vietnamese. `AGENTS.md` holds the binding project rules (summarized below); `CLI.md` documents every script's flags; `README.md` explains behavior and safety limits.

## Commands

Environment: Conda env `ai-macbook` (Python 3.10). `mlx-vlm` is installed separately (`python -m pip install mlx-vlm`), not in `requirements.txt`.

```bash
# Full test suite (the verification command from AGENTS.md)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /opt/anaconda3/envs/ai-macbook/bin/python -m unittest discover -s tests -p "test_*.py" -v

# Single file / single test (unittest, run from repo root with PYTHONPATH=.)
PYTHONPATH=. python -m unittest tests.test_fusion -v
PYTHONPATH=. python -m unittest tests.test_fusion.TestFusion.test_synchronizer_stale_depth -v

# Lint as CI does (only E9,F63,F7,F82 fail the build)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Run the app (webcam 0 | video path | dummy); keys: Space=OCR, Q=VQA, C=back to warning mode, S=stop speech, Esc=quit
python app.py --source 0
python app.py --source dummy --no-gui --duration 5   # smoke test, no camera

# Pipeline smoke/eval used in CI
python scripts/evaluate_video.py --video dummy --duration 2.0 --device cpu --out evaluation/results/ci_eval_report.json

# Model preparation (needs network; do NOT set offline env vars for these)
python scripts/download_models.py
python scripts/download_models.py --mlx-vlm
python scripts/inspect_environment.py
```

Runtime is offline by default: `src/runtime/model_loading.py` treats `SECOND_EYE_OFFLINE` as `1` unless set otherwise, so every Hugging Face `from_pretrained` call must go through `pretrained_kwargs()` (→ `local_files_only`). Only download scripts call `enable_model_downloads()`.

CI (`.github/workflows/cicd.yml`) runs on Ubuntu with CPU torch, so tests must not require MPS, MLX, Apple Vision, a camera, or the `say` command — those are mocked/skipped in tests.

## Architecture

`app.py` loads the three YAML files in `configs/` (`app_config.yaml`, `model_config.yaml`, `fusion_rules.yaml`), builds every component, and hands them to `SystemCoordinator` (`src/runtime/system_coordinator.py`), which is the heart of the runtime. `AppRunner` (`src/ui/app_runner.py`) owns the OpenCV window/key loop and draws via `HUDRenderer`.

**Threads and data flow** (all owned by `SystemCoordinator` unless noted):
- `CameraManager` thread produces immutable `FramePacket`s and fans them out to two independent `BoundedBuffer`s with **drop-oldest** semantics (one for detection, one for depth), so slow consumers never queue stale frames.
- `DepthWorker` (~8 Hz, deliberately throttled to leave GPU time for detection): Depth Anything V2 → `DepthMap` → `Synchronizer.register_depth`.
- `DetectionWorker`: YOLOv8n → ByteTrack → `Synchronizer.synchronize` (pairs detection with depth, checks skew/age limits from `fusion_rules.yaml → synchronizer`, sets `DataQuality`) → `RiskFSM.update` → `_post_safety_alert` → `AlertAggregator` → `AudioCoordinator`.
- On-demand thread (one at a time, guarded by a token so cancelled/superseded tasks can't publish results): OCR (`src/ocr/`, Apple Vision with EasyOCR fallback, image-quality gate first) or VQA (`src/vqa/`, backend chosen by `create_vqa_backend`: `mlx_vlm` default, `legacy_caption` BLIP+MarianMT, `disabled`). VQA is lazy-loaded and falls back to detection context; `src/vqa/safety.py` guardrails reject claims that the path is safe.
- `AudioCoordinator` thread: priority queue (`HIGH_RISK` < `SYSTEM_STATUS` < `ON_DEMAND` < `INFO`), dedup window, expiry, interruption; speaks via macOS `say` (voice `Linh`) and WAVs in `assets/audio/`.
- `Watchdog`/`Metrics` in `src/runtime/` record heartbeats and latencies.

**Modes**: `_selected_mode` is `WARNING` | `OCR` | `VQA`; `_current_mode` (shown on HUD) adds `OBSERVATION`/`DEGRADED`. Detection, depth and risk assessment keep running in every mode, but obstacle audio is emitted only in `WARNING`. Choosing OCR/VQA mutes all obstacle alerts, including HIGH (`audio_coordinator.set_safety_alerts_enabled(False)`), until the user presses C; finishing an OCR/VQA task does not switch back. Alerts are gated *before* aggregation so muted modes don't consume cooldowns, and only assessments newer than `_warning_since` are alerted after re-entering WARNING.

**Risk rules** (`src/fusion/`): only classes in `model_config.yaml → detection.alert_classes` whose bbox overlaps the trapezoidal `walking_corridor` (≥ `min_overlap_ratio`, intersection / smaller area) and whose ROI depth (80th percentile of an eroded bbox, `src/depth/roi_extractor.py`) reaches `depth_threshold_high` produce audio. Depth is *relative* (normalized 0–1, larger = closer), never meters. MEDIUM/LOW are computed and drawn on the HUD but never trigger sound, banners or cooldown. The FSM de-escalates immediately when the bbox leaves the corridor or proximity drops; LEFT/CENTER/RIGHT (`spatial_zones.py`) only describe position. Stale/degraded data must never be interpreted as "safe".

**Contracts** (`src/contracts/`): frozen dataclasses (`FramePacket`, `DetectionResult`/`Detection`/`BoundingBox`, `DepthMap`, `RiskAssessment` with separate `DataQuality` and `RiskLevel`, `AudioTask`, OCR/VQA request/result) shared across modules. Timestamps are `time.monotonic()`.

## Rules (from AGENTS.md)

- Detection, depth and risk evaluation must keep running in all modes; obstacle alerts only in Warning mode; OCR/VQA strictly on-demand.
- Never use the VLM to assert that the path is safe.
- Don't change Detection, Depth or Risk contracts without updating every consumer.
- Preserve drop-oldest camera buffering. No cloud APIs or telemetry.
- Prefer Core ML / MLX / MPS; no CUDA-only dependencies in the main runtime; large models must lazy-load and have a fallback.
- For runtime changes, check for deadlocks, clean shutdown and thread safety (`_state_lock` guards coordinator state). For safety changes, add tests covering failures and stale data.
- Keep changes small and scoped; don't overwrite the user's uncommitted changes; report which tests were run and remaining limitations.
