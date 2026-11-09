#!/usr/bin/env python3
"""Chạy app.py thật trên một video (không GUI), TTS ghi log thay vì phát loa.

Kịch bản: t=20 s OCR, t=35 s VQA (câu mặc định của phím Q), t=60 s về Cảnh báo; ghi FPS,
latency, RAM (ru_maxrss), mọi câu nói và các luồng còn sống khi tắt.
Video dùng trong báo cáo: ghép 2585 frame TUM fr3_long_office_household thành MP4 30 fps.

Chạy: python eval_app_e2e.py VIDEO.mp4 [--out app_e2e.json]
"""
import json
import os
import resource
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.audio import tts_engine  # noqa: E402

T0 = time.monotonic()
LOG = []
_speaking_until = [0.0]


def speak(self, text):
    # giả lập thời lượng nói ~ 0.07 s/ký tự như giọng Linh @200 wpm
    LOG.append((round(time.monotonic() - T0, 1), text))
    _speaking_until[0] = time.monotonic() + min(8.0, 0.06 * len(text))
    return True


tts_engine.TTSEngine.speak = speak
tts_engine.TTSEngine.play_sound = lambda self, f: True
tts_engine.TTSEngine.stop = lambda self: _speaking_until.__setitem__(0, 0.0)
tts_engine.TTSEngine.is_speaking = lambda self: time.monotonic() < _speaking_until[0]
tts_engine.TTSEngine.wait_until_done = lambda self, timeout=None: True

from src.runtime import system_coordinator as sc  # noqa: E402
from src.ui.app_runner import DEFAULT_VQA_QUESTION  # noqa: E402

holder = {}
_orig_init = sc.SystemCoordinator.__init__


def init(self, *a, **k):
    _orig_init(self, *a, **k)
    holder["c"] = self


sc.SystemCoordinator.__init__ = init

samples = []


def driver():
    while "c" not in holder:
        time.sleep(0.1)
    c = holder["c"]
    plan = {20: ("OCR", c.trigger_ocr), 35: ("VQA", lambda: c.trigger_vqa(DEFAULT_VQA_QUESTION)),
            60: ("WARN", c.enter_warning_mode)}
    start = time.monotonic()
    done = set()
    while time.monotonic() - start < 88:
        el = int(time.monotonic() - start)
        for t, (name, fn) in plan.items():
            if el >= t and t not in done:
                done.add(t)
                LOG.append((round(time.monotonic() - T0, 1), f"<<{name}: {fn()}>>"))
        s = c.get_runtime_snapshot()
        m = s["metrics"]
        samples.append((el, s["mode"], m.get("fps"), m.get("detection_ms", {}).get("p95") if isinstance(m.get("detection_ms"), dict) else None,
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9))
        time.sleep(1.0)


threading.Thread(target=driver, daemon=True).start()
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
sys.argv = ["app.py", "--source", sys.argv[1], "--no-gui", "--duration", "90"]
import runpy  # noqa: E402

t_stop = None
try:
    runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
finally:
    t_stop = time.monotonic() - T0
print("\n===== CÂU NÓI =====")
for t, x in LOG:
    print(f"t={t:>5}  {x}")
print("\n===== MẪU MỖI 5 s (giây, mode, fps, det_p95, RSS GB) =====")
for s in samples[::5]:
    print(s)
c = holder["c"]
print("\nmetrics cuối:", c.metrics.get_summary())
print("thread còn sống:", [t.name for t in threading.enumerate() if t.name != "MainThread"])
print("dừng ở t =", round(t_stop, 1))
if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"spoken": [{"t": t, "text": x} for t, x in LOG],
                   "samples": [dict(zip(("t", "mode", "fps", "det_p95", "rss_gb"), s)) for s in samples],
                   "final_metrics": c.metrics.get_summary(),
                   "threads_alive_after_stop": [t.name for t in threading.enumerate() if t.name != "MainThread"],
                   "stopped_at_sec": t_stop}, f, ensure_ascii=False, indent=1)
