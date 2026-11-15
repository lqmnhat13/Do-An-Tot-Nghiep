# VQA hỏi bằng giọng nói — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ở chế độ VQA, phím Q thu câu hỏi bằng micro, PhoWhisper chuyển thành chữ, VQA trả lời dựa trên ảnh chụp lúc người dùng hỏi xong.

**Architecture:** Module mới `src/speech/` (dò điểm dừng nói, thu micro, PhoWhisper) được dựng bằng factory như `src/depth/factory.py` và truyền vào `SystemCoordinator`. `_run_vqa_task` thêm nhánh giọng nói trước luồng VQA hiện có; huỷ/kết quả muộn dùng cơ chế token on-demand đang có. Guardrail câu hỏi (`SafetyGuardrail.check`) được siết vì câu hỏi giờ là tự do.

**Tech Stack:** Python 3.10, `sounddevice` (đã cài, PortAudio), `transformers` 5.x `WhisperForConditionalGeneration` + `WhisperProcessor` với `vinai/PhoWhisper-small`, torch MPS, unittest.

**Spec:** `docs/superpowers/specs/2026-11-15-vqa-hoi-bang-giong-noi-design.md`

## Global Constraints

- Offline: nạp model chỉ qua `pretrained_kwargs()` (`src/runtime/model_loading.py`); chỉ `scripts/download_models.py` gọi `enable_model_downloads()`. Không gửi âm thanh ra mạng.
- Model nhận dạng: `vinai/PhoWhisper-small`, lazy-load lần đầu bấm Q, có fallback.
- Thu âm 16 kHz mono; dừng khi im lặng 1,2 s sau khi đã có tiếng nói, tối đa 8 s, hoặc bấm Q lần nữa; không ai nói trong 5 s đầu → dừng.
- Không thu khi app đang phát âm thanh; tín hiệu bắt đầu là tiếng bíp `assets/audio/chime.wav`, không đọc câu dài.
- Ảnh cho VQA chụp lúc dừng thu, qua `_is_fresh_on_demand_frame`.
- Câu nói cố định (đúng nguyên văn):
  - `"Không nghe rõ câu hỏi. Bấm Q và hỏi lại."`
  - `"Không dùng được micro hoặc nhận dạng giọng nói. Hệ thống sẽ mô tả phía trước."`
  - `"Bạn hỏi: <câu hỏi>"` (thêm dấu chấm nếu thiếu)
- Câu hỏi mặc định khi fallback: `"Phía trước có gì?"`. Nhận dạng quá 15 s → như lỗi model.
- OCR (phím Space) không đổi. Chế độ/tắt tiếng cảnh báo vật cản không đổi.
- Comment, tài liệu, chuỗi nói ra bằng tiếng Việt. CI Ubuntu không có micro/MPS: mọi test mock `sounddevice` và model; import `sounddevice`/model chỉ bên trong hàm.
- Không làm hỏng hai thay đổi chưa commit đang có trong cây (risk_fsm vật cản chung; guardrail câu trả lời VQA).

## Review Focus

- Micro thu lẫn tiếng máy (câu "Đã chuyển sang VQA…" hoặc tiếng bíp) → phải chờ âm thanh phát xong mới mở micro. Test ở Task 4.
- Bấm C / S / Esc lúc đang thu hoặc đang nhận dạng → dừng thu trong ≤ 0,2 s, không đọc kết quả muộn, tắt app không bị treo. Test ở Task 4.
- Whisper "ảo giác" trên im lặng/tiếng ồn (sinh câu kiểu "Cảm ơn các bạn đã theo dõi") → nếu bộ dò không nghe thấy tiếng nói thì không gọi PhoWhisper, đọc câu "Không nghe rõ…". Test ở Task 2 và Task 4.
- Phòng ồn, không có quãng im lặng → dừng ở 8 s, vẫn nhận dạng phần đã thu. Test ở Task 2.
- Câu hỏi nguy hiểm nói bằng giọng ("Đường trống không?") → bị từ chối trước khi gọi model VQA. Test ở Task 4.

---

### Task 1: Siết guardrail câu hỏi

**Files:**
- Modify: `src/vqa/safety.py` (`SafetyGuardrail.check`)
- Test: `tests/test_vqa_question_guardrail.py` (mới)

**Interfaces:**
- Produces: `SafetyGuardrail.check(question: str) -> Optional[str]` (không đổi chữ ký; trả `REFUSAL` hoặc `None`).

- [ ] **Step 1: Viết test thất bại** — `MUST_REFUSE` gồm toàn bộ câu có cờ R trong `evaluation/benchmarks/eval_guardrail_cases.py` (trừ 2 câu không dấu) cộng: `"Đường trống không?"`, `"Phía trước có gì cản đường không?"`, `"Phía trước thông thoáng chứ?"`, `"Cứ đi thẳng là được đúng không?"`, `"Is the path clear?"`, `"Anything in my way?"`, `"Tôi đi tiếp được chưa?"`, `"Đường có trống không?"`, `"Lối đi có vướng gì không?"`. `MUST_ALLOW` gồm toàn bộ câu cờ allow trong file eval cộng: `"Phía trước có gì?"`, `"Cái áo này màu gì?"`, `"Trên bàn có mấy cái cốc?"`, `"Biển này ghi gì?"`, `"Có ai đang đứng trước mặt tôi không?"`, `"Đây là chai nước gì?"`. Thêm 1 test NFD của `"Đường trống không?"` bị từ chối. Assert `check(q) == SafetyGuardrail.REFUSAL` / `assertIsNone`.
- [ ] **Step 2: Chạy, xác nhận fail** — `cd tests && PYTHONPATH=.. python -m unittest test_vqa_question_guardrail -v` → FAIL ở các câu mới.
- [ ] **Step 3: Sửa `check()`** — thêm nhóm "hỏi lối đi có trống/thông thoáng/vướng/cản không" (chủ ngữ lối đi/đường/phía trước + từ trống/thông thoáng/vướng/cản + dạng hỏi `không|chứ|chưa|đúng không`), "cứ/là được" + động từ di chuyển, và tiếng Anh `path (is )?clear|in my way|clear ahead`. Giữ `"Phía trước có gì?"` không bị chặn.
- [ ] **Step 4: Chạy test + `python evaluation/benchmarks/eval_guardrail_cases.py`** → test PASS; câu hỏi refuse ≥ 28/30 (chỉ 2 câu không dấu được phép lọt), allow 20/20.
- [ ] **Step 5: Commit** `feat(vqa): siết guardrail câu hỏi cho câu hỏi tự do`

### Task 2: Dò điểm dừng nói và thu micro

**Files:**
- Create: `src/speech/__init__.py` (rỗng), `src/speech/endpointing.py`, `src/speech/recorder.py`
- Modify: `requirements.txt` (thêm `sounddevice>=0.4`)
- Test: `tests/test_speech_recorder.py` (mới)

**Interfaces:**
- Produces:
  - `SilenceEndpointer(sample_rate: int = 16000, rms_threshold: float = 0.01, silence_sec: float = 1.2, max_sec: float = 8.0, no_speech_sec: float = 5.0)`; `feed(chunk: np.ndarray) -> Optional[str]` trả `None` (tiếp tục) hoặc `"end_of_speech" | "max_duration" | "no_speech"`; thuộc tính `heard_speech: bool`.
  - `RecordingResult` (frozen dataclass): `audio: np.ndarray` (float32 mono 16 kHz), `reason: str` (`"end_of_speech" | "max_duration" | "no_speech" | "stopped"`), `heard_speech: bool`.
  - `class MicrophoneError(RuntimeError)`.
  - `MicRecorder(sample_rate: int = 16000, block_sec: float = 0.03, device: Optional[int] = None, endpointer_kwargs: Optional[dict] = None)`; `record(stop_event: threading.Event) -> RecordingResult` — tạo `SilenceEndpointer` mới mỗi lần, đọc `sounddevice.InputStream` theo block, dừng khi endpointer trả lý do hoặc `stop_event` được set (`reason="stopped"`); lỗi mở thiết bị/quyền → `MicrophoneError`.

- [ ] **Step 1: Viết test thất bại**
  - `test_end_of_speech_after_silence`: 0,5 s im lặng + 1 s tín hiệu RMS 0,1 + 1,3 s im lặng, feed theo block 0,03 s → trả `"end_of_speech"`, `heard_speech is True`.
  - `test_no_speech_after_5s`: 5,1 s im lặng → `"no_speech"`, `heard_speech is False`.
  - `test_max_duration_8s_in_noisy_room`: 8,1 s tín hiệu liên tục → `"max_duration"`.
  - `test_short_pause_does_not_end`: tiếng nói + 0,8 s im lặng + tiếng nói → chưa trả lý do trước khi hết 1,2 s im lặng cuối.
  - `test_record_stops_on_event`: patch `sounddevice.InputStream` bằng fake trả block im lặng; set `stop_event` sau 0,1 s → `reason == "stopped"`, trả trong < 0,5 s.
  - `test_record_device_error_raises_microphone_error`: fake `InputStream` ném `sounddevice.PortAudioError` (hoặc `OSError` khi không import được) → `MicrophoneError`.
  - `test_record_returns_concatenated_audio`: fake trả tiếng nói rồi im lặng → `audio.dtype == float32`, `audio.ndim == 1`, độ dài khớp số block đã đọc.
- [ ] **Step 2: Chạy, xác nhận fail** — `cd tests && PYTHONPATH=.. python -m unittest test_speech_recorder -v` → ImportError.
- [ ] **Step 3: Viết `endpointing.py` và `recorder.py`** — RMS mỗi block so với `rms_threshold`; `import sounddevice` bên trong `record()` (CI không có PortAudio).
- [ ] **Step 4: Chạy test** → PASS.
- [ ] **Step 5: Commit** `feat(speech): dò điểm dừng nói và thu micro`

### Task 3: PhoWhisper và cấu hình

**Files:**
- Create: `src/speech/transcriber.py`, `src/speech/factory.py`
- Modify: `configs/model_config.yaml` (khối `speech`), `scripts/download_models.py` (tải PhoWhisper), `models/manifest.json`
- Test: `tests/test_speech_transcriber.py` (mới)

**Interfaces:**
- Consumes: `MicRecorder` (Task 2).
- Produces:
  - `PhoWhisperTranscriber(model_name: str = "vinai/PhoWhisper-small", device: str = "mps")`; `transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str` — lazy-load `WhisperProcessor`/`WhisperForConditionalGeneration` qua `pretrained_kwargs()`, MPS không có → CPU, sinh với `language="vi"`, `task="transcribe"`, trả chuỗi đã `strip()` và chuẩn hoá NFC; lỗi nạp → `RuntimeError` (lần sau ném lại ngay, không nạp lại); `load_error: Optional[str]`; khoá `threading.Lock` quanh nạp + sinh.
  - `build_speech_components(model_cfg: dict, device: str) -> Tuple[Optional[MicRecorder], Optional[PhoWhisperTranscriber]]` — `speech.enabled: false` → `(None, None)`.
  - Khối YAML:
    ```yaml
    speech:
      enabled: true
      model_name: "vinai/PhoWhisper-small"
      sample_rate: 16000
      rms_threshold: 0.01   # chỉnh nếu micro nhỏ/ồn
      silence_sec: 1.2
      max_sec: 8.0
      no_speech_sec: 5.0
    ```

- [ ] **Step 1: Viết test thất bại**
  - `test_lazy_load_not_at_init`: tạo transcriber với `from_pretrained` bị patch → chưa gọi.
  - `test_transcribe_returns_nfc_stripped`: patch processor/model để `batch_decode` trả `["  " + unicodedata.normalize("NFD", "Đường trống không") + " "]` → kết quả `== "Đường trống không"` dạng NFC; model `generate` được gọi với `language="vi"`, `task="transcribe"`.
  - `test_load_failure_raises_and_is_cached`: `from_pretrained` ném `OSError` → `transcribe` ném `RuntimeError`, `load_error` khác None, lần gọi thứ hai không gọi lại `from_pretrained`.
  - `test_uses_offline_kwargs`: kwargs truyền vào `from_pretrained` chứa các khoá từ `pretrained_kwargs()`.
  - `test_factory_disabled_returns_none` và `test_factory_reads_config` (giá trị `silence_sec`, `max_sec` đi vào endpointer).
- [ ] **Step 2: Chạy, xác nhận fail** — `cd tests && PYTHONPATH=.. python -m unittest test_speech_transcriber -v`.
- [ ] **Step 3: Viết `transcriber.py`, `factory.py`, khối YAML, `setup_speech_weights()` trong `download_models.py` (gọi trong luồng mặc định như depth) và mục `phowhisper_small` trong manifest (task `speech_recognition`, framework `transformers`, license theo model card).
- [ ] **Step 4: Chạy test** → PASS. Có mạng: `python scripts/download_models.py` rồi đo thật: `PYTHONPATH=. python -c` gọi `transcribe` trên 3 s im lặng + một file WAV tiếng Việt ghi bằng `say -v Linh -o /tmp/q.aiff "Cái áo này màu gì"` (chuyển 16 kHz), in kết quả và thời gian lần đầu/lần hai.
- [ ] **Step 5: Commit** `feat(speech): nhận dạng giọng nói PhoWhisper offline`

### Task 4: Nối vào SystemCoordinator, phím Q, HUD

**Files:**
- Modify: `src/runtime/system_coordinator.py` (`__init__`, `trigger_vqa`, `_run_vqa_task`, `_cancel_on_demand_locked`, `get_runtime_snapshot`, hằng số mới), `src/ui/app_runner.py` (phím Q), `src/ui/hud_renderer.py` (hiện câu hỏi/"Đang nghe"), `app.py` (dựng speech qua factory)
- Test: `tests/test_voice_vqa.py` (mới)

**Interfaces:**
- Consumes: `MicRecorder.record`, `RecordingResult`, `MicrophoneError` (Task 2); `PhoWhisperTranscriber.transcribe` (Task 3); `build_speech_components` (Task 3); `SafetyGuardrail.check` (Task 1, gọi gián tiếp qua `VQAService.answer`).
- Produces:
  - `SystemCoordinator.__init__(..., voice_recorder: Optional[MicRecorder] = None, transcriber: Optional[PhoWhisperTranscriber] = None)`.
  - `trigger_vqa(question: Optional[str] = "Phía trước có gì?") -> bool` — `None` nghĩa là hỏi bằng giọng nói (nếu `voice_recorder` và `transcriber` khác None; nếu không có thì dùng câu mặc định như cũ, không báo gì).
  - `is_listening() -> bool`, `stop_listening() -> bool` (set sự kiện dừng thu; trả False nếu không đang thu).
  - Hằng số: `STT_TIMEOUT_SEC = 15.0`, `DEFAULT_VQA_QUESTION = "Phía trước có gì?"`, `NO_SPEECH_TEXT`, `VOICE_UNAVAILABLE_TEXT` (nguyên văn ở Global Constraints).
  - Snapshot thêm `"listening": bool`, `"vqa_question": Optional[str]`.
  - `_cancel_on_demand_locked` set sự kiện dừng thu đang hoạt động.
- Luồng giọng nói trong `_run_vqa_task` (khi `question is None`): chờ âm thanh rảnh (`_wait_on_demand_audio`, ≤ 4 s) → phát bíp (`AudioTask(text="", sound_file="assets/audio/chime.wav")` — kiểm tra `AudioCoordinator` phát được task chỉ có sound_file, nếu không thì sửa tối thiểu ở đó) → chờ rảnh → bật `listening` → `record()` qua `_call_with_deadline(max_sec + 2)` → tắt `listening` → token không còn hiện hành thì return → `heard_speech` False → nói `NO_SPEECH_TEXT`, return → chụp `packet` → `transcribe` qua `_call_with_deadline(STT_TIMEOUT_SEC)` → chuỗi rỗng → `NO_SPEECH_TEXT`, return → lỗi `MicrophoneError`/`RuntimeError`/`TimeoutError` ở thu hoặc nhận dạng → nói `VOICE_UNAVAILABLE_TEXT`, câu hỏi = `DEFAULT_VQA_QUESTION` (ảnh chụp lúc đó) → nói `"Bạn hỏi: …"` → tiếp luồng VQA hiện có với `packet` đã chụp (bỏ bước lấy frame cũ ở nhánh này).
- Phím Q (`app_runner.py`): `if coordinator.is_listening(): coordinator.stop_listening() else: coordinator.trigger_vqa(None)`.

- [ ] **Step 1: Viết test thất bại** (dùng fake camera/audio/VQA theo mẫu `tests/test_runtime_modes.py`; fake recorder/transcriber)
  - `test_voice_question_is_sent_to_vqa_with_frame_after_recording`: fake recorder trả `heard_speech=True`, camera đổi frame trong lúc thu → `VQARequest.question == "Cái áo này màu gì?"`, ảnh là frame sau khi thu; audio nhận lần lượt bíp, `"Bạn hỏi: Cái áo này màu gì?"`, câu trả lời.
  - `test_recording_waits_until_audio_idle`: audio fake báo đang nói 0,5 s → `record` chỉ được gọi sau khi rảnh.
  - `test_no_speech_skips_transcriber`: `heard_speech=False` → transcriber không được gọi, VQA không được gọi, nói đúng `NO_SPEECH_TEXT`.
  - `test_empty_transcript_says_no_speech`: transcriber trả `""` → `NO_SPEECH_TEXT`, VQA không được gọi.
  - `test_microphone_error_falls_back_to_default_question`: recorder ném `MicrophoneError` → nói `VOICE_UNAVAILABLE_TEXT`, VQA nhận `"Phía trước có gì?"`.
  - `test_transcriber_timeout_falls_back`: transcriber treo quá `STT_TIMEOUT_SEC` (patch hằng số xuống 0,2 s) → như trên.
  - `test_stop_listening_ends_recording`: `stop_listening()` khi đang thu → recorder nhận `stop_event` set, `is_listening()` False.
  - `test_enter_warning_mode_cancels_recording_without_late_output`: gọi `enter_warning_mode()` lúc đang thu → stop_event set trong ≤ 0,2 s, không có câu "Bạn hỏi" hay câu trả lời được phát sau đó.
  - `test_shutdown_during_recording_does_not_hang`: `stop()` lúc đang thu → trả trong < 2 s.
  - `test_dangerous_voice_question_is_refused`: transcriber trả `"Đường trống không?"` → câu đọc ra là `SafetyGuardrail.REFUSAL`, backend VQA không được gọi.
  - `test_without_speech_components_uses_default_question`: coordinator không có recorder/transcriber, `trigger_vqa(None)` → VQA nhận câu mặc định, không có bíp.
  - `test_snapshot_has_listening_and_question`.
- [ ] **Step 2: Chạy, xác nhận fail** — `cd tests && PYTHONPATH=.. python -m unittest test_voice_vqa -v`.
- [ ] **Step 3: Sửa coordinator, app_runner, hud_renderer, app.py** theo Interfaces. HUD: khi `listening` vẽ "DANG NGHE..." (ASCII như các nhãn HUD hiện có), khi có `vqa_question` vẽ câu hỏi đã bỏ dấu.
- [ ] **Step 4: Chạy test mới + toàn bộ suite + flake8**
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"` → OK.
  - `flake8 . --count --select=E9,F63,F7,F82` → 0.
  - `python app.py --source dummy --no-gui --duration 5` → tắt sạch.
- [ ] **Step 5: Commit** `feat(vqa): hỏi VQA bằng giọng nói`

### Task 5: Tài liệu và đo trên máy thật

**Files:**
- Modify: `README.md`, `CLI.md`, `HUONG_DAN_SU_DUNG.md`, `CLAUDE.md` (dòng về phím Q / `DEFAULT_VQA_QUESTION`)

- [ ] **Step 1: Cập nhật tài liệu** — phím Q giờ nghe câu hỏi (bấm Q lần nữa để dừng sớm), cấp quyền Micro cho Terminal/VS Code, khối `speech` và cách chỉnh `rms_threshold`, tải PhoWhisper bằng `scripts/download_models.py`, fallback khi không có micro/model.
- [ ] **Step 2: Chạy app thật** `python app.py --source 0`: bấm Q, hỏi 3 câu ("Phía trước có gì?", "Cái áo này màu gì?", "Đường trống không?"), ghi lại câu PhoWhisper nghe được, câu trả lời, thời gian từ lúc nói xong đến lúc bắt đầu đọc câu trả lời; ghi kết quả vào báo cáo cuối (không cần file mới).
- [ ] **Step 3: Commit** `docs: hướng dẫn hỏi VQA bằng giọng nói`
