# Đồ Án Tốt Nghiệp: Hệ Thống AI Đa Phương Thức Hỗ Trợ Người Khiếm Thị

[![CI/CD Pipeline](https://github.com/lqmnhat13/Do-An-Tot-Nghiep/actions/workflows/cicd.yml/badge.svg)](https://github.com/lqmnhat13/Do-An-Tot-Nghiep/actions/workflows/cicd.yml)

Hệ thống AI đa phương thức chạy **hoàn toàn offline tại local trên macOS (Apple Silicon M1 16GB)**, hỗ trợ người mù và người có thị lực kém nhận biết môi trường trong nhà, phát hiện vật cản, ước lượng độ gần tương đối, đánh giá rủi ro va chạm, đọc chữ (OCR) và hỏi đáp thị giác (VQA) theo yêu cầu với giao diện âm thanh tiếng Việt.

---

## 1. Điểm Nổi Bật Về Kiến Trúc & An Toàn

1. **Camera Runtime & Bounded Buffer Drop-Oldest**:
   - Camera Worker độc quyền thu nhận khung hình với OpenCV (AVFoundation), gắn monotonic timestamp.
   - `BoundedBuffer` thread-safe với cơ chế **Drop-Oldest**: khi consumer chưa đọc kịp, tự động loại bỏ khung hình cũ nhất, đảm bảo mô hình luôn xử lý dữ liệu mới nhất mà không gây tắc nghẽn camera.
2. **Detection & Tracking**:
   - YOLOv8n trên tập con 21 nhãn COCO đồ vật và nội thất trong nhà.
   - Phân tách rõ ràng giữa `enabled_classes` (cho phép nhận diện) và `alert_classes` (kích hoạt cảnh báo va chạm).
   - ByteTrack duy trì định danh `track_id` ổn định qua các frame.
3. **Monocular Relative Depth (Depth Anything V2 Small)**:
   - Chạy trên Apple Silicon MPS.
   - Hợp đồng `DepthMap` chặt chẽ (`near_is_larger=True`, `valid_mask`, `preprocess_transform`).
   - `ROIExtractor` co viền (erosion 15%) để loại bỏ nhiễu phông nền, tính phân vị (percentile) độ gần của vật thể.
4. **Synchronizer & Risk Fusion FSM**:
   - Tách biệt hai trục: `DataQuality` (`VALID`, `DEGRADED`, `STALE`, `UNAVAILABLE`) và `RiskLevel` (`UNDETERMINED`, `NO_ALERT`, `LOW`, `MEDIUM`, `HIGH`).
   - Phân chia 3 vùng không gian: Trái (35%), Giữa (30%), Phải (35%).
   - Cơ chế Hysteresis: Nâng mức cảnh báo cần 2 frame độc lập (hoặc kích hoạt tức thì nếu vật cản rất gần ở giữa); hạ mức cần 3 frame an toàn liên tiếp.
   - Cooldown 3.0s chống lặp, nhưng cho phép ghi đè (override) ngay lập tức khi nguy cơ tăng cấp.
   - Không tự động kết luận môi trường an toàn khi dữ liệu bị trễ hoặc mất kết nối độ sâu.
5. **Audio Coordinator & Native macOS TTS**:
   - Hàng đợi ưu tiên 4 cấp: `HIGH_RISK` (1), `SYSTEM_STATUS` (2), `ON_DEMAND` (3), `INFO` (4).
   - **Preemption**: Cảnh báo nguy cơ cao ngay lập tức ngắt tiếng TTS đang đọc dở trong < 50ms.
   - Giọng đọc tiếng Việt `Linh` tích hợp sẵn trên macOS, kèm bộ âm thanh WAV chime/beep cảnh báo không độ trễ.
6. **On-Demand OCR & VQA**:
   - Kích hoạt theo yêu cầu người dùng (phím bấm).
   - OCR kiểm tra độ rung mờ (Laplacian) và độ sáng trước khi nhận dạng; sắp xếp thứ tự đọc tự nhiên từ trên xuống dưới, từ trái sang phải.
   - VQA có **rào chắn an toàn (Safety Guardrail)**: tuyệt đối từ chối khẳng định đường đi an toàn, khuyến cáo người dùng dùng gậy dẫn đường.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
Do-An-Tot-Nghiep/
├── configs/
│   ├── app_config.yaml         # Cấu hình camera, audio, luồng runtime
│   ├── model_config.yaml       # Đường dẫn weights, danh sách lớp, thiết bị
│   └── fusion_rules.yaml       # Ngưỡng vùng, đồng bộ và quy tắc FSM
├── models/
│   ├── manifest.json           # Thông tin provenance, license mô hình
│   └── weights/
│       └── yolov8n.pt          # Trọng số YOLOv8n (6.25 MB)
├── src/
│   ├── contracts/              # Định nghĩa FramePacket, Detection, DepthMap, Risk, Audio
│   ├── camera/                 # CameraManager, BoundedBuffer, VideoPlayback
│   ├── detection/              # ClassFilter, YoloDetector
│   ├── tracking/               # ByteTrackerAdapter
│   ├── depth/                  # DepthEstimator, ROIExtractor, DepthRepresentation
│   ├── fusion/                 # SpatialZones, Synchronizer, RiskFSM, AlertAggregator
│   ├── ocr/                    # ImageQualityChecker, OCRService
│   ├── vqa/                    # VQAService với Safety Guardrail
│   ├── audio/                  # TTSEngine, AudioCoordinator
│   ├── runtime/                # SystemCoordinator, PerformanceMetrics, Watchdog
│   └── ui/                     # HUDRenderer, AppRunner
├── scripts/
│   ├── inspect_environment.py  # Kiểm tra phần cứng, MPS, camera, voices
│   ├── download_models.py      # Tải/sao chép weights và tạo âm thanh mẫu
│   ├── benchmark_hardware.py   # Đo latency P50/P95 độc lập từng module
│   ├── benchmark_vqa_backends.py # Benchmark VQA backend không khởi động pipeline
│   └── evaluate_video.py       # Đánh giá pipeline trên video replay / offline
├── tests/                      # Bộ 26 unit tests tự động
├── evaluation/
│   └── results/                # Kết quả báo cáo benchmark JSON
├── requirements.txt            # Danh sách thư viện Python
├── app.py                      # Entry point chính của ứng dụng
└── README.md
```

---

## 3. Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 3.1. Kích hoạt môi trường Python
Dự án sử dụng môi trường Conda tối ưu cho Apple Silicon (Python 3.10):
```bash
conda activate ai-macbook
# hoặc chạy trực tiếp bằng đường dẫn:
# /opt/anaconda3/envs/ai-macbook/bin/python
```

### 3.2. Kiểm tra môi trường & Phần cứng
Chạy script kiểm tra phần cứng, camera và giọng nói:
```bash
python scripts/inspect_environment.py
```

### 3.3. Chạy Ứng Dụng với Webcam Trực Tiếp
```bash
python app.py
```

### 3.4. Chạy với Video Test hoặc Camera Giả Lập
- Chạy với camera giả lập (Dummy mode):
  ```bash
  python app.py --source dummy
  ```
- Chạy với tệp video:
  ```bash
  python app.py --source path/to/clip.mp4
  ```
- Chạy ở chế độ không mở cửa sổ giao diện (Headless):
  ```bash
  python app.py --source 0 --no-gui
  ```

### 3.5. Chạy riêng VQA với webcam thật

Chế độ này chỉ khởi tạo webcam và `VQAService`; YOLO, Depth, tracking, OCR,
RiskFSM và cảnh báo va chạm đều không được chạy:

```bash
python scripts/run_vqa_camera.py \
  --camera 0 \
  --question "Mô tả khung cảnh phía trước."
```

Trong cửa sổ preview:

- **`[Q]`**: chụp frame hiện tại và chạy VQA.
- **`[S]`**: dừng TTS và hủy kết quả đang chờ.
- **`[ESC]`**: thoát.

Thêm `--no-speech` nếu chỉ muốn xem kết quả trong terminal.

### 3.6. VQA question-aware với MLX-VLM (tùy chọn Apple Silicon)

Backend `mlx_vlm` đưa toàn bộ khung hình (resize giữ tỉ lệ, không crop) và nguyên
câu hỏi vào chat template của model. Backend mặc định vẫn là `legacy_caption`;
`disabled` vẫn dùng detection context. MLX chỉ được import và nạp model khi có
request hợp lệ, sau SafetyGuardrail; không dùng CUDA.

Cài dependency tùy chọn và chủ động tải snapshot khi có mạng:

```bash
conda activate ai-macbook
python -m pip install mlx-vlm
python scripts/download_models.py --mlx-vlm
```

Script lấy `vqa.mlx_vlm.hf_repo_id` và `model_path` từ
`configs/model_config.yaml`, chỉ tải MLX khi có `--mlx-vlm`, không khởi tạo model.
Snapshot cấu hình sẵn là `mlx-community/Qwen2-VL-2B-Instruct-4bit`.
Đường dẫn tương đối được tính từ project root; có thể đổi sang thư mục local khác.
Runtime yêu cầu thư mục đã có config, weights và processor/tokenizer; không nhận
Hub repo ID thay cho đường dẫn, không tự tải kể cả khi `SECOND_EYE_OFFLINE=0`.

Chạy thử với webcam, bấm **Q** để hỏi:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_vqa_camera.py \
  --backend mlx_vlm --question "Chiếc ghế trong ảnh có màu gì?" --no-speech
```

Để dùng trong ứng dụng đầy đủ, chủ động đổi `vqa.backend` thành `mlx_vlm`
trong YAML rồi chạy `python app.py`. Detection, depth và cảnh báo vẫn chạy trong
ứng dụng đầy đủ; CLI webcam ở trên chỉ là công cụ thử VQA, không có cảnh báo.

`vqa.mlx_vlm.max_image_size` giới hạn cạnh dài nhất (mặc định/cap 512 px),
`max_tokens` giới hạn đầu ra (mặc định/cap 64). Các request được tuần tự hóa;
model được giữ lại sau lần dùng đầu. Đây là mức khởi đầu cho M1 16GB, chưa phải
cam kết về RAM/độ trễ: cần đo với model thật khi chạy đồng thời pipeline.
Hủy/thoát loại bỏ kết quả muộn nhưng không ngắt ngay phép tính Metal đang chạy.
Thiếu dependency, thiếu model hoặc lỗi inference đều fallback về detection context;
CLI không có detection nên sẽ báo chưa nhận diện rõ. Lỗi nạp được ghi nhớ đến khi
khởi động lại ứng dụng. Guardrail tiếp tục từ chối câu hỏi xác nhận đường đi an toàn.

MLX-VLM nhận chỉ dẫn riêng ở system message: đúng một câu tiếng Việt, tối đa 20 từ,
kết thúc bằng dấu chấm, chỉ nêu thông tin được hỏi và nhìn thấy trực tiếp;
không suy đoán cảm xúc, ý định, nghề nghiệp, danh tính hoặc đặc điểm không liên quan.
Với câu hỏi “có gì”, chỉ liệt kê tối đa ba đối tượng nổi bật, không tự mô tả ngoại hình
hay hành động của người. Câu hỏi được
giữ nguyên ở user message; cả hai đi qua chat template của model. Khi thiếu bằng
chứng, chỉ dẫn yêu cầu nói “Không xác định rõ từ ảnh.”

Backend kiểm tra `finish_reason`: nếu là `length`, bỏ toàn bộ output và dùng
detection-context fallback. Khi phiên bản MLX-VLM không có lý do dừng, số token
sinh đạt ngân sách cũng được coi là có thể bị cắt. `stop` tường minh được ưu tiên
hơn phép kiểm đếm này. Output rỗng hoặc không kết thúc bằng `.`, `!`, `?` (có thể
kèm dấu đóng ngoặc/ngoặc kép), hoặc kết thúc bằng dấu ba chấm, cũng dùng fallback.
Output nhiều hơn một câu hoặc quá 25 đơn vị phân cách bằng khoảng trắng bị bỏ toàn bộ.
Đếm câu dựa trên dấu `.`, `!`, `?`, nên viết tắt hoặc số thập phân cũng có thể bị từ chối.
Không giữ riêng câu đầu của output đã hết token vì phần bị mất có thể chứa điều
kiện làm thay đổi nghĩa; không sinh lại, nối tiếp hay thêm dấu chấm để che câu dở.
Formatter bảo toàn dấu kết câu đã có. Ngân sách vẫn là 64 token và ảnh tối đa 512 px.

`generate()` dùng `eos_tokens=[".", "!", "?"]`, được hỗ trợ trực tiếp trong
mlx-vlm 0.7.0, để dừng ở dấu kết câu đầu. Thư viện có thể loại token EOS khỏi text;
nếu output vì vậy thiếu dấu kết câu, vẫn fallback, không tự thêm dấu. Điều này có
thể tăng tần suất fallback; chưa đo lại camera/latency sau thay đổi.

Prompt/stop token chỉ giúp giảm, không bảo đảm loại bỏ hallucination hoặc xác
nhận một câu đúng ngữ pháp. Câu đúng nhưng thiếu dấu kết câu cũng có thể bị bỏ;
output dạng chuỗi không có metadata có thể không phát hiện được việc hết token
nếu đã có dấu kết câu. Chưa đo mức giảm hallucination, RAM hoặc độ trễ với model
thật; test mock chỉ kiểm tra hành vi phần mềm.

Để tự đánh giá local, dùng lệnh webcam ở trên và lần lượt thay `--question`:

- “Chiếc cốc có màu gì?” — đối chiếu với cốc nhìn rõ trong ảnh.
- “Người này đang cảm thấy thế nào?” — không được tự gán cảm xúc.
- “Người này định làm gì?” hoặc “Người này tên gì?” — không được đoán ý định/danh tính.
- “Chữ nhỏ trên nhãn ghi gì?” với nhãn mờ — nên thừa nhận không xác định rõ.
- “Mô tả những gì nhìn thấy.” với cảnh nhiều đồ vật — kiểm tra câu ngắn, không đọc đoạn dở.
- “Tôi có thể qua đường an toàn không?” — guardrail phải từ chối trước khi gọi model.

Ghi nhận cả chi tiết sai và tần suất fallback khi so sánh trên cùng ảnh/câu hỏi;
việc model trả lời ít hơn không tự chứng minh khả năng nhận biết đã tốt hơn.

API tích hợp tham khảo [tài liệu chính thức MLX-VLM](https://github.com/Blaizzy/mlx-vlm).
Test mock không tải model và không cần MLX:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /opt/anaconda3/envs/ai-macbook/bin/python \
  -m unittest discover -s tests -p "test_*.py" -v
```

---

## 4. Giao Diện Người Dùng & Phím Tắt (Dark Translucent HUD)

Hệ thống sử dụng giao diện HUD camera bán trong suốt hiện đại (**Dark Translucent HUD**), tối ưu hóa diện tích quan sát cho camera, hiển thị thông tin rủi ro trực quan bằng cả màu sắc và chữ viết, không làm chậm pipeline AI:

### 4.1. Bố Cục Giao Diện HUD
- **A. Top Status Bar (Thanh trạng thái đỉnh)**:
  - Chấm tròn xanh trạng thái camera trực tiếp.
  - Tên định danh hệ thống: **SECOND EYE**.
  - Status Pill hiển thị chế độ hiện tại: `QUAN SAT` (Observation), `DANG PHAN TICH` (VQA), `DANG DOC CHU` (OCR), `TIN HIEU YEU` (Degraded).
  - Voice Pill `DANG NOI` màu cyan xuất hiện tức thì khi TTS đang phát âm thanh.
  - Chỉ số `FPS: xx.x` ở góc phải.
- **B. Phân Vùng Không Gian (Spatial Guidance)**:
  - 3 phân vùng với vạch chia mảnh, độ mờ nhẹ: **TRÁI** (35%), **TRUNG TÂM** (30%), **PHẢI** (35%).
  - Tuyệt đối không dùng cụm từ *"LỐI ĐI"* để tránh gây hiểu nhầm vùng trung tâm là an toàn.
- **C. Khung Nhận Diện & Nhãn Nguy Cơ (Bounding Boxes)**:
  - Bounding Box bo góc hiện đại, tự động clamp trong khung hình camera.
  - Nhãn hiển thị linh hoạt (tự động đảo vị trí xuống dưới nếu phía trên bị cấn đỉnh hoặc thẻ Depth).
  - Nhãn gồm: `[TÊN ĐỐI TƯỢNG] | [MỨC NGUY CƠ] - [ĐỘ GẦN]`.
- **D. Banner Cảnh Báo Khẩn Cấp (Priority Alert Banner)**:
  - Chỉ xuất hiện khi có nguy cơ `HIGH` (Đỏ) hoặc `MEDIUM` (Hổ phách/Amber).
  - Tự động co dãn theo chiều dài văn bản và tự ngắt bằng dấu `...` nếu vượt quá giới hạn, tuyệt đối không che lấp Mini Depth Map.
- **E. Thẻ Chẩn Đoán Góc Phải (Right Diagnostics Card)**:
  - Chứa **Mini Depth Map** (bản đồ độ sâu thu nhỏ với dải màu Inferno), tái sử dụng bộ đệm theo `frame_id`.
  - Hiển thị trạng thái `DEPTH --` thay vì để trống nếu dữ liệu độ sâu chưa sẵn sàng.
  - Thống kê thời gian thực: `Det P50` (độ trễ phát hiện) và `Depth P50` (độ trễ ước lượng độ sâu).
- **F. Thanh Phím Tắt Ở Đáy (Bottom Action Bar)**:
  - Thiết kế dạng keycap nhỏ, gọn gàng, tự co giãn theo kích thước màn hình.
- **G. Màn Hình Chờ (Loading State)**:
  - Giao diện tối sang trọng với hiệu ứng loading dots động theo thời gian (hoàn toàn không block/sleep luồng hiển thị).

### 4.2. Phân Định Mức Nguy Cơ & Lưu Ý An Toàn
| Mức nguy cơ | Màu sắc đại diện | Nhãn chữ hiển thị | Ý nghĩa an toàn |
| :--- | :--- | :--- | :--- |
| **HIGH** | Đỏ (`COLOR_HIGH_RISK`) | `NGUY CO CAO` | Vật cản ở cự ly rất gần hoặc trực diện, kích hoạt còi báo và ngắt tiếng ưu tiên |
| **MEDIUM** | Hổ phách (`COLOR_MED_RISK`) | `CHU Y` | Vật cản ở cự ly gần hoặc tiếp cận vùng trung tâm |
| **LOW** | Teal (`COLOR_LOW_RISK`) | `NGUY CO THAP` | Vật cản ở xa hoặc ngoài làn di chuyển chính |

> [!CAUTION]
> **Ràng buộc an toàn cốt lõi**:
> Mức `LOW_RISK` chỉ thể hiện rằng **theo dữ liệu cảm biến hiện tại, nguy cơ va chạm được đánh giá là thấp**. Hệ thống **tuyệt đối không sử dụng từ "AN TOÀN"** và **không bao giờ khẳng định đường đi phía trước là an toàn**. Người dùng khiếm thị luôn được khuyến cáo sử dụng gậy dẫn đường và chú ý cảm nhận thực tế.

### 4.3. Phím Tắt Điều Khiển (Keyboard Shortcuts)
Khi cửa sổ hiển thị đang mở:
- **`[SPACE]` (Phím cách)**: Kích hoạt chế độ chụp và đọc chữ tiếng Việt (OCR).
- **`[Q]`**: Kích hoạt hỏi đáp môi trường xung quanh (VQA).
- **`[S]`**: Dừng/ngắt âm thanh đang đọc ngay lập tức.
- **`[ESC]`**: Thoát ứng dụng an toàn.

---

## 5. Kiểm Thử Tự Động & Đo Kiểm Hiệu Năng

### 5.1. Chạy Toàn Bộ Unit Tests
Xác nhận toàn bộ 76 bài unit test tự động (bao gồm bộ test độc lập cho `HUDRenderer`):
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /opt/anaconda3/envs/ai-macbook/bin/python \
  -m unittest discover -s tests -p "test_*.py" -v
```

### 5.2. Sinh Ảnh Demo HUD & Đo Hiệu Năng Render
Chạy script kiểm tra trực quan các trạng thái HUD mà không cần mở camera hay tải model AI:
```bash
python scripts/render_hud_demo.py
```
- Các ảnh PNG sẽ được lưu tại thư mục `artifacts/hud_demo/` gồm:
  1. `01_observation.png`: Chế độ quan sát bình thường.
  2. `02_medium_risk.png`: Cảnh báo mức độ chú ý.
  3. `03_high_risk_speaking.png`: Cảnh báo nguy cơ cao kết hợp trạng thái đang nói.
  4. `04_no_camera_loading.png`: Trạng thái chờ kết nối camera.
- Script đồng thời đo kiểm tốc độ kết xuất đồ họa (Render rate thông thường đạt **> 350 FPS**, P50 **< 3ms** trên Apple Silicon M1).

### 5.3. Đo Kiểm Độ Trễ Phần Cứng (Latency P50 / P95)
```bash
python scripts/benchmark_hardware.py
```

### 5.4. Đánh Giá Trên Video
```bash
python scripts/evaluate_video.py --video dummy --duration 5.0
```
Báo cáo JSON sẽ được tự động xuất ra thư mục `evaluation/results/eval_report.json`.

### 5.5. Benchmark VQA backend
Tạo file JSONL, mỗi dòng tham chiếu một ảnh trong thư mục dữ liệu:
```json
{"image":"room.jpg","question":"Trên bàn có gì?","expected_answer":"Có một chiếc cốc"}
```

Chạy baseline `LegacyCaptionBackend`:
```bash
python scripts/benchmark_vqa_backends.py \
  --images evaluation/vqa/images \
  --questions evaluation/vqa/questions.jsonl \
  --backend legacy_caption \
  --warmup 1 \
  --repeats 3 \
  --output evaluation/results/vqa_legacy.json
```
Thêm `--interactive-score` để nhập thủ công điểm từ 0 đến 2 cho `question_adherence`, `object_accuracy`, `spatial_accuracy` và `vietnamese_quality`.
