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

---

## 4. Phím Tắt Điều Khiển (Keyboard Shortcuts)

Khi cửa sổ hiển thị đang hoạt động, sử dụng các phím tắt sau:
- **`[SPACE]` (Phím cách)**: Kích hoạt chế độ chụp và đọc chữ tiếng Việt (OCR).
- **`[Q]`**: Kích hoạt hỏi đáp môi trường xung quanh (VQA).
- **`[S]`**: Dừng/ngắt âm thanh đang đọc ngay lập tức.
- **`[ESC]`**: Thoát ứng dụng an toàn.

---

## 5. Kiểm Thử Tự Động & Đo Kiểm Hiệu Năng

### 5.1. Chạy Toàn Bộ Unit Tests (26 bài test)
```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py" -v
```

### 5.2. Đo Kiểm Độ Trễ Phần Cứng (Latency P50 / P95)
```bash
python scripts/benchmark_hardware.py
```

### 5.3. Đánh Giá Trên Video
```bash
python scripts/evaluate_video.py --video dummy --duration 5.0
```
Báo cáo JSON sẽ được tự động xuất ra thư mục `evaluation/results/eval_report.json`.
