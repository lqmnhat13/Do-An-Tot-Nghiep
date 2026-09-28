# CLI của dự án

Tài liệu này liệt kê các lệnh có thể chạy trực tiếp từ thư mục gốc dự án. Ví dụ dùng môi trường `ai-macbook` trên macOS Apple Silicon.

## Chuẩn bị

```bash
cd /Users/lenhat/Do-An-Tot-Nghiep
conda activate ai-macbook
python -m pip install -r requirements.txt
```

Nếu dùng VQA backend `mlx_vlm`, cài thêm `python -m pip install mlx-vlm` và chuẩn bị model bằng lệnh ở mục `download_models.py`. Sau khi model đã có sẵn, có thể đặt các biến sau để buộc runtime không truy cập mạng:

```bash
export SECOND_EYE_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1
```

Không dùng các biến offline khi **chủ động tải model**. Các đường dẫn tương đối bên dưới đều tính từ thư mục gốc dự án.

## CLI chính: ứng dụng đầy đủ

```bash
python app.py --source 0
```

`app.py` khởi tạo camera, detection, depth và cảnh báo liên tục; OCR/VQA chỉ chạy khi yêu cầu. Trong cửa sổ GUI: `Space` để OCR, `Q` để VQA, `S` để hủy/dừng đọc, `Esc` để thoát. VQA không được dùng để xác nhận đường đi an toàn.

| Tùy chọn | Ý nghĩa |
| --- | --- |
| `--source 0` | Webcam mặc định. Có thể thay `0` bằng đường dẫn video hoặc `dummy`. Nếu bỏ qua, lấy nguồn trong `configs/app_config.yaml`. |
| `--weights PATH` | Đường dẫn trọng số YOLO; mặc định lấy từ `configs/model_config.yaml`. |
| `--device mps\|cpu` | Thiết bị suy luận; nếu bỏ qua, lấy từ cấu hình ứng dụng. |
| `--no-gui` | Không mở cửa sổ; phím tắt GUI không khả dụng. |
| `--duration GIÂY` | Tự động dừng sau số giây chỉ định. |
| `--help` | Xem trợ giúp tham số. |

Ví dụ:

```bash
python app.py --source dummy --no-gui --duration 5
python app.py --source /duong/dan/video.mp4 --device mps
```

`dummy` chỉ là smoke test, không phải bằng chứng hoạt động tốt với camera thật.

## CLI phụ

### Kiểm tra OCR

```bash
python scripts/run_ocr_camera.py --image /duong/dan/anh.jpg --no-speech
python scripts/run_ocr_camera.py --camera 0
```

Tùy chọn: `--camera N`, `--image PATH`, `--width PIXEL`, `--height PIXEL`, `--device mps|cpu`, `--min-confidence SỐ`, `--no-speech`, `--help`. Khi dùng camera: `Space` để chụp/đọc, `S` để dừng đọc, `Esc` để thoát. `--image` xử lý ảnh tĩnh, không cần camera. Script này **không** chạy detection/depth/cảnh báo.

### Kiểm tra VQA

```bash
python scripts/run_vqa_camera.py --camera 0 --question "Mô tả khung cảnh phía trước."
python scripts/run_vqa_camera.py --camera 0 --backend mlx_vlm --no-speech
```

Tùy chọn: `--camera N`, `--width PIXEL`, `--height PIXEL`, `--device mps|cpu`, `--backend mlx_vlm|legacy_caption|disabled`, `--question TEXT`, `--no-speech`, `--help`. Nếu bỏ qua `--backend`, script dùng backend trong `configs/model_config.yaml`. Trong cửa sổ: `Q` để hỏi, `S` để hủy/dừng đọc, `Esc` để thoát. Script này chỉ nhận ảnh từ webcam, **không có `--image`** và **không** chạy detection/depth/cảnh báo.

### Chuẩn bị model và âm thanh

```bash
python scripts/download_models.py
python scripts/download_models.py --mlx-vlm
```

Lệnh đầu kiểm tra/tải YOLO, Depth Anything, BLIP/MarianMT, EasyOCR và tạo các WAV cảnh báo trong `assets/audio/`. Lệnh `--mlx-vlm` **chỉ** tải snapshot MLX-VLM được khai báo trong `configs/model_config.yaml`. Hai lệnh cần mạng khi model chưa có; lệnh đầu có thể ghi lại các WAV mẫu. Chỉ chạy khi cần chuẩn bị tài nguyên.

### Kiểm tra môi trường

```bash
python scripts/inspect_environment.py
```

In thông tin macOS/MPS, thư viện, giọng TTS, model và thử mở camera `0`; không có tham số riêng. Script có thể phát tiếng nói thử. Đường dẫn kiểm tra YOLO trong script được gắn cố định theo thư mục home, nên đối chiếu lại với `models/weights/` nếu báo thiếu.

### Đánh giá pipeline trên video

```bash
python scripts/evaluate_video.py --video /duong/dan/video.mp4 --duration 30 --out evaluation/results/eval_report.json
python scripts/evaluate_video.py --video dummy --duration 5
```

Tùy chọn: `--video PATH|dummy` (mặc định `dummy`), `--duration GIÂY` (mặc định `5`), `--device mps|cpu` (mặc định `mps`), `--out PATH` (mặc định `evaluation/results/eval_report.json`), `--help`. Tạo báo cáo JSON về frame và metric; giá trị trên `dummy` không đại diện cho độ chính xác với video thật.

### Benchmark VQA bằng bộ ảnh

```bash
python scripts/benchmark_vqa_backends.py \
  --images /duong/dan/thu-muc-anh \
  --questions /duong/dan/questions.jsonl \
  --backend legacy_caption \
  --output evaluation/results/vqa_benchmark.json
```

Mỗi dòng JSONL cần ít nhất `image` (tên file trong thư mục ảnh) và `question`; có thể thêm `expected_answer` và `manual_scores`. Tùy chọn: `--device mps|cpu`, `--warmup N` (mặc định `1`), `--repeats N` (mặc định `1`), `--interactive-score`, `--help`. Script đo latency và ghi JSON, không chạy camera/cảnh báo. Hiện tại script không truyền `model_path` vào `mlx_vlm`, nên **không dùng `--backend mlx_vlm`** để benchmark cho đến khi bổ sung cấu hình đường dẫn model.

### Benchmark phần cứng

```bash
python scripts/benchmark_hardware.py
```

Đo latency YOLO, Depth Anything và TTS trên thiết bị hiện tại; không có tham số CLI. Sử dụng ảnh ngẫu nhiên, vì vậy đây là phép đo hiệu năng, không phải đo độ chính xác.

### Tạo ảnh demo HUD

```bash
python scripts/render_hud_demo.py
```

Tạo ảnh HUD giả lập trong `artifacts/hud_demo/` và đo thời gian render; không cần camera và không có tham số CLI. Ảnh giả lập không phải bằng chứng về detection/depth trên cảnh thật.

## Lệnh chạy test

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
/opt/anaconda3/envs/ai-macbook/bin/python \
-m unittest discover -s tests -p 'test_*.py' -v
```

Đây là lệnh kiểm chứng trong `AGENTS.md`. Các unit test không thay thế việc thử với camera, ảnh và video thật.
