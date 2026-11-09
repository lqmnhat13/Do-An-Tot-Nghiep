# Kết quả đánh giá ngày 2026-11-09

Lượt kiểm thử nghiệm thu trên dữ liệu thật, trước và sau khi sửa 5 lỗi phát hiện được
(xem [../../../REAL_DATA_EVALUATION.md](../../../REAL_DATA_EVALUATION.md)). Kết quả của lượt
2026-10-23 vẫn nằm ở thư mục cha và `../baseline/`; không file cũ nào bị ghi đè.

## Môi trường

| Mục | Giá trị |
|---|---|
| Máy | Apple M1, 16 GB, macOS 27.0.1 |
| Python / env | 3.10, conda `ai-macbook` (torch 2.12.0, ultralytics 8.4.138, transformers 5.17.0, mlx-vlm 0.7.0, scipy 1.14.1, pyobjc 12.2.2) |
| Detection | YOLOv8n COCO (`models/weights/yolov8n.pt`), conf 0.25, MPS |
| Depth | Depth-Anything-V2-Metric-Indoor-Small, 256×256, `metric_scale` 0.60, MPS |
| OCR | Apple Vision (`vi-VT`, `en-US`), EasyOCR dự phòng |
| VQA | Qwen2.5-VL-3B-Instruct-4bit qua MLX, temperature 0, ảnh ≤ 512 px, ≤ 128 token |
| Cấu hình | `configs/*.yaml` tại thời điểm chạy (commit `fe99ac3` + các bản sửa ngày 2026-11-09) |

## Danh sách file

Các file ở thư mục này là **sau khi sửa** (hoặc không bị ảnh hưởng bởi bản sửa); `before_fix/` là trước khi sửa.

| File | Dữ liệu (nguồn) | Mẫu | Chỉ số chính | Script |
|---|---|---|---|---|
| `detection_coco.json` | COCO val2017 | 500 ảnh, 2510 vật, 24 lớp | mAP, P/R theo lớp, độ bền (tối, mờ chuyển động, JPEG) | `eval_detection_coco.py` |
| `depth_warning_nyu.json` | NYU Depth V2 test (Kinect) | 654 ảnh, 1019 vật trong hành lang | AbsRel/δ1, P/R của HIGH theo khoảng cách thật, vật cản chung | `eval_depth_warning_nyu.py` |
| `warning_tum_*.json` (4) | TUM RGB-D fr1_desk, fr3_walking_xyz, fr3_nostructure_notexture_near_withloop, fr3_long_office_household | 596 / 827 / 1093 / 2488 frame | P/R mức vật và mức frame, số câu cảnh báo, độ trễ | `eval_warning_tum.py` |
| `depth_scale_tum.json` | 4 chuỗi TUM trên | 1/15 frame | hệ số tỉ lệ tối ưu so với 0.60 | `eval_depth_scale_tum.py` |
| `ocr.json` | VinText, ICDAR 2015, SROIE, văn bản VI tổng hợp | 150/150/150/60 | F1 theo từ (có dấu, bỏ dấu), thời gian | `eval_ocr.py` |
| `mcocr.json` | MC-OCR 2021 (hoá đơn VI thật) | 120 ảnh | tỉ lệ đọc được từng trường (tên, địa chỉ, ngày, tổng tiền) | `eval_mcocr.py` |
| `vqa.json` | OpenViVQA dev, KTVIC test | 100 câu + 50 ảnh | tỉ lệ hợp lệ, token recall, thời gian, guardrail | `eval_vqa.py` |
| `vqa_describe_nyu.json` | NYU (40 ảnh) | 40 × 2 câu hỏi | câu bị loại, câu trùng lặp (ảo giác), khẳng định lối đi trống | `eval_vqa_describe_nyu.py` |
| `vizwiz.json`, `vizwiz_manual_grades.json` | VizWiz-VQA val (ảnh của người khiếm thị) | 34 câu (20 trả lời được, 14 không) | chấm tay, so sánh OCR | `eval_vizwiz.py` |
| `guardrail_cases.json` | 50 câu hỏi + 11 câu trả lời tự soạn | 61 | đúng/sai theo nhóm (dễ, khó, không dấu, tiếng Anh) | `eval_guardrail_cases.py` |
| `fault_scenarios.json` | mô phỏng (không cần model) | 2 kịch bản | câu nói và thời điểm khi depth treo / rút webcam | `eval_fault_scenarios.py` |
| `app_e2e_long_office.json` | `app.py` trên video TUM fr3_long_office (86 s) | 90 s | câu nói, FPS, độ trễ, RAM, tắt sạch | `eval_app_e2e.py` |
| `before_fix/*` | như trên, trước khi sửa | | | |

## Tái lập

```bash
W=/duong/dan/workdir      # ngoài repo, ~9 GB
bash evaluation/benchmarks/setup_data.sh $W
cd evaluation/benchmarks
export PYTHONPATH=$W/pyfix HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
PY=/opt/anaconda3/envs/ai-macbook/bin/python
OUT=$PWD/../results/real_data/2026-11-09   # --out nhận đường dẫn tuyệt đối
$PY eval_detection_coco.py --coco-dir $W/data/coco --num-images 500 --out $OUT/detection_coco.json
$PY eval_depth_warning_nyu.py --nyu-dir $W/data/nyu
for s in $W/data/tum/rgbd_dataset_*; do $PY eval_warning_tum.py --seq-dir $s; done
$PY eval_depth_scale_tum.py --tum-dir $W/data/tum --out $OUT/depth_scale_tum.json
$PY eval_ocr.py --data-dir $W/data --n 150
$PY eval_mcocr.py --parquet $W/data/mcocr/val.parquet --n 120 --out $OUT/mcocr.json
$PY eval_vqa.py --data-dir $W/data --n-qa 100 --n-describe 50 --image-out $W/vqa_images
$PY eval_vqa_describe_nyu.py --nyu-dir $W/data/nyu --out $OUT/vqa_describe_nyu.json
$PY eval_vizwiz.py --parquet $W/data/vizwiz/val0.parquet --out $OUT/vizwiz.json
$PY eval_guardrail_cases.py --out $OUT/guardrail_cases.json
(cd ../.. && PYTHONPATH=. $PY evaluation/benchmarks/eval_fault_scenarios.py --out $OUT/fault_scenarios.json)
# Video cho eval_app_e2e.py: ghép rgb/*.png của fr3_long_office_household thành MP4 640x480 30 fps
$PY eval_app_e2e.py $W/long_office.mp4 --out $OUT/app_e2e_long_office.json
```

`eval_detection_coco.py`, `eval_depth_warning_nyu.py`, `eval_warning_tum.py`, `eval_ocr.py` và `eval_vqa.py`
ghi vào `evaluation/results/real_data/` theo tên cố định (sẽ ghi đè bản 2026-10-23 ở đó) — chép sang thư mục
ngày sau khi chạy. Chạy tuần tự để số đo thời gian không bị nhiễu; trong lượt này OCR và VizWiz từng chạy
song song nên thời gian của hai bộ đó có thể cao hơn thực tế.

## Lưu ý khi trích dẫn

- Chấm tay VizWiz và câu hỏi dịch sang tiếng Việt do **một người** thực hiện; nên chấm lại độc lập trước khi công bố.
- Bộ ca guardrail là dữ liệu tự soạn, không phải bộ chuẩn công khai.
- Khoảng cách thật lấy từ Kinect (không đo được < ~0.5 m). TUM là camera cầm tay, không phải người khiếm thị đi bộ.
- `metric_scale` 0.60 được fit trên một nửa NYU; số liệu NYU có tách nửa giữ lại (`system_HIGH_heldout`).
