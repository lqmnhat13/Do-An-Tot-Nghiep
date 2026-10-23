# Tiến độ đánh giá trên dữ liệu thật

Hoàn tất 2026-10-23. Báo cáo: [../REAL_DATA_EVALUATION.md](../REAL_DATA_EVALUATION.md). Số liệu: [../results/real_data/](../results/real_data/).

| Bước | Script | Trạng thái |
|---|---|---|
| Nhận diện (COCO val2017) | `eval_detection_coco.py` | Xong (đã chạy lại sau khi sửa lỗi gán tên lớp) |
| Độ sâu + cảnh báo (NYU) | `eval_depth_warning_nyu.py` | Xong |
| Cảnh báo theo thời gian (TUM) | `eval_warning_tum.py` | Xong |
| Pipeline song song (TUM video) | `scripts/evaluate_video.py` | Xong → `realtime_pipeline_tum.json` |
| OCR | `eval_ocr.py` | Xong |
| VQA | `eval_vqa.py` | Xong (+ chấm tay 24 mẫu, ghi trong báo cáo) |

Lệnh chạy lại: xem mục "Chạy lại" trong báo cáo.
