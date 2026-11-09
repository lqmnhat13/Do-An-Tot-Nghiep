# Tiến độ đánh giá trên dữ liệu thật

Báo cáo: [../REAL_DATA_EVALUATION.md](../REAL_DATA_EVALUATION.md).

## Lượt 2026-11-09 (kiểm thử nghiệm thu, trước/sau khi sửa 5 lỗi)

Số liệu: [../results/real_data/2026-11-09/](../results/real_data/2026-11-09/) — mục lục và lệnh tái lập trong README của thư mục đó.

| Bước | Script | Trạng thái |
|---|---|---|
| Nhận diện (COCO val2017) | `eval_detection_coco.py` | Xong |
| Độ sâu + cảnh báo (NYU) | `eval_depth_warning_nyu.py` | Xong (depth metric) |
| Cảnh báo theo thời gian (4 chuỗi TUM) | `eval_warning_tum.py` | Xong |
| Hệ số tỉ lệ theo cảnh (TUM) | `eval_depth_scale_tum.py` | Mới, xong |
| OCR (VinText, ICDAR 2015, SROIE, VI tổng hợp) | `eval_ocr.py` | Xong |
| OCR hoá đơn tiếng Việt (MC-OCR 2021) | `eval_mcocr.py` | Mới, xong |
| VQA (OpenViVQA, KTVIC) | `eval_vqa.py` | Xong (sau khi sửa) |
| VQA câu hỏi phím Q (NYU) | `eval_vqa_describe_nyu.py` | Mới, xong (câu cũ và mới) |
| VQA ảnh người khiếm thị (VizWiz) | `eval_vizwiz.py` | Mới, xong (+ chấm tay) |
| Guardrail (61 ca) | `eval_guardrail_cases.py` | Mới, xong |
| Sự cố camera/depth | `eval_fault_scenarios.py` | Mới, xong |
| Toàn bộ app trên video | `eval_app_e2e.py` | Mới, xong |

## Lượt 2026-10-23

Số liệu ở [../results/real_data/](../results/real_data/) (file gốc) và `baseline/`. Đo trước khi chuyển sang depth metric.
