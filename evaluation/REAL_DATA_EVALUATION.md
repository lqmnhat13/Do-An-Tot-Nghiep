# Đánh giá hệ thống trên dữ liệu thật

Cập nhật: **2026-11-09** · Máy: Apple M1 16 GB, macOS 27.0.1 · Môi trường `ai-macbook` (Python 3.10)

Tài liệu ghi lại kết quả kiểm thử từng chức năng (nhận diện, độ sâu + cảnh báo, OCR, VQA, độ tin cậy runtime) trên các bộ dữ liệu công khai có nhãn, case dễ lẫn case khó trong phạm vi đề tài (trong nhà, tiếng Việt, M1 offline), cùng các lỗi phát hiện được và kết quả sau khi sửa.

- Số liệu gốc lượt này: [results/real_data/2026-11-09/](results/real_data/2026-11-09/) (mục lục, môi trường, lệnh tái lập trong README của thư mục đó).
- Lượt trước (2026-10-23, trước khi chuyển sang depth metric): [results/real_data/](results/real_data/) và [results/real_data/baseline/](results/real_data/baseline/).
- Script: [benchmarks/](benchmarks/).

## Tóm tắt

| Chức năng | Case dễ | Case khó | Đánh giá |
|---|---|---|---|
| Nhận diện (COCO, 500 ảnh) | Người AP50 0.80, mèo 0.91, bồn cầu 0.81 | Ghế R 0.30, bàn R 0.33; mờ chuyển động: R 0.54 → 0.23 | Tốt với người/vật lớn; yếu với ghế, bàn, ảnh mờ |
| Cảnh báo HIGH vật ≤ 1.5 m (NYU, 654 ảnh) | **P 0.80 / R 0.72**; vật ≤ 1 m: R 0.94 | Vật thật 1.4 m bị ước lượng 1.7 m (sát ngưỡng) | Tăng mạnh so với bản relative (R 0.16) |
| Cảnh báo theo video (TUM) | Bàn làm việc P/R 0.98/0.98; văn phòng 87 s P/R 0.93/0.93 | Người đi lại: **R 0.31**; mặt phẳng trơn: vật cản chung đúng 34% | Phụ thuộc hệ số tỉ lệ của cảnh |
| OCR Apple Vision | Văn bản in F1 0.99; hoá đơn EN 0.92 | Biển hiệu VI 0.72; hoá đơn VI: tổng tiền 98%, tên cửa hàng đúng dấu 57% | Tốt; sai dấu trên giấy in nhiệt |
| VQA Qwen2.5-VL-3B | OpenViVQA hợp lệ 98%; VizWiz 11/20 đúng + 7 đúng một phần | 12/14 ảnh không trả lời được thì trung thực; bịa chữ khi đọc sách/biển | Dùng được để mô tả; đọc chữ nên dùng OCR |
| Runtime | 24 FPS, detection p95 69 ms, RAM đỉnh 2.2 GB (chưa nạp VQA) | Rút webcam được báo sau 0.6 s; depth treo được báo sau 3.2 s | Đạt thời gian thực, có báo lỗi bằng giọng nói |

## Lỗi phát hiện ngày 2026-11-09 và kết quả sau khi sửa

| # | Lỗi (trước khi sửa) | Bằng chứng | Sửa | Sau khi sửa |
|---|---|---|---|---|
| 1 | Phím Q hỏi "Mô tả khung cảnh phía trước" làm Qwen2.5-VL-3B-4bit sinh câu ảo giác cố định | 3/40 ảnh phòng khách NYU (có sofa, không có người) → "Cô gái mặc áo dài đứng trên cầu thang."; tái lập được ở tiến trình mới; 15/40 câu trả lời bị loại | Phím Q hỏi "Phía trước có gì?" (câu đã được đánh giá) | 0 câu ảo giác lặp lại; 8/40 bị loại (20%) |
| 2a | Câu dự phòng "Chưa phát hiện được khung cảnh hoặc vật thể rõ ràng phía trước" nghe như phía trước trống | Xuất hiện ở mọi câu trả lời bị loại (tới 37,5% với câu hỏi cũ) | "Chưa mô tả được khung cảnh lúc này. Vui lòng thử lại và tiếp tục dùng gậy dò đường." | — |
| 2b | Validator coi dấu chấm trong số là hết câu → loại "25.000 đồng", "12.05.2027", "1.29" | VizWiz: hỏi giá → câu dự phòng | Bỏ qua dấu chấm/phẩy giữa hai chữ số | VizWiz: "$1.29"; OpenViVQA hợp lệ 0.96 → 0.98 |
| 2c | Câu trả lời tiếng Anh bị giọng tiếng Việt đọc | 2/40 mô tả NYU | Loại câu ≥ 4 từ không có ký tự tiếng Việt | Chuyển sang câu dự phòng |
| 2d | Kiểm tra câu trả lời bỏ lọt "Lối đi thông thoáng.", "Phía trước trống…" | 8/11 ca câu trả lời | Chặn khẳng định trống/thông thoáng khi chủ ngữ là lối đi/đường/phía trước/hành lang | 10/11 (ca còn lại: chặn nhầm "an toàn thực phẩm") |
| 3a | Mất webcam: ~12 s im lặng rồi **chuyển sang ảnh giả "Chair Simulation"**, pipeline chạy trên ảnh giả | Mô phỏng webcam lỗi | Camera thật chỉ thử kết nối lại, không bao giờ dùng ảnh giả | Thông báo "Mất tín hiệu camera…" sau 0.6 s, nhắc lại mỗi 10 s, báo khi phục hồi |
| 3b | Depth treo/stale: người đứng 0.9 m phía trước suốt 10 s nhưng không có tiếng nào (chỉ HUD hiện DEGRADED) | Mô phỏng depth treo trên `SystemCoordinator` thật | Luồng giám sát sức khỏe (watchdog + thời điểm có dữ liệu VALID gần nhất + lỗi nạp model depth) | "Dữ liệu độ sâu bị gián đoạn…" sau 3.2 s |
| 4 | OCR Apple Vision treo > 40 s trong app, khoá luôn VQA đến khi khởi động lại | 1/4 lần chạy app, 1/2 lần chạy test suite; không tái lập khi chạy riêng (0.14 s) | Vision có `ocr.read_timeout_sec` (10 s) → EasyOCR; OCR/VQA có giới hạn 30/60 s → báo "quá thời gian" và giải phóng worker | Lần chạy app sau khi sửa: OCR 1.3 s, VQA 4.4 s (gồm nạp model) |
| 5 | Báo cáo này ghi "Vision OCR đang hỏng", "lỗi chặn chưa sửa", R = 0.16 | Lỗi đã sửa trong commit `003578c` | Viết lại báo cáo | — |

Nguyên nhân gốc của lần Vision treo chưa xác định được (không phải do PyTorch MPS chạy song song: đã thử YOLO trên MPS cùng lúc, Vision vẫn 0.08–0.22 s). Bản sửa chỉ chặn hậu quả.

## Dữ liệu

| Chức năng | Bộ dữ liệu (nguồn) | Mẫu dùng |
|---|---|---|
| Nhận diện | COCO val2017 (cocodataset.org) | 500 ảnh, 2510 vật, 24 lớp |
| Độ sâu, cảnh báo | NYU Depth V2 test (Silberman et al., ECCV 2012), độ sâu Kinect | 654 ảnh, 1019 vật trong hành lang |
| Cảnh báo theo thời gian | TUM RGB-D (Sturm et al., IROS 2012): `fr1_desk` (camera cầm tay quanh bàn), `fr3_walking_xyz` (người đi lại), `fr3_nostructure_notexture_near_withloop` (mặt phẳng trơn sát camera, không vật COCO), `fr3_long_office_household` (đi vòng văn phòng) | 596 + 827 + 1093 + 2488 frame (≈ 174 s) |
| OCR tiếng Việt | VinText test (VinAI, CVPR 2021); MC-OCR 2021 (hoá đơn bán lẻ, RIVF 2021) | 150 + 120 ảnh |
| OCR tiếng Anh | ICDAR 2015 Incidental Scene Text; SROIE (ICDAR 2019) hoá đơn | 150 + 150 ảnh |
| OCR văn bản VI | Tổng hợp ([make_vi_documents.py](benchmarks/make_vi_documents.py)), 10 đoạn × 6 điều kiện chụp | 60 ảnh — chỉ để bổ sung |
| VQA | OpenViVQA dev (UIT, VLSP 2023); KTVIC test; **VizWiz-VQA** val (ảnh và câu hỏi thật của người khiếm thị, Gurari et al., CVPR 2018); NYU (40 ảnh, câu hỏi của phím Q) | 100 + 50 + 34 + 40 |
| Guardrail | 50 câu hỏi + 11 câu trả lời tự soạn (dễ, khó, không dấu, tiếng Anh, câu không được chặn) | 61 ca |
| Runtime | `app.py` trên video ghép từ TUM `fr3_long_office_household`; mô phỏng sự cố trên `SystemCoordinator` | 90 s; 2 kịch bản |

## 1. Nhận diện vật thể (YOLOv8n)

Chạy qua `YoloDetector` + `ClassFilter`, tham số từ `configs/model_config.yaml`. Kết quả trùng khớp lượt 2026-10-23 (suy luận tất định).

| Chỉ số (24 lớp) | Giá trị |
|---|---|
| mAP50-95 / mAP50 / mAP75 | 0.420 / 0.596 / 0.456 |
| AP vật nhỏ / vừa / lớn | 0.153 / 0.422 / 0.611 |
| P / R tại conf 0.25 (app) | 0.743 / 0.545 |
| P / R tại conf 0.45 | 0.865 / 0.431 |
| Riêng 13 `alert_classes`: mAP50 | 0.627 |

Theo lớp cảnh báo (AP50 · R tại conf 0.25): người 0.80 · 0.68; mèo 0.91 · 0.89; bồn cầu 0.81 · 0.84; laptop 0.72 · 0.71; tủ lạnh 0.71 · 0.50; bồn rửa 0.64 · 0.56; giường 0.61 · 0.54; sofa 0.56 · 0.54; ti vi 0.55 · 0.46; chó 0.57 · 0.56; điện thoại 0.45 · 0.38; **bàn ăn 0.42 · 0.33; ghế 0.39 · 0.30**.

Độ bền (150 ảnh, mAP50 / recall): gốc 0.629 / 0.54 · thiếu sáng 0.491 / 0.33 · JPEG q15 0.343 / 0.28 · **mờ chuyển động 0.285 / 0.23**.

Trên video TUM, hệ thống nhận nhầm mặt bàn thành "giường" (`fr1_desk`, 30 frame) và tủ/vách văn phòng thành "tủ lạnh" (4/22 câu cảnh báo ở `long_office`).

## 2. Độ sâu và chế độ Cảnh báo

### 2.1 Độ chính xác độ sâu (NYU, Eigen crop)

| Cấu hình | AbsRel | RMSE (m) | δ1 | Thời gian |
|---|---|---|---|---|
| **Metric-Indoor-Small 256×256 × 0.60 (app, không căn theo GT)** | 0.156 | 0.577 | 0.746 | 54 ms |
| Small (relative) 256×256, căn scale+shift theo GT từng ảnh (cận trên) | 0.079 | 0.350 | 0.939 | 56 ms |

Với vật trong hành lang, sai số khoảng cách tuyệt đối (phân vị 20 trong ROI) có trung vị **0.22 m**, p90 0.69 m. Thứ tự gần–xa: Spearman 0.90.

### 2.2 Cảnh báo HIGH so với khoảng cách thật (NYU, 1019 vật)

| Vật gần thật | Bản relative (2026-10-23) P / R | **Bản metric hiện tại P / R** | Nửa giữ lại (không dùng để fit 0.60) P / R |
|---|---|---|---|
| ≤ 1.0 m | 0.44 / 0.48 | 0.18 / **0.94** | 0.17 / 0.91 |
| ≤ 1.5 m | 0.92 / 0.16 | **0.80 / 0.72** | 0.81 / 0.70 |
| ≤ 2.0 m | 1.00 / 0.09 | 0.98 / 0.46 | 0.99 / 0.44 |

(P ở dòng ≤ 1.0 m thấp vì ngưỡng HIGH là 1.5 m, nên vật 1–1.5 m được tính là "sai" ở dòng đó.)

Theo lớp, vật ≤ 1.5 m được HIGH: bồn cầu 9/9, bồn rửa 23/29, giường 37/47, ghế 58/84, bàn ăn 38/56, sofa 25/36. Vật bị bỏ sót có khoảng cách thật trung vị 1.37 m nhưng ước lượng 1.67 m (sát ngưỡng). Lưu ý: con số trên chỉ tính vật **đã được YOLO nhận diện**; với ghế (recall nhận diện 0.30) tỉ lệ cảnh báo từ đầu đến cuối thấp hơn nhiều.

"Vật cản chung" (không cần lớp COCO): bật ở 3/654 ảnh NYU, 2/3 có vật thật < 1 m.

### 2.3 Cảnh báo theo thời gian (TUM, pipeline đầy đủ)

Detection + tracking mỗi frame, depth mỗi 4 frame (≈ 7.5 Hz), Synchronizer/RiskFSM/AlertAggregator theo timestamp thật. Mọi frame đều `VALID`.

| Chuỗi (độ dài) | Vật ≤ 1.5 m P / R | Frame có vật < 1 m được HIGH | Frame không có vật gần mà vẫn HIGH | Vật cản chung: số frame (đúng) | Câu cảnh báo |
|---|---|---|---|---|---|
| fr1_desk (20 s) | 0.98 / 0.98 | 310/414 (75%) | 84/182 | 215 (99%) | 6 (18/phút) |
| fr3_long_office (87 s) | 0.93 / 0.93 | 616/761 (81%) | 682/1727 | 464 (76%) | 22 (15/phút) |
| fr3_walking_xyz (29 s) | 0.99 / **0.31** | 8/8 | 106/819 | 0 | 4 |
| fr3_nostructure_near (38 s) | — (không vật COCO) | **79/325 (24%)** | 150/768 | 229 (**34%**) | 6 |

Phần lớn frame "không có vật gần mà vẫn HIGH" có vật thật ở 1–1.5 m (ngưỡng HIGH là 1.5 m, còn mức frame dùng 1.0 m).

### 2.4 Hệ số tỉ lệ theo cảnh (TUM)

`metric_scale` = 0.60 được fit trên NYU. Hệ số tối ưu (trung vị GT/dự đoán thô) trên từng chuỗi TUM:

| Chuỗi | Hệ số tối ưu (p10–p90 theo frame) | Sai số trung vị với 0.60 → với hệ số tối ưu |
|---|---|---|
| fr1_desk | 0.53 (0.47–0.61) | 0.16 → 0.10 m |
| fr3_long_office | 0.65 (0.53–0.76) | 0.18 → 0.18 m |
| fr3_nostructure_near | 0.57 (0.45–0.77) | 0.25 → 0.24 m |
| fr3_walking_xyz | **0.49** (0.34–0.58) | **0.46 → 0.36 m** |

Hệ số dao động 0.49–0.65 giữa các cảnh và 0.34–0.77 giữa các frame trong cùng chuỗi. Ở `walking_xyz`, vật thật 1–2 m bị ước lượng trung vị 1.86 m, giải thích recall 0.31. Cần hiệu chỉnh lại cho camera dùng khi demo (`scripts/calibrate_depth.py`), nhưng một hệ số cố định không khắc phục được sai lệch theo cảnh.

## 3. OCR

Chạy qua `OCRService.process` (kiểm tra chất lượng → tiền xử lý → nhận dạng → ghép dòng). Apple Vision chạy được trực tiếp, không cần vá (`vision_unpatched` = `vision`).

| Bộ dữ liệu | Engine | F1 | P | R | F1 bỏ dấu | Thời gian trung vị |
|---|---|---|---|---|---|---|
| VinText (VI, biển hiệu thật) | **Vision** | **0.724** | 0.77 | 0.68 | 0.806 | 0.15 s |
| | EasyOCR | 0.439 | 0.66 | 0.33 | 0.530 | 1.40 s |
| Văn bản VI (tổng hợp) | **Vision** | **0.989** | 0.99 | 0.99 | 1.000 | 0.16 s |
| | EasyOCR | 0.946 | 0.95 | 0.94 | 0.985 | 0.74 s |
| ICDAR 2015 (EN, chữ cảnh) | **Vision** | **0.497** | 0.45 | 0.56 | 0.499 | 0.13 s |
| | EasyOCR | 0.365 | 0.48 | 0.29 | 0.369 | 0.56 s |
| SROIE (EN, hoá đơn) | **Vision** | **0.922** | 0.91 | 0.94 | 0.922 | 0.27 s |
| | EasyOCR | 0.765 | 0.81 | 0.73 | 0.766 | 4.25 s |

Bước kiểm tra chất lượng (sau khi sửa ngày 2026-10-23) chỉ còn chặn 1/510 ảnh; SROIE tăng từ 0.67 lên 0.92 nhờ không còn chặn oan hoá đơn giấy trắng.

**Hoá đơn tiếng Việt thật (MC-OCR, 120 ảnh, Vision)** — tỉ lệ trường đọc được nguyên văn (có dấu / bỏ dấu): tổng tiền **0.98 / 0.98**, ngày **0.92 / 0.92**, tên cửa hàng 0.57 / 0.97, địa chỉ 0.25 / 0.65; 0.21 s/ảnh. Lỗi chủ yếu là dấu trên giấy in nhiệt: "PHỞ" → "PHÔ", "Cẩm Phả" → "Cẩm Phá", "CẨM" → "CẦM". Người nghe vẫn hiểu được phần lớn nội dung; số tiền và ngày gần như luôn đúng.

## 4. VQA

### 4.1 OpenViVQA và KTVIC (sau khi sửa)

| Chỉ số | Qwen2.5-VL-3B 4bit (MLX, mặc định) | BLIP + MarianMT (cũ) |
|---|---|---|
| Hỏi đáp: tỉ lệ hợp lệ | **0.98** (2026-10-23: 0.96) | 1.00 |
| Hỏi đáp: token recall so với đáp án | **0.51** (0.49) | 0.19 |
| Mô tả "Phía trước có gì?": tỉ lệ hợp lệ | 0.94 | 1.00 |
| Mô tả: content precision | 0.36 | 0.34 |
| Thời gian trung vị (sau khi nạp) | 2.7–2.8 s; lần đầu gồm nạp model 3–7 s | 0.87 s |

Backend cũ bỏ qua câu hỏi và chỉ mô tả ảnh chung chung.

### 4.2 Câu hỏi của phím Q (NYU, 40 ảnh trong nhà)

| | "Mô tả khung cảnh phía trước" (trước) | "Phía trước có gì?" (sau) |
|---|---|---|
| Câu trả lời bị loại | 15/40 (9 chạm giới hạn token, 4 sai số câu, 2 tiếng Anh) | 8/40 (6 chạm giới hạn token, 2 sai số câu) |
| Câu ảo giác lặp lại ở nhiều ảnh | "Cô gái mặc áo dài đứng trên cầu thang." × 3 | 0 |
| Khẳng định lối đi trống/an toàn | 0 | 0 |

### 4.3 VizWiz — ảnh và câu hỏi của người khiếm thị (34 câu, chấm tay)

Câu hỏi được dịch sang tiếng Việt; chấm tay so với đáp án gốc tiếng Anh (một người chấm, [vizwiz_manual_grades.json](results/real_data/2026-11-09/vizwiz_manual_grades.json)).

| Nhóm | Kết quả |
|---|---|
| 20 câu trả lời được (thuốc, đồ uống, sách, biển, thực đơn, mô tả vật) | 11 đúng, 7 đúng một phần, 2 sai/từ chối (trước khi sửa: 11 / 6 / 3) |
| 14 ảnh "không trả lời được" (mờ, lệch khung) | 12 trung thực ("Không xác định rõ"), 2 bịa ("HP", "Update") |
| 6 câu đọc chữ trả lời được, chạy qua OCR | OCR đúng 6/6; VQA bịa tên sách ("Live and Let Die" thay vì Clive Cussler) và đọc sai tấm biển |

Kết luận: khi người dùng muốn **đọc chữ** nên dùng OCR (phím Space), không dùng VQA. Lỗi điển hình của VQA: gọi thuốc ho "Night Time" là "thuốc ngủ", "Nestea" → "Nestlé", thêm chuột/bàn phím không có trong ảnh.

### 4.4 Guardrail an toàn (61 ca tự soạn)

| Nhóm | Kết quả |
|---|---|
| Câu hỏi phải từ chối (dễ + khó) | 22/30. Bỏ lọt: "Đường trống không?", "Phía trước có gì cản đường không?", "Phía trước thông thoáng chứ?", "Cứ đi thẳng là được đúng không?", 2 câu không dấu, 2 câu tiếng Anh ("Is the path clear?", "Anything in my way?") |
| Câu hỏi không được chặn (đọc chữ, đếm, mô tả) | 20/20 |
| Câu trả lời của VLM | 10/11 (trước khi sửa 8/11). Còn chặn nhầm "Tờ giấy nói về an toàn thực phẩm." |
| Bộ guardrail của `eval_vqa.py` | 24/24 |

Trong app, phím Q luôn gửi một câu cố định (không có nhập giọng nói), nên lớp kiểm tra **câu trả lời** là lớp bảo vệ thực sự; guardrail câu hỏi chỉ có tác dụng với `scripts/run_vqa_camera.py --question`.

## 5. Runtime và độ tin cậy

`app.py` trên video `fr3_long_office` (86 s, lặp), TTS ghi log thay vì phát loa; bấm OCR ở giây 20, VQA ở giây 35, về Cảnh báo ở giây 60.

| Chỉ số | Trước khi sửa | Sau khi sửa |
|---|---|---|
| FPS detection | 24.0 | 23.8 |
| Detection p50 / p95 | 33 / 69 ms | 36 / 69 ms |
| Depth p50 / p95 | 84 / 102 ms | 85 / 96 ms |
| RAM | RSS 1.16 GB, peak footprint 2.2 GB (VQA chưa nạp) | — |
| OCR | "Đang xử lý…" rồi treo > 40 s; VQA bị từ chối | Kết quả sau 1.3 s |
| VQA | không chạy được | Trả lời sau 4.4 s (gồm nạp model) |
| Câu cảnh báo vật cản trong 90 s (~65 s ở chế độ Cảnh báo) | 11 | 10 |
| Tắt máy | sạch (chỉ còn luồng `tqdm_monitor` của thư viện) | sạch |

Kịch bản sự cố trên `SystemCoordinator` thật (detector/depth giả, [eval_fault_scenarios.py](benchmarks/eval_fault_scenarios.py)):

| Kịch bản | Trước khi sửa | Sau khi sửa |
|---|---|---|
| Depth treo, người đứng 0.9 m phía trước | Im lặng suốt thời gian quan sát; chỉ HUD hiện DEGRADED | Giây 11.2 (treo từ giây 8): "Dữ liệu độ sâu bị gián đoạn. Cảnh báo vật cản đang tạm ngừng, hãy dừng lại và dùng gậy dò đường." |
| Rút webcam | ~12 s im lặng, rồi pipeline chạy trên ảnh giả "Chair Simulation" | Giây 8.6 (rút ở giây 8): "Mất tín hiệu camera…"; cắm lại ở giây 16 → giây 17.3: "Cảnh báo vật cản đã hoạt động trở lại." |

Unit test: 181/181 đạt (thêm 20 test trong [tests/test_acceptance_fixes.py](../tests/test_acceptance_fixes.py)).

## 6. Hạn chế còn lại và hướng cải thiện

1. **Hệ số tỉ lệ độ sâu phụ thuộc cảnh** (0.49–0.65): recall HIGH giảm còn 0.31 ở cảnh người đi lại. Hiệu chỉnh `metric_scale` cho camera demo; hướng lâu dài: tinh chỉnh model trên dữ liệu camera thật hoặc dùng cảm biến độ sâu.
2. **Mặt phẳng trơn sát camera**: luật vật cản chung chỉ đúng 34% và bắt được 24% frame có vật < 1 m.
3. **Nhận diện ghế/bàn** (recall 0.30–0.33) và **mờ chuyển động** (recall 0.23): fine-tune YOLO trên dữ liệu trong nhà có tăng cường mờ/thiếu sáng, hoặc dùng YOLOv8s. Nhận nhầm "giường", "tủ lạnh" tạo câu cảnh báo sai tên vật.
4. **Tần suất cảnh báo 15–18 câu/phút** trong văn phòng, bị giới hạn bởi `global_alert_interval_sec` 3.5 s; cần thử nghiệm với người dùng để chọn nhịp phù hợp.
5. **VQA**: 20% câu trả lời bị loại (chủ yếu chạm 128 token); vẫn bịa chữ khi đọc sách/biển. Guardrail câu hỏi bỏ lọt 8/30 cách diễn đạt.
6. **Vision OCR treo**: chưa rõ nguyên nhân gốc; hiện chỉ có giới hạn thời gian và dự phòng EasyOCR (chậm hơn, F1 thấp hơn).
7. Tên `ByteTrackerAdapter` nhưng thực chất là ghép IoU tham lam một tầng (không Kalman, không ghép lần hai cho detection điểm thấp).

## 7. Giới hạn của đánh giá

- Chưa có video người khiếm thị đi lại thật. TUM là camera cầm tay; NYU là ảnh tĩnh.
- Khoảng cách thật lấy từ Kinect (không đo được < ~0.5 m).
- Văn bản tiếng Việt in là dữ liệu tổng hợp; MC-OCR chỉ có nhãn theo trường.
- VizWiz và mô tả cảnh được chấm tay bởi một người; câu hỏi VizWiz được dịch sang tiếng Việt. Bộ ca guardrail tự soạn.
- `metric_scale` fit trên một nửa NYU (có báo cáo nửa giữ lại).
- Kịch bản sự cố dùng detector/depth giả để tái lập chính xác; không thay thế phép thử rút camera thật.
- Trong lượt này OCR và VizWiz có lúc chạy song song nên thời gian của hai bộ này có thể cao hơn thực tế.

## Chạy lại

Xem [results/real_data/2026-11-09/README.md](results/real_data/2026-11-09/README.md).
