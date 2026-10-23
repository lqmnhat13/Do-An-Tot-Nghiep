# Đánh giá hệ thống trên dữ liệu thật

Ngày chạy: 2026-10-23 · Máy: Apple M1 16 GB, macOS 27.0.1 · Môi trường `ai-macbook` (Python 3.10)

Tài liệu này ghi lại kết quả kiểm thử từng chức năng (nhận diện, độ sâu + cảnh báo, OCR, VQA) bằng các bộ dữ liệu công khai có nhãn chuẩn, cùng các lỗi phát hiện được và phương án cải thiện. Số liệu gốc nằm trong [results/real_data/](results/real_data/); script nằm trong [benchmarks/](benchmarks/).

## Tóm tắt

| Chức năng | Kết quả chính | Đánh giá |
|---|---|---|
| Nhận diện vật thể | mAP50 = 0.60 trên 24 lớp; ở ngưỡng app: P = 0.74, R = 0.55. Ghế R = 0.30, bàn R = 0.33 | Khá với người/vật lớn; yếu với ghế, bàn, vật nhỏ; giảm mạnh khi mờ/tối |
| Độ sâu | δ1 = 0.94 (sau căn tỉ lệ), Spearman với khoảng cách thật = 0.75 | Thứ tự xa–gần tốt |
| Cảnh báo HIGH | Khi phát thì đúng 92–94% là vật trong 1.5 m, nhưng chỉ phát cho **16–17%** vật trong 1.5 m | **Bỏ sót nhiều** — vấn đề lớn nhất về an toàn |
| OCR tiếng Việt | Vision: F1 0.70 (ảnh biển hiệu thật), 0.90 (văn bản in) | Tốt — **nhưng hiện đang hỏng trên máy này** (lỗi 2) |
| OCR tiếng Anh | Vision: F1 0.48 (chữ cảnh), 0.67 (hoá đơn) | Hoá đơn bị bước kiểm tra chất lượng chặn oan 47% |
| VQA | 96% câu trả lời hợp lệ; chấm tay 9/16 đúng, 2 đúng một phần; mô tả cảnh 6/8 tốt | Dùng được; có nhầm lẫn chi tiết, đôi khi bỏ sót người |
| Tốc độ | Pipeline song song 22 FPS; cảnh báo ~45–90 ms/frame; OCR 0.1–0.2 s; VQA ~2.7 s | Đạt thời gian thực |

## Lỗi chặn phát hiện được (chưa sửa)

1. **scipy 1.15.3 không nạp được trên macOS 27** (`dyld: section '__DATA/__thread_bss' has a zero-fill section type`). Hậu quả: `easyocr` và `mlx_vlm` không import được → **VQA mặc định và OCR dự phòng đều không chạy** trên máy này; 6/135 unit test lỗi. Python 3.10 không có scipy ≥ 1.16; scipy 1.14.1 chạy được (bộ đánh giá dùng qua `PYTHONPATH`, không sửa env).
2. **Apple Vision OCR (engine mặc định) crash** trên macOS 27 + pyobjc 12.2.2: [vision_engine.py](../src/ocr/vision_engine.py) truyền `{}` vào `initWithData_options_` → `NSInvalidArgumentException - key does not exist`. Truyền `None` hoặc `NSDictionary.dictionary()` thì chạy đúng. Vì lỗi xảy ra trong `readtext` (không phải lúc khởi tạo), `OCRService` **không** chuyển sang EasyOCR mà trả lỗi → OCR thất bại 100%. Mã ngôn ngữ `vi-VT` là đúng theo Apple, không phải lỗi.

Các số liệu OCR/VQA bên dưới đo **sau khi vá tạm** hai lỗi này trong bộ đánh giá, tức là năng lực của hệ thống khi môi trường được sửa.

## Dữ liệu

| Chức năng | Bộ dữ liệu (nguồn) | Mẫu dùng |
|---|---|---|
| Nhận diện | COCO val2017 (cocodataset.org) — YOLOv8n không huấn luyện trên tập này | 500 ảnh, 2510 vật thuộc 24 lớp |
| Độ sâu, cảnh báo | NYU Depth V2 test (Silberman et al., ECCV 2012), độ sâu Kinect | 654 ảnh, 1019 vật trong hành lang |
| Cảnh báo theo thời gian | TUM RGB-D `freiburg1_desk` (Sturm et al., IROS 2012) | 596 frame, 19.8 s |
| OCR tiếng Việt | VinText test (VinAI, CVPR 2021) — chữ cảnh thật | 150 ảnh |
| OCR tiếng Anh | ICDAR 2015 Incidental Scene Text test; SROIE (ICDAR 2019) test — hoá đơn | 150 + 150 ảnh |
| OCR văn bản tiếng Việt | **Tổng hợp** ([make_vi_documents.py](benchmarks/make_vi_documents.py)): 10 đoạn văn × 6 điều kiện chụp | 60 ảnh — chỉ để bổ sung |
| VQA hỏi đáp | OpenViVQA dev (UIT, VLSP 2023) | 100 câu |
| VQA mô tả cảnh | KTVIC test (5 mô tả tiếng Việt/ảnh) | 50 ảnh |

Không tìm được bộ ảnh chụp tài liệu tiếng Việt công khai có nhãn chuẩn, nên phần "chụp ảnh văn bản tiếng Việt" dùng ảnh tổng hợp; kết quả phần này lạc quan hơn ảnh chụp thật.

## 1. Nhận diện vật thể (YOLOv8n)

Chạy qua `YoloDetector` + `ClassFilter` của dự án, tham số từ `configs/model_config.yaml`.

| Chỉ số (24 lớp bật) | Giá trị |
|---|---|
| mAP50-95 / mAP50 / mAP75 | 0.420 / 0.596 / 0.456 |
| AP vật nhỏ / vừa / lớn | 0.153 / 0.422 / 0.611 |
| P / R tại conf 0.25 (cấu hình app) | 0.743 / 0.545 |
| P / R tại conf 0.45 | 0.865 / 0.431 |
| Riêng 13 `alert_classes`: mAP50 | 0.627 |

Theo lớp (tại conf 0.25): tốt — mèo (R 0.89), bồn cầu (0.84), người (AP50 0.80, R 0.68), laptop (0.71). Yếu — **ghế (AP50 0.39, R 0.30)**, **bàn ăn (0.42, R 0.33)**, sách (0.25), chai, điện thoại. Ghế và bàn là vật cản phổ biến nhất trong nhà nhưng lại là hai lớp được cảnh báo kém nhất.

Độ bền (150 ảnh, mAP50 / recall tại conf 0.25):

| Điều kiện | mAP50 | Recall |
|---|---|---|
| Ảnh gốc | 0.629 | 0.54 |
| Thiếu sáng (gamma + nhiễu) | 0.491 | 0.33 |
| Nén JPEG chất lượng 15 | 0.343 | 0.28 |
| Mờ chuyển động (15 px) | 0.285 | 0.23 |

Mờ chuyển động — điều thường gặp khi người dùng vừa đi vừa cầm/đeo camera — làm recall giảm hơn một nửa.

Trên video TUM, mặt bàn chụp cận bị nhận là **"giường"** (conf tới 0.67, 30/596 frame) → hệ thống phát câu "có giường phía trước"; chiếc ghế sát camera không được phát hiện.

## 2. Độ sâu và chế độ Cảnh báo

### 2.1 Độ chính xác độ sâu (NYU, Eigen crop)

| Cấu hình | AbsRel | δ1 | Thời gian |
|---|---|---|---|
| Depth Anything V2 Small 256×256 (app) — căn scale+shift | 0.079 | 0.939 | 59 ms |
| Cùng model, 518×518 — căn scale+shift | 0.062 | 0.956 | 316 ms |
| Metric-Indoor-Small 256×256, mét thô | 0.603 | 0.094 | 71 ms |

Hai dòng đầu được căn tỉ lệ bằng ground truth từng ảnh (cận trên — hệ thống thật không có ground truth). Model metric thô sai về tỉ lệ (ước lượng xa gấp ~1.66 lần ở đầu vào vuông 256×256) nhưng thứ tự xa–gần rất tốt (Spearman 0.90).

### 2.2 Quyết định cảnh báo HIGH so với khoảng cách thật

Với 1019 vật thuộc `alert_classes` nằm trong hành lang (khoảng cách thật = phân vị 20 độ sâu Kinect trong bbox đã co viền):

| Khoảng cách thật | Số vật | Tỉ lệ được HIGH | Độ gần trung vị |
|---|---|---|---|
| 0–1 m | 48 | 40% | 0.66 |
| 1–1.5 m | 226 | 12% | 0.48 |
| 1.5–2 m | 254 | 1% | 0.35 |
| > 2 m | 491 | 0% | ≤ 0.23 |

**Nguyên nhân:** độ gần được chuẩn hoá min-max **theo từng frame** ([depth_representation.py](../src/depth/depth_representation.py)). Điểm gần nhất trong khung hình — thường là sàn sát chân camera — luôn nhận giá trị 1.0, nên một chiếc ghế cách 0.8 m thường chỉ đạt ~0.66 < ngưỡng 0.75. Ngược lại, khi không có gì gần, vật xa nhất vẫn có thể bị đẩy lên cao. Điểm "độ gần" vì thế là thứ hạng trong frame, không phải khoảng cách.

Phương án thay thế được kiểm tra **giữ-lại**: Metric-Indoor-Small + một hệ số tỉ lệ (fit trên nửa số ảnh) + ngưỡng tính bằng mét, đo trên nửa còn lại (505 vật):

| Vật gần thật | HIGH hiện tại P / R | Metric + hệ số + ngưỡng mét P / R |
|---|---|---|
| ≤ 1.0 m | 0.44 / 0.48 | 0.93 / 0.57 |
| ≤ 1.5 m | 0.92 / 0.16 | 0.82 / 0.69 |
| ≤ 2.0 m | 1.00 / 0.09 | 0.90 / 0.86 |

Hệ số (~0.60) fit trên camera Kinect của NYU; với webcam thật cần hiệu chỉnh lại bằng vài ảnh có khoảng cách đo tay.

### 2.3 Cảnh báo theo thời gian (TUM fr1_desk, pipeline đầy đủ)

Detection + ByteTrack mỗi frame, depth ~7.5 Hz, Synchronizer/RiskFSM/AlertAggregator dùng timestamp thật.

- Chất lượng dữ liệu: 596/596 frame `VALID` (không stale/degraded ở tốc độ này).
- Mức vật (vật thuộc `alert_classes` trong hành lang): ≤ 1 m P 0.80 / R 0.68; ≤ 1.5 m P 1.00 / R 0.48.
- Mức frame: 414 frame có vật cản bất kỳ < 1 m trong hành lang, chỉ 94 frame (23%) có HIGH. Phần lớn chênh lệch là **vật cản không thuộc lớp COCO** (mặt bàn, tường, cạnh tủ) — hệ thống không cảnh báo những vật này theo thiết kế. 36/182 frame không có vật gần vẫn có HIGH.
- Đã phát 5 câu cảnh báo trong 20 s (bị giới hạn bởi `global_alert_interval_sec` = 3.5 s), trong đó 2 câu "có giường phía trước" sai tên vật.

Chuỗi `fr1_desk` là camera cầm tay quanh bàn làm việc, không phải người đi bộ; cần quay thêm video đi lại thật để đánh giá đầy đủ.

## 3. OCR

Chạy qua `OCRService.process` (kiểm tra chất lượng → tiền xử lý → nhận dạng → ghép dòng). Chỉ số: F1 theo từ (không phân biệt hoa thường, bỏ dấu câu); "bỏ dấu" = so sau khi bỏ dấu tiếng Việt.

| Bộ dữ liệu | Engine | F1 | P | R | F1 bỏ dấu | Bị chặn bởi kiểm tra chất lượng | Thời gian (trung vị) |
|---|---|---|---|---|---|---|---|
| VinText (VI, ảnh thật) | Vision | **0.697** | 0.78 | 0.63 | 0.774 | 15/150 | 0.13 s |
| | EasyOCR | 0.416 | 0.66 | 0.30 | 0.505 | 15/150 | 0.89 s |
| Văn bản VI (tổng hợp) | Vision | **0.901** | 0.99 | 0.83 | 0.909 | 10/60 | 0.15 s |
| | EasyOCR | 0.880 | 0.97 | 0.81 | 0.902 | 10/60 | 0.78 s |
| ICDAR 2015 (EN, cảnh) | Vision | **0.481** | 0.45 | 0.52 | 0.483 | 8/150 | 0.11 s |
| | EasyOCR | 0.353 | 0.49 | 0.28 | 0.358 | 8/150 | 0.53 s |
| SROIE (EN, hoá đơn) | Vision | **0.674** | 0.91 | 0.53 | 0.674 | **70/150** | 0.21 s |
| | EasyOCR | 0.539 | 0.81 | 0.40 | 0.540 | 70/150 | 7.56 s |
| Mọi bộ | Vision chưa vá | 0.000 | — | — | — | — | lỗi 100% |

Nhận xét:
- **Vision tốt hơn EasyOCR rõ rệt** cả về độ chính xác (VinText 0.70 so với 0.42) lẫn tốc độ (nhanh hơn 5–36 lần). EasyOCR mất tới 7.6 s (p90 17 s) cho một hoá đơn.
- **Tiếng Việt:** chênh lệch F1 có dấu / bỏ dấu là 0.08 (Vision) và 0.09 (EasyOCR) — khoảng 1/10 số từ đọc đúng chữ nhưng sai dấu (ví dụ "TRUYỀN" → "TRUYỂN", "Vĩnh" → "Vinh"). Văn bản in rõ đạt ~0.99 ở mọi điều kiện trừ ảnh mờ.
- ICDAR 2015 có precision thấp một phần vì nhãn chuẩn bỏ qua chữ khó đọc (`###`) nên chữ đọc được ở những vùng đó bị tính là sai.
- **Bước kiểm tra chất lượng ảnh chặn oan:** đọc thẳng những ảnh bị chặn thì được F1 = **0.907** (70 hoá đơn SROIE), 0.983 (10 văn bản mờ tổng hợp), 0.624 (15 ảnh VinText). Hai nguyên nhân:
  - Ngưỡng "lóa sáng" (`max_brightness` 240) dùng **độ sáng trung bình cả ảnh** — giấy trắng chụp tốt vốn có trung bình 241–253.
  - Ngưỡng mờ (Laplacian 60) tính trên **ảnh độ phân giải gốc**; ảnh 3264 px nét vẫn chỉ đạt 30–36.

## 4. VQA

Câu hỏi giữ nguyên tiếng Việt từ bộ dữ liệu; câu mô tả dùng câu mặc định của app "Phía trước có gì?". Lần gọi đầu gồm nạp model (11.1 s với MLX).

| Chỉ số | Qwen2.5-VL-3B 4bit (MLX, mặc định) | BLIP + MarianMT (cũ) |
|---|---|---|
| Hỏi đáp: tỉ lệ trả lời hợp lệ | 0.96 | 1.00 |
| Hỏi đáp: token recall so với đáp án | **0.447** | 0.188 |
| Mô tả: tỉ lệ hợp lệ | 0.84 | 1.00 |
| Thời gian trung vị (sau khi nạp) | 2.6–2.7 s | 0.86 s |

Backend cũ bỏ qua câu hỏi và chỉ mô tả ảnh chung chung (hỏi "đây là trường đại học nào?" → "Đó là tòa nhà nơi khách sạn được đặt"). Qwen trả lời đúng trọng tâm và đọc được chữ trong ảnh ("Trường Đại học Mở TP. Hồ Chí Minh", "xe máy", "Phường 8").

**Chấm tay** (xem ảnh, mẫu ngẫu nhiên): hỏi đáp 16 câu — 9 đúng, 2 đúng một phần, 4 sai, 1 từ chối ("Không xác định" với ảnh ban đêm rất rõ). Lỗi điển hình: đếm sai (2 cầu thay vì 1), nhầm "đội nón" thành "cầm nón", sai màu áo. Mô tả 8 ảnh — 6 tốt, 2 đúng một phần: cổng làng bị gọi là "cổng nhà thờ" và **bỏ sót con trâu cùng người đang đi tới**; một ảnh khác bỏ sót người đứng trước đền.

Câu trả lời bị loại (rồi chuyển sang fallback): 4% hỏi đáp, 16% mô tả — chủ yếu vì quy tắc "đúng một hoặc hai câu" trong `MLXVLMBackend._validated_answer` loại cả câu trả lời hợp lý nhưng dài ba câu.

**Guardrail an toàn** (16 câu hỏi): đúng 11/16.
- Không chặn: "Phía trước có vật cản không?", "Tôi có thể đi tiếp không?", "Đi về phía trước được không?" (bộ kiểm tra câu trả lời `check_answer` vẫn còn là lớp bảo vệ thứ hai).
- Chặn nhầm: "Biển báo ghi là đi thẳng hay rẽ trái?", "Tờ giấy này viết về an toàn thực phẩm phải không?" — do so khớp từ khoá "đi thẳng", "an toàn".

## 5. Tốc độ (M1, MPS)

| Thành phần | Thời gian |
|---|---|
| Detection YOLOv8n (chạy riêng) | 25 ms trung vị, p90 39 ms |
| Depth 256×256 (chạy riêng) | 60 ms |
| **Pipeline song song đầy đủ** (camera → detection → tracking → fusion, depth song song) | **22.3 FPS**; detection 42 ms (p95 70 ms), depth 86 ms (p95 101 ms) |
| OCR Vision / EasyOCR | 0.11–0.21 s / 0.5–7.6 s |
| VQA MLX | 2.6 s trung vị (p90 3.1 s), lần đầu 11 s |

Khi chạy song song, detection chậm hơn ~70% so với chạy riêng do tranh chấp GPU với depth, nhưng vẫn đạt thời gian thực. Độ trễ từ lúc chụp đến lúc ra quyết định cảnh báo khoảng 45–90 ms.

## 6. Phương án cải thiện (theo mức ưu tiên)

1. **Sửa môi trường và Vision OCR** (bắt buộc, công sức nhỏ):
   - Ghim `scipy==1.14.1` trong `requirements.txt` cho Python 3.10, hoặc nâng lên Python ≥ 3.11 để dùng scipy ≥ 1.16.
   - Trong `vision_engine.py`, truyền `None` thay cho `{}`.
   - Bắt lỗi trong `readtext` để chuyển sang EasyOCR thay vì báo lỗi.
2. **Đổi cách tính độ gần** (ảnh hưởng an toàn lớn nhất): bỏ chuẩn hoá min-max theo frame.
   - Phương án đã kiểm chứng: Metric-Indoor-Small + hệ số hiệu chỉnh theo camera + ngưỡng mét (ví dụ ≤ 1.5 m). Recall tăng 16% → 69%, model này chỉ chậm hơn 12 ms.
   - Nếu giữ model tương đối: chuẩn hoá theo phân vị cố định của vùng hành lang phía trên sàn thay cho min/max cả ảnh.
3. **Sửa bước kiểm tra chất lượng OCR:**
   - Đo độ mờ sau khi thu ảnh về khoảng 1000 px.
   - Đo lóa sáng bằng tỉ lệ pixel bão hoà (> 250) thay vì độ sáng trung bình.
   - Hoặc chỉ cảnh báo bằng lời mà vẫn đọc. Riêng hoá đơn, cách này tăng F1 từ 0.67 lên khoảng 0.9.
4. **Nhận diện:**
   - Fine-tune YOLOv8n/s trên dữ liệu trong nhà (ghế, bàn, cửa, cầu thang), vì hai lớp ghế/bàn đang có recall 0.30–0.33.
   - Cân nhắc YOLOv8s: mAP cao hơn, vẫn đủ nhanh trên M1.
   - Tăng cường dữ liệu với mờ chuyển động và thiếu sáng.
   - Bỏ lớp `bed` khỏi `alert_classes` hoặc đòi conf cao hơn, vì mặt bàn hay bị nhận nhầm là giường.
5. **Cảnh báo vật cản ngoài lớp COCO:** thêm luật "bất kỳ vùng nào trong hành lang (trừ sàn) có độ sâu metric < ngưỡng thì phát cảnh báo chung 'có vật cản phía trước'". Luật này dựa trên depth và không cần detection. Trên TUM, 77% frame có vật cản gần nhưng không có cảnh báo.
6. **VQA:**
   - Nới quy tắc "đúng 1–2 câu" (cắt lấy hai câu đầu thay vì loại cả câu trả lời).
   - Thêm vào prompt yêu cầu luôn nhắc người và con vật nếu có.
   - Viết lại guardrail theo ý định câu hỏi (hỏi về việc di chuyển của người dùng) thay vì khớp từ khoá, và bổ sung test cho các câu hỏi bị bỏ sót hoặc chặn nhầm ở trên.
7. **Tốc độ:**
   - Giữ depth 256×256: bản 518×518 chính xác hơn một chút (δ1 0.956) nhưng chậm gấp 5,4 lần.
   - Có thể chuyển YOLO sang Core ML để giảm tranh chấp GPU với depth.
   - Nạp sẵn model VQA khi máy rảnh để tránh lần đầu mất 11 s.

## 7. Giới hạn của đánh giá

- Chưa có video người khiếm thị đi lại thật; TUM là camera cầm tay quanh bàn, NYU là ảnh tĩnh.
- Khoảng cách thật lấy từ cảm biến Kinect (tối thiểu ~0.5–0.7 m), nên không đánh giá được vật sát dưới 0.7 m.
- Văn bản tiếng Việt chụp là dữ liệu tổng hợp; VinText là chữ cảnh, không phải tài liệu.
- VQA chỉ được chấm tay trên mẫu nhỏ (24 câu), bởi một người chấm.
- Hệ số hiệu chỉnh metric fit trên NYU; với webcam thật phải đo lại.
- Số liệu OCR/VQA đo sau khi vá tạm hai lỗi môi trường; trên máy hiện tại, nếu chưa sửa thì hai chức năng này không chạy.

## Chạy lại

```bash
W=/duong/dan/workdir                       # ngoài repo, ~5 GB
bash evaluation/benchmarks/setup_data.sh $W
cd evaluation/benchmarks
export PYTHONPATH=$W/pyfix HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
PY=/opt/anaconda3/envs/ai-macbook/bin/python
$PY eval_detection_coco.py --coco-dir $W/data/coco --num-images 500
$PY eval_depth_warning_nyu.py --nyu-dir $W/data/nyu
$PY eval_warning_tum.py --seq-dir $W/data/tum/rgbd_dataset_freiburg1_desk
$PY eval_ocr.py --data-dir $W/data --n 150
$PY eval_vqa.py --data-dir $W/data --n-qa 100 --n-describe 50 --image-out $W/vqa_images
```

Chạy tuần tự để số đo tốc độ không bị nhiễu. Số liệu pipeline song song lấy từ `scripts/evaluate_video.py` trên video ghép từ chuỗi TUM (TTS được thay bằng `sleep` cùng thời lượng để không phát tiếng).
