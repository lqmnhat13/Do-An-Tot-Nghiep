# Kế hoạch triển khai: sửa lời báo độ gần và chặn cảnh báo từ depth cũ

> Tài liệu giao việc cho AI sửa code trong repository này. Thực hiện hai mục tiêu dưới đây; không mở rộng sang dẫn đường, thay model hoặc thêm tính năng cloud. Đọc `AGENTS.md` trước khi sửa. Đây là kế hoạch, chưa phải bằng chứng hệ thống đã an toàn để sử dụng ngoài thực tế.

## 1. Mục tiêu và phạm vi

1. Xóa mọi lời khẳng định khoảng cách theo mét được suy ra từ **Depth Anything V2 relative depth**. Giữ điểm `relative_proximity` để pipeline hiện tại tiếp tục hoạt động, nhưng lời nói và HUD phải thể hiện đây chỉ là độ gần **tương đối**.
2. Không tạo hoặc bắt đầu phát cảnh báo va chạm dựa trên depth đã cũ, detection đã cũ, hoặc cặp detection–depth lệch thời gian quá ngưỡng. Khi thiếu dữ liệu hợp lệ, thể hiện trạng thái không đánh giá được độ gần; không suy ra `LOW`/`NO_ALERT` hoặc khẳng định đường đi an toàn.
3. Bảo toàn detection, depth và đường cảnh báo trong mọi chế độ, kể cả OCR/VQA on-demand. Giữ cơ chế camera buffer `drop-oldest`, ưu tiên cảnh báo hợp lệ, khả năng shutdown và chạy offline trên M1 16 GB.

Không đổi public fields/ý nghĩa của `Detection`, `DepthMap`, `RiskAssessment` và `AudioTask` chỉ để hoàn thành việc này. Nếu thật sự phải đổi contract, tìm và cập nhật toàn bộ consumer cùng test trong cùng thay đổi.

## 2. Hiện trạng đã kiểm tra trong mã

- `src/depth/roi_extractor.py`: `extract_proximity()` chuẩn hóa relative depth theo **từng frame**, rồi dùng `0.4 + (1 - score) * 3.6` để tạo số mét như `~0.8m` và `> 3m`. Công thức này chưa được hiệu chuẩn bằng khoảng cách thật.
- `src/fusion/synchronizer.py`: có `DataQuality.VALID/DEGRADED/STALE/UNAVAILABLE`, nhưng ngưỡng `STALE` 2000 ms đang viết cứng; `max_alert_age_ms` được lưu nhưng chưa dùng. `register_depth()` và `synchronize()` được gọi từ các worker khác nhau.
- `src/fusion/risk_fsm.py`: vẫn trích ROI khi có `depth_map` dù chất lượng cặp là `DEGRADED` hoặc `STALE`; nhánh từ chối hiện chỉ kiểm tra `UNAVAILABLE` hoặc điểm bằng 0. Bộ đếm hysteresis cần được xử lý khi dữ liệu mất hiệu lực.
- `src/fusion/alert_aggregator.py`: lọc theo mức `HIGH/MEDIUM` mà chưa loại assessment có chất lượng kém hoặc hết hạn; `AudioTask.expires_at` hiện được tính từ thời điểm tạo task, không ràng buộc với tuổi dữ liệu nguồn.
- `src/audio/audio_coordinator.py`: kiểm tra task hết hạn khi vào queue và khi lấy ra, nhưng cần kiểm tra lại **sau khi phát chime và trước khi gọi TTS** vì chime có thể làm task hết hạn.
- `configs/fusion_rules.yaml`: có các ngưỡng `max_pair_skew_ms`, `max_detection_age_ms`, `max_depth_age_ms`, `max_alert_age_ms`; `app.py` chưa truyền `max_alert_age_ms` vào `Synchronizer`. `scripts/evaluate_video.py` khởi tạo các thành phần bằng giá trị mặc định.
- `src/ui/hud_renderer.py`, `src/runtime/system_coordinator.py`, `README.md`, `HUONG_DAN_SU_DUNG.md`, `PHAN_TICH_DU_AN.md` dùng hoặc giải thích `proximity_desc`; rà lại các chuỗi hiển thị sau khi sửa.

## 3. Quyết định kỹ thuật bắt buộc

### 3.1. Ý nghĩa của điểm và lời báo độ gần

- **Giữ** `relative_proximity: float [0, 1]`, hướng `near_is_larger`, phép lấy percentile ROI, `valid_mask` và cách xử lý ROI không hợp lệ. Không thay các ngưỡng RiskFSM trong nhiệm vụ này nếu test không chứng minh cần thiết.
- **Bỏ** phép chuyển `relative_proximity` sang mét và mọi chuỗi `~Xm`, `> Xm` bắt nguồn từ nó. `reason` dùng để debug cũng không được ngụ ý đây là khoảng cách mét.
- `proximity_desc` chỉ được mô tả theo thang tương đối, ví dụ `độ gần tương đối cao/trung bình/thấp` hoặc `chưa rõ`. Chọn một bộ từ cố định, dễ đọc bằng TTS, và dùng nhất quán trong HUD, log và tài liệu.
- Cảnh báo âm thanh phải ngắn: ví dụ `Chú ý, có ghế phía trước` cho HIGH; có thể đọc `độ gần tương đối cao` nếu cần, nhưng **không** đọc số mét và không nói `đường đi an toàn`. Không dùng `rất gần` như một khẳng định khoảng cách vật lý tuyệt đối.
- Chỉ khi sau này có nguồn metric depth đã hiệu chuẩn, đánh giá sai số và nhánh xử lý riêng mới được cân nhắc phát khoảng cách mét. Việc đó nằm ngoài phạm vi kế hoạch này.

### 3.2. Chính sách chất lượng dữ liệu cho cảnh báo dựa trên depth

| Chất lượng cặp | Xử lý RiskFSM | Xử lý âm thanh |
| --- | --- | --- |
| `VALID` | Tính độ gần và nguy cơ như hiện tại. | Có thể cảnh báo, nhưng phải còn hạn tại lúc bắt đầu phát. |
| `DEGRADED` | Không dùng depth của cặp này để tạo nguy cơ dựa trên độ gần. Trả `UNDETERMINED` cho đánh giá không đủ dữ liệu. | Không phát cảnh báo va chạm từ cặp này; trạng thái suy giảm riêng có thể được báo ngắn, chống lặp. |
| `STALE` hoặc `UNAVAILABLE` | Không trích ROI; trả `UNDETERMINED`. | Không phát cảnh báo va chạm từ cặp này. |

`DEGRADED` có thể do một nguồn quá tuổi hoặc hai nguồn lệch nhau. Cả hai trường hợp đều không đủ cơ sở gắn độ gần của một frame cho detection từ frame khác. **Không tắt detection hoặc depth worker** khi từ chối cảnh báo.

### 3.3. Hạn dùng của cảnh báo

Áp dụng kiểm tra ở ba điểm: sau đồng bộ, trước khi tạo `AudioTask`, và ngay trước khi TTS bắt đầu nói. Dùng đồng hồ `time.monotonic()` và cho phép truyền `current_mono` trong test để tránh test phụ thuộc sleep.

- `Synchronizer` phải tính chất lượng theo các ngưỡng cấu hình, bỏ ngưỡng `STALE = 2000 ms` viết cứng hoặc giải thích rõ ranh giới `DEGRADED`/`STALE` bằng cấu hình. Mọi cặp quá `max_detection_age_ms`, `max_depth_age_ms` hoặc `max_pair_skew_ms` đều không được dùng cho cảnh báo depth. Chỉ cần `VALID` mới được đi tiếp.
- Khi có `RiskAssessment`, `AlertAggregator` chỉ chọn assessment `data_quality == VALID`, chưa hết `RiskAssessment.expires_at`, và chưa quá hạn từ `source_timestamp`. Hạn tối đa cho cảnh báo phải **không dài hơn** tuổi tối đa của detection, depth và `max_alert_age_ms`; có thể dùng `min(max_detection_age_ms, max_depth_age_ms, max_alert_age_ms)` tính từ `source_timestamp` vì timestamp này là thời điểm của nguồn cũ hơn. Đây là ràng buộc bảo thủ, dễ kiểm chứng.
- Đặt `AudioTask.expires_at` bằng thời điểm **sớm nhất** giữa hạn của assessment, hạn dữ liệu nguồn và giới hạn thời gian chờ task hiện có. Không gia hạn dữ liệu bằng cách đặt `now + 3 s` nếu hạn nguồn đến trước.
- Nếu task hết hạn trong lúc phát chime, bỏ câu TTS của task. Phạm vi bảo đảm là dữ liệu còn hạn **khi câu cảnh báo bắt đầu được nói**; không cần ngắt một câu ngắn đã bắt đầu hợp lệ chỉ vì hạn trôi qua giữa câu.
- Assessment bị loại vì dữ liệu cũ không được tiêu thụ cooldown của track. Khi dữ liệu mới trở lại, cảnh báo hợp lệ phải có thể phát theo quy tắc cooldown hiện có.

## 4. Trình tự sửa code

### Bước A — Lời báo độ gần

1. Sửa `src/depth/roi_extractor.py`: giữ score, xóa công thức mét, trả mô tả tương đối và `reason` phù hợp. Với ROI không hợp lệ, vẫn trả `(0.0, "chưa rõ", lý do)`.
2. Sửa `src/fusion/alert_aggregator.py` để câu HIGH/MEDIUM không diễn giải nhãn tương đối thành khoảng cách thật. Giữ hướng trái/giữa/phải và thứ tự ưu tiên hiện có.
3. Rà `src/ui/hud_renderer.py`, `src/runtime/system_coordinator.py` và các tài liệu liên quan. Tìm toàn repo các chuỗi `~...m`, `> 3m`, `khoảng cách` gần `relative_proximity` và sửa các ví dụ/tuyên bố sai. Không xóa số mét từ nguồn dữ liệu metric thật nếu có.

### Bước B — Ngăn stale trong pipeline

1. Sửa `src/fusion/synchronizer.py` để sử dụng ngưỡng cấu hình nhất quán. Cân nhắc lock tối thiểu quanh việc cập nhật/đọc `_latest_depth` và `_depth_history`; lấy snapshot dưới lock, xử lý ngoài lock. Không giữ lock trong inference hoặc TTS. Không cho kết quả depth cũ đến muộn ghi đè `_latest_depth` mới hơn.
2. Sửa `src/fusion/risk_fsm.py`: chỉ trích ROI khi cặp `VALID`; khi không hợp lệ, trả `UNDETERMINED`, `proximity_desc="chưa rõ"`, và reset bộ đếm xác nhận phụ thuộc depth cho track đó. Khi có depth mới, xác nhận phải bắt đầu từ mẫu mới; giữ đường kích hoạt HIGH ở giữa cho **mẫu mới hợp lệ**.
3. Sửa `src/fusion/alert_aggregator.py`: lọc `data_quality`, hạn assessment và hạn nguồn **trước** khi xếp hạng hoặc đánh dấu cooldown. Dùng một chính sách hạn được nạp từ `configs/fusion_rules.yaml`; đừng để ngưỡng trong `app.py` khác ngưỡng trong `scripts/evaluate_video.py`.
4. Sửa `src/audio/audio_coordinator.py`: kiểm tra lại `task.is_expired()` sau chime, ngay trước `tts_engine.speak()`; bảo toàn preemption và shutdown.
5. Sửa wiring trong `app.py` và `scripts/evaluate_video.py` cho mọi tham số mới. Không thêm dependency hoặc tác vụ mạng.

### Bước C — Tài liệu và báo cáo

1. Cập nhật `README.md`, `HUONG_DAN_SU_DUNG.md` và `PHAN_TICH_DU_AN.md` ở các đoạn nói về depth tương đối, khoảng cách và chất lượng dữ liệu.
2. Ghi rõ giới hạn: xử lý stale và bỏ số mét giả **không chứng minh** độ chính xác vật lý của các ngưỡng RiskFSM, khả năng phát hiện mọi vật cản hoặc an toàn di chuyển.

## 5. Test bắt buộc

Mở rộng test đang có, không chỉ viết test kiểm tra chuỗi nội bộ của một hàm:

| File | Tình huống và kỳ vọng |
| --- | --- |
| `tests/test_depth_roi.py` | Score cao/trung bình/thấp và ROI invalid; mô tả/reason không chứa khoảng cách mét giả; score và chiều gần/xa giữ nguyên. |
| `tests/test_fusion.py` | Depth/detection mới và đồng bộ vẫn tạo HIGH; depth cũ, detection cũ, cùng `frame_id` nhưng timestamp cũ, cặp lệch quá ngưỡng đều `UNDETERMINED` và không tạo cảnh báo; depth phục hồi bằng frame mới thì cảnh báo hợp lệ trở lại. Kiểm tra không dùng lại bộ đếm hysteresis cũ. |
| `tests/test_fusion.py` hoặc test aggregator riêng | Assessment `STALE/DEGRADED/UNAVAILABLE`, assessment hết hạn, hoặc nguồn quá hạn đều bị bỏ; không đánh dấu cooldown; `AudioTask.expires_at` không muộn hơn hạn nguồn. |
| `tests/test_audio_priority.py` | Task hết hạn trong thời gian chime hoặc nằm chờ trong queue không được gọi `speak`; HIGH hợp lệ vẫn ngắt tác vụ ưu tiên thấp hơn. |
| `tests/test_end_to_end.py` | Trong lúc OCR/VQA on-demand, HIGH hợp lệ vẫn được phát; HIGH tạo từ depth stale không được phát/ngắt tiếng; shutdown không deadlock. |
| Test đồng thời có kiểm soát | `register_depth()` và `synchronize()` chạy từ hai luồng, không lỗi/cặp nửa cập nhật; kết quả cũ đến muộn không thay thế depth mới. Dùng event/barrier thay cho sleep dài. |

Không cần thay ngưỡng rủi ro chỉ để test dễ qua. Nếu test lộ ra thiếu sót khác, sửa tối thiểu và giải thích trong báo cáo.

## 6. Kiểm chứng cuối cùng

Chạy đúng lệnh dự án:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
/opt/anaconda3/envs/ai-macbook/bin/python \
-m unittest discover -s tests -p "test_*.py" -v
```

Sau đó kiểm tra thủ công bằng video hoặc webcam trên M1 với: (1) vật cản và depth mới; (2) depth đứng hình/chậm nhưng detection vẫn chạy; (3) depth phục hồi; (4) VQA/OCR chạy đồng thời. Quan sát HUD, câu TTS, thời điểm bắt đầu phát và shutdown. Nếu không có camera/video thật, ghi rõ chưa kiểm chứng phần này; không coi kết quả từ `dummy` là bằng chứng sử dụng thực tế.

## 7. Tiêu chí hoàn thành và báo cáo của AI triển khai

- Không còn số mét suy ra từ relative depth trong HUD, TTS, log người dùng và ví dụ tài liệu.
- Không có cảnh báo va chạm dựa trên depth/detection quá tuổi, cặp quá lệch, assessment quá hạn hoặc task hết hạn trước khi TTS bắt đầu.
- Dữ liệu không hợp lệ được thể hiện là không đánh giá được độ gần; detection/depth tiếp tục chạy, và cảnh báo từ dữ liệu mới phục hồi bình thường.
- OCR/VQA vẫn on-demand; HIGH hợp lệ vẫn ưu tiên; camera buffer vẫn drop-oldest; không thêm cloud/telemetry.
- Test toàn bộ chạy qua. Báo cáo file đã sửa, test đã chạy và kết quả, tình huống thử trên M1, cùng các giới hạn chưa kiểm chứng. Không tuyên bố hệ thống bảo đảm đường đi an toàn.
