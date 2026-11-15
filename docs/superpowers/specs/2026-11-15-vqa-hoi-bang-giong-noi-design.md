# VQA hỏi bằng giọng nói — thiết kế

Ngày: 2026-11-15

## Mục đích

Hiện phím Q luôn gửi câu cố định "Phía trước có gì?". Người khiếm thị cần hỏi điều mình
muốn biết về khung cảnh trước mặt. Ở chế độ VQA, hệ thống nghe câu hỏi bằng giọng nói,
chuyển thành chữ và để model VQA trả lời dựa trên ảnh camera chụp đúng lúc hỏi.

## Mục tiêu / Không làm / Không được xảy ra

- Mục tiêu: bấm Q → nói câu hỏi tiếng Việt → nghe câu trả lời cho đúng câu đó, offline.
- Không làm: thay đổi OCR (phím Space giữ nguyên); từ khoá đánh thức, nghe liên tục;
  hỏi tiếng Anh; nhập câu hỏi bằng bàn phím.
- Không được xảy ra: gửi âm thanh lên cloud; VQA khẳng định đường đi an toàn (kể cả khi
  câu hỏi tự do hỏi trực tiếp); app treo hoặc im lặng khi micro/model lỗi.

## Ràng buộc

- Offline (`SECOND_EYE_OFFLINE`), nạp model qua `pretrained_kwargs()`; chỉ script tải model
  được tải mạng. Apple Speech bị loại vì tiếng Việt không có bản on-device (đã kiểm tra).
- M1 16 GB chạy cùng YOLO + depth + Qwen; model lớn phải lazy-load và có fallback.
- CI Ubuntu không có micro/MPS: test phải mock micro và model.

## Các cách đã cân nhắc

| Cách | Đánh đổi |
|---|---|
| **A. PhoWhisper-small (VinAI) qua `transformers` — ĐÃ CHỌN** | Tối ưu cho tiếng Việt, dùng thư viện đã có, ~1 GB, MPS. Chỉ tiếng Việt. |
| B. Whisper large-v3-turbo qua `mlx-whisper` | Đa ngôn ngữ, tốt nhưng thêm dependency, ~1,6 GB, tranh MLX/RAM với Qwen. |
| C. Whisper small qua `mlx-whisper` | Nhẹ nhưng tiếng Việt kém, sai câu hỏi thì sai câu trả lời. |

Người dùng chọn A vì câu hỏi bằng tiếng Việt và không thêm thư viện mới.

## Luồng sử dụng (đề xuất của Claude)

1. Người dùng bấm **Q** (đang ở bất kỳ chế độ nào → chọn VQA như hiện tại, cảnh báo vật
   cản tắt tiếng như hiện tại).
2. App phát tiếng "bíp" ngắn (`assets/audio/chime.wav`), không đọc câu dài để micro không thu
   lẫn giọng máy. Mọi âm thanh khác dừng trong lúc thu.
3. Thu micro 16 kHz mono. Dừng khi: im lặng ~1,2 s sau khi đã có tiếng nói, hoặc tối đa
   8 s, hoặc bấm Q lần nữa. Không ai nói trong 5 s đầu → dừng.
4. **Chụp ảnh lúc dừng thu** (thời điểm người dùng hỏi xong), qua kiểm tra ảnh mới như hiện tại.
5. PhoWhisper chuyển âm thanh thành chữ (lazy-load lần đầu bấm Q; chuẩn hoá NFC).
6. Đọc lại ngắn "Bạn hỏi: <câu hỏi>." để người dùng biết máy nghe đúng không; HUD hiện câu hỏi. Chỉ
   đọc lại khi câu hỏi đến từ giọng nói; nhánh fallback (xem "Khi có lỗi") không đọc lại.
7. Guardrail câu hỏi → model VQA với ảnh ở bước 4 → đọc câu trả lời (luồng VQA hiện có).

## Khi có lỗi

| Tình huống | App nói |
|---|---|
| Không nghe được tiếng nói / chữ rỗng | "Không nghe rõ câu hỏi. Bấm Q và hỏi lại." — không chạy VQA. |
| Micro bị từ chối quyền hoặc lỗi (kể cả bản ghi toàn số 0 khi macOS chặn quyền Micro), hoặc chưa có model PhoWhisper | "Không dùng được micro hoặc nhận dạng giọng nói. Hệ thống sẽ mô tả phía trước." rồi dùng câu mặc định "Phía trước có gì?" (fallback). Nhánh này không đọc lại "Bạn hỏi: …" vì người dùng không nói câu đó. |
| Nhận dạng quá hạn (15 s) | Như lỗi model: báo rồi dùng câu mặc định, không đọc lại "Bạn hỏi". |
| Bấm S / C trong lúc thu hoặc nhận dạng | Huỷ, không đọc kết quả muộn (cơ chế token hiện có). |

## Guardrail câu hỏi

Câu hỏi giờ là tự do nên `SafetyGuardrail.check()` thành lớp chặn đầu tiên. Sửa trong cùng
việc này: 8/30 câu hỏi bị lọt trong `eval_guardrail_cases.py` (ví dụ "Đường trống không?",
"Phía trước thông thoáng chứ?", "Cứ đi thẳng là được đúng không?", câu tiếng Anh), giữ
20/20 câu hỏi thường không bị chặn. Câu không dấu không cần xử lý vì PhoWhisper luôn ra
chữ có dấu. Lớp kiểm tra câu trả lời giữ nguyên.

## Phần để người làm quyết định

- Cách đo im lặng (ngưỡng năng lượng RMS đơn giản là đủ), tham số trong `configs/app_config.yaml`
  hoặc `model_config.yaml` (khối `speech`).
- Vị trí module (gợi ý `src/speech/`), cách nối vào `SystemCoordinator._run_vqa_task`,
  phím Q khi đang thu.
- Thêm PhoWhisper vào `scripts/download_models.py` và `models/manifest.json`; cập nhật README,
  CLI.md, HUONG_DAN_SU_DUNG.md (quyền Micro cho Terminal).
- Đo độ trễ nhận dạng trên M1 và ghi vào báo cáo.
