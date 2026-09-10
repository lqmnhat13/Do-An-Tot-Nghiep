# Project context

Hệ thống hỗ trợ người khiếm thị, chạy offline trên macOS Apple Silicon M1 16GB.

## Architecture invariants

- Detection, depth và cảnh báo an toàn phải tiếp tục hoạt động trong mọi chế độ.
- VQA và OCR chỉ chạy on-demand.
- Không bao giờ dùng VLM để khẳng định đường đi an toàn.
- Không thay đổi Detection, Depth hoặc Risk contracts nếu chưa cập nhật consumer.
- Bảo toàn drop-oldest camera buffering.
- Không thêm cloud API hoặc telemetry.

## Hardware constraints

- Target device: Apple M1, 16GB unified memory.
- Ưu tiên Core ML, MLX hoặc MPS.
- Không đưa dependency chỉ hoạt động với CUDA vào runtime chính.
- Model lớn phải lazy-load và có fallback.

## Verification

- Run: PYTHONPATH=. /opt/anaconda3/envs/ai-macbook/bin/python \
  -m unittest discover -s tests -p "test_*.py" -v
- Với thay đổi runtime, kiểm tra deadlock, shutdown và thread safety.
- Với thay đổi safety, thêm test cho lỗi và dữ liệu stale.

## Working rules

- Giữ thay đổi nhỏ và tập trung.
- Không sửa file ngoài phạm vi nếu không cần thiết.
- Không ghi đè thay đổi chưa commit của người dùng.
- Báo cáo test đã chạy và giới hạn còn lại.