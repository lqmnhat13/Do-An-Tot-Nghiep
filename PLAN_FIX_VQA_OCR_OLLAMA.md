# KẾ HOẠCH NÂNG CẤP VÀ KHẮC PHỤC CHẤT LƯỢNG OCR & VQA

> **Mục tiêu:** Khắc phục triệt để tình trạng OCR đọc chữ đảo lộn/vô nghĩa trên lọ thuốc và VQA mô tả sai hoặc trả lời "Không xác định rõ từ ảnh", đồng thời tích hợp thêm backend Ollama (Local Vision LLM) làm tùy chọn nâng cao cho chất lượng vượt trội trên macOS Apple Silicon M1 16GB.

---

## 1. PHÂN TÍCH HIỆN TRẠNG & ĐIỂM NGHẼN

### 1.1. Điểm nghẽn OCR (Nhận diện chữ trên nhãn thuốc/tài liệu)
1. **Lỗi gom dòng thô sơ (`src/ocr/ocr_service.py`):**
   - Thuật toán cũ: `valid_items.sort(key=lambda x: (int(x[0] // 20), x[1]))`.
   - Bất cứ khi nào nhãn thuốc bị nghiêng nhẹ hoặc chia theo cột (Thành phần - Hàm lượng), các từ bị nhảy cóc giữa các dòng và cột khác nhau.
   - Nối toàn bộ bằng khoảng trắng đơn `" ".join(...)` khiến câu chữ dính chùm, giọng đọc TTS không thể ngắt nghỉ tự nhiên.
2. **Thiếu tiền xử lý cho camera tiêu cự cố định (Fixed Focus):**
   - Webcam laptop không có Macro lấy nét cận cảnh. Ký tự 6–8pt trên lọ thuốc bị nhòe và thấp hơn 10 pixel. EasyOCR cần ảnh được phóng đại cục bộ và tăng tương phản (CLAHE) trước khi đưa vào mô hình nhận dạng CRAFT.

### 1.2. Điểm nghẽn VQA (Mô tả cảnh và hỏi đáp hình ảnh)
1. **Prompt ức chế suy luận (`src/vqa/backend.py`):**
   - Quy tắc `"Nếu không chắc chắn, trả lời ‘Không xác định rõ từ ảnh.’"` khiến mô hình Qwen2-VL-2B (vốn nhỏ và độ tự tin thấp) chọn ngay câu này làm lối thoát thay vì quan sát và mô tả đồ vật.
2. **Bộ lọc đầu ra quá cứng nhắc (`_validated_answer`):**
   - Giới hạn cứng $\le 25$ từ và bắt buộc duy nhất 1 dấu chấm câu.
   - Khi mô hình sinh câu mô tả chi tiết 26–28 từ hoặc chia làm 2 câu ngắn (ví dụ: *"Trước mặt có cái bàn. Trên bàn có lọ thuốc."*), hệ thống lập tức vứt bỏ và báo lỗi fallback về câu vô định.

### 1.3. Nhu cầu bổ sung Ollama (Local Vision LLM)
- Các mô hình Vision trên Ollama (đặc biệt là `minicpm-v:2.6` và `llama3.2-vision`) có năng lực đọc hiểu tài liệu, bảng biểu và nhãn thuốc tiếng Việt ở đẳng cấp vượt trội so với EasyOCR truyền thống.
- Ollama chạy cục bộ 100% qua Metal (Apple Silicon), quản lý VRAM thông minh và hoàn toàn không vi phạm nguyên tắc offline của đồ án.

---

## 2. KIẾN TRÚC & RÀNG BUỘC HỆ THỐNG (INVARIANTS)

Để đảm bảo hệ thống vận hành an toàn trên Mac M1 16GB:
1. **Offline & Zero Cloud:** Không gọi bất kỳ cloud API nào. Ollama chỉ giao tiếp qua `http://127.0.0.1:11434` trên máy local.
2. **Ưu tiên an toàn:** Pipeline phát hiện vật cản (Detection + Depth + Cảnh báo rủi ro) vẫn là luồng ưu tiên số 1, chạy liên tục và không bị khóa bởi OCR/VQA.
3. **On-Demand Single-Worker:** OCR và VQA chỉ chạy khi người dùng bấm phím hoặc gọi lệnh CLI. Không chạy nền ngầm tiêu tốn RAM.
4. **Tự động giải phóng RAM:** Backend Ollama phải cấu hình `keep_alive` hợp lý (hoặc giải phóng ngay sau khi suy luận) để không giữ VRAM của GPU M1.
5. **Tương thích ngược:** Tất cả các backend cũ (`mlx_vlm`, `legacy_caption`, `disabled`, `easyocr`) phải tiếp tục hoạt động bình thường; test suite 91 tests phải luôn xanh.

---

## 3. LỘ TRÌNH TRIỂN KHAI CHI TIẾT (4 GIAI ĐOẠN)

### GIAI ĐOẠN 1: Cải tiến thuật toán gom dòng & Tiền xử lý OCR (`src/ocr/ocr_service.py`)
* **Nhiệm vụ 1.1 - Thuật toán phân cụm dòng hình học (Geometric Line Clustering):**
  - Trích xuất toạ độ đầy đủ của bounding box: `top_y, bottom_y, left_x, right_x, center_y, height`.
  - Gom các từ vào cùng một dòng nếu độ phủ theo trục dọc ($IoU_y$) vượt ngưỡng hoặc tâm $y$ nằm trong khoảng chiều cao của dòng.
  - Sắp xếp các từ trong cùng một dòng từ trái sang phải theo `left_x`.
  - Sắp xếp các dòng từ trên xuống dưới theo `top_y`.
* **Nhiệm vụ 1.2 - Định dạng ngắt câu cho giọng đọc TTS:**
  - Nối các từ trong dòng bằng khoảng trắng.
  - Nối các dòng bằng dấu chấm câu và xuống dòng (`".\n"`) để khi đọc to, TTS tự động ngắt nghỉ giữa các mục (Tên thuốc -> Thành phần -> Hướng dẫn).
* **Nhiệm vụ 1.3 - Tự động phóng đại và tăng tương phản:**
  - Tự động áp dụng bộ lọc CLAHE (Contrast Limited Adaptive Histogram Equalization) khi ảnh có độ tương phản thấp.
  - Hỗ trợ upscale nhẹ (1.5x) với ảnh chữ nhỏ để EasyOCR nhận diện rõ nét từng ký tự.

### GIAI ĐOẠN 2: Nới lỏng bộ lọc & Tinh chỉnh Prompt cho VQA (`src/vqa/backend.py`)
* **Nhiệm vụ 2.1 - Cải tiến System Prompt:**
  - Giảm tính tiêu cực của prompt: Thay vì ép *"Nếu không chắc chắn, trả lời 'Không xác định rõ từ ảnh'"*, hướng dẫn mô hình: *"Hãy mô tả trung thực những đồ vật, màu sắc và chữ nhìn thấy được trong ảnh."*.
* **Nhiệm vụ 2.2 - Nới lỏng bộ lọc `_validated_answer`:**
  - Nâng giới hạn ngân sách từ lên **35 từ** (thay vì 25 từ).
  - Cho phép 1 đến 2 câu ngắn hoàn chỉnh (thay vì bắt buộc đúng 1 câu duy nhất).
  - Giữ nguyên các cơ chế chống cắt cụt giữa chừng và chống suy diễn cảm xúc/nghề nghiệp.

### GIAI ĐOẠN 3: Xây dựng Backend Ollama (`OllamaVLMBackend`)
* **Nhiệm vụ 3.1 - Xây dựng `OllamaVLMBackend` trong `src/vqa/backend.py`:**
  - Sử dụng thư viện chuẩn `urllib.request` và `json` (không cần cài thêm package bên thứ ba).
  - Kết nối đến `http://127.0.0.1:11434/api/chat`.
  - Mã hoá ảnh sang Base64 chuẩn định dạng của Ollama.
  - Gửi prompt hỏi đáp và nhận câu trả lời dạng JSON hoàn chỉnh.
  - Đặt timeout an toàn (25s) và cơ chế báo lỗi thân thiện nếu chưa bật dịch vụ Ollama.
* **Nhiệm vụ 3.2 - Cấu hình trong `configs/model_config.yaml`:**
  - Thêm cấu hình cho backend `ollama`:
    ```yaml
    vqa:
      backend: "mlx_vlm" # mlx_vlm | ollama | legacy_caption | disabled
      ollama:
        host: "http://127.0.0.1:11434"
        model: "minicpm-v:latest" # hoặc llama3.2-vision:latest
        timeout_sec: 25.0
        keep_alive: "5m"
    ```
* **Nhiệm vụ 3.3 - Hỗ trợ OCR qua Ollama trong CLI `scripts/run_ocr_camera.py`:**
  - Thêm cờ `--backend ollama` vào CLI đọc chữ. Khi kích hoạt, VLM sẽ đọc nhãn thuốc theo ngữ cảnh thông minh thay vì chỉ nhận diện từng từ rời rạc.

### GIAI ĐOẠN 4: Kiểm thử, Tối ưu hoá & Tài liệu hoá
* **Nhiệm vụ 4.1 - Unit Tests:**
  - Thêm `tests/test_ocr_clustering.py` kiểm tra thuật toán gom dòng không bị xáo trộn vị trí.
  - Thêm test case cho `OllamaVLMBackend` (dùng mock response để test luôn chạy offline được).
  - Đảm bảo 100% unit tests (91+ tests) đều xanh.
* **Nhiệm vụ 4.2 - Hướng dẫn sử dụng:**
  - Cập nhật [HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md) hướng dẫn cách cài đặt model Ollama (`ollama pull minicpm-v`) và cách chuyển đổi giữa các backend.

---

## 4. MA TRẬN SO SÁNH VÀ KẾ HOẠCH BẢO VỆ ĐỒ ÁN

| Phương án | Điểm mạnh | Hạn chế | Ý nghĩa trong Đồ án |
|---|---|---|---|
| **EasyOCR cải tiến** (Giai đoạn 1) | Siêu nhẹ, 0 extra RAM, chạy tức thì trong app | Không hiểu ngữ nghĩa sâu của thuốc | Phương án cơ sở nhẹ nhàng (Lightweight baseline) |
| **MLX Qwen2-VL 2B** (Giai đoạn 2) | Chạy native Metal, không cần phần mềm phụ, phản hồi nhanh (1.8s) | Mô hình 2B nhỏ nên khả năng đọc chữ chi tiết ở mức trung bình | Cân bằng tốt nhất giữa tốc độ và tài nguyên |
| **Ollama MiniCPM-V** (Giai đoạn 3) | Đọc nhãn thuốc, thành phần cực kỳ thông minh và trôi chảy | Cần cài Ollama, tốn 5GB RAM, mất 8–12s trên M1 | Minh chứng công nghệ VLM hiện đại cho báo cáo |

Việc tích hợp cả 3 phương án vào một hệ thống thống nhất sẽ tạo ra một đề tài đồ án có tính học thuật và thực tiễn rất cao, thể hiện rõ tư duy phân tích sự đánh đổi kỹ thuật (Engineering Trade-offs).
