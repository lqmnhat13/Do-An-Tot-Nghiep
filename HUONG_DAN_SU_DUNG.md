# Hướng dẫn sử dụng hệ thống AI đa phương thức hỗ trợ người khiếm thị

> Đối chiếu với mã nguồn ngày 17/09/2026, commit `12be6a7ab13cdafc3e6dd60fa1353ade3eeaa17d`. Thiết bị mục tiêu: macOS trên Apple Silicon M1, 16 GB RAM. Tài liệu hướng dẫn thao tác với phiên bản hiện tại; không giả định các chức năng mới chỉ được ghi trong cấu hình đã hoạt động đầy đủ.

## Mục lục

1. [Hệ thống làm được gì?](#gioi-thieu)
2. [Chuẩn bị trước khi cài đặt](#chuan-bi)
3. [Cài môi trường Python và thư viện](#cai-dat)
4. [Chuẩn bị model để sử dụng offline](#model)
5. [Kiểm tra camera, âm thanh và môi trường](#kiem-tra)
6. [Khởi động ứng dụng](#khoi-dong)
7. [Sử dụng các chức năng hằng ngày](#su-dung)
8. [Đọc thông tin trên giao diện](#giao-dien)
9. [Thay đổi cấu hình](#cau-hinh)
10. [Thử riêng VQA với câu hỏi tùy chọn](#vqa-rieng)
11. [Kiểm thử, benchmark và lưu log](#danh-gia)
12. [Xử lý lỗi thường gặp](#xu-ly-loi)
13. [Kịch bản trình diễn và bảng thao tác nhanh](#trinh-dien)
14. [Phạm vi xác minh của hướng dẫn](#xac-minh)

<a id="gioi-thieu"></a>
## 1. Hệ thống làm được gì?

Ứng dụng dùng camera để nhận biết một số vật thể, ước lượng độ gần tương đối và đưa ra cảnh báo bằng giọng nói tiếng Việt. Người dùng còn có thể nhấn phím để đọc chữ trong ảnh hoặc yêu cầu mô tả khung cảnh.

| Chức năng | Cách kích hoạt trong ứng dụng chính | Kết quả |
|---|---|---|
| Quan sát và cảnh báo | Tự chạy sau khi khởi động | Vật thể xuất hiện trên giao diện; một số tình huống tạo âm báo/lời cảnh báo |
| Đọc chữ — OCR | Nhấn `SPACE` | Chụp ảnh hiện tại, nhận dạng chữ và đọc kết quả |
| Mô tả cảnh — VQA | Nhấn `Q` | Chụp ảnh hiện tại và xử lý yêu cầu “Mô tả khung cảnh phía trước” |
| Dừng lời nói/yêu cầu hiện tại | Nhấn `S` | Ngắt âm thanh và vô hiệu kết quả on-demand đang chờ |
| Thoát | Nhấn `ESC` | Dừng ứng dụng và đóng cửa sổ |

Detection, depth và đánh giá nguy cơ tiếp tục hoạt động trong lúc OCR/VQA xử lý. Cảnh báo HIGH có thể chen ngang lời đọc; cảnh báo MEDIUM tạm không chen ngang trong chế độ on-demand.

Đây là bản mẫu phục vụ nghiên cứu và trình diễn. Mức nguy cơ thấp hoặc không có lời cảnh báo không xác nhận rằng đường đi an toàn. Các con số mét xuất hiện trên giao diện/lời đọc hiện là ước lượng theo quy tắc chưa hiệu chuẩn. Khi thử nghiệm, nên quan sát tại chỗ hoặc trong khu vực được kiểm soát, có người hỗ trợ khi cần; không dùng kết quả để thay thế phương tiện hỗ trợ di chuyển đang có.

**Cách đọc tài liệu:** người cài đặt lần đầu đọc từ mục 2 đến 6. Khi máy đã sẵn sàng, người vận hành có thể bắt đầu từ mục 7 và dùng bảng cuối tài liệu. Hướng dẫn kiến trúc và giải thích kỹ thuật chi tiết nằm trong [PHAN_TICH_DU_AN.md](PHAN_TICH_DU_AN.md).

<a id="chuan-bi"></a>
## 2. Chuẩn bị trước khi cài đặt

### 2.1. Thiết bị và phần mềm

Chuẩn bị:

- Máy Mac Apple Silicon; cấu hình mục tiêu của đề tài là M1 với 16 GB bộ nhớ hợp nhất.
- Webcam tích hợp hoặc camera ngoài được macOS nhận diện.
- Loa hoặc thiết bị phát âm thanh đang hoạt động.
- Conda dành cho Apple Silicon và môi trường Python 3.10.
- Mã nguồn dự án đã có trên máy.
- Kết nối mạng cho lần cài thư viện, tải model và tải giọng đọc nếu máy chưa có.
- Dung lượng đĩa cho môi trường Python và nhiều bộ trọng số. Repository chưa cung cấp số dung lượng tổng đã được xác minh; theo dõi dung lượng còn trống trong quá trình chuẩn bị.

Không cần CUDA. Nhánh PyTorch dùng MPS nếu khả dụng; backend VQA MLX dành cho Apple Silicon. Chạy trên Windows/Linux không phải đường sử dụng đầy đủ được hướng dẫn ở đây vì phần camera và TTS phụ thuộc macOS.

### 2.2. Xác định thư mục gốc

Trên máy hiện tại, mở Terminal và chạy:

```bash
cd /Users/lenhat/Do-An-Tot-Nghiep
pwd
ls app.py requirements.txt configs
```

Kết quả cần thấy là đường dẫn dự án và các tệp/thư mục nêu trên. Nếu dùng máy khác, thay đường dẫn sau `cd` bằng vị trí thực tế đã lưu mã nguồn.

**Các lệnh bên dưới đều được chạy từ thư mục gốc này**, trừ khi được ghi khác. Chạy ở thư mục khác có thể khiến ứng dụng không đọc đúng YAML, weights hoặc âm thanh do nhiều đường dẫn là tương đối.

### 2.3. Chọn cách chuẩn bị môi trường

Nếu đã có môi trường `ai-macbook`, dùng môi trường đó để kiểm tra trước. Không cần tạo lại hoặc nâng cấp tất cả thư viện chỉ để mở ứng dụng.

Nếu cài trên máy mới, tạo môi trường riêng ở mục 3. Môi trường cũ của dự án từng gặp lỗi native MLX/nanobind khi chạy full pipeline; tạo môi trường mới chỉ giúp cô lập cài đặt, chưa phải cách sửa lỗi đã được chứng minh.

<a id="cai-dat"></a>
## 3. Cài môi trường Python và thư viện

### 3.1. Dùng môi trường đã có

```bash
conda env list
conda activate ai-macbook
python --version
python -c 'import sys, platform; print(sys.executable); print(platform.machine())'
```

Cần kiểm tra:

- Tên môi trường xuất hiện trong `conda env list`.
- Python thuộc đúng môi trường vừa kích hoạt.
- Phiên bản dùng cho dự án là Python 3.10.
- Kiến trúc Python nên là `arm64` trên Apple Silicon. Nếu hiện `x86_64`, kiểm tra lại bản Conda/Python đang chạy trước khi cài MLX.

Nếu môi trường đã có nhưng Terminal chưa kích hoạt được, trên máy hiện tại có thể gọi trực tiếp:

```bash
/opt/anaconda3/envs/ai-macbook/bin/python --version
```

Đường dẫn này chỉ đúng khi môi trường được cài tại vị trí đó. Trong các lệnh tiếp theo, có thể thay `python` bằng đường dẫn đầy đủ này.

### 3.2. Tạo môi trường mới khi chưa có

Chỉ chạy bước này nếu `ai-macbook` chưa tồn tại:

```bash
conda create -n ai-macbook python=3.10
conda activate ai-macbook
```

Conda sẽ liệt kê các gói cần cài và yêu cầu xác nhận. Nếu muốn thử môi trường mới mà giữ nguyên môi trường cũ, dùng một tên khác, chẳng hạn `second-eye-test`, ở cả lệnh tạo và kích hoạt. Cú pháp quản lý môi trường tham khảo [tài liệu Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

Nếu Terminal báo `conda: command not found`, cần cài hoặc khởi tạo Conda cho shell trước. Có thể dùng [hướng dẫn cài đặt Conda chính thức](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), chọn bản tương ứng Apple Silicon; sau đó mở lại Terminal.

### 3.3. Cài các thư viện chính

Khi có mạng, từ thư mục dự án:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Dùng `python -m pip` giúp cài vào đúng interpreter đang sử dụng. `pip check` kiểm tra xung đột dependency được khai báo; kết quả không có lỗi chưa đủ bảo đảm mọi thư viện native tương thích khi chạy chung.

### 3.4. Cài MLX-VLM nếu sử dụng cấu hình VQA mặc định

YAML hiện chọn `vqa.backend: mlx_vlm`. Thư viện tương ứng chưa nằm trong `requirements.txt`, nên cần cài thêm:

```bash
python -m pip install mlx-vlm
python -m pip check
```

Nếu chỉ muốn thử camera/detection/depth và câu mô tả dự phòng, có thể chọn backend `disabled` theo mục 9.3 và chưa cài MLX-VLM. Không nên hiểu `disabled` là tắt toàn bộ nút VQA: ứng dụng chính vẫn có thể trả thông tin từ kết quả nhận diện.

Repository chưa có bộ phiên bản khóa đầy đủ đã được xác nhận ổn định. Vì vậy các lệnh cài đặt trên là quy trình theo khai báo của dự án, không phải cam kết rằng bất kỳ phiên bản mới nhất nào cũng hoạt động cùng nhau.

<a id="model"></a>
## 4. Chuẩn bị model để sử dụng offline

### 4.1. Tải nhóm model cơ bản

Khi máy có mạng và môi trường đã được kích hoạt:

```bash
python scripts/download_models.py
```

Lệnh thực hiện các công việc sau:

| Thành phần | Việc script thực hiện |
|---|---|
| YOLOv8n | Giữ file đã có, tìm bản sao tại một số thư mục backup, hoặc tải rồi đặt vào `models/weights/yolov8n.pt` |
| Depth Anything V2 Small | Chuẩn bị processor và model trong cache Hugging Face |
| BLIP và MarianMT | Chuẩn bị model cho backend mô tả ảnh cũ, kể cả khi YAML đang chọn MLX |
| EasyOCR vi/en | Chuẩn bị model OCR bằng Reader chạy CPU trong bước tải |
| Âm thanh | Tạo lại `alert_high.wav`, `chime.wav`, `ready.wav` trong `assets/audio/` |

Nếu đã tùy chỉnh các tệp WAV trên, sao lưu trước khi chạy vì script tạo lại chúng. Script hiện chưa có tùy chọn CLI riêng để chỉ tải depth hoặc chỉ tải OCR.

**Đọc từng dòng `[OK]` và `[FAIL]`.** Ở nhánh chuẩn bị cơ bản, một số hàm bắt lỗi rồi cho script chạy tiếp. Vì vậy dòng `[ALL DONE]` cuối cùng hoặc mã thoát 0 không có nghĩa tất cả model đều đã sẵn sàng.

### 4.2. Tải model VQA MLX riêng

Nếu dùng backend mặc định `mlx_vlm`, chạy thêm:

```bash
python scripts/download_models.py --mlx-vlm
```

Lệnh này **chỉ tải snapshot MLX**, không thay thế lệnh chuẩn bị cơ bản. Script đọc tên repository và thư mục đích trong `configs/model_config.yaml`:

```yaml
vqa:
  mlx_vlm:
    hf_repo_id: "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model_path: "models/weights/qwen2_vl_2b_4bit"
```

Đây là đoạn cấu hình minh họa các khóa liên quan; không dùng nó để ghi đè toàn bộ tệp YAML.

### 4.3. Kiểm tra tài nguyên local

```bash
ls -lh models/weights/yolov8n.pt
ls -lh assets/audio/alert_high.wav assets/audio/chime.wav assets/audio/ready.wav
ls models/weights/qwen2_vl_2b_4bit
```

Với snapshot MLX, cần có `config.json`, trọng số `.safetensors` và các tệp processor/tokenizer được tải cùng snapshot. Không chỉ sao chép một tệp weights rời. Kiểm tra thư mục tồn tại mới là bước đầu; kiểm chứng đầy đủ cần gọi VQA thực tế khi offline.

Các model không cùng nằm trong một thư mục:

- YOLO: đường dẫn trong dự án.
- MLX-VLM: thư mục snapshot trong dự án hoặc đường dẫn local đã cấu hình.
- Depth, BLIP, MarianMT: cache Hugging Face của tài khoản chạy chương trình.
- OCR: cache do EasyOCR quản lý.

Nếu tải bằng một tài khoản nhưng chạy bằng tài khoản khác, hoặc sao chép dự án sang máy mới mà không mang theo cache, ứng dụng có thể vẫn báo thiếu model. Khi bàn giao máy, nên chuẩn bị model bằng chính tài khoản và môi trường sẽ vận hành, rồi kiểm tra lại không có mạng.

### 4.4. Xác nhận khả năng chạy offline

Sau khi tải xong, chạy ứng dụng với các cờ offline:

```bash
SECOND_EYE_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python app.py --source 0
```

Trong lần kiểm tra này, lần lượt thử quan sát, `SPACE` và `Q`. Chỉ khởi động được cửa sổ chưa xác nhận OCR/VQA có model, vì hai nhóm này nạp chậm ở lần sử dụng đầu tiên.

Có thể ngắt mạng để thử lại sau khi chuẩn bị xong. Khi thiếu tài nguyên, dừng ứng dụng, kết nối mạng và chạy script chuẩn bị; sau đó khởi động lại. Không cần tắt chính sách offline trong ứng dụng chính để khắc phục thiếu model. Script tải chủ động bật quyền tải trong tiến trình của nó.

<a id="kiem-tra"></a>
## 5. Kiểm tra camera, âm thanh và môi trường

### 5.1. Cấp quyền camera

Khi macOS hỏi quyền truy cập camera, cho phép ứng dụng đang chạy Python. Nếu trước đó đã từ chối:

1. Mở **System Settings — Cài đặt hệ thống**.
2. Vào **Privacy & Security — Quyền riêng tư & Bảo mật**.
3. Chọn **Camera**.
4. Bật quyền cho ứng dụng được macOS liệt kê tương ứng với nơi chạy lệnh, chẳng hạn Terminal hoặc IDE.
5. Đóng và mở lại ứng dụng chạy lệnh nếu được yêu cầu, rồi thử camera lần nữa.

Tên mục có thể khác theo ngôn ngữ macOS. Tham khảo [hướng dẫn quyền riêng tư của Apple](https://support.apple.com/en-au/guide/mac-help/-mchl211c911f/mac). Phiên bản hiện tại điều khiển bằng bàn phím, không cần microphone để nhận lệnh giọng nói.

### 5.2. Kiểm tra giọng tiếng Việt và âm báo

Liệt kê giọng đọc có trên máy:

```bash
say -v '?'
```

Tìm giọng `Linh` hoặc giọng tiếng Việt được cài, sau đó thử:

```bash
say -v Linh -r 180 "Xin chào. Đây là phần kiểm tra âm thanh của hệ thống."
afplay assets/audio/chime.wav
```

Cần nghe được câu đọc và âm báo. Nếu không có giọng `Linh`, tìm phần cài đặt giọng nói trong Trợ năng của macOS để bổ sung giọng tiếng Việt khi có mạng, hoặc chọn tên giọng thực có trong cấu hình `audio.voice`. Sau khi chuẩn bị, thử lại trong điều kiện offline.

Nếu câu lệnh có lỗi, chạy trực tiếp như trên để đọc thông báo trong Terminal. TTS của ứng dụng ẩn stderr từ tiến trình `say`, nên cửa sổ ứng dụng có thể không giải thích rõ lỗi tên giọng.

### 5.3. Chạy công cụ kiểm tra môi trường

```bash
python scripts/inspect_environment.py
```

Công cụ sẽ thử import thư viện, kiểm tra MPS, giọng đọc, một số tài nguyên model và mở camera số 0. Lệnh này thực sự sử dụng camera và có thể phát tiếng thử.

| Nội dung cần xem | Cách diễn giải |
|---|---|
| `MPS Available: True` | Môi trường thấy backend MPS; vẫn cần chạy model thật để xác nhận toàn pipeline |
| Camera đọc được frame | Nguồn số 0 hoạt động tại thời điểm kiểm tra |
| Giọng Linh gọi được | Lệnh TTS thử không báo lỗi; vẫn cần nghe xác nhận đúng thiết bị âm thanh |
| `[FAIL]` khi import | Xem lại dependency trong đúng môi trường Python |

Script chưa kiểm tra đầy đủ snapshot MLX, OCR và mọi backend. Phần tìm YOLO còn dùng đường dẫn `~/Do-An-Tot-Nghiep`, nên nếu dự án ở nơi khác, cảnh báo thiếu weights có thể chỉ do đường dẫn kiểm tra không đúng. Đối chiếu lại bằng các lệnh ở mục 4.3.

<a id="khoi-dong"></a>
## 6. Khởi động ứng dụng

### 6.1. Chạy thường ngày với webcam

```bash
cd /Users/lenhat/Do-An-Tot-Nghiep
conda activate ai-macbook
SECOND_EYE_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python app.py --source 0
```

Ứng dụng in nguồn camera, thiết bị tính toán và đường dẫn YOLO. Sau bước nạp model, cửa sổ camera xuất hiện và chương trình có thể đọc lời chào.

Để kiểm tra đang nhận camera thật, đưa tay hoặc một vật vào khung hình và xác nhận ảnh thay đổi tương ứng. Không lấy riêng lời chào “sẵn sàng” hoặc cửa sổ còn mở làm bằng chứng tất cả model và camera đều khỏe. Nếu nhìn thấy `Dummy Cam`, nguồn hiện tại là ảnh giả lập.

### 6.2. Chạy camera khác

```bash
python app.py --source 1
```

Chỉ số camera phụ thuộc cách macOS/OpenCV nhận thiết bị; số 1 là ví dụ. Nếu không mở được, kiểm tra thiết bị đã kết nối, quyền truy cập và ứng dụng khác đang sử dụng camera.

### 6.3. Chạy với video có sẵn

```bash
python app.py --source "/duong/dan/toi/video-thu-nghiem.mp4"
```

Thay đường dẫn bằng tệp thật, giữ dấu ngoặc kép nếu có khoảng trắng. Video được phát lặp trong triển khai hiện tại; dùng `ESC` hoặc `--duration` để kết thúc. Video phù hợp để lặp lại một tình huống, nhưng kết quả vẫn phụ thuộc khả năng model và điều kiện xử lý.

### 6.4. Chạy nguồn giả lập

```bash
python app.py --source dummy --duration 15
```

Nguồn dummy tạo nền và hình chữ nhật chuyển động. Nó giúp kiểm tra cửa sổ, worker và việc đóng ứng dụng, **không thay thế model bằng mock**. YOLO và depth vẫn được khởi tạo; lỗi thiếu weights hoặc lỗi thư viện native vẫn có thể xảy ra. Hình chữ nhật ghi “Chair Simulation” cũng không bảo đảm YOLO nhận ra một chiếc ghế thật.

### 6.5. Chạy không mở giao diện

```bash
python app.py --source 0 --no-gui --duration 30
```

Chương trình chạy pipeline trong khoảng thời lượng được đặt sau bước khởi động worker; thời gian nạp model trước đó và thời gian dừng có thể làm tổng thời gian lệnh dài hơn.

Trong chế độ này, **không có cơ chế nhận các phím `SPACE`, `Q`, `S`, `ESC` của cửa sổ OpenCV**. Đây là chế độ quan sát/đánh giá tự động, không phải cách dùng OCR/VQA bằng bàn phím trong Terminal. Muốn dừng sớm, nhấn `Ctrl+C` tại Terminal.

### 6.6. Thử nhánh CPU

```bash
python app.py --source 0 --device cpu
```

Tham số này chọn CPU cho các thành phần nhận device chung như YOLO và depth, nên tốc độ có thể giảm. Nó không chuyển backend MLX sang một backend VQA chạy PyTorch CPU. Nếu muốn loại nhánh MLX khi chẩn đoán, đổi `vqa.backend` thành `disabled` theo mục 9.3 rồi khởi động lại; cách này cũng chưa bảo đảm tránh được mọi lỗi import native ở thư viện khác.

### 6.7. Các tham số của ứng dụng chính

```bash
python app.py --help
```

| Tham số | Ý nghĩa | Ví dụ |
|---|---|---|
| `--source` | Camera, video hoặc dummy; ghi đè nguồn trong YAML | `--source 0` |
| `--weights` | File YOLO local; ghi đè đường dẫn trong YAML | `--weights models/weights/yolov8n.pt` |
| `--device` | Thiết bị chung cho các thành phần có dùng tham số này | `--device mps` |
| `--no-gui` | Không mở cửa sổ | `--no-gui` |
| `--duration` | Thời lượng vòng chạy, tính bằng giây; dùng số dương | `--duration 30` |

Ứng dụng chính không có các tham số `--question`, `--backend` hoặc `--no-speech`; các tham số đó thuộc công cụ thử VQA riêng.

<a id="su-dung"></a>
## 7. Sử dụng các chức năng hằng ngày

### 7.1. Quan sát và nghe cảnh báo

1. Khởi động ứng dụng với webcam và đợi ảnh hiện ra.
2. Bấm vào cửa sổ camera để cửa sổ nhận bàn phím.
3. Hướng camera về khu vực muốn quan sát, giữ tương đối ổn định.
4. Đưa một vật quen thuộc như ghế, chai hoặc cốc vào cảnh; không cần tiến sát để cố tạo cảnh báo.
5. Quan sát bbox và nghe thông báo nếu hệ thống tạo cảnh báo phù hợp.

Không phải mọi lớp được nhận diện đều thuộc nhóm phát cảnh báo. Một vật có bbox nhưng không có tiếng nói có thể do mức nguy cơ chưa đủ cao, không thuộc `alert_classes`, còn cooldown hoặc tác vụ âm thanh đã hết hạn. Không có âm thanh cũng có thể do camera/model/loa lỗi; kiểm tra Terminal khi kết quả khác kỳ vọng.

### 7.2. Đọc chữ bằng OCR

1. Đặt tờ giấy hoặc nhãn trước camera, để chữ đủ lớn trong ảnh.
2. Giữ giấy tương đối phẳng; tránh ánh sáng phản chiếu và che khuất chữ.
3. Giữ camera ổn định, bấm vào cửa sổ rồi nhấn `SPACE` một lần.
4. Chờ thông báo và kết quả đọc. Lần đầu có thể chậm hơn do nạp EasyOCR.
5. Nếu ảnh bị báo mờ/tối/lóa, điều chỉnh vị trí và ánh sáng rồi thử lại khi tác vụ trước đã kết thúc.

OCR xử lý ảnh tại thời điểm yêu cầu, không liên tục đọc theo việc xoay camera sau đó. Trong một lần thử đầu, nên dùng một hoặc hai dòng chữ in rõ thay vì trang nhiều cột. Không có cơ chế xác nhận độ đúng của toàn bộ nội dung OCR; tên riêng, dấu tiếng Việt hoặc thứ tự đoạn có thể bị nhận sai.

Nếu hệ thống báo không phát hiện chữ, điều đó có nghĩa không tìm thấy kết quả đạt điều kiện trong ảnh đã chụp, không chứng minh ảnh hoàn toàn không có chữ.

### 7.3. Mô tả cảnh bằng VQA

1. Hướng camera về cảnh cần mô tả.
2. Nhấn `Q` trong cửa sổ ứng dụng.
3. Chờ hệ thống xử lý ảnh và đọc câu trả lời.
4. Đối chiếu kết quả với cảnh thật khi thử nghiệm; ghi nhận trường hợp mô tả sai hoặc chỉ trả câu dự phòng.

Phím `Q` gửi câu hỏi cố định “Mô tả khung cảnh phía trước”. Nó không bật microphone hoặc hộp nhập câu hỏi. Muốn thử câu hỏi như “Chiếc cốc có màu gì?”, dùng công cụ ở mục 10.

Nếu VLM thiếu model, lỗi suy luận hoặc đầu ra không đạt giới hạn định dạng, ứng dụng có thể mô tả các vật thể đã nhận diện thay vì trả lời từ VLM. Một câu nói về vật thể chưa đủ chứng minh Qwen đã chạy thành công; xem Terminal để biết có thông báo fallback hay không.

### 7.4. Khi đang bận hoặc có cảnh báo chen ngang

Ứng dụng chỉ cho một worker OCR/VQA hoạt động tại một thời điểm. Bấm liên tiếp không tạo hàng đợi nhiều câu hỏi; yêu cầu mới có thể bị từ chối khi worker cũ chưa xong.

Nếu cảnh báo HIGH chen ngang lời OCR/VQA, phần đang nói có thể bị dừng. Không nên chờ ứng dụng tự đọc tiếp đúng vị trí bị ngắt. Sau khi xử lý tình huống và tác vụ cũ kết thúc, có thể yêu cầu lại nếu cần.

### 7.5. Dừng lời nói và thoát

- Nhấn `S` để ngắt âm thanh hiện tại và hủy hiệu lực kết quả on-demand đang chờ. Đây không phải công tắc tắt tiếng vĩnh viễn: cảnh báo mới vẫn có thể phát sau đó.
- Sau khi nhấn `S`, model có thể còn tính toán trong nền; yêu cầu mới có thể chưa được nhận ngay.
- Nhấn `ESC` để thoát từ cửa sổ. Đây là thao tác thoát được vòng lặp xử lý rõ ràng; không nên chỉ dựa vào nút đóng cửa sổ của macOS.
- Khi chạy trong Terminal, `Ctrl+C` được vòng chạy chính xử lý để dừng. Nếu bị kẹt trong bước khởi tạo native, đường dừng có thể khác với lúc vòng chạy đã hoạt động.

Khi kết thúc bình thường, Terminal in `[AppRunner] Đã dừng toàn bộ hệ thống.`. Có thể đợi một lúc ngắn vì chương trình đang chờ worker và dừng tiến trình phát âm thanh.

<a id="giao-dien"></a>
## 8. Đọc thông tin trên giao diện

### 8.1. Trạng thái hoạt động

| Nhãn HUD | Cách hiểu |
|---|---|
| `QUAN SAT` | Đang ở chế độ quan sát thông thường; không phải chứng nhận tất cả nguồn dữ liệu đều tốt |
| `DANG DOC CHU` | Đang trong tác vụ OCR |
| `DANG PHAN TICH` | Đang trong tác vụ VQA |
| `TIN HIEU YEU` | Dữ liệu ghép detection/depth đang bị đánh dấu suy giảm hoặc cũ |
| `DANG NOI` | Bộ âm thanh đang phát |
| `DEPTH --` | HUD chưa có bản đồ depth để hiển thị; cần xem thêm log để biết nguyên nhân |

Các nhãn có thể không có dấu vì được vẽ bằng font OpenCV. Thanh FPS và các số P50 là thông tin chẩn đoán, không phải thước đo trực tiếp về độ chính xác.

### 8.2. Vùng không gian và mức nguy cơ

Vạch chia biểu thị trái, giữa, phải của **khung hình camera**. Nếu xoay camera lệch hướng cơ thể, “phía trước” của ảnh không nhất thiết là hướng đang di chuyển của người dùng.

| Nhãn chính | Ý nghĩa sử dụng |
|---|---|
| `NGUY CO CAO` | Thuật toán đang đánh giá vật thể ở mức HIGH |
| `CHU Y` | Thuật toán đang đánh giá ở mức MEDIUM |
| `NGUY CO THAP` | Nhãn giao diện cho nhóm còn lại trong logic vẽ bbox hiện tại; không nên dùng nó để suy ra dữ liệu luôn hợp lệ |

Trong contract còn có `UNDETERMINED` và `NO_ALERT`; HUD hiện chưa hiển thị tách biệt mọi trạng thái này trên nhãn bbox. Vì vậy phải đọc kết hợp trạng thái, depth và log, không chỉ dựa vào màu hoặc nhãn thấp.

Các chuỗi như `~1.3m` được tạo từ độ gần tương đối, chưa phải số đo khoảng cách vật lý đã hiệu chuẩn. Bản đồ depth thu nhỏ giúp quan sát kết quả mô hình; màu sắc không cho biết tự động nơi nào có thể đi qua.

<a id="cau-hinh"></a>
## 9. Thay đổi cấu hình

### 9.1. Quy trình chỉnh sửa

1. Dừng ứng dụng.
2. Sao lưu tệp cấu hình định sửa hoặc ghi lại giá trị cũ.
3. Chỉnh đúng khóa trong tệp đang có; YAML dùng khoảng trắng để thụt lề, không dùng tab.
4. Lưu tệp và chạy lại ứng dụng.
5. Kiểm tra hành vi thực tế vì ứng dụng không tự nạp lại YAML khi đang chạy.

Các đoạn YAML dưới đây là **phần trích để chỉ vị trí cần sửa**, không phải toàn bộ nội dung thay thế.

### 9.2. Camera, giọng đọc và kích thước cửa sổ

Trong `configs/app_config.yaml`:

```yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 30

audio:
  voice: "Linh"
  speech_rate_wpm: 180

runtime:
  preview_width: 800
  preview_height: 600
```

`180` là ví dụ giảm tốc độ đọc từ giá trị mặc định hiện tại `200`. Chọn theo khả năng nghe của người dùng và thử bằng một câu ngắn. Giá trị FPS camera là mục tiêu yêu cầu, không bảo đảm model xử lý được từng ấy frame mỗi giây.

Có thể sửa `ui.show_depth_map` và `ui.show_spatial_guides` để thay đổi phần hiển thị tương ứng. Nếu chỉ muốn đổi camera trong một lần chạy, dùng `--source` để không cần sửa YAML.

### 9.3. Chọn backend VQA

Trong mục `vqa` của `configs/model_config.yaml`, giữ `use_vlm: true` nếu muốn lựa chọn backend có model có hiệu lực, rồi chọn một trong các giá trị:

| Giá trị `backend` | Điều kiện | Kết quả |
|---|---|---|
| `mlx_vlm` | Cài MLX-VLM và có snapshot local | Xử lý ảnh và câu hỏi bằng Qwen qua MLX |
| `legacy_caption` | Có BLIP và MarianMT trong cache | Sinh caption rồi dịch; triển khai này không dùng câu hỏi để điều khiển nội dung |
| `disabled` | Không cần model VQA | Ứng dụng chính dùng context detection làm câu dự phòng |

Ví dụ muốn thử không nạp VLM:

```yaml
vqa:
  backend: "disabled"
```

Chỉ sửa khóa `backend` đang có, không xóa các khóa còn lại. Muốn bật lại, đổi về `mlx_vlm` rồi khởi động lại.

Nhánh MLX hiện giới hạn cứng cạnh ảnh tối đa 512 pixel và tối đa 64 token. Tăng YAML vượt các giá trị này không làm backend tăng giới hạn. Model được giữ lại sau lần dùng đầu; chọn `S` không đồng nghĩa giải phóng model khỏi RAM.

### 9.4. Những cấu hình dễ gây hiểu nhầm

| Khóa/nhóm khóa | Lưu ý trong phiên bản hiện tại |
|---|---|
| `audio.enabled`, `audio.engine`, `audio.enable_chimes` | Chưa được áp dụng đầy đủ; không dùng để kỳ vọng tắt toàn bộ âm thanh |
| `runtime.enable_preview` | Dùng CLI `--no-gui` để tắt cửa sổ |
| `camera.buffer_size` và các khóa reconnect | Chưa được truyền từ YAML vào đầy đủ runtime |
| Interval detection/depth | Không phải tất cả giá trị YAML đều điều khiển worker thực tế |
| `ocr.min_confidence`, các ngưỡng chất lượng OCR | Ứng dụng chính còn dùng mặc định service cho nhiều giá trị |
| `risk_fsm.depth_threshold_*` | FSM hiện có ngưỡng viết trực tiếp trong mã |
| Device riêng của từng model | Chưa được sử dụng như bộ điều khiển độc lập cho từng model |
| `ui.show_diagnostics` | Chưa điều khiển HUD như tên gọi gợi ý |

Không nên điều chỉnh ngưỡng nguy cơ chỉ để làm bản trình diễn phát cảnh báo nhiều hơn. Nếu thay quy tắc xử lý hoặc danh sách lớp cảnh báo để nghiên cứu, cần kiểm thử lại với tình huống có đối chiếu. Chi tiết chênh lệch cấu hình nằm trong [tài liệu phân tích](PHAN_TICH_DU_AN.md).

<a id="vqa-rieng"></a>
## 10. Thử riêng VQA với câu hỏi tùy chọn

Công cụ `scripts/run_vqa_camera.py` chỉ chạy camera, VQA và TTS tùy chọn. **Nó không chạy detection, depth, OCR hoặc cảnh báo va chạm.** Đóng ứng dụng chính trước khi mở công cụ này để tránh hai phiên cùng sử dụng camera và tài nguyên.

### 10.1. Chạy với một câu hỏi cụ thể

```bash
SECOND_EYE_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_vqa_camera.py \
  --camera 0 \
  --backend mlx_vlm \
  --question "Chiếc cốc trong ảnh có màu gì?"
```

Đưa một chiếc cốc nhìn rõ vào ảnh rồi nhấn `Q` trong cửa sổ để gửi yêu cầu. Câu hỏi không tự chạy ngay khi mở chương trình. Mỗi lần nhấn `Q` dùng lại câu hỏi đã truyền; muốn đổi câu hỏi, thoát rồi chạy lại lệnh với nội dung khác.

Trong cửa sổ này: `Q` hỏi, `S` dừng/hủy kết quả chờ, `ESC` thoát. Không có chức năng OCR bằng `SPACE`.

### 10.2. Chỉ xem kết quả trong Terminal

```bash
python scripts/run_vqa_camera.py \
  --backend mlx_vlm \
  --question "Mô tả những gì nhìn thấy." \
  --no-speech
```

`--no-speech` chỉ bỏ TTS, cửa sổ preview vẫn mở. Công cụ này không có cờ `--no-gui` trong parser hiện tại.

### 10.3. Diễn giải câu trả lời dự phòng

Công cụ riêng không có context detection. Nếu backend lỗi hoặc đầu ra bị loại, nó có thể nói chưa phát hiện rõ khung cảnh. Điều này không đủ để kết luận camera hỏng: xem Terminal để phân biệt thiếu model, lỗi API hoặc câu trả lời bị bộ kiểm tra từ chối.

Một số câu hỏi thử hữu ích là “Chiếc cốc có màu gì?” và “Trong ảnh có những gì?”. Để kiểm tra guardrail, có thể dùng “Tôi có thể qua đường an toàn không?”; hệ thống phải trả thông báo từ chối thuộc phạm vi an toàn trước khi gọi model.

<a id="danh-gia"></a>
## 11. Kiểm thử, benchmark và lưu log

Phần này dành cho người cài đặt, phát triển hoặc chuẩn bị báo cáo. Không cần chạy lại toàn bộ công cụ trước mỗi lần sử dụng.

### 11.1. Kiểm thử tự động

Lệnh chuẩn của dự án trên máy hiện tại:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /opt/anaconda3/envs/ai-macbook/bin/python \
  -m unittest discover -s tests -p "test_*.py" -v
```

Trên máy khác, dùng đường dẫn Python của môi trường thực tế hoặc `python` sau khi kích hoạt môi trường. Khi thành công, unittest in tổng số test và `OK`. Nếu có `FAILED`, `ERROR` hoặc tiến trình bị abort, không xem suite là đã qua.

Ghi nhận từ lần chạy ngày 17/09/2026 trong phiên biên tập tài liệu phân tích: 21 test báo `ok`, sau đó suite dừng tại `test_full_pipeline_dummy`, mã thoát 134, với lỗi MLX/nanobind. Hướng dẫn này không khẳng định lỗi đó đã được sửa.

Muốn kiểm tra riêng logic ưu tiên âm thanh:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
python -m unittest discover -s tests -p "test_audio_priority.py" -v
```

Test phần mềm âm thanh không thay thế việc nghe thử loa và giọng đọc thật.

### 11.2. Kiểm tra giao diện bằng dữ liệu minh họa

```bash
python scripts/render_hud_demo.py
```

Công cụ tạo ảnh minh họa trong `artifacts/hud_demo/`, không cần camera thật hoặc nạp model AI để tạo các tình huống. Đây là cách xem bố cục cảnh báo; không phải phép đánh giá chất lượng nhận diện.

### 11.3. Đo thử pipeline với dummy hoặc video

```bash
python scripts/evaluate_video.py \
  --video dummy \
  --duration 5 \
  --device mps \
  --out evaluation/results/eval_demo_local.json
```

Hoặc dùng video thật:

```bash
python scripts/evaluate_video.py \
  --video "/duong/dan/toi/video-thu-nghiem.mp4" \
  --duration 30 \
  --device mps \
  --out evaluation/results/eval_video_local.json
```

Chọn tên output mới để tránh ghi đè báo cáo cũ. Công cụ có thể phát âm thanh và vẫn cần model. Nó khởi tạo nhiều thành phần bằng giá trị mặc định của lớp, không đọc đầy đủ ba YAML như `app.py`; vì vậy không được mặc định coi kết quả là benchmark đúng cấu hình ứng dụng đang dùng.

FPS, P50 và P95 cho biết đặc điểm xử lý, không xác nhận độ chính xác. Số `end_to_end` hiện chưa bao gồm toàn bộ thời gian từ camera đến lúc người dùng nghe hết câu cảnh báo.

### 11.4. Benchmark VQA với tập ảnh

Chuẩn bị các ảnh thật, chẳng hạn `evaluation/vqa_images/cup.jpg`, và một tệp `evaluation/vqa_questions.jsonl`. Mỗi dòng JSONL là một đối tượng JSON độc lập:

```jsonl
{"image":"cup.jpg","question":"Chiếc cốc có màu gì?","expected_answer":"Chiếc cốc màu trắng."}
```

`expected_answer` phải do người chuẩn bị dữ liệu đối chiếu ảnh thật, không sao chép ví dụ nếu màu cốc khác. Sau khi các tệp đã tồn tại:

```bash
python scripts/benchmark_vqa_backends.py \
  --images evaluation/vqa_images \
  --questions evaluation/vqa_questions.jsonl \
  --backend mlx_vlm \
  --device mps \
  --warmup 1 \
  --repeats 3 \
  --output evaluation/results/vqa_mlx_local.json
```

Thêm `--interactive-score` nếu muốn chấm chất lượng theo câu hỏi của chương trình. Script đo trực tiếp backend, không chạy toàn bộ dịch vụ VQA/guardrail hay pipeline cảnh báo; dùng các câu hỏi mô tả thông thường để đánh giá và không suy ra khả năng bảo vệ an toàn của ứng dụng từ báo cáo này.

### 11.5. Lưu log khi gặp lỗi

```bash
mkdir -p evaluation/results
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
python -u -m unittest discover -s tests -p "test_*.py" -v \
  > evaluation/results/test_local.log 2>&1
```

Lệnh ghi cả stdout và stderr vào file thay vì hiện toàn bộ ở Terminal. Dùng tên khác nếu muốn giữ log trước đó.

Ghi lại môi trường cùng log:

```bash
python --version
python -m pip check
python -m pip freeze > evaluation/results/environment_local.txt
```

Danh sách phiên bản là thông tin chẩn đoán của lần chạy; không tự biến thành bộ dependency đã được xác nhận hoạt động. Khi báo lỗi, gửi kèm lệnh đã chạy, nguồn camera/video, backend, thời điểm lỗi và đoạn thông báo cuối cùng.

<a id="xu-ly-loi"></a>
## 12. Xử lý lỗi thường gặp

### 12.1. Không tìm thấy thư viện hoặc dùng nhầm Python

**Biểu hiện:** `ModuleNotFoundError`, cài xong nhưng chạy vẫn báo thiếu package.

**Thực hiện:** kiểm tra `conda activate ai-macbook`, xem `sys.executable` bằng lệnh mục 3.1 rồi cài bằng `python -m pip` của đúng môi trường. Nếu chạy từ IDE, chọn cùng interpreter. Với MLX-VLM, nhớ đây là dependency cài riêng.

### 12.2. Camera đen, không mở được hoặc xuất hiện Dummy Cam

**Thực hiện theo thứ tự:**

1. Kiểm tra quyền camera ở macOS.
2. Đóng các phiên ứng dụng chính/công cụ VQA đang chạy đồng thời và phần mềm khác có thể đang dùng camera.
3. Kiểm tra `--source` hoặc camera index.
4. Thử `python scripts/inspect_environment.py` để xem có đọc được frame không.
5. Dừng và chạy lại ứng dụng sau khi sửa quyền hoặc kết nối.

Camera manager có thể chuyển sang dummy sau nhiều lần mở thất bại. Khi thấy `Dummy Cam`, ứng dụng không còn hiển thị quan sát thật. Việc chuyển này không phải đã khôi phục được camera.

### 12.3. Thiếu YOLO hoặc depth

**YOLO:** nếu báo không tìm thấy weights, kiểm tra đang ở thư mục gốc và chạy `ls -lh models/weights/yolov8n.pt`. Chuẩn bị lại bằng script khi có mạng hoặc truyền `--weights` đến file local đúng.

**Depth:** nếu log báo thiếu cache/model, chạy bước chuẩn bị cơ bản bằng đúng tài khoản và môi trường. Ứng dụng có thể vẫn mở nhưng tạo depth không hợp lệ; không xem việc cửa sổ chạy được là đã sửa lỗi. Khởi động lại sau khi chuẩn bị.

### 12.4. OCR báo mờ, tối, lóa hoặc không thấy chữ

Nếu là lỗi chất lượng ảnh, tăng độ rõ bằng cách giữ camera yên, đưa chữ vào vùng đủ lớn, thay hướng ánh sáng và tránh phản chiếu. Chờ tác vụ cũ kết thúc rồi nhấn lại `SPACE`.

Nếu thông báo là thiếu EasyOCR/model, chỉnh ảnh sẽ không giải quyết được: quay lại cài dependency và chuẩn bị model. Nên khởi động lại sau khi sửa vì service ghi nhớ lần thử nạp thất bại.

### 12.5. VQA không trả lời đúng câu hỏi hoặc luôn dùng câu dự phòng

Kiểm tra theo thứ tự:

1. Trong ứng dụng chính, `Q` chỉ hỏi mô tả cảnh mặc định; muốn câu hỏi khác dùng công cụ riêng.
2. `backend` có đang là `disabled` hoặc `legacy_caption` không?
3. Có snapshot local đầy đủ và thư viện `mlx-vlm` trong đúng môi trường không?
4. Terminal có báo thiếu đường dẫn, lỗi inference hoặc đầu ra bị loại vì giới hạn câu/token không?
5. Nếu vừa bổ sung model hoặc sửa dependency, đã khởi động lại chưa?

Không đổi `model_path` thành tên Hub repository để mong runtime tự tải. Adapter MLX yêu cầu thư mục local. Một câu đúng nhưng thiếu dấu kết thúc cũng có thể bị loại; tăng token trong YAML vượt 64 không thay đổi giới hạn cứng.

### 12.6. Có hình nhưng không nghe tiếng

Thử lần lượt `say` và `afplay` như mục 5.2. Kiểm tra âm lượng, thiết bị đầu ra, tên giọng và file WAV. Khi lệnh độc lập hoạt động, thử OCR với ảnh chữ rõ để tạo một nội dung có thể nghe được.

Không phải mọi frame hay vật thể đều tạo lời nói. `S` chỉ dừng hiện tại; còn `audio.enabled: false` chưa phải cách tắt/bật âm thanh đáng tin cậy trong phiên bản này. Công cụ VQA riêng chạy với `--no-speech` thì chủ động không phát TTS.

### 12.7. Bấm phím không có tác dụng

Bấm vào cửa sổ camera để nó nhận bàn phím. Nếu đang chạy `--no-gui`, không có listener cho các phím chức năng. Nếu OCR/VQA đang bận hoặc worker cũ chưa kết thúc sau khi nhấn `S`, yêu cầu mới có thể bị từ chối; xem log và đợi worker kết thúc.

### 12.8. Ứng dụng chậm khi gọi VQA

Đóng các tác vụ nặng không cần thiết, thử lại sau lần nạp đầu và quan sát bộ nhớ bằng công cụ hệ thống. Để so sánh, dừng ứng dụng, đổi backend thành `disabled` rồi chạy cùng cảnh. Nếu tốc độ thay đổi rõ, cần đo mức cạnh tranh tài nguyên của VQA.

Không có cam kết thời gian phản hồi cố định cho M1 16 GB. Dùng `--device cpu` có thể giúp chẩn đoán một số vấn đề thiết bị, nhưng thường không phải cách tăng tốc và không chuyển MLX sang CPU theo tham số đó.

### 12.9. Crash nanobind/MLX, mã thoát 134

Thông báo đã ghi nhận:

```text
Critical nanobind error: refusing to add duplicate key "cpu"
to enumeration "mlx.core.DeviceType"!
```

Đây là lỗi native, không phải lỗi gõ sai phím hoặc test assertion. Lưu log và danh sách phiên bản trước khi thay đổi môi trường. Kiểm tra lại đúng interpreter và `pip check`; nếu cần thử cài đặt khác, dùng môi trường tách biệt để giữ lại môi trường hiện có.

Chưa có tổ hợp phiên bản hoặc lệnh sửa một bước đã được xác minh trong hướng dẫn này. Không nên nâng/hạ hàng loạt package trong môi trường đang dùng rồi coi lỗi đã hết nếu chưa chạy lại full suite và pipeline thật. Chạy dummy hoặc chọn CPU cũng không bảo đảm tránh lỗi, vì cả hai vẫn có bước nạp model/thư viện.

<a id="trinh-dien"></a>
## 13. Kịch bản trình diễn và bảng thao tác nhanh

### 13.1. Chuẩn bị trước buổi trình diễn

Chuẩn bị model và giọng đọc khi còn mạng; chạy thử đầy đủ OCR/VQA offline ít nhất một lượt trên chính máy sẽ trình diễn. Chọn một camera và không mở đồng thời công cụ VQA riêng với ứng dụng chính.

Chuẩn bị một chiếc ghế hoặc đồ vật quen thuộc, một tờ giấy có hai dòng chữ rõ và một cảnh đơn giản để mô tả. Đặt máy ở vị trí ổn định, thử loa, xác nhận nguồn là camera thật. Ghi lại lỗi còn tồn tại thay vì chỉ dựa vào một lần giao diện mở thành công.

### 13.2. Trình tự trình diễn gợi ý

| Bước | Thao tác | Điều cần quan sát |
|---|---|---|
| 1 | Mở ứng dụng với `--source 0` | Ảnh thật thay đổi theo cảnh, log nạp model |
| 2 | Đưa một vật quen thuộc vào trái/giữa/phải | Nhãn và hướng cập nhật; nhận diện có thể không thành công mọi lần |
| 3 | Đưa tờ giấy vào khung và nhấn `SPACE` | Kết quả đọc hoặc thông báo chất lượng ảnh |
| 4 | Đợi xong, nhấn `Q` | Mô tả cảnh; kiểm tra log để phân biệt model và fallback |
| 5 | Nhấn `S` khi đang có lời đọc | Âm thanh dừng, yêu cầu cũ không phát lại kết quả muộn |
| 6 | Nhấn `ESC` | Cửa sổ đóng và chương trình kết thúc |

Không cần tạo tình huống va chạm thật để chứng minh cảnh báo. Nếu cần minh họa bố cục mức HIGH/MEDIUM mà nhận diện trực tiếp không ổn định, dùng ảnh từ `render_hud_demo.py` và ghi rõ đó là dữ liệu minh họa.

### 13.3. Bảng tra nhanh

| Nhu cầu | Lệnh hoặc thao tác |
|---|---|
| Vào dự án | `cd /Users/lenhat/Do-An-Tot-Nghiep` |
| Kích hoạt môi trường | `conda activate ai-macbook` |
| Chuẩn bị model cơ bản khi có mạng | `python scripts/download_models.py` |
| Chuẩn bị MLX riêng | `python scripts/download_models.py --mlx-vlm` |
| Kiểm tra máy/camera/TTS | `python scripts/inspect_environment.py` |
| Mở ứng dụng | `python app.py --source 0` |
| Đọc chữ | `SPACE` trong cửa sổ |
| Mô tả cảnh | `Q` trong cửa sổ |
| Dừng lời/yêu cầu hiện tại | `S` trong cửa sổ |
| Thoát | `ESC` trong cửa sổ; `Ctrl+C` tại Terminal khi cần |
| Không mở cửa sổ trong 30 giây | `python app.py --no-gui --duration 30` |
| Xem tùy chọn | `python app.py --help` |

<a id="xac-minh"></a>
## 14. Phạm vi xác minh của hướng dẫn

Các lệnh, phím tắt, cấu hình và hành vi được đối chiếu với [app.py](app.py), [AppRunner](src/ui/app_runner.py), [SystemCoordinator](src/runtime/system_coordinator.py), [script tải model](scripts/download_models.py), [công cụ VQA camera](scripts/run_vqa_camera.py) và các script đánh giá. Liên kết nội bộ và cú pháp các đoạn lệnh được kiểm tra khi soạn tài liệu.

Tài liệu này không ghi nhận một lần cài mới toàn bộ môi trường, tải lại model hay thử camera/loa thật. Kết quả full suite được dẫn ở mục 11 là lần chạy đã thực hiện ngày 17/09/2026 trong phiên biên tập tài liệu phân tích; lỗi native chưa được sửa trong công việc viết hướng dẫn. Khi triển khai trên máy khác, cần kiểm tra lại các bước vận hành thực tế và ghi rõ môi trường sử dụng.
