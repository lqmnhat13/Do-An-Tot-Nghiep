# Phân tích và giải thích thiết kế hệ thống AI đa phương thức hỗ trợ người khiếm thị

> Tài liệu được biên tập và đối chiếu với mã nguồn ngày 17/09/2026, tại commit `12be6a7ab13cdafc3e6dd60fa1353ade3eeaa17d`. Nội dung trình bày cả ý tưởng thiết kế, cách chương trình đang hoạt động và những điểm còn cần hoàn thiện.

## 1. Bài toán và hướng tiếp cận của đề tài

### 1.1. Hệ thống cần hỗ trợ người dùng như thế nào?

Trong đề tài này, em xây dựng một ứng dụng sử dụng camera để hỗ trợ người khiếm thị nhận biết môi trường xung quanh. Thay vì chỉ hiển thị kết quả trên màn hình, ứng dụng chuyển những thông tin cần thiết thành lời nói và âm báo. Người dùng có thể biết có vật thể nào phía trước, vật thể nằm ở hướng nào, hoặc yêu cầu đọc chữ và mô tả khung cảnh.

Ví dụ, khi camera nhìn thấy một chiếc ghế ở giữa khung hình, chỉ nhận ra “đây là ghế” là chưa đủ. Hệ thống còn cần ước lượng chiếc ghế đang gần hay xa, xem thông tin đó có còn mới không, rồi quyết định có nên phát cảnh báo. Khi người dùng muốn đọc chữ trên một tờ giấy, ứng dụng lại cần một cách xử lý khác: kiểm tra ảnh có rõ không, nhận dạng chữ và đọc kết quả thành tiếng.

Từ hai tình huống trên, em chia chức năng thành hai nhóm. Nhóm thứ nhất là **quan sát và cảnh báo liên tục**, gồm phát hiện vật thể, ước lượng độ sâu và đánh giá nguy cơ. Nhóm thứ hai là **xử lý theo yêu cầu**, gồm đọc chữ và trả lời câu hỏi về ảnh. Việc phân chia này giúp ứng dụng dành tài nguyên thường xuyên cho quan sát, trong khi những tác vụ nặng hơn chỉ được gọi khi cần.

### 1.2. Vì sao chọn chạy cục bộ trên máy Mac?

Thiết bị mục tiêu là Mac dùng Apple Silicon M1 với 16 GB bộ nhớ hợp nhất. Ứng dụng được thiết kế để xử lý ảnh ngay trên máy sau khi đã chuẩn bị đầy đủ mô hình. Cách triển khai này phù hợp với mục tiêu không phụ thuộc đường truyền và không gửi hình ảnh sinh hoạt của người dùng đến dịch vụ bên ngoài.

Đổi lại, tất cả thành phần phải chia sẻ tài nguyên của một máy. Camera, mô hình nhận diện, mô hình độ sâu và mô hình ngôn ngữ thị giác không thể được lựa chọn riêng rẽ chỉ theo độ chính xác. Một mô hình lớn có thể trả lời tốt hơn trong một số trường hợp, nhưng nếu làm chậm cảnh báo thì chưa chắc phù hợp với toàn hệ thống. Vì vậy, tiêu chí của đề tài là cân bằng khả năng nhận biết, thời gian phản hồi, bộ nhớ và mức độ thuận tiện khi triển khai trên M1.

“Chạy offline” ở đây nói đến giai đoạn sử dụng ứng dụng. Giai đoạn chuẩn bị vẫn cần tải thư viện và trọng số mô hình khi có mạng. Dự án tách việc tải này vào script riêng; chương trình chính mặc định tìm mô hình đã có trên máy.

### 1.3. Phạm vi và nguyên tắc thiết kế

Đây là một bản mẫu nghiên cứu hỗ trợ nhận biết môi trường, chưa phải một thiết bị dẫn đường đã được kiểm chứng ngoài thực tế. Hệ thống hiện tập trung vào một số vật thể quen thuộc, nhiều đối tượng thuộc môi trường trong nhà. Không phát hiện vật thể không đồng nghĩa với việc phía trước không có chướng ngại.

Ba nguyên tắc xuyên suốt là:

1. Detection, depth và đánh giá nguy cơ tiếp tục chạy khi người dùng gọi OCR hoặc VQA.
2. Mô hình sinh câu trả lời không được dùng làm căn cứ khẳng định đường đi an toàn.
3. Khi dữ liệu thiếu hoặc cũ, hệ thống cần thể hiện được sự không chắc chắn, thay vì đánh đồng với không có nguy cơ.

Trong tài liệu, phần “lý do lựa chọn” là diễn giải về sự phù hợp của thiết kế với bài toán. Đây không phải tuyên bố rằng dự án đã thử nghiệm tất cả công nghệ thay thế và chứng minh phương án hiện tại tốt nhất. Hành vi cụ thể được đối chiếu với mã nguồn; những con số hiệu năng chỉ được xem là kết quả thực nghiệm khi có dữ liệu đi kèm.

## 2. Những khái niệm cần hiểu trước khi đọc kiến trúc

Các thuật ngữ dưới đây được dùng nhiều trong dự án. Có thể hiểu chúng thông qua câu hỏi mà mỗi thành phần trả lời.

| Khái niệm | Cách hiểu trong đề tài | Ví dụ |
|---|---|---|
| Detection — phát hiện đối tượng | Xác định trong ảnh có vật gì và vùng ảnh chứa vật đó | Tìm thấy một chiếc ghế, khoanh bằng hình chữ nhật |
| Bounding box, viết tắt bbox | Hình chữ nhật bao quanh đối tượng | Bốn tọa độ giới hạn vùng chiếc ghế |
| Tracking — theo dõi đối tượng | Ghép cùng một vật qua nhiều lần quan sát | Ghế ở hai ảnh liên tiếp vẫn mang mã `track_id=7` |
| Depth — độ sâu | Mô tả quan hệ gần, xa của các điểm trong ảnh | Mặt ghế gần camera hơn bức tường phía sau |
| ROI — vùng quan tâm | Phần ảnh đang cần phân tích riêng | Vùng bên trong bbox chiếc ghế |
| Fusion — kết hợp kết quả | Ghép tên vật thể, hướng, độ gần và thời gian để đánh giá | Ghế ở giữa, tương đối gần, dữ liệu còn mới |
| OCR — nhận dạng ký tự | Chuyển chữ trong ảnh thành văn bản | Đọc dòng chữ trên một biển báo |
| VQA — hỏi đáp về ảnh | Nhận ảnh và câu hỏi, trả lời bằng ngôn ngữ | “Trong ảnh có những gì?” |
| VLM — mô hình thị giác và ngôn ngữ | Mô hình xử lý đồng thời thông tin hình ảnh và văn bản | Mô hình được gọi bên trong dịch vụ VQA |
| TTS — chuyển văn bản thành tiếng nói | Đọc một chuỗi văn bản thành âm thanh | Đọc lời cảnh báo bằng giọng tiếng Việt |
| Suy luận | Dùng mô hình đã có trọng số để tạo kết quả từ dữ liệu mới | Đưa ảnh camera vào YOLO |
| Backend | Phần triển khai thực hiện tác vụ phía sau một giao diện chung | VQA có thể dùng MLX-VLM hoặc bộ sinh chú thích cũ |

Cần phân biệt **công nghệ thực thi** với **mô hình**. Chẳng hạn, PyTorch là thư viện dùng để thực hiện các phép tính học sâu; YOLOv8n là mô hình phát hiện vật thể. Tương tự, MLX-VLM là thư viện chạy mô hình thị giác và ngôn ngữ, còn Qwen2-VL là mô hình được lựa chọn cho nhánh đó.

## 3. Công nghệ nền tảng và lý do sử dụng

### 3.1. Python: kết nối các thành phần của ứng dụng

Python là ngôn ngữ chính của dự án. Nó được dùng để đọc cấu hình, quản lý camera, gọi mô hình, xử lý kết quả và điều phối âm thanh. Cấu hình CI hiện dùng Python 3.10; môi trường kiểm thử được chỉ định của dự án cũng sử dụng phiên bản này.

Lợi ích của Python trong đề tài là các thư viện cần thiết đều có giao diện Python, nên em có thể tập trung vào cách phối hợp chúng. So với viết toàn bộ bằng C++ hoặc Swift ngay từ đầu, cách này giảm lượng mã tích hợp và thuận tiện hơn khi thử nghiệm nhiều mô hình. Những phép tính nặng phần lớn được chuyển cho thư viện nền và thiết bị tính toán, thay vì thực hiện bằng vòng lặp Python thuần.

Hạn chế là Python không tự bảo đảm xử lý thời gian thực. Việc tạo nhiều luồng cũng không có nghĩa mọi đoạn mã sẽ chạy song song hiệu quả. Dự án dùng luồng chủ yếu để tách các công việc có nhịp hoạt động khác nhau; mức tăng tốc thực tế còn phụ thuộc thư viện, GPU và tranh chấp tài nguyên.

### 3.2. NumPy, OpenCV và Pillow: ba vai trò khác nhau trong xử lý ảnh

**NumPy** biểu diễn ảnh và bản đồ độ sâu bằng mảng số. Nhờ đó, chương trình có thể cắt một vùng ảnh, lọc pixel hợp lệ hoặc tính phân vị độ gần trên cả vùng. Nếu thao tác từng pixel bằng danh sách và vòng lặp thông thường, mã sẽ dài và khó tối ưu hơn. Trong dự án, NumPy là dạng dữ liệu chung giúp các mô-đun trao đổi ảnh thuận tiện.

**OpenCV** phụ trách những việc gắn với camera và xử lý ảnh truyền thống: mở webcam qua AVFoundation trên macOS, đọc video, thay đổi kích thước ảnh, kiểm tra độ mờ và vẽ giao diện quan sát. Dùng cùng một thư viện cho camera, video thử nghiệm và cửa sổ hiển thị giúp bản mẫu đơn giản hơn việc xây riêng một giao diện desktop hoàn chỉnh. Tuy nhiên, HUD OpenCV chủ yếu phục vụ theo dõi và trình diễn; đây chưa phải giao diện tiếp cận hoàn thiện cho người khiếm thị.

**Pillow** làm cầu nối khi các bộ xử lý ảnh của mô hình nhận đối tượng `PIL.Image`. Một chi tiết cần chú ý là OpenCV thường dùng thứ tự kênh màu BGR, trong khi đầu vào của các mô hình ở đây được chuyển sang RGB. Nếu bỏ qua bước chuyển này, hình vẫn có đủ ba kênh nhưng màu sắc bị diễn giải sai.

Ba thư viện không thay thế hoàn toàn cho nhau: NumPy là nơi chứa và tính toán trên dữ liệu, OpenCV quản lý nhiều thao tác thị giác và camera, còn Pillow hỗ trợ định dạng ảnh mà một số API mô hình yêu cầu.

### 3.3. PyTorch và MPS: sử dụng GPU của máy Mac

PyTorch cung cấp các phép tính tensor và môi trường chạy mô hình học sâu. Trong dự án, nhánh YOLO và Depth Anything sử dụng hệ sinh thái này. Khi chọn thiết bị `mps`, chương trình hướng các phép tính được hỗ trợ sang GPU thông qua backend dành cho macOS. MPS ở đây là một đường thực thi của PyTorch, không phải một mô hình riêng.

Lựa chọn này phù hợp vì thiết bị đích là Apple Silicon, không phải GPU NVIDIA. Các adapter detection và depth có kiểm tra khả năng dùng MPS và chuyển sang CPU khi MPS không sẵn sàng. Đây là dự phòng khi chọn thiết bị, không nên hiểu là mọi lỗi phát sinh giữa lúc suy luận đều được tự động khắc phục bằng CPU.

Core ML là một hướng tối ưu đáng cân nhắc trong quá trình phát triển, nhưng mã hiện tại chưa chạy các mô hình qua Core ML. Giữ đường PyTorch ở giai đoạn này giúp tái sử dụng trực tiếp cách nạp mô hình đang có; việc chuyển sang Core ML sẽ cần thêm bước chuyển đổi và kiểm tra tính tương đương của đầu ra.

### 3.4. Ultralytics và Transformers: giảm công việc tích hợp mô hình

Thư viện **Ultralytics** cung cấp API nạp YOLO và suy luận detection. Adapter của dự án gọi API này, lấy bbox, nhãn và độ tin cậy rồi chuyển sang kiểu dữ liệu nội bộ. Cách làm này tránh phải tự triển khai toàn bộ phần tiền xử lý và giải mã đầu ra của YOLO. [Tài liệu YOLOv8 của Ultralytics](https://docs.ultralytics.com/models/yolov8/).

**Hugging Face Transformers** được dùng để nạp bộ xử lý ảnh và mô hình Depth Anything, đồng thời hỗ trợ BLIP và MarianMT trong nhánh VQA cũ. Vai trò của thư viện là thống nhất cách khởi tạo, tiền xử lý và gọi mô hình. Trong runtime offline, dự án truyền `local_files_only=True` để tìm tài nguyên đã có trên máy.

Việc sử dụng thư viện tích hợp sẵn giúp mã ngắn hơn, nhưng cũng làm dự án phụ thuộc vào API và phiên bản của chúng. Vì `requirements.txt` chủ yếu quy định phiên bản tối thiểu, hai lần cài đặt ở những thời điểm khác nhau chưa chắc tạo ra cùng một môi trường.

### 3.5. MLX và MLX-VLM: phục vụ nhánh hỏi đáp trên Apple Silicon

MLX là framework tính toán hướng tới Apple Silicon. MLX-VLM xây dựng trên MLX để chạy các mô hình kết hợp ảnh và ngôn ngữ. Dự án dùng nhánh này cho backend VQA mặc định trong YAML. [Tài liệu MLX](https://ml-explore.github.io/mlx/build/html/index.html) và [repository MLX-VLM](https://github.com/Blaizzy/mlx-vlm).

Trong bối cảnh M1 16 GB, hướng tiếp cận này cho phép sử dụng bản mô hình đã lượng tử hóa và tích hợp qua một adapter riêng. Lý do lựa chọn là sự phù hợp với nền tảng đích, chưa phải kết luận MLX luôn nhanh hơn PyTorch cho mọi tác vụ. Muốn kết luận về tốc độ cần đo cùng ảnh, cùng câu hỏi và cùng điều kiện tải.

MLX và PyTorch cùng tồn tại trong tiến trình nhưng không có nghĩa ứng dụng đã có bộ lập lịch GPU chung. Khi VQA hoạt động, tài nguyên vẫn có thể bị cạnh tranh với detection và depth. Việc tách worker giữ các vòng xử lý tiếp tục hoạt động về mặt kiến trúc; nó chưa bảo đảm độ trễ của chúng không tăng.

### 3.6. YAML, dataclass và các công cụ hỗ trợ

**YAML** lưu những tham số như nguồn camera, tên model, ngưỡng confidence và giọng đọc. PyYAML đọc các tệp này thành cấu trúc dữ liệu Python. So với đặt mọi giá trị trực tiếp trong mã, cấu hình riêng dễ kiểm tra và điều chỉnh hơn. Tuy nhiên, một khóa chỉ có hiệu lực khi chương trình thực sự đọc và truyền nó đến thành phần tương ứng; dự án hiện còn một số khóa chưa được nối đầy đủ.

**Dataclass và enum** giúp mô tả dữ liệu chung. Ví dụ, mức rủi ro được biểu diễn bằng `RiskLevel`, thay vì mỗi mô-đun tự đặt một chuỗi khác nhau. Các tiện ích `threading`, hàng đợi và khóa được dùng để điều phối công việc giữa các luồng. `time.monotonic()` đo khoảng thời gian mà không phụ thuộc việc người dùng chỉnh giờ hệ thống.

Ngoài các thư viện chính, `requirements.txt` còn khai báo `torchvision`, `scipy`, `scikit-image` và `tqdm`. Đây là các thành phần hỗ trợ hệ sinh thái xử lý ảnh, mô hình và công cụ; không nên trình bày chúng như bốn nhánh chức năng độc lập của ứng dụng.

### 3.7. Âm thanh macOS và kiểm thử tự động

Ứng dụng dùng lệnh `say` để đọc văn bản và `afplay` để phát tệp âm thanh. Giải pháp tận dụng công cụ đã có trên macOS, giảm nhu cầu bổ sung một mô hình tổng hợp tiếng nói riêng. Giọng `Linh` và tốc độ đọc được truyền từ cấu hình; khả năng dùng giọng phụ thuộc môi trường máy thực tế.

Dự án sử dụng `unittest` cho kiểm thử Python và GitHub Actions để tự động kiểm tra mã, chạy test, thử pipeline, đóng gói và tạo bản phát hành theo tag. Những kiểm tra này giúp phát hiện lỗi hồi quy, nhưng CI chạy CPU trên Ubuntu không đại diện cho trải nghiệm camera, giọng nói và GPU trên M1.

## 4. Các mô hình được sử dụng và sự đánh đổi khi lựa chọn

### 4.1. YOLOv8n: nhận biết vật thể với chi phí phù hợp

YOLOv8n là phiên bản nano của dòng YOLOv8. Mô hình nhận ảnh và trả về các vùng có vật thể, tên lớp và confidence. Tài liệu chính thức liệt kê khoảng 3,2 triệu tham số cho bản nano, thấp hơn các bản s, m, l và x trong cùng dòng. Bản nhỏ là một điểm khởi đầu phù hợp khi cần chừa tài nguyên cho các mô hình khác. [Thông số YOLOv8](https://docs.ultralytics.com/models/yolov8/).

Trong dự án, trọng số được nạp từ `models/weights/yolov8n.pt`. Cấu hình hiện giữ 24 lớp đối tượng, dù chú thích YAML còn ghi 21. Danh sách `enabled_classes` quyết định lớp nào được giữ lại sau nhận diện; `alert_classes` là tập con được xét phát cảnh báo. Nhận ra một cuốn sách để mô tả cảnh và phát cảnh báo va chạm là hai nhu cầu khác nhau, nên cần hai danh sách này.

So với chọn một bản YOLO lớn hơn, bản nano hướng tới giảm chi phí tính toán. Đổi lại, mô hình nhỏ có thể bỏ sót vật thể nhỏ, bị che hoặc khó nhận dạng. Không có phép đo trong tài liệu này chứng minh YOLOv8n tốt hơn mọi mô hình detection khác trên cảnh sử dụng của đề tài. Lựa chọn hiện tại chủ yếu hợp lý ở khía cạnh tích hợp và ngân sách tài nguyên.

Ngưỡng confidence đang được `app.py` lấy từ YAML là 0,25. Confidence thể hiện mức tự tin của mô hình đối với nhận diện, không phải xác suất đường đi an toàn. Giảm ngưỡng có thể giữ lại nhiều dự đoán hơn nhưng cũng cần kiểm soát nhận nhầm bằng đánh giá thực tế.

### 4.2. Depth Anything V2 Small: bổ sung thông tin gần, xa từ một camera

Detection cho biết “vật gì, ở đâu trong ảnh”, còn Depth Anything V2 Small bổ sung quan hệ gần, xa. Đây là mô hình ước lượng độ sâu đơn mục, tức dùng một ảnh từ một camera. Bản Small là biến thể nhỏ của họ mô hình; checkpoint dự án dùng qua Transformers là `depth-anything/Depth-Anything-V2-Small-hf`. [Model card chính thức](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).

Điểm phù hợp với đề tài là không cần thêm camera thứ hai hoặc cảm biến đo khoảng cách chuyên dụng. So với triển khai một hệ thống camera stereo, ứng dụng giữ được phần cứng đơn giản hơn. So với các biến thể lớn trong cùng họ, lựa chọn Small nhằm giảm áp lực tính toán. Những ưu điểm về triển khai này không đồng nghĩa với việc camera thường đo khoảng cách chính xác như cảm biến đã hiệu chuẩn.

Adapter yêu cầu kích thước xử lý 256 × 256, sau đó nội suy kết quả về kích thước ảnh gốc để đối chiếu với bbox. Kích thước nhỏ giúp giới hạn công việc suy luận, nhưng có thể làm mất chi tiết của chướng ngại nhỏ. Nhịp depth được điều hòa bằng khoảng thời gian 0,125 giây trong worker; 8 lần mỗi giây là mục tiêu điều phối, không phải tốc độ luôn đạt được.

Điều quan trọng nhất là **đầu ra hiện được dùng như độ gần tương đối**. Ví dụ, điểm 0,8 cho chiếc ghế và 0,3 cho bức tường biểu thị quan hệ gần hơn trong khung hình sau chuẩn hóa. Không thể suy ra chiếc ghế cách người dùng đúng 0,8 mét. Khi thành phần trong cảnh thay đổi, thang chuẩn hóa cũng có thể thay đổi.

Nếu mô hình không nạp được, adapter trả bản đồ có `valid_mask` toàn false. Các số 0 trong mảng lúc này chỉ là dữ liệu giữ chỗ; chúng không được coi là phép đo độ sâu hợp lệ.

### 4.3. EasyOCR: đọc chữ theo yêu cầu

EasyOCR là bộ công cụ nhận dạng văn bản, gồm bước tìm vùng chữ và đọc nội dung trong các vùng đó. Thư viện hỗ trợ nhiều ngôn ngữ; dự án khởi tạo Reader với tiếng Việt và tiếng Anh. [Repository EasyOCR](https://github.com/JaidedAI/EasyOCR).

Lý do phù hợp với bản mẫu là Reader cung cấp trực tiếp vị trí, văn bản và confidence, giúp ứng dụng tập trung vào chất lượng ảnh và thứ tự đọc. So với việc tự huấn luyện cả hệ thống OCR, cách này giảm nhiều công việc chuẩn bị dữ liệu. So với dùng VLM để đọc mọi văn bản, một nhánh OCR riêng giúp quy trình dễ kiểm tra hơn và trả được các vùng chữ đã nhận dạng.

PaddleOCR hoặc các công cụ OCR khác vẫn là phương án có thể đánh giá, nhưng mã hiện chỉ triển khai EasyOCR. Khóa `engine: easyocr | paddleocr` trong chú thích cấu hình chưa tạo ra cơ chế chuyển engine thực tế. Dự án cũng chưa ghim rõ phiên bản từng trọng số con của EasyOCR trong manifest, nên cần tránh mô tả quá cụ thể một kiến trúc OCR con mà repository chưa xác nhận.

Reader chỉ được nạp khi có yêu cầu đầu tiên đi qua kiểm tra chất lượng ảnh. Mã truyền `gpu=True` khi thiết bị ứng dụng là `mps`; điều này chưa đủ chứng minh OCR thực sự chạy trên MPS ở môi trường hiện tại. Muốn ghi kết luận đó cần kiểm tra Reader và đo trên máy đích.

### 4.4. Qwen2-VL-2B-Instruct 4-bit: trả lời câu hỏi về ảnh

Qwen2-VL là mô hình kết hợp thông tin thị giác và ngôn ngữ. Biến thể Instruct được thiết kế để nhận chỉ dẫn hoặc câu hỏi; vì vậy phù hợp hơn một bộ chỉ sinh chú thích ảnh khi người dùng muốn hỏi nội dung cụ thể. [Model card Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).

YAML chọn snapshot `mlx-community/Qwen2-VL-2B-Instruct-4bit`, lưu tại `models/weights/qwen2_vl_2b_4bit`. “4-bit” nói đến lượng tử hóa: một phần trọng số được biểu diễn với độ chính xác số thấp hơn để giảm dung lượng. Lợi ích kỳ vọng là tiết kiệm bộ nhớ so với bản có độ chính xác cao hơn; đổi lại có thể ảnh hưởng chất lượng. Tổng RAM khi chạy còn gồm bộ xử lý ảnh, dữ liệu trung gian và các model khác, nên không thể tính chỉ từ kích thước trọng số.

Chọn bản 2B thay vì một VLM lớn hơn là đánh đổi theo tài nguyên M1 16 GB. Dự án còn thu nhỏ cạnh dài ảnh xuống tối đa 512 pixel và giới hạn sinh tối đa 64 token. Token là đơn vị xử lý văn bản của mô hình, không tương đương một từ tiếng Việt. Bộ kiểm tra riêng tiếp tục giới hạn câu trả lời của backend ở một câu và tối đa 25 đơn vị phân tách bằng khoảng trắng.

So với backend BLIP cũ, điểm khác biệt trong mã là Qwen nhận cả ảnh và câu hỏi. Tuy nhiên, mô hình vẫn có thể mô tả sai hoặc suy diễn chi tiết không có trong ảnh. Do đó, câu trả lời chỉ hỗ trợ mô tả; quyết định cảnh báo không đi qua VLM.

### 4.5. BLIP và MarianMT: phương án mô tả ảnh được giữ lại

Backend `legacy_caption` gồm hai bước: BLIP tạo chú thích ảnh bằng tiếng Anh, sau đó MarianMT với checkpoint `Helsinki-NLP/opus-mt-en-vi` dịch sang tiếng Việt. Hai mô hình tương ứng với hai nhiệm vụ rõ ràng: hiểu ảnh để sinh mô tả và dịch văn bản. [Model card BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), [model card OPUS-MT Anh–Việt](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi).

Cách chia này thuận tiện cho một chức năng mô tả cảnh tổng quát và tận dụng các checkpoint có sẵn. Hạn chế là sai sót có thể xuất hiện ở cả bước mô tả lẫn bước dịch. Hơn nữa, triển khai legacy hiện không dùng câu hỏi để điều khiển nội dung caption, nên không nên xem nó tương đương một hệ hỏi đáp ảnh đầy đủ.

Backend legacy là lựa chọn cấu hình được giữ lại, không phải mô hình tự động được nạp khi Qwen lỗi. Khi backend đang chọn thất bại, dịch vụ VQA dùng thông tin detection/risk để tạo câu dự phòng, tránh tự động nạp thêm một nhóm mô hình nặng.

### 4.6. Bảng đối chiếu vai trò và thời điểm nạp

| Thành phần | Nhiệm vụ | Thời điểm nạp | Giới hạn chính |
|---|---|---|---|
| YOLOv8n | Tìm và phân loại vật thể | Khi khởi tạo ứng dụng | Chỉ nhận biết trong phạm vi lớp và khả năng của model; thiếu weights làm khởi động thất bại |
| Depth Anything V2 Small | Ước lượng độ gần tương đối | Thử nạp khi khởi tạo | Chưa cung cấp khoảng cách mét đã hiệu chuẩn; lỗi nạp tạo depth không hợp lệ |
| EasyOCR | Đọc văn bản vi/en | Lần yêu cầu phù hợp đầu tiên | Phụ thuộc chất lượng ảnh, trọng số local và môi trường thư viện |
| Qwen2-VL qua MLX-VLM | Trả lời hoặc mô tả ảnh | Lần yêu cầu hợp lệ đầu tiên nếu chọn backend này | Có thể suy diễn sai; tranh chấp tài nguyên với pipeline liên tục |
| BLIP + MarianMT | Mô tả ảnh rồi dịch | Khi gọi backend legacy lần đầu | Không điều kiện hóa câu trả lời theo câu hỏi trong triển khai hiện tại |

Nạp chậm, hay *lazy loading*, chỉ trì hoãn lúc đưa model vào bộ nhớ. Mã hiện giữ model để tái sử dụng; không nên hiểu là hoàn thành mỗi yêu cầu thì model được tự động giải phóng.

## 5. Kiến trúc tổng thể và lý do tổ chức

### 5.1. Một ứng dụng, nhiều mô-đun và nhiều luồng

Dự án được tổ chức dưới dạng ứng dụng nguyên khối có mô-đun: mọi thành phần chạy trong một tiến trình Python, nhưng được chia thành các nhóm mã có trách nhiệm riêng. Không có máy chủ trung gian hoặc dịch vụ cloud trong đường xử lý chính.

Cách tổ chức này phù hợp với một thiết bị cục bộ. Ảnh NumPy có thể được chuyển giữa các thành phần mà không cần xây giao thức mạng. So với tách thành nhiều dịch vụ ngay từ đầu, thiết kế hiện tại giảm công việc triển khai và giao tiếp. Đổi lại, các mô-đun chia sẻ tài nguyên và một lỗi native nghiêm trọng có thể làm dừng cả tiến trình.

Các công việc chạy theo những nhịp khác nhau: camera liên tục thu ảnh; detection nhận diện rồi đánh giá nguy cơ; depth tạo bản đồ gần, xa; audio phát âm thanh. OCR hoặc VQA có worker theo yêu cầu. `AppRunner` vận hành vòng lặp giao diện và nhận phím bấm.

Fusion là một bước logic riêng nhưng hiện chạy trong **worker detection**, không có một worker fusion độc lập. Phân biệt này giúp đọc sơ đồ chức năng mà không nhầm mỗi khối đều tương ứng với một luồng.

### 5.2. Phân chia trách nhiệm

| Nhóm | Trách nhiệm | Lý do tách riêng |
|---|---|---|
| Thu nhận ảnh | Mở nguồn ảnh, gắn thời gian, phát frame | Tránh nhiều thành phần tranh quyền đọc camera |
| Nhận biết | Detection, tracking và depth | Mỗi loại mô hình có đầu ra và tốc độ khác nhau |
| Đánh giá nguy cơ | Đồng bộ, lấy độ gần, xác định hướng, duy trì trạng thái | Giữ quy tắc cảnh báo có thể kiểm tra độc lập với model |
| Tương tác | OCR và VQA theo yêu cầu | Hạn chế tác vụ nặng chạy thường xuyên |
| Đầu ra | Chọn thứ tự âm thanh và vẽ HUD | Tránh nhiều thành phần tự nói cùng lúc |
| Điều phối | Quản lý worker, trạng thái, số đo hiệu năng và dừng ứng dụng | Có một nơi chịu trách nhiệm về vòng đời hệ thống |

Một lợi ích của cách chia này là có thể thay mô hình mà ít ảnh hưởng phần còn lại, miễn adapter mới vẫn trả đúng contract. Ví dụ, thay bộ ước lượng độ sâu không chỉ cần trả một ma trận: nó còn phải khai báo rõ chiều gần/xa, dữ liệu hợp lệ và thời điểm chụp.

### 5.3. Vì sao camera có hai bộ đệm riêng?

Detection và depth không xử lý nhanh như nhau. Nếu cùng lấy ảnh từ một hàng đợi, nhánh này có thể lấy mất ảnh mà nhánh kia cần. Vì vậy camera phát mỗi packet vào hai bộ đệm độc lập, mỗi bộ đệm có sức chứa 2 trong `SystemCoordinator`.

Khi đầy, bộ đệm bỏ frame cũ nhất để nhận frame mới, gọi là **drop-oldest**. Chẳng hạn bộ đệm đang giữ frame 101 và 102, camera gửi frame 103 thì frame 101 bị loại. Đây là sự đánh đổi có chủ đích: ứng dụng ưu tiên quan sát gần hiện tại hơn việc xử lý đầy đủ mọi ảnh đã chụp.

Worker hiện đọc theo thứ tự FIFO bằng `get()`, nghĩa là lấy phần tử cũ nhất còn trong bộ đệm. Vì thế không nên mô tả rằng worker luôn lấy đúng frame mới nhất. Bộ đệm nhỏ chỉ giúp hạn chế tồn đọng; không tự bảo đảm một giới hạn độ trễ toàn hệ thống.

### 5.4. Contracts: thống nhất ý nghĩa trước khi trao đổi dữ liệu

Có thể hình dung contract như một mẫu phiếu chung. Camera gửi “ảnh nào, chụp lúc nào”; detector gửi “tìm thấy gì trong ảnh đó”; depth gửi “giá trị này mang ý nghĩa gần hay xa”. Nhờ mẫu chung, các mô-đun không phải đoán ý nghĩa của dữ liệu nhận được.

| Contract | Thông tin chính | Quan hệ sử dụng |
|---|---|---|
| `FramePacket` | Ảnh BGR, mã frame, thời điểm chụp, kích thước | Camera tạo; các nhánh nhận ảnh sử dụng |
| `DetectionResult` | Bbox, nhãn, confidence, track ID và metadata | Detector/tracker tạo; fusion và HUD đọc |
| `DepthMap` | Mảng depth, mask hợp lệ, quy ước gần/xa, thời gian | Depth tạo; ROI, synchronizer và HUD đọc |
| `RiskAssessment` | Mức rủi ro, chất lượng dữ liệu, hướng, độ gần, hạn dùng | FSM tạo; aggregator, HUD và context VQA đọc |
| `AudioTask` | Văn bản, âm báo, ưu tiên, hạn dùng, khả năng bị ngắt | Các dịch vụ tạo; audio coordinator thực thi |
| `OCRRequest/Result`, `VQARequest/Result` | Nội dung yêu cầu, kết quả và thời gian xử lý | Coordinator trao đổi với dịch vụ on-demand |

Nhiều dataclass được khai báo `frozen=True` để hạn chế việc gán lại thuộc tính. Tuy nhiên, mảng NumPy bên trong vẫn có thể bị sửa tại chỗ. Camera dùng `copy()` khi đóng gói ảnh để giảm nguy cơ dùng lại vùng nhớ cũ; điều này không thay thế hoàn toàn kỷ luật quản lý dữ liệu chia sẻ.

## 6. Cấu trúc thư mục và vai trò từng tệp

### 6.1. Cách đọc repository

Mã chạy chính nằm trong `src/`; cấu hình nằm trong `configs/`; công cụ chuẩn bị và đo đạc nằm trong `scripts/`; kiểm thử nằm trong `tests/`. Sự phân chia này giúp phân biệt mã dùng khi vận hành với mã chỉ phục vụ phát triển hoặc đánh giá.

Nếu đọc dự án lần đầu, nên bắt đầu từ `app.py`, sau đó đến `SystemCoordinator`, rồi theo từng nhánh camera, detection, depth và fusion. Cách đọc này cho thấy các thành phần được kết nối ra sao trước khi đi vào thuật toán của từng phần.

### 6.2. Các tệp ở cấp dự án và cấu hình

| Tệp hoặc thư mục | Vai trò |
|---|---|
| [`app.py`](app.py) | Điểm vào ứng dụng; đọc YAML và tham số dòng lệnh, tạo các đối tượng, kết nối chúng và chạy giao diện |
| [`README.md`](README.md) | Hướng dẫn giới thiệu và sử dụng; một số số liệu cần đối chiếu lại với mã và kết quả chạy |
| [`AGENTS.md`](AGENTS.md) | Quy tắc phát triển dự án, gồm bất biến kiến trúc và lệnh kiểm thử |
| `PHAN_TICH_DU_AN.md` | Giải thích bài toán, lựa chọn kỹ thuật, triển khai và giới hạn |
| [`requirements.txt`](requirements.txt) | Danh sách dependency Python chính; chưa khóa đầy đủ phiên bản môi trường |
| [`configs/app_config.yaml`](configs/app_config.yaml) | Cấu hình ứng dụng, camera, runtime, âm thanh và HUD |
| [`configs/model_config.yaml`](configs/model_config.yaml) | Tên, đường dẫn và tham số các mô hình; danh sách lớp detection |
| [`configs/fusion_rules.yaml`](configs/fusion_rules.yaml) | Tham số phân vùng, đồng bộ, FSM và nhãn tiếng Việt |
| [`models/manifest.json`](models/manifest.json) | Metadata mô hình do dự án khai báo; hiện cần đồng bộ với cấu hình mới |
| `models/weights/` | Trọng số được lưu theo đường dẫn dự án, gồm YOLO và snapshot MLX khi đã chuẩn bị |
| `assets/audio/` | Âm thanh báo sẵn sàng, thông báo và cảnh báo |
| [`evaluation/results/eval_report.json`](evaluation/results/eval_report.json) | Báo cáo đo đạc lưu sẵn, dùng làm bằng chứng tham khảo |
| [`.github/workflows/cicd.yml`](.github/workflows/cicd.yml) | Quy trình lint, test, smoke test, kiểm tra tài nguyên, đóng gói và release |

Không phải mọi model đều được nạp trực tiếp từ `models/weights/`. Depth và các model Transformers cũ được gọi bằng tên checkpoint rồi tìm trong cache local; EasyOCR cũng có cơ chế lưu model riêng. Vì vậy chỉ kiểm tra một thư mục weights là chưa đủ để kết luận máy đã sẵn sàng chạy offline.

### 6.3. `src/contracts/`: định nghĩa dữ liệu dùng chung

| Tệp | Vai trò |
|---|---|
| `frame_packet.py` | Định nghĩa gói ảnh, kích thước và cách tính tuổi frame |
| `detection.py` | Định nghĩa bbox, một detection và tập kết quả; cung cấp thao tác tọa độ cơ bản |
| `depth_map.py` | Định nghĩa kiểu biểu diễn depth, giá trị, mask và metadata |
| `risk.py` | Định nghĩa hướng, mức rủi ro, chất lượng dữ liệu và assessment có hạn dùng |
| `audio.py` | Định nghĩa mức ưu tiên, tác vụ âm thanh và kiểm tra hết hạn |
| `request.py` | Định nghĩa đầu vào/đầu ra OCR và VQA |
| `__init__.py` | Xuất lại các kiểu dữ liệu để các mô-đun khác import thuận tiện |

Đặt các kiểu này riêng giúp detection không phải biết cách audio hoạt động và audio không phải hiểu cấu trúc model. Mỗi bên chỉ cần tôn trọng dữ liệu chung tại ranh giới giao tiếp.

### 6.4. Camera, detection, tracking và depth

Các đường dẫn trong bảng dưới đây tính từ `src/`.

| Tệp | Vai trò |
|---|---|
| `camera/camera_manager.py` | Sở hữu nguồn camera/video/dummy, đọc frame, gắn ID và thời gian, lưu ảnh mới nhất, phân phối cho subscriber |
| `camera/bounded_buffer.py` | Bộ đệm hữu hạn, hỗ trợ đọc có timeout và bỏ phần tử cũ khi đầy |
| `camera/video_playback.py` | Thành phần hỗ trợ đọc video; cần phân biệt lớp hỗ trợ này với đường khởi tạo camera chính trong `app.py` |
| `detection/yolo_detector.py` | Nạp YOLO, gọi suy luận và chuyển đầu ra về `DetectionResult` |
| `detection/class_filter.py` | Lọc lớp nhận diện, xác định lớp cảnh báo và ánh xạ tên tiếng Việt |
| `tracking/byte_tracker.py` | Gán và duy trì track ID bằng ghép cùng lớp theo độ chồng lấp bbox; tên file chưa phản ánh đúng mức triển khai ByteTrack |
| `depth/depth_estimator.py` | Nạp Depth Anything, tiền xử lý, suy luận, nội suy và tạo `DepthMap` |
| `depth/depth_representation.py` | Chuẩn hóa depth, xử lý quy ước số lớn hơn là gần hay xa |
| `depth/roi_extractor.py` | Co bbox, kiểm tra mask và tính điểm độ gần đại diện cho từng vật thể |

### 6.5. Fusion, âm thanh và tương tác theo yêu cầu

| Tệp trong `src/` | Vai trò |
|---|---|
| `fusion/synchronizer.py` | Ghép detection với depth theo frame/thời gian; đánh dấu chất lượng cặp dữ liệu |
| `fusion/spatial_zones.py` | Xác định vật thể thuộc trái, giữa hay phải |
| `fusion/risk_fsm.py` | Tính mức nguy cơ và duy trì lịch sử theo track để giảm dao động |
| `fusion/alert_aggregator.py` | Chọn cảnh báo cần ưu tiên và tạo `AudioTask` |
| `audio/audio_coordinator.py` | Sở hữu hàng đợi âm thanh, xử lý ưu tiên, hết hạn, trùng lặp và ngắt lời |
| `audio/tts_engine.py` | Thực thi `say`/`afplay`, quản lý và dừng tiến trình phát âm thanh |
| `ocr/image_quality.py` | Kiểm tra kích thước, độ sáng và độ mờ trước nhận dạng |
| `ocr/ocr_service.py` | Nạp EasyOCR theo yêu cầu, nhận dạng, lọc confidence và sắp xếp văn bản |
| `vqa/safety.py` | Chặn những câu hỏi thuộc nhóm xác nhận an toàn bằng quy tắc từ khóa |
| `vqa/backend.py` | Định nghĩa giao diện backend, nhánh BLIP/MarianMT, nhánh MLX-VLM và hàm chọn backend |
| `vqa/formatting.py` | Ghép mô tả với context vật thể hoặc tạo câu dự phòng khi backend không có kết quả |
| `vqa/vqa_service.py` | Điều phối guardrail, backend và formatter, trả về `VQAResult` |

### 6.6. Runtime và giao diện

| Tệp trong `src/` | Vai trò |
|---|---|
| `runtime/system_coordinator.py` | Kết nối toàn pipeline, quản lý worker, mode, yêu cầu on-demand, trạng thái mới nhất và shutdown |
| `runtime/model_loading.py` | Chính sách offline, tham số nạp model và thông báo thiếu tài nguyên |
| `runtime/metrics.py` | Thu thập thời gian xử lý, FPS và các thống kê phục vụ quan sát |
| `runtime/watchdog.py` | Cung cấp cơ chế ghi nhịp hoạt động và kiểm tra sức khỏe worker; vòng giám sát thực chưa được nối đầy đủ |
| `ui/app_runner.py` | Vòng lặp chương trình, xử lý phím bấm, thời lượng chạy và đóng cửa sổ |
| `ui/hud_renderer.py` | Vẽ ảnh, bbox, depth, trạng thái nguy cơ và thông tin chẩn đoán trên HUD |

HUD đọc trạng thái hiện có để hiển thị, không phải nơi ra quyết định nguy cơ. Tách như vậy giúp việc thay đổi giao diện ít ảnh hưởng quy tắc cảnh báo.

### 6.7. Các script phục vụ phát triển

| Tệp trong `scripts/` | Vai trò |
|---|---|
| `download_models.py` | Chuẩn bị model khi có mạng và tạo tài nguyên âm thanh; có tùy chọn chuẩn bị snapshot MLX-VLM |
| `inspect_environment.py` | Kiểm tra hệ điều hành, thư viện, model, TTS và camera |
| `benchmark_hardware.py` | Đo thử các thành phần như YOLO, depth và TTS trên thiết bị |
| `benchmark_vqa_backends.py` | Chạy tập ảnh/câu hỏi để so sánh backend VQA, ghi thời gian và hỗ trợ chấm chất lượng thủ công |
| `evaluate_video.py` | Chạy đánh giá với nguồn video hoặc dummy và lưu thống kê |
| `run_vqa_camera.py` | Chạy thử VQA với camera như công cụ riêng; không đại diện cho toàn pipeline cảnh báo |
| `render_hud_demo.py` | Tạo dữ liệu minh họa để kiểm tra cách hiển thị HUD |

### 6.8. Các tệp kiểm thử

| Tệp trong `tests/` | Phạm vi chính |
|---|---|
| `test_bounded_buffer.py` | Bộ đệm, drop-oldest và timeout |
| `test_contracts.py` | Các kiểu dữ liệu và thao tác của contract |
| `test_detection_tracking.py` | Lọc lớp và theo dõi đối tượng cơ bản |
| `test_depth_roi.py` | Quy ước depth, chuẩn hóa và trích xuất vùng quan tâm |
| `test_fusion.py` | Đồng bộ, phân vùng và quy tắc nguy cơ |
| `test_audio_priority.py` | Thứ tự ưu tiên, ngắt lời và hết hạn tác vụ |
| `test_ocr_vqa.py` | Dịch vụ OCR/VQA và các hành vi dự phòng |
| `test_mlx_vlm.py` | Adapter MLX-VLM, giới hạn đầu ra và xử lý gọi đồng thời, chủ yếu bằng mô phỏng |
| `test_offline_model_loading.py` | Nạp model theo chính sách offline |
| `test_hud_renderer.py` | Kết quả hiển thị và bảo toàn dữ liệu đầu vào |
| `test_end_to_end.py` | Ghép pipeline, điều phối on-demand, hủy yêu cầu và shutdown |
| `test_vqa_camera_cli.py` | Công cụ thử VQA bằng camera và tham số dòng lệnh |
| `test_benchmark_vqa_backends.py` | Công cụ benchmark VQA và kiểm tra dữ liệu đánh giá |

## 7. Luồng quan sát và cảnh báo: từ ảnh camera đến lời nói

### 7.1. Bước 1 — Thu nhận và đánh dấu ảnh

`CameraManager` là chủ sở hữu duy nhất của camera trong ứng dụng chính. Nguồn có thể là webcam, video có sẵn hoặc ảnh dummy phục vụ thử nghiệm. Mỗi frame được đóng thành `FramePacket`, mang số thứ tự tăng dần và timestamp lấy từ `time.monotonic()`.

Timestamp biểu thị thời điểm ảnh được thu nhận. Nhờ đó, dù suy luận mất thời gian, các thành phần phía sau vẫn biết mình đang đánh giá cảnh mới hay cảnh đã cũ. Camera gửi packet vào hai buffer và giữ ảnh mới nhất để HUD, OCR, VQA có thể truy cập.

### 7.2. Bước 2 — Nhận diện và gán mã theo dõi

Worker detection lấy frame, gọi YOLO rồi lọc theo `enabled_classes`. Mỗi vật thể được biểu diễn bằng nhãn, bbox và confidence. Sau đó tracker so sánh với các vật thể đã thấy ở lần trước để duy trì `track_id`.

Tracker hiện dùng **IoU**, tức tỷ lệ diện tích giao nhau trên diện tích hợp của hai bbox. Nếu hai hộp cùng lớp chồng lên nhau đủ nhiều, chương trình có cơ sở đơn giản để xem chúng là cùng một vật. Ngưỡng được truyền từ `app.py` là 0,3; bộ đếm mất dấu dùng mặc định tối đa 5 trước khi loại track.

Ví dụ, nếu chiếc ghế chỉ dịch sang phải một ít giữa hai ảnh, bbox cũ và mới vẫn chồng nhau đáng kể và có thể giữ cùng ID. FSM sau đó dùng ID này để nhớ nguy cơ trước đó và thời điểm đã phát cảnh báo.

Mặc dù lớp mang tên `ByteTrackerAdapter`, triển khai hiện tại là ghép tham lam theo class và IoU. Mã chưa có bộ dự đoán chuyển động, Kalman filter hoặc các giai đoạn ghép đặc trưng của ByteTrack đầy đủ. Cách đơn giản này nhẹ và dễ theo dõi, nhưng dễ đổi ID khi vật bị che, chuyển động nhanh hoặc hai vật cùng lớp đi qua nhau.

### 7.3. Bước 3 — Ước lượng độ sâu trên nhánh riêng

Trong khi detection hoạt động, worker depth lấy ảnh từ buffer của nó. Adapter chuyển BGR sang RGB, chạy bộ xử lý đầu vào và Depth Anything, rồi nội suy đầu ra về kích thước ảnh gốc.

`DepthMap` trả về gồm ma trận giá trị và `valid_mask`. Mask cho biết pixel nào có thể dùng, ví dụ loại những giá trị không hữu hạn. Cờ `near_is_larger=True` ghi rõ rằng giá trị lớn hơn được diễn giải là gần hơn. Nếu không lưu quy ước này, một consumer có thể vô tình đảo ngược gần và xa.

Hai nhánh có thể đã xử lý những frame khác nhau do tốc độ và việc bỏ frame. Vì vậy không thể ghép kết quả chỉ vì chúng vừa hoàn thành gần nhau; cần kiểm tra ID và thời điểm chụp.

### 7.4. Bước 4 — Đồng bộ kết quả và đánh giá độ mới

`Synchronizer` lưu depth mới nhất và lịch sử tối đa 20 frame ID. Khi có detection, nó ưu tiên depth cùng ID. Nếu chưa có bản khớp, nó dùng depth mới nhất đã nhận để tránh phải chờ vô hạn. Đây là phép ghép gần đúng theo thời gian, không có bước bù chuyển động giữa hai ảnh.

Giả sử detection của frame 120 hoàn thành trước depth của frame 120. Nếu depth mới nhất là frame 118, hệ thống có thể ghép chúng rồi đánh dấu chất lượng dựa trên tuổi và độ lệch. Quy tắc không đợi nhánh chậm giúp duy trì hoạt động, nhưng có thể lấy độ sâu tại vị trí mà vật thể đã di chuyển khỏi đó.

| Trạng thái | Ý nghĩa trong mã |
|---|---|
| `VALID` | Các mốc thời gian nằm trong ngưỡng được cấu hình |
| `DEGRADED` | Detection/depth bị lệch hoặc có nguồn vượt ngưỡng tuổi thông thường |
| `STALE` | Tuổi nguồn cũ hơn hoặc tuổi depth vượt 2 giây |
| `UNAVAILABLE` | Chưa có depth để ghép |

`VALID` ở đây xác nhận điều kiện thời gian, không xác nhận model dự đoán đúng. Chất lượng thời gian được giữ riêng với `RiskLevel` để phân biệt “chưa đủ thông tin” và “mức nguy cơ thấp”. Tuy nhiên, FSM hiện vẫn có thể tính nguy cơ từ `STALE` hoặc `DEGRADED`; đây là giới hạn của chính sách xử lý hiện tại.

### 7.5. Bước 5 — Lấy độ gần đại diện cho từng vật thể

Một bbox có thể chứa cả vật thể và nền. `ROIExtractor` vì vậy lùi vào 15% chiều rộng ở mỗi cạnh trái/phải và 15% chiều cao ở mỗi cạnh trên/dưới. Phần ở giữa có khả năng đại diện cho vật thể tốt hơn viền hộp, dù cách này vẫn có thể bỏ mất chi tiết nhỏ hoặc vật có hình dạng mảnh.

Sau khi giới hạn ROI trong ảnh, chương trình yêu cầu tối thiểu 15 pixel hợp lệ và tỷ lệ hợp lệ ít nhất 25%. Nếu không đủ, kết quả là “chưa rõ”. Khi đủ dữ liệu, toàn bản đồ depth được chuẩn hóa về khoảng 0 đến 1, rồi lấy phân vị 80 của các pixel hợp lệ trong ROI.

Có thể hiểu phân vị 80 là một giá trị nghiêng về nhóm pixel gần hơn, nhưng ít phụ thuộc vào một pixel cực đại duy nhất. So với trung bình, cách này giúp phần gần của vật thể có ảnh hưởng rõ hơn. Nếu các giá trị depth hợp lệ gần như bằng nhau, helper hiện gán 0,5; đó là quy ước số học, không phải bằng chứng rằng toàn cảnh nằm ở một khoảng cách vừa phải.

Mã còn dùng công thức `0.4 + (1 - score) * 3.6`, giới hạn trong khoảng 0,5–4,5 để tạo số mét minh họa. Công thức này là quy tắc tự đặt, chưa được hiệu chuẩn bằng khoảng cách thật. Vì thế các chuỗi như `~1.3m` hoặc `> 3m` không được xem là phép đo vật lý đáng tin cậy.

### 7.6. Bước 6 — Xác định hướng và mức nguy cơ tức thời

Khung hình được chia thành vùng trái 35%, giữa 30% và phải 35%. `SpatialZones` xét tâm bbox, đồng thời xem vật thể có phủ hơn 40% vùng giữa không. Quy tắc bổ sung này giúp một vật rộng nằm lệch tâm nhưng vẫn chắn phía trước được xếp về giữa.

FSM sau đó xét lớp vật thể, hướng và điểm độ gần. Những ngưỡng sau là giá trị đang viết trong mã, chưa lấy từ bảng ngưỡng YAML:

| Điều kiện độ gần | Ở giữa | Ở hai bên |
|---|---|---|
| Từ 0,75 trở lên | `HIGH` | `HIGH` |
| Từ 0,55 đến dưới 0,75 | `HIGH` | `MEDIUM` |
| Từ 0,35 đến dưới 0,55 | `MEDIUM` | `LOW` |
| Từ 0,20 đến dưới 0,35 | `LOW` | `LOW` |
| Dưới 0,20 nhưng lớn hơn 0 | `NO_ALERT` | `NO_ALERT` |

Trước bảng này, nếu không có dữ liệu depth hoặc điểm độ gần bằng 0, kết quả là `UNDETERMINED`. Nếu có dữ liệu nhưng lớp không thuộc nhóm cảnh báo thì kết quả là `NO_ALERT`. Các tên trạng thái mô tả quyết định của thuật toán, không phải chứng nhận cảnh quan sát đã an toàn.

### 7.7. Bước 7 — Dùng lịch sử để giảm cảnh báo dao động

**FSM**, hay máy trạng thái hữu hạn, giúp quyết định dựa cả vào hiện tại lẫn trạng thái trước đó. **Hysteresis** có thể hiểu là đặt điều kiện chuyển trạng thái khác nhau để tránh mức cảnh báo nhảy lên xuống liên tục khi đầu vào dao động quanh ngưỡng.

Trong triển khai này, vật rất gần ở giữa có thể lên HIGH ngay. Với trường hợp HIGH khác, bộ đếm xác nhận tham gia quyết định. Khi đang HIGH hoặc MEDIUM mà mức ứng viên xuống LOW/NO_ALERT, cần 3 mẫu depth mới liên tiếp theo cấu hình hiện tại mới hạ xuống. Cùng một `depth_frame_id` không được đếm nhiều lần như nhiều quan sát độc lập.

Cần diễn đạt chính xác rằng không phải mọi lần hạ mức đều chờ 3 mẫu: ứng viên MEDIUM hiện được trả về trực tiếp, kể cả khi trạng thái trước đó là HIGH. Quy tắc trì hoãn chủ yếu áp dụng khi xuống LOW hoặc NO_ALERT.

### 7.8. Bước 8 — Chọn cảnh báo và phát âm thanh

`AlertAggregator` không đọc tất cả vật thể cùng lúc. Nó ưu tiên HIGH ở giữa, HIGH ở hai bên, MEDIUM ở giữa, rồi MEDIUM ở hai bên. Trong cùng hạng, vật có điểm độ gần lớn hơn được ưu tiên. Cách chọn này hướng tới cung cấp một thông tin cần chú ý nhất tại mỗi lượt.

Cooldown theo track, hiện cấu hình 2,5 giây, giảm việc lặp lại cùng một cảnh báo quá thường xuyên. Tuy nhiên, cơ chế cho phép tăng lên HIGH vượt qua cooldown có dấu hiệu không đạt ý định: FSM đã cập nhật trạng thái thành HIGH trước khi hàm kiểm tra override xét trạng thái cũ. Vì vậy chưa thể khẳng định mọi lần tăng nguy cơ đều phát ngay.

Cuối cùng, cảnh báo được chuyển thành `AudioTask`. Audio coordinator quyết định có ngắt lời đang nói không, kiểm tra hạn dùng rồi gọi bộ phát âm thanh. HUD đồng thời đọc snapshot để hiển thị kết quả; snapshot có thể chứa các nguồn có thời điểm khác nhau, không phải một ảnh chụp nguyên tử của toàn bộ pipeline AI.

## 8. Luồng OCR, VQA và điều phối âm thanh

### 8.1. OCR: từ yêu cầu đọc chữ đến văn bản được đọc ra

Khi người dùng nhấn `SPACE`, coordinator kiểm tra xem còn worker on-demand nào hoạt động không. Nếu đang bận, yêu cầu mới không được chạy chồng lên. Nếu nhận yêu cầu, chương trình ngắt âm thanh cũ, lấy frame mới nhất và bắt đầu xử lý OCR.

Các bước tiếp theo là:

1. **Kiểm tra chất lượng ảnh.** Ảnh quá nhỏ, tối, lóa hoặc mờ bị từ chối sớm. Độ mờ được ước lượng bằng phương sai Laplacian: ảnh thiếu cạnh rõ thường có giá trị thấp. Đây là phép kiểm tra sơ bộ, không bảo đảm mọi chữ trong ảnh đều đọc được.
2. **Nạp Reader nếu cần.** EasyOCR chỉ được khởi tạo ở lần dùng đầu tiên phù hợp. Trong chế độ offline, `download_enabled=False` ngăn tự tải trọng số.
3. **Nhận dạng các vùng chữ.** Reader trả vị trí, nội dung và confidence.
4. **Lọc và sắp xếp.** Ngưỡng service hiện dùng là 0,35. Các vùng còn lại được xếp theo dải tọa độ Y rộng 20 pixel, rồi từ trái sang phải. Đây là cách sắp xếp đơn giản; tài liệu nhiều cột hoặc bố cục phức tạp có thể bị đọc sai thứ tự.
5. **Trả và đọc kết quả.** Văn bản được tạo thành tác vụ ưu tiên `ON_DEMAND`. Nếu không có vùng chữ đạt điều kiện, dịch vụ thông báo không phát hiện chữ thay vì tự bổ sung nội dung.

Trong suốt quá trình này, detection và depth tiếp tục xử lý. Cảnh báo HIGH vẫn có thể được đưa vào hàng đợi ưu tiên để ngắt nội dung đang đọc.

### 8.2. VQA: từ câu hỏi đến câu trả lời có giới hạn

Phím `Q` của ứng dụng chính gọi yêu cầu mô tả cảnh mặc định. Dịch vụ VQA nhận được câu hỏi qua API nội bộ; giao diện hiện tại chưa cung cấp một luồng hội thoại giọng nói tự do đầy đủ.

Sau khi lấy ảnh, coordinator chụp lại context từ các assessment mới nhất: tên vật thể, hướng, mô tả độ gần và mức nguy cơ. Context này có thể khác thời điểm với ảnh vừa lấy. Trong triển khai hiện tại, backend nhận ảnh và câu hỏi; context được dùng ở bước formatter, không phải một cơ chế kiểm chứng từng chi tiết do mô hình sinh ra.

**Trước khi gọi model**, `SafetyGuardrail` kiểm tra từ khóa của câu hỏi. Các cách hỏi chứa nhóm từ như “an toàn”, “đi thẳng” hoặc “qua đường” có thể nhận câu từ chối cố định. Việc chặn trước backend còn tránh nạp VLM cho một yêu cầu nằm ngoài phạm vi cho phép.

Nếu câu hỏi được chấp nhận, backend MLX-VLM thực hiện lần lượt:

1. Kiểm tra đường dẫn local, `config.json` và trọng số `.safetensors` trước khi import/nạp MLX-VLM.
2. Nạp model một lần và dùng khóa để không có hai lần load hoặc generation chạy chồng lên nhau trong backend.
3. Chuyển ảnh sang RGB, giữ tỷ lệ và toàn khung hình, thu nhỏ cạnh dài tối đa 512 pixel.
4. Tạo prompt từ chỉ dẫn và câu hỏi, sinh tối đa 64 token với temperature bằng 0.
5. Kiểm tra đầu ra: loại kết quả rỗng, có dấu hiệu bị cắt, thiếu dấu kết câu, không đúng một câu hoặc quá 25 đơn vị phân tách bằng khoảng trắng.

Giới hạn một câu áp dụng cho **đầu ra backend MLX**, không phải toàn bộ lời nói cuối cùng. Formatter còn thêm tiền tố và thông tin của tối đa ba vật thể từ context, nên câu trả lời hoàn chỉnh có thể dài hơn.

Nếu backend lỗi hoặc đầu ra bị loại, formatter tạo câu dự phòng dựa trên detection/risk. Ví dụ, nó có thể thông báo một chiếc ghế được phát hiện ở giữa. Phương án này giúp ứng dụng tiếp tục phản hồi, nhưng cũng kế thừa giới hạn và độ cũ của context; nó không tương đương một câu trả lời đầy đủ cho mọi câu hỏi.

### 8.3. Vì sao không dùng VQA để quyết định an toàn?

Một mô hình ngôn ngữ có thể tạo ra câu trôi chảy ngay cả khi nhận biết ảnh chưa đúng. Các kiểm tra dấu câu hoặc số token chỉ giúp loại một số đầu ra không đạt định dạng, không chứng minh nội dung đúng với thực tế.

Vì vậy, nhánh tạo `RiskAssessment` chỉ đi qua detection, depth và quy tắc fusion. VQA không được phép thay thế nhánh này để nói rằng người dùng có thể đi tiếp. Guardrail từ khóa là một lớp hạn chế bổ sung, nhưng vẫn có thể bỏ sót cách diễn đạt khác hoặc chặn nhầm câu hỏi; đây chưa phải cơ chế bảo đảm an toàn hoàn chỉnh.

### 8.4. Một nơi điều phối toàn bộ âm thanh

Nếu OCR, VQA và cảnh báo đều tự gọi TTS, người dùng có thể nghe các câu chồng lên nhau. `AudioCoordinator` vì vậy là nơi duy nhất nhận và sắp xếp tác vụ phát âm thanh trong ứng dụng chính.

| Ưu tiên | Giá trị | Ví dụ |
|---|---:|---|
| `HIGH_RISK` | 1 | Cảnh báo nguy cơ cao |
| `SYSTEM_STATUS` | 2 | Thông báo trạng thái hoặc cảnh báo MEDIUM |
| `ON_DEMAND` | 3 | Kết quả OCR/VQA |
| `INFO` | 4 | Thông báo khởi động |

Số càng nhỏ thì ưu tiên càng cao. Tác vụ mới có ưu tiên cao hơn có thể dừng tác vụ hiện tại nếu tác vụ đó cho phép ngắt. Các tác vụ hết hạn bị bỏ; thông báo trùng không thuộc HIGH_RISK bị hạn chế trong cửa sổ 1,5 giây.

Trong mode on-demand, coordinator chỉ cho cảnh báo HIGH đi qua từ safety pipeline, tạm giữ MEDIUM khỏi chen ngang câu trả lời. Việc đánh giá nguy cơ vẫn tiếp tục; quy tắc này chỉ thay đổi đầu ra âm thanh. Cũng cần lưu ý lời gọi `interrupt()` khi bắt đầu on-demand làm sạch âm thanh cũ, nên ưu tiên HIGH không có nghĩa một cảnh báo đã phát trước đó tuyệt đối không bị ảnh hưởng bởi chuyển mode.

### 8.5. Hủy yêu cầu mà không phát nhầm kết quả đến muộn

Một tác vụ suy luận đang chạy trong thư viện native không phải lúc nào cũng dừng tức thì khi nhấn `S`. Coordinator dùng **generation token**, có thể hiểu là số phiên của yêu cầu. Kết quả chỉ được phát nếu số phiên còn trùng với yêu cầu đang có hiệu lực.

Ví dụ, người dùng gọi VQA rồi nhấn dừng khi model vẫn đang tính. Khi câu trả lời cũ xuất hiện, token đã bị vô hiệu nên nó không được phát âm thanh hoặc đổi mode của một phiên khác. Cơ chế này hủy tác động của kết quả muộn; nó không tuyên bố đã hủy ngay phép tính GPU.

Coordinator còn kiểm tra worker trước đã kết thúc chưa trước khi nhận yêu cầu mới. Điều này hạn chế tích lũy nhiều tác vụ nặng sau những lần người dùng bấm liên tiếp.

## 9. Khởi động, cấu hình và kết thúc chương trình

### 9.1. Trình tự khởi động

`app.py` đọc ba tệp YAML, sau đó áp dụng các giá trị CLI được hỗ trợ. Camera manager và các dịch vụ được khởi tạo, YOLO được nạp ngay từ file local, còn depth thử nạp từ tài nguyên local. Nếu thiếu weights YOLO, khởi động thất bại; nếu depth không nạp được, ứng dụng có thể tiếp tục với dữ liệu độ sâu không khả dụng.

OCR và VQA mới tạo đối tượng dịch vụ, chưa nạp các model nặng. Coordinator đăng ký hai buffer với camera. Khi `start()` chạy, audio worker, camera worker, detection worker và depth worker được khởi động; `AppRunner` tiếp tục vòng lặp giao diện hoặc chế độ không mở cửa sổ.

### 9.2. Các cách chạy chính

Thực hiện từ thư mục gốc dự án, trong môi trường đã cài dependency và chuẩn bị model:

```bash
# Webcam với cấu hình hiện tại
python app.py

# Nguồn ảnh giả để kiểm tra luồng chương trình
python app.py --source dummy

# Video có sẵn
python app.py --source path/to/video.mp4

# Không mở cửa sổ, chạy trong 30 giây
python app.py --source 0 --no-gui --duration 30

# Chọn CPU và đường dẫn weights cụ thể
python app.py --device cpu --weights models/weights/yolov8n.pt
```

Trong cửa sổ GUI, `SPACE` gọi OCR, `Q` gọi mô tả cảnh, `S` dừng lời nói/on-demand và `ESC` thoát. Nguồn dummy phù hợp để kiểm tra cách các thành phần phối hợp, không đánh giá được chất lượng nhận biết cảnh thật.

### 9.3. Cấu hình đã được nối và cấu hình còn thiếu

Những nhóm tham số được `app.py` truyền gồm nguồn/kích thước/FPS camera, thiết bị chung, weights và confidence YOLO, danh sách lớp, kích thước đầu vào depth, tham số ROI, tỷ lệ vùng, một số ngưỡng đồng bộ và FSM, giọng/tốc độ đọc, backend VQA, thông số MLX và đối tượng cấu hình HUD.

Tuy nhiên, nhiều khóa khác chỉ đang tồn tại trong YAML. Ví dụ, chỉnh ngưỡng depth ở `fusion_rules.yaml` chưa thay đổi các ngưỡng viết trực tiếp trong FSM. Vì vậy, khi giải thích tác dụng một tham số, cần theo dấu từ YAML qua `app.py` đến nơi sử dụng, thay vì chỉ đọc tên khóa. Bảng ở mục 11 ghi lại các chênh lệch cụ thể để thuận tiện kiểm tra.

### 9.4. Dừng ứng dụng và vấn đề luồng

Khi dừng, coordinator đặt cờ chạy về false, vô hiệu token on-demand, dừng camera và chờ các worker. Detection/depth và worker on-demand có thời gian `join` tối đa 1,5 giây theo từng lời gọi. Sau đó hệ thống dừng audio, ngắt TTS và vòng giao diện đóng cửa sổ OpenCV.

Thời gian chờ hữu hạn giúp tránh treo vô thời hạn khi một worker chưa kết thúc. Đổi lại, `stop()` trả về chưa có nghĩa mọi phép tính native đã hoàn tất. Các khóa và token hỗ trợ bảo vệ trạng thái chung, nhưng không nên từ đó kết luận toàn bộ chương trình đã được chứng minh không có race condition hoặc deadlock.

## 10. Kiểm thử và cách đọc số liệu đánh giá

### 10.1. Kiểm thử xác nhận được điều gì?

Các test hiện tập trung vào hành vi phần mềm: frame cũ có bị bỏ đúng không, depth có được diễn giải đúng chiều không, cảnh báo ưu tiên có ngắt lời được không, yêu cầu đã hủy có phát kết quả muộn không. Nhiều bài dùng mock, tức đối tượng mô phỏng thay cho model thật, để kiểm tra quy tắc nhanh và ổn định.

Test mock rất hữu ích cho logic điều phối, nhưng không chứng minh model nhìn đúng cảnh hay chạy đủ nhanh trên M1. Cần phân biệt ba tầng đánh giá: kiểm tra quy tắc Python, kiểm tra tích hợp thật giữa thư viện/model và đánh giá chất lượng trong tình huống sử dụng.

### 10.2. Lệnh kiểm thử chuẩn và kết quả lần biên tập

Lệnh được chạy lại ngày 17/09/2026:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /opt/anaconda3/envs/ai-macbook/bin/python \
  -m unittest discover -s tests -p "test_*.py" -v
```

Lần chạy này có 21 test báo `ok`, sau đó dừng tại `test_full_pipeline_dummy` với mã thoát 134. Lỗi native xuất hiện sau khi bước nạp trọng số depth hoàn thành:

```text
Critical nanobind error: refusing to add duplicate key "cpu"
to enumeration "mlx.core.DeviceType"!
```

Đây không phải một assertion báo sai kết quả mong đợi; tiến trình bị dừng ở lớp thư viện native. Vì vậy không thể ghi rằng toàn bộ bộ kiểm thử đã thành công. Cần cô lập môi trường, phiên bản và thứ tự import để tìm nguyên nhân; thông báo hiện có chưa đủ quy trách nhiệm cho một package cụ thể.

Bản phân tích trước ngày 14/09/2026 ghi nhận 21 test báo `ok` trước khi suite dừng, và 75/76 method thành công khi chạy tách tiến trình, riêng full pipeline vẫn lỗi. Các con số chạy tách này được giữ như **ghi nhận lịch sử từ tài liệu gốc**, không phải kết quả được chạy lại trong lần biên tập hiện tại.

### 10.3. Báo cáo benchmark lưu sẵn

Tệp [`evaluation/results/eval_report.json`](evaluation/results/eval_report.json) ghi một lần chạy nguồn dummy trong 5 giây, có 135 frame và FPS được báo là khoảng 26,6. Thời gian detection có P50 khoảng 16,4 ms, P95 khoảng 49,8 ms; depth có P50 khoảng 63,1 ms và P95 khoảng 102,3 ms.

P50 là trung vị: khoảng một nửa mẫu đo không vượt giá trị này. P95 cho biết khoảng 95% mẫu không vượt ngưỡng đó, nên giúp nhìn các lần xử lý chậm hơn thường lệ. Chỉ nhìn thời gian trung bình có thể che mất những đợt chậm đáng chú ý.

Đây là artifact có sẵn, không được benchmark lại trong lần sửa tài liệu. Báo cáo chưa ghi đủ phần cứng, commit, phiên bản dependency và tải đồng thời của VQA. Ngoài ra, thời gian mang tên `end_to_end` trong coordinator được đo quanh lượt detection/fusion sau khi lấy packet, chưa bao gồm toàn bộ chờ camera, hàng đợi âm thanh và thời gian phát hết lời nói. Vì vậy không nên dùng nó như độ trễ từ cảnh thực đến tai người dùng.

### 10.4. Những đánh giá còn cần bổ sung

Để đi từ bản mẫu đến một kết quả nghiên cứu có sức thuyết phục hơn, cần tập video hoặc tình huống có nhãn đối chiếu: vật thể thật, vị trí, khoảng cách đo được và thời điểm cần cảnh báo. Detection cần đo bỏ sót/nhận nhầm; OCR cần đo sai ký tự; VQA cần chấm mức đúng và bám ảnh; cảnh báo cần đo thời gian phản hồi và tần suất cảnh báo không cần thiết.

Trên máy M1 16 GB, cũng cần đo RAM và độ trễ khi đồng thời chạy detection, depth và VQA. Một model chạy tốt một mình chưa đủ chứng minh toàn ứng dụng hoạt động ổn định khi người dùng gọi nhiều chức năng.

## 11. Giới hạn hiện tại và những điểm cần hoàn thiện

### 11.1. Giới hạn ảnh hưởng trực tiếp đến cách diễn giải kết quả

**Khoảng cách chưa được hiệu chuẩn.** Điểm độ gần sau chuẩn hóa phụ thuộc khung hình. Các con số mét do ROI tạo ra chỉ là heuristic. Trước khi có quy trình hiệu chuẩn và đánh giá, cách mô tả “gần” hoặc “rất gần” phù hợp với ý nghĩa dữ liệu hơn.

**Dữ liệu cũ chưa bị loại khỏi mọi quyết định.** Synchronizer có đánh dấu STALE/DEGRADED, nhưng FSM vẫn có thể chấm risk. `RiskAssessment.expires_at` cũng chưa được aggregator kiểm tra. Cần hoàn thiện chính sách khi thiếu dữ liệu hoặc dữ liệu đã cũ, đồng thời thông báo trạng thái để người dùng không hiểu nhầm sự im lặng là an toàn.

**Cơ chế cảnh báo và giám sát chưa hoàn chỉnh.** Override cooldown khi tăng lên HIGH có vấn đề về thứ tự cập nhật trạng thái. Watchdog nhận heartbeat nhưng chưa thấy vòng gọi kiểm tra sức khỏe và xử lý phục hồi được nối vào runtime. Camera có thể chuyển sang dummy sau nhiều lần mở thất bại; nếu không báo rõ, việc giao diện còn chạy có thể gây hiểu nhầm rằng vẫn đang quan sát thật.

**VQA và tracker đều có giới hạn nhận biết.** Guardrail từ khóa không bao phủ mọi cách hỏi; một câu đúng định dạng vẫn có thể sai. Tracker IoU đơn giản cũng có thể đổi ID, ảnh hưởng lịch sử nguy cơ và chống lặp cảnh báo. Đây là những điểm cần đánh giá bằng tình huống cụ thể, không chỉ sửa cách gọi tên.

### 11.2. Bảng đối chiếu giữa cấu hình và triển khai

| Điểm cần kiểm tra | Hiện trạng trong mã | Ý nghĩa khi sử dụng hoặc phát triển |
|---|---|---|
| Tên `ByteTrackerAdapter` | Ghép tham lam theo class và IoU | Cần đổi tên cho đúng hoặc triển khai ByteTrack đầy đủ nếu bài toán cần |
| Ngưỡng depth trong YAML | YAML ghi 0,60/0,40/0,20; FSM dùng 0,55/0,35/0,20 và ngưỡng bên 0,75 | Chỉnh bảng YAML chưa thay đổi quyết định tương ứng |
| `max_alert_age_ms` | Constructor synchronizer lưu nhưng chưa dùng; app cũng chưa truyền khóa này | Chưa có cổng chặn tuổi cảnh báo theo cấu hình đó |
| `STALE`/`DEGRADED` | Vẫn có thể tạo mức risk từ depth đã cũ | Cần chính sách xử lý riêng, không chỉ gắn nhãn |
| Override cooldown | Trạng thái HIGH đã được gán trước khi kiểm tra tăng cấp | Có thể chặn cảnh báo tăng từ MEDIUM lên HIGH trong cooldown |
| Số mét từ relative depth | Ánh xạ tuyến tính tự đặt | Chưa thể coi là khoảng cách đo được |
| `camera.buffer_size` | Hai buffer được tạo trực tiếp với `maxsize=2` | YAML hiện trùng nhưng thay cấu hình chưa có tác dụng |
| Tham số reconnect camera | App không truyền; class dùng mặc định 2 giây/5 lần | Chỉnh YAML không thay đổi chính sách reconnect |
| Interval của worker | Depth dùng trực tiếp 0,125 giây; detection không dùng interval khai báo | Tần suất thật phụ thuộc worker và tốc độ suy luận |
| Watchdog | Có `beat()`, chưa nối vòng `check_health()`/phục hồi | Chưa thể nói hệ thống tự phát hiện và phục hồi mọi worker bị treo |
| Cấu hình audio | `enabled`, `engine`, `enable_chimes`, `min_alert_interval_sec` chưa được áp dụng đầy đủ | Không thể chỉ đổi YAML để tắt âm thanh hoặc thay engine |
| Cấu hình OCR | App chỉ truyền `use_gpu`; service giữ mặc định vi/en, confidence 0,35, blur 50, sáng tối đa 235 | Khác các giá trị YAML 0,3, 60 và 240 |
| IoU/input size detector | Chưa truyền từ YAML, dùng mặc định 0,45/640 | Hiện trùng giá trị nhưng có thể lệch khi chỉnh cấu hình |
| Device từng model | App truyền thiết bị chung cho các thành phần có nhận tham số đó; backend MLX dùng đường riêng | Các khóa device từng model không phải cơ chế chọn thiết bị độc lập; `--device cpu` không chuyển backend MLX sang PyTorch CPU |
| `ui.show_diagnostics` | Chưa được dùng; các cờ depth và spatial guide có được đọc | Không nên hứa rằng cờ này bật/tắt diagnostics |
| Manifest | Còn ghi 21 lớp và chưa liệt kê backend MLX hiện tại | Metadata chưa phản ánh đủ cấu hình 24 lớp và model mới |
| Dependency | Nhiều cận dưới, MLX-VLM chưa nằm trong requirements chính và chưa được pin | Khó tái lập chính xác môi trường và kiểm chứng backend thật |
| Camera fallback | Có thể chuyển nguồn dummy sau nhiều lần thất bại | Cần thông báo rõ mất nguồn quan sát thật |
| `RiskAssessment.expires_at` | Aggregator chưa kiểm tra | Consumer chưa bảo vệ đầy đủ ý nghĩa hạn dùng của contract |

Bảng này ghi nhận hiện trạng phục vụ giải thích và phát triển tiếp. Lần biên tập tài liệu không sửa các hành vi runtime nêu trên.

### 11.3. Thứ tự ưu tiên phát triển tiếp

Trước hết cần khôi phục một lần chạy full suite ổn định và xác định môi trường dependency có thể tái lập. Khi nền tích hợp còn crash, các phép đo mô hình đồng thời khó có ý nghĩa đầy đủ.

Tiếp theo là hoàn thiện các điều kiện liên quan đến cảnh báo: chính sách dữ liệu stale, hạn dùng tại consumer, override cooldown và thông báo mất camera. Sau đó nối các cấu hình đang còn rời rạc, đồng bộ manifest/README/CI và làm rõ tên tracker.

Cuối cùng, thực hiện đánh giá trên cảnh thật và thiết bị đích, đặc biệt trong thời gian VQA hoạt động. Việc đổi model, tăng kích thước model hoặc chuyển sang Core ML nên dựa trên kết quả đo này, thay vì chỉ dựa vào thông số riêng của từng mô hình.

## 12. Nhìn lại ý tưởng thiết kế của đề tài

Điểm em muốn thể hiện trong đề tài là cách phối hợp nhiều công nghệ để giải quyết các nhu cầu khác nhau của người dùng. Detection trả lời có vật gì; depth bổ sung quan hệ gần, xa; fusion chuyển các kết quả đó thành thông tin cần chú ý; OCR và VQA hỗ trợ khi người dùng chủ động tìm hiểu thêm; âm thanh đưa kết quả đến người dùng theo thứ tự ưu tiên.

Giá trị của thiết kế nằm cả ở phần mô hình lẫn cách quản lý dữ liệu và thời gian. Một nhận diện đúng nhưng đến quá muộn, một độ sâu bị hiểu sai đơn vị hoặc một cảnh báo bị chìm trong lời đọc đều làm giảm ích lợi của hệ thống. Vì vậy, buffer hữu hạn, timestamp, contracts và cơ chế ngắt lời là các phần thiết yếu của giải pháp.

Phiên bản hiện tại đã thể hiện được hướng tổ chức đó, nhưng vẫn còn khoảng cách giữa bản mẫu hoạt động và hệ thống đủ tin cậy trong sử dụng thực tế. Trình bày rõ phần đã làm, phần được kiểm thử và phần chưa xác minh giúp đề tài có cơ sở để tiếp tục hoàn thiện mà không phóng đại khả năng hiện có.

## 13. Nguồn đối chiếu và gợi ý đọc tiếp

Để kiểm tra hành vi chương trình, ưu tiên đọc [`app.py`](app.py), [`SystemCoordinator`](src/runtime/system_coordinator.py), các adapter dưới [`src/`](src/), ba tệp [`configs/`](configs/) và các bài [`tests/`](tests/). Những vị trí cụ thể đã được chỉ ra trong bảng cấu trúc tệp và các phần luồng xử lý.

Các nguồn chính thức được dẫn gần phần mô tả công nghệ/model dùng để giải thích khả năng và giao diện của chúng. Những nguồn đó không chứng minh độ chính xác, mức sử dụng RAM hoặc độ trễ của chính ứng dụng này trên M1. Với các kết luận về dự án, cần kết hợp mã nguồn với log kiểm thử và báo cáo đánh giá có ghi rõ điều kiện thực nghiệm.
