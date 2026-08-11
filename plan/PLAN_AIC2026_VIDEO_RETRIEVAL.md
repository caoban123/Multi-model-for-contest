# KẾ HOẠCH XÂY DỰNG HỆ THỐNG TRUY XUẤT VIDEO CHO AIC 2026

> Tài liệu ngữ cảnh và kế hoạch triển khai dành cho Codex Agent  
> Trạng thái: **Planning only – chưa được phép viết mã nguồn**  
> Ngôn ngữ làm việc chính: **Tiếng Việt**  
> Phiên bản tài liệu: **1.0**  
> Phạm vi: **Vòng sơ tuyển AIC 2026 – Textual KIS, Q&A và TRAKE**

---

## 0. CHỈ DẪN BẮT BUỘC DÀNH CHO CODEX AGENT

Tài liệu này là nguồn ngữ cảnh trung tâm của dự án. Codex phải đọc toàn bộ tài liệu trước khi phân tích repository, đề xuất kiến trúc chi tiết hoặc thực hiện bất kỳ thay đổi nào.

Ở thời điểm hiện tại, Codex **không được viết mã nguồn**, không tạo project skeleton, không cài thư viện, không tải model, không tải dữ liệu và không thay đổi cấu trúc repository. Nhiệm vụ trước mắt chỉ gồm: hiểu bài toán, kiểm tra các giả định, phân rã hệ thống, lập kế hoạch và báo cáo những điểm còn thiếu.

Mọi thông tin trong tài liệu được chia thành ba mức độ:

1. **Thông tin chính thức từ BTC**: được lấy từ tài liệu “Thông tin vòng Sơ tuyển AIC2026”, ba buổi tập huấn và bảng “[AIC 2026] TỔNG HỢP THẮC MẮC THÍ SINH”.
2. **Suy luận kỹ thuật của nhóm**: được rút ra từ cấu trúc dữ liệu thực tế và cơ chế chấm điểm.
3. **Đề xuất kiến trúc**: được tham khảo từ tư duy chung của các hệ thống AIC/SOICT, Lifelog Search Challenge, Video Browser Showdown, VISIONE và vitrivr. Đây không phải yêu cầu bắt buộc của BTC.

Codex không được biến suy luận hoặc đề xuất thành “quy định của BTC”. Khi báo cáo, phải ghi rõ đâu là dữ kiện chính thức, đâu là giả định và đâu là đề xuất.

Trước khi được cho phép triển khai, Codex phải hoàn thành các đầu ra sau:

- Bản kiểm kê repository hiện tại.
- Bản kiểm kê dữ liệu hiện có trên máy.
- Sơ đồ dependency giữa các module dự kiến.
- Danh sách quyết định kỹ thuật cần người dùng phê duyệt.
- Danh sách rủi ro và điểm chưa được BTC trả lời.
- Lộ trình triển khai theo từng milestone.
- Tiêu chí nghiệm thu cho từng milestone.
- Đề xuất cấu hình tối thiểu và cấu hình mở rộng.
- Ước lượng tài nguyên lưu trữ, RAM, CPU, GPU và thời gian tiền xử lý ở mức tương đối.
- Kế hoạch kiểm thử không phụ thuộc vào ground truth chính thức.

Codex phải hỏi lại người dùng trước những hành động sau:

- Tải model hoặc dữ liệu có dung lượng lớn.
- Thay đổi framework backend hoặc frontend.
- Chọn vector database ngoài FAISS.
- Chọn dịch vụ cloud hoặc API trả phí.
- Re-encode video hoặc thay đổi frame rate.
- Xóa, di chuyển hoặc giải nén hàng loạt dữ liệu.
- Thêm mô hình nặng vào đường chạy online.
- Thiết kế hệ thống phụ thuộc Internet trong lúc thi.

---

# 1. BỐI CẢNH DỰ ÁN

## 1.1. Mục tiêu tổng quát

Xây dựng một hệ thống tìm kiếm và hiểu video phục vụ vòng sơ tuyển AIC 2026. Hệ thống phải hỗ trợ ba dạng truy vấn:

- **Textual Known Item Search – Textual KIS**: tìm đúng video và một frame nằm trong đoạn đáp án dựa trên mô tả bằng văn bản.
- **Q&A**: tìm đúng video, đúng đoạn frame và trả lời đúng nội dung câu hỏi.
- **TRAKE – Temporal Retrieval and Alignment of Key Events**: tìm đúng video chứa một chuỗi sự kiện, sau đó căn chỉnh một semantic keyframe cho từng sự kiện con theo đúng thứ tự thời gian.

Hệ thống không chỉ là một model. Đây phải là một **video retrieval system hoàn chỉnh**, gồm dữ liệu, chỉ mục, nhiều bộ truy xuất, cơ chế fusion, reranking, temporal reasoning, giao diện tương tác, xuất kết quả và công cụ đánh giá nội bộ.

## 1.2. Mục tiêu cạnh tranh

Do cách chấm điểm sử dụng nhiều mức xếp hạng, hệ thống phải tối ưu đồng thời:

- **Top-1 precision**: đưa đáp án tốt nhất lên vị trí đầu.
- **Top-5 precision**: giữ đáp án đúng trong nhóm đầu.
- **Recall@20, Recall@50 và Recall@100**: không bỏ sót video đúng ở tầng candidate generation.
- **Tốc độ thao tác**: người dùng phải kiểm tra và điều chỉnh kết quả nhanh.
- **Temporal accuracy**: đặc biệt quan trọng với TRAKE.
- **Khả năng giải thích và kiểm soát**: người dùng phải biết vì sao một kết quả được xếp hạng cao và có thể sửa hướng tìm kiếm.

## 1.3. Non-goals trong giai đoạn đầu

Các nội dung sau chưa phải ưu tiên của phiên bản đầu:

- Huấn luyện một vision-language model mới từ đầu.
- Fine-tune model lớn trên toàn bộ video.
- Xây dựng Agent tự trị hoàn toàn.
- Tự động trả lời tất cả câu Q&A không cần con người.
- Xử lý toàn bộ video bằng LVLM trong lúc truy vấn.
- Phụ thuộc vào một API thương mại duy nhất.
- Xây dựng hệ thống phân tán nhiều server ngay từ đầu.
- Tối ưu cực hạn trước khi có baseline đo được.
- Re-encode toàn bộ video nếu chưa xác minh frame mapping.

---

# 2. THÔNG TIN CHÍNH THỨC TỪ BTC

## 2.1. Các loại truy vấn

### Textual KIS

Đầu vào là một mô tả bằng ngôn ngữ tự nhiên về một sự kiện. Đội thi phải xác định đúng video và nộp một frame bất kỳ nằm trong đoạn đáp án đúng.

Định dạng logic:

- `video_id`
- `frame_id`

Một kết quả được xem là đúng khi:

- `video_id` khớp ground truth.
- `frame_id` nằm trong đoạn `[s, e]`.

### Q&A

Đầu vào gồm mô tả sự kiện và một câu hỏi về thông tin xuất hiện trong sự kiện. Đội thi phải:

- Tìm đúng video.
- Chọn đúng frame hoặc đoạn frame theo quy định.
- Trả lời đúng về mặt ngữ nghĩa.

Câu trả lời có thể bằng tiếng Việt hoặc tiếng Anh theo tài liệu BTC.

Định dạng logic:

- `video_id`
- `frame_id`
- `answer`

### TRAKE

Đầu vào mô tả một chuỗi sự kiện có cấu trúc. Hệ thống phải:

1. Tìm đúng một video chứa chuỗi sự kiện.
2. Với mỗi sự kiện con, chọn đúng một semantic keyframe.
3. Bảo đảm các frame tương ứng đúng nội dung và đúng thứ tự thời gian.

Semantic keyframe là khoảnh khắc có ý nghĩa nội dung, không phải I-frame kỹ thuật trong nén video.

## 2.2. Cách chấm điểm

Mỗi truy vấn được phép gửi tối đa **100 câu trả lời**.

Với Textual KIS và Q&A, R-Score về cơ bản là nhị phân:

- Đúng đầy đủ điều kiện: 1.
- Sai một trong các điều kiện bắt buộc: 0.

Với TRAKE:

- Sai video: toàn bộ câu trả lời nhận 0.
- Đúng video: điểm bằng tỷ lệ semantic keyframe đúng trên tổng số sự kiện con.

BTC tính các mức:

- R@1
- R@5
- R@20
- R@50
- R@100

Điểm cuối cùng của một truy vấn là trung bình của năm mức trên.

### Hệ quả kỹ thuật

Cơ chế chấm điểm buộc hệ thống phải có hai tầng rõ ràng:

- **Candidate generation** ưu tiên recall để đáp án đúng không bị loại khỏi Top-100.
- **Reranking** ưu tiên precision để đáp án đúng được đẩy lên Top-1 và Top-5.

Chỉ xây một retriever rồi sắp xếp bằng cosine similarity là chưa đủ.

## 2.3. Dữ liệu BTC cung cấp

Từ tài liệu và danh sách tải dữ liệu hiện tại, Batch 1 có các nhóm dữ liệu:

- Videos.
- Keyframes.
- Objects.
- CLIP Features.
- Mapping giữa CLIP feature, keyframe, video và frame.
- Media information hoặc metadata.

Các file được chia theo nhóm video như L21, L22, …, L30. Một số nhóm video hoặc keyframe được chia thành nhiều archive con như `_a`, `_b`, `_c`.

Các file hỗ trợ toàn batch gồm dạng tương tự:

- `clip-features-32-aic25-b1.zip`
- `map-keyframes-aic25-b1.zip`
- `media-info-aic25-b1.zip`
- `objects-aic25-b1.zip`

BTC cho biết CLIP feature mẫu được trích xuất bằng **CLIP ViT-B/32** và lưu theo thứ tự keyframe.

Objects được trích xuất bởi Faster R-CNN pretrained trên OpenImages V4 theo tài liệu vòng sơ tuyển.

Metadata có nguồn từ YouTube và có thể thiếu ở một số video.

Video là dữ liệu gốc của cuộc thi. Keyframes, Objects, CLIP Features và Metadata là dữ liệu hỗ trợ truy xuất.

## 2.4. Chính sách công cụ và tài nguyên

Theo bảng Q&A của BTC:

- BTC không giới hạn model, thuật toán hoặc công cụ dùng để xây dựng hệ thống.
- Có thể xây hệ thống tự động hoặc hệ thống có người tương tác.
- AI Agent chỉ là định hướng, không bắt buộc.
- BTC không cung cấp API AI.
- BTC không cung cấp máy ảo.
- Đội thi tự chuẩn bị tài nguyên.
- Vòng sơ tuyển yêu cầu nộp submission kèm báo cáo mô tả giải pháp hoặc hệ thống.
- Không bắt buộc phải tải và lưu đầy đủ toàn bộ dữ liệu.
- Nếu không lưu dữ liệu nào đó thì đội có nguy cơ không truy xuất được phần dữ liệu đó.
- BTC không hỗ trợ lưu trữ riêng cho từng đội.

## 2.5. Những điểm BTC chưa trả lời rõ

Tại thời điểm lập tài liệu này, các nội dung sau chưa có câu trả lời chính thức đầy đủ:

- Có được dùng Internet trong lúc thi hay không.
- Có được gọi API model bên ngoài trong lúc xử lý truy vấn hay không.
- Định dạng submission cuối cùng là CSV, JSON hay định dạng khác.
- Frame ID bắt đầu từ 0 hay 1.
- Cửa sổ `[s, e]` của Textual KIS rộng bao nhiêu.
- Số lượng truy vấn chính thức.
- Batch 2 công bố lúc nào.
- Tổng dung lượng và thời lượng Batch 2.
- Batch 2 có giữ nguyên hoàn toàn cấu trúc Batch 1 hay không.
- Cách chấm “khớp ngữ nghĩa” của câu trả lời Q&A.
- Có submission thử hay không.
- Keyframe BTC được trích xuất bằng thuật toán cụ thể nào.
- Với TRAKE, đề sẽ cho mô tả riêng từng sự kiện hay một mô tả tổng.
- Quy tắc chính xác khi có nhiều frame gần tương đương.
- Có baseline chính thức hay không.

### Quy tắc thiết kế do các điểm chưa rõ

Hệ thống phải:

- Có khả năng chạy offline.
- Không phụ thuộc bắt buộc vào API bên ngoài.
- Tách cấu hình submission khỏi logic retrieval.
- Không hard-code frame indexing.
- Cho phép adapter khi Batch 2 có cấu trúc khác.
- Cho phép thay đổi answer normalization.
- Không giả định audio luôn tồn tại.
- Không giả định metadata luôn đầy đủ.

---

# 3. BÀI HỌC KIẾN TRÚC TỪ CÁC HỆ THỐNG CÔNG KHAI

## 3.1. Tư duy chung

Các hệ thống mạnh trong AIC/SOICT, LSC và VBS không xem bài toán là “chọn một model mạnh nhất”. Họ xây dựng một hệ thống gồm nhiều kênh tìm kiếm chuyên biệt.

Mỗi modality giải một loại tín hiệu:

- Dense visual-text embedding: ngữ nghĩa tổng quát.
- Object detection: vật thể, số lượng, vị trí.
- OCR: chữ trên màn hình, biển hiệu, số, tên riêng.
- ASR: lời nói và âm thanh.
- Metadata: tiêu đề, kênh, thời gian, nguồn.
- Color feature: màu sắc nổi bật.
- Temporal index: thứ tự, khoảng cách và chuỗi sự kiện.
- Human feedback: sửa hướng tìm kiếm trong thời gian ngắn.

## 3.2. Bài học từ VISIONE và VBS

Tư duy quan trọng:

- Tiền xử lý nặng được thực hiện offline.
- Online query chỉ encode truy vấn và tìm trên index.
- Nhiều feature được tìm riêng rồi kết hợp.
- Search engine và UI là thành phần quan trọng ngang model.
- Kết quả phải được group theo video và giảm trùng lặp.
- Cần browsing theo scene, frame lân cận và timeline.
- Search-by-example là một thao tác rất mạnh.
- Có thể dùng FAISS cho vector và một lexical engine cho object, OCR, tag, metadata.

## 3.3. Bài học từ vitrivr và LSC

Tư duy quan trọng:

- Backend retrieval, database và UI nên tách module.
- Mỗi feature trả về một partial ranking.
- Fusion nên có trọng số điều chỉnh được.
- Người dùng phải có thể chuyển giữa exploration và exploitation.
- Metadata thời gian và location có thể quan trọng ngang embedding.
- Giao diện không chỉ hiển thị; giao diện tham gia vào ranking thông qua feedback.
- Event grouping giúp tránh nhiều keyframe gần giống nhau.
- Conversational search có ích nhưng không được thay thế retrieval core.

## 3.4. Bài học từ các hệ thống AIC/SOICT

Tư duy quan trọng:

- Một truy vấn phức tạp nên được phân rã thành nhiều sự kiện hoặc điều kiện.
- Mỗi điều kiện có thể chọn một modality tìm kiếm khác nhau.
- OCR, object, color và spatial relation có thể quyết định kết quả.
- Temporal retrieval cần tìm từng event rồi kiểm tra cùng video và đúng thứ tự.
- LLM phù hợp cho query rewriting, dịch, phân tích intent và tạo filter.
- LLM không nên là nơi duy nhất quyết định frame cuối cùng.

---

# 4. NGUYÊN TẮC THIẾT KẾ CỦA DỰ ÁN

## 4.1. Offline-heavy, online-light

Mọi công việc tốn tài nguyên nên được đẩy về offline:

- Trích xuất hoặc nạp feature.
- Chuẩn hóa mapping.
- Tạo index.
- OCR toàn bộ keyframe nếu đủ tài nguyên.
- ASR toàn bộ video nếu đủ tài nguyên.
- Captioning chọn lọc.
- Object normalization.
- Tạo temporal neighborhood.
- Tạo thumbnail và cache.

Đường online phải ngắn:

1. Phân tích query.
2. Encode query.
3. Truy xuất từ các index.
4. Fusion.
5. Rerank tập nhỏ.
6. Hiển thị hoặc xuất kết quả.

## 4.2. Retrieval-first

Ưu tiên đầu tiên là xây retrieval core có thể đo được. Agent, LVLM và Q&A automation chỉ được thêm sau khi retrieval baseline ổn định.

## 4.3. Multi-index, không single-model

Hệ thống phải coi mỗi kênh là một nguồn bằng chứng:

- Semantic evidence.
- Object evidence.
- OCR evidence.
- ASR evidence.
- Metadata evidence.
- Temporal evidence.
- Human feedback evidence.

Không có kênh nào được xem là luôn đúng.

## 4.4. Hai tầng recall và precision

Tầng 1 tìm rộng.  
Tầng 2 đánh giá kỹ.

Tầng 1 phải đủ nhanh để lấy hàng trăm hoặc hàng nghìn ứng viên.  
Tầng 2 chỉ chạy trên tập nhỏ để kiểm tra ràng buộc phức tạp.

## 4.5. Human-in-the-loop

Hệ thống vòng sơ tuyển có thể dùng người vận hành. Vì vậy UI phải hỗ trợ:

- Xem nhiều candidate nhanh.
- Chọn positive và negative example.
- Sửa trọng số.
- Mở frame lân cận.
- Nhảy đến video/timeline.
- Search-by-image.
- Ghim một video để tìm sâu bên trong.
- So sánh nhiều candidate.
- Xác nhận đáp án trước khi export.

## 4.6. Không phá frame mapping

Không được re-encode video một cách tùy tiện. Bất kỳ proxy video nào cũng phải có mapping rõ với video gốc. Kết quả cuối cùng phải quy về `video_id` và `frame_id` của dữ liệu gốc.

## 4.7. Cấu hình thay cho hard-code

Tất cả đường dẫn, batch, model, index, số lượng candidate, fusion weight, top-k và submission format phải nằm trong cấu hình.

---

# 5. KIẾN TRÚC HỆ THỐNG ĐỀ XUẤT

## 5.1. Sơ đồ logic tổng quát

Hệ thống gồm tám lớp:

1. **Data Registry Layer**
2. **Offline Feature and Index Layer**
3. **Query Understanding Layer**
4. **Candidate Retrieval Layer**
5. **Fusion and Video Aggregation Layer**
6. **Reranking and Reasoning Layer**
7. **Task-specific Solver Layer**
8. **User Interface and Submission Layer**

## 5.2. Data Registry Layer

Đây là nguồn sự thật duy nhất về dữ liệu.

### Video Registry

Mỗi video cần có các thông tin logic:

- Batch.
- `video_id`.
- Đường dẫn hoặc archive chứa video.
- Thời lượng.
- FPS.
- Tổng số frame.
- Có audio hay không.
- Metadata hiện có.
- Nhóm L21–L30.
- Trạng thái đã tải hay chưa.
- Trạng thái đã index hay chưa.
- Trạng thái ASR/OCR/caption.
- Checksum nếu có.

### Keyframe Registry

Mỗi keyframe cần có:

- Keyframe ID nội bộ.
- `video_id`.
- `frame_id`.
- Timestamp suy ra từ FPS nếu hợp lệ.
- Đường dẫn ảnh bên trong ZIP hoặc đường dẫn đã giải nén.
- Chỉ số tương ứng trong CLIP feature.
- Batch.
- Scene hoặc shot ID nếu có.
- Previous/next keyframe.
- Neighborhood frame range.
- Trạng thái ảnh có thể đọc được.
- Trạng thái OCR/object/caption.

### Feature Registry

Mỗi loại feature phải có:

- Tên feature.
- Model tạo feature.
- Kích thước vector.
- Chuẩn normalize.
- Metric dùng để tìm kiếm.
- Thứ tự vector.
- Mapping version.
- Batch coverage.
- Index version.

### Yêu cầu

Không được dựa vào tên file một cách ngầm định. Tất cả mapping phải được kiểm tra tính nhất quán.

## 5.3. Offline Feature and Index Layer

### Dense Visual-Text Index

Baseline bắt buộc sử dụng CLIP ViT-B/32 do BTC cung cấp.

Nhiệm vụ:

- Nạp feature.
- Kiểm tra shape.
- Kiểm tra số vector khớp mapping.
- Kiểm tra normalization.
- Tạo FAISS index.
- Lưu index version.
- Hỗ trợ truy vấn text-to-image.
- Hỗ trợ image-to-image khi có query image.

### Metadata Index

Lưu các trường:

- Video title.
- Channel.
- Description.
- Tags.
- Publish time nếu có.
- Group/batch.
- Duration.
- Audio availability.
- Các trường khác từ media-info.

Có thể dùng SQLite hoặc DuckDB trước. Elasticsearch/OpenSearch chỉ cần khi lexical search phức tạp hoặc dữ liệu lớn.

### Object Index

BTC đã cung cấp object output. Hệ thống phải chuẩn hóa:

- Label.
- Confidence.
- Bounding box.
- Relative position.
- Area ratio.
- Count.
- Keyframe ID.
- Video ID.
- Frame ID.

Cần hỗ trợ:

- Có/không có object.
- Số lượng object.
- Vị trí trái/phải/trên/dưới/trung tâm.
- Kích thước tương đối.
- Đồng xuất hiện nhiều object.
- Khoảng cách hoặc quan hệ không gian ở mức gần đúng.

### OCR Index

OCR chưa chắc được BTC cung cấp. Đây là module mở rộng có giá trị cao.

Mục tiêu:

- Trích chữ trên keyframe.
- Chuẩn hóa Unicode.
- Giữ cả text gốc và text normalized.
- Lưu bounding box và confidence.
- Hỗ trợ exact match, fuzzy match và BM25.
- Hỗ trợ chữ Việt và Anh.
- Không chạy OCR online trên toàn dataset.

### ASR Index

ASR chỉ áp dụng cho video có audio.

Mục tiêu:

- Transcript theo segment.
- Timestamp start/end.
- Ngôn ngữ.
- Confidence nếu có.
- Mapping transcript về frame range.
- Search bằng lexical index.
- Cho phép mở video đúng đoạn lời nói.

### Caption Index

Captioning là optional.

Chỉ nên chạy:

- Trên toàn keyframe khi có GPU và thời gian.
- Hoặc trên candidate thường xuyên xuất hiện.
- Hoặc trên các scene mà CLIP yếu.

Caption không thay thế embedding. Caption là một kênh lexical bổ sung.

### Temporal Index

Cần lưu:

- Thứ tự keyframe trong video.
- Frame ID.
- Scene/shot.
- Khoảng cách thời gian.
- Neighborhood.
- Các event candidate đã được tìm.
- Liên kết giữa transcript, OCR và keyframe.

Temporal index là nền tảng cho TRAKE.

---

# 6. CÁC MÔ HÌNH VÀ VAI TRÒ

## 6.1. CLIP ViT-B/32

Trạng thái: **Baseline bắt buộc** vì feature đã được BTC cung cấp.

Vai trò:

- Text-to-keyframe retrieval.
- Image-to-keyframe retrieval.
- Candidate generation nhanh.
- Baseline để đánh giá các model mới.

Hạn chế:

- Yếu với chữ nhỏ.
- Yếu với đếm.
- Yếu với quan hệ thời gian.
- Có thể nhầm hành động gần giống.
- Không bảo đảm hiểu ràng buộc AND phức tạp.

## 6.2. Model embedding mạnh hơn

Ứng viên có thể gồm SigLIP, OpenCLIP, EVA-CLIP hoặc video-text encoder.

Vai trò:

- Tạo index bổ sung.
- Rerank candidate.
- So sánh ensemble.

Không được thêm vào baseline trước khi:

- Có benchmark nội bộ.
- Có storage budget.
- Có thời gian tạo feature.
- Có kế hoạch versioning.

## 6.3. Cross-encoder hoặc reranker

Vai trò:

- Đánh giá lại Top-N candidate.
- Hiểu toàn bộ câu mô tả tốt hơn cosine similarity.
- Kiểm tra ràng buộc object, action, context ở mức sâu hơn.

Reranker không được chạy trên toàn dataset.

## 6.4. OCR

Ứng viên: PaddleOCR hoặc model OCR tương đương.

Vai trò:

- Biển hiệu.
- Tên người/tổ chức.
- Phụ đề.
- Số.
- Giá.
- Logo dạng chữ.
- Chữ trên giao diện máy tính.

## 6.5. ASR

Ứng viên: faster-whisper hoặc model tương đương.

Vai trò:

- Tìm câu nói.
- Tìm tên riêng.
- Trả lời Q&A từ audio.
- Căn đoạn video theo lời thoại.

ASR chỉ là optional cho video có audio.

## 6.6. Object detector hoặc open-vocabulary detector

BTC đã có object Faster R-CNN/OpenImages.

Model bổ sung như Grounding DINO hoặc YOLO-World chỉ cần khi:

- Label BTC không đủ.
- Query có object hiếm.
- Cần open-vocabulary detection trên candidate.

Không chạy model này trên toàn bộ keyframe trước khi có bằng chứng cần thiết.

## 6.7. LVLM

Ứng viên: Qwen-VL, LLaVA, BLIP-2 hoặc model tương đương.

Vai trò phù hợp:

- Rerank Top candidate.
- Trả lời Q&A trên một đoạn ngắn.
- Mô tả chi tiết candidate.
- Xác minh quan hệ giữa nhiều frame.
- Hỗ trợ người dùng đọc kết quả.

Vai trò không phù hợp:

- Quét toàn bộ dataset online.
- Thay thế FAISS.
- Tự đoán video mà không có retrieval.
- Là nguồn duy nhất cho frame ID.

## 6.8. LLM Query Planner

Vai trò:

- Dịch truy vấn Việt–Anh.
- Tách subject, action, object, place, color, text, audio cue.
- Tách chuỗi TRAKE thành event.
- Chọn modality tìm kiếm.
- Sinh synonym.
- Viết lại truy vấn.
- Tạo structured filter.
- Gợi ý câu hỏi làm rõ.

LLM Query Planner phải có fallback rule-based. Nếu API/Internet không dùng được, hệ thống vẫn phải hoạt động.

---

# 7. PIPELINE XỬ LÝ OFFLINE

## 7.1. Bước O1 – Data audit

Kiểm tra:

- File nào đã tải.
- Archive nào hỏng.
- Dung lượng.
- Cấu trúc thư mục trong ZIP.
- Số video.
- Số keyframe.
- Số vector.
- Số mapping row.
- Coverage object.
- Coverage metadata.
- Audio availability.
- Batch coverage.

Đầu ra:

- Data inventory.
- Missing-data report.
- Corruption report.
- Storage estimate.
- Manifest draft.

## 7.2. Bước O2 – Chuẩn hóa mapping

Mục tiêu:

- Một keyframe phải ánh xạ được tới đúng video và frame.
- Một vector phải ánh xạ được tới đúng keyframe.
- Không trùng ID.
- Không mất thứ tự.
- Không có vector orphan.
- Không có keyframe orphan.

Đây là bước có độ ưu tiên cao nhất trước khi xây index.

## 7.3. Bước O3 – Tạo baseline index

Tạo FAISS index từ CLIP feature BTC.

Đầu ra logic:

- Index file.
- Mapping table.
- Index metadata.
- Version.
- Metric.
- Normalization status.
- Build report.

## 7.4. Bước O4 – Xây structured stores

Tạo store cho:

- Video.
- Keyframe.
- Object.
- Metadata.
- Temporal order.

Không cần dùng database phức tạp ngay từ đầu.

## 7.5. Bước O5 – Optional enrichments

Theo thứ tự ưu tiên:

1. OCR.
2. ASR.
3. Caption.
4. Additional embedding.
5. Open-vocabulary object.
6. Scene relation.

Mỗi enrichment phải có benchmark chứng minh giá trị trước khi xử lý toàn bộ dataset.

---

# 8. PIPELINE XỬ LÝ TRUY VẤN ONLINE

## 8.1. Bước Q1 – Nhận dạng loại truy vấn

Hệ thống cần biết query thuộc:

- Textual KIS.
- Q&A.
- TRAKE.
- Search-by-image.
- Metadata search.
- Object search.
- Hybrid search.

Người dùng phải có thể sửa loại query nếu hệ thống phân loại sai.

## 8.2. Bước Q2 – Query understanding

Tách truy vấn thành:

- Main subject.
- Secondary subjects.
- Action.
- Object.
- Color.
- Clothing.
- Location.
- Scene.
- Text clue.
- Audio clue.
- Count.
- Spatial relation.
- Temporal relation.
- Negative constraint.
- Uncertainty.

Ví dụ logic:

“Người đàn ông mặc áo đỏ bước vào cửa hàng, sau đó cầm điện thoại, phía sau có biển Samsung.”

Phân rã:

- Subject: man.
- Clothing/color: red shirt.
- Event 1: enter store.
- Event 2: hold phone.
- OCR clue: Samsung.
- Temporal relation: event 1 before event 2.
- Context: indoor/store.

## 8.3. Bước Q3 – Query expansion

Sinh:

- Bản tiếng Việt.
- Bản tiếng Anh.
- Synonym.
- Câu ngắn tập trung object.
- Câu ngắn tập trung action.
- Câu ngắn tập trung scene.
- Negative prompt nếu cần.
- Structured filters.

Không được sinh quá nhiều query expansion không kiểm soát. Mỗi expansion phải có nguồn và trọng số.

## 8.4. Bước Q4 – Parallel retrieval

Chạy các kênh:

- CLIP semantic search.
- Metadata BM25.
- Object filter.
- OCR search.
- ASR search.
- Search-by-image nếu có.
- Temporal event search nếu là TRAKE.

Mỗi kênh trả về:

- Candidate.
- Score thô.
- Rank trong kênh.
- Evidence.
- Query variant tạo candidate.

## 8.5. Bước Q5 – Fusion

Các phương án:

- Weighted score fusion.
- Reciprocal Rank Fusion.
- Rank voting.
- Rule-based boost.
- Hybrid fusion theo query type.

Baseline nên bắt đầu với RRF vì các score giữa modality khó so sánh trực tiếp.

Sau khi có dữ liệu đánh giá, có thể học hoặc tinh chỉnh trọng số.

## 8.6. Bước Q6 – Video aggregation

Không nên để Top-100 chứa quá nhiều frame của cùng một video.

Cần:

- Group keyframe theo video.
- Tính video-level score.
- Giữ một số frame đại diện mỗi video.
- Bảo toàn frame tốt nhất.
- Giữ diversity giữa video.
- Cho phép mở thêm frame của một video khi người dùng chọn.

## 8.7. Bước Q7 – Reranking

Rerank Top-N candidate bằng:

- Model embedding mạnh hơn.
- Cross-encoder.
- LVLM.
- Structured rule.
- Object/OCR/ASR evidence.
- Temporal consistency.

Cần lưu cả score trước và sau rerank để phân tích lỗi.

## 8.8. Bước Q8 – Human feedback

Cho phép:

- Chọn result relevant.
- Chọn result irrelevant.
- Search similar.
- Tăng/giảm trọng số modality.
- Giới hạn video.
- Mở temporal neighborhood.
- Thêm filter.
- Bỏ filter.
- Ghim candidate.

Feedback phải tạo truy vấn mới hoặc rerank lại mà không phá trạng thái cũ.

---

# 9. PIPELINE RIÊNG CHO TEXTUAL KIS

## 9.1. Mục tiêu

Tìm đúng video và một frame thuộc đoạn đáp án.

## 9.2. Quy trình

1. Query understanding.
2. Sinh 2–5 query variants có kiểm soát.
3. CLIP retrieval Top-K lớn.
4. Object/OCR/metadata retrieval nếu query chứa tín hiệu tương ứng.
5. Fusion.
6. Group theo video.
7. Diversity.
8. Rerank Top video và Top frame.
9. Hiển thị grid.
10. Người dùng mở timeline hoặc frame lân cận.
11. Chọn frame.
12. Export tối đa 100 result theo thứ hạng.

## 9.3. Chiến lược ranking

Ưu tiên:

- Candidate đúng video ở Top-100.
- Đúng scene ở Top-20.
- Đúng frame ở Top-5.
- Tốt nhất ở Top-1.

## 9.4. Kiểm tra lỗi

Phân loại lỗi:

- Sai video nhưng đúng semantic.
- Đúng video, sai scene.
- Đúng scene, frame quá sớm hoặc quá muộn.
- Query expansion làm lệch nghĩa.
- Object/OCR filter loại mất đáp án.
- Nhiều frame trùng nhau chiếm ranking.
- CLIP nhầm action.
- Metadata boost quá mạnh.

---

# 10. PIPELINE RIÊNG CHO Q&A

## 10.1. Mục tiêu

Không chỉ trả lời câu hỏi; phải định vị đúng video và frame trước.

## 10.2. Quy trình

1. Tách mô tả sự kiện khỏi câu hỏi.
2. Retrieval bằng phần mô tả.
3. Chọn Top video.
4. Mở đoạn video hoặc chuỗi frame quanh candidate.
5. Xác định modality trả lời:
   - Visual.
   - OCR.
   - ASR.
   - Counting.
   - Temporal.
6. Chạy answer module trên đoạn nhỏ.
7. Sinh answer candidate.
8. Chuẩn hóa answer.
9. Hiển thị evidence.
10. Cho người dùng xác nhận.
11. Export `video_id`, `frame_id`, `answer`.

## 10.3. Answer normalization

Cần có lớp normalization cho:

- Viết hoa/thường.
- Dấu tiếng Việt.
- Số chữ và số ký tự.
- Đơn vị.
- Màu sắc.
- Tên riêng.
- Synonym.
- Tiếng Việt và tiếng Anh.

Vì BTC chưa công bố chi tiết cách chấm semantic match, phải lưu:

- Answer gốc.
- Answer normalized.
- Alternative forms.
- Evidence frame.
- Evidence transcript/OCR.

## 10.4. Counting

Không tin trực tiếp vào LVLM khi đếm.

Cần:

- Xem nhiều frame.
- Tránh đếm lặp cùng object.
- Dùng detector hoặc tracking nếu cần.
- Cho người dùng kiểm tra.

## 10.5. Temporal Q&A

Nếu câu hỏi có “sau đó”, “trước khi”, “ai tiếp theo”, “bao nhiêu lần”:

- Lấy đoạn video dài hơn.
- Xem sequence frame.
- Dùng temporal reasoning.
- Không trả lời từ một keyframe đơn.

---

# 11. PIPELINE RIÊNG CHO TRAKE

## 11.1. Mục tiêu

Tìm đúng video trước, sau đó tìm đúng semantic keyframe cho từng event.

## 11.2. Query decomposition

Tách chuỗi thành:

- Event 1.
- Event 2.
- …
- Event N.

Mỗi event có:

- Text description.
- Object.
- Action.
- Scene.
- Modality.
- Confidence.
- Ràng buộc với event khác.

## 11.3. Candidate retrieval cho từng event

Mỗi event có thể dùng kênh khác nhau:

- Event hành động: semantic embedding.
- Event có chữ: OCR.
- Event có lời nói: ASR.
- Event vật thể cụ thể: object search.
- Event scene: CLIP + metadata.
- Event chuyển cảnh: temporal neighborhood.

## 11.4. Video-level selection

Tính điểm video dựa trên:

- Có candidate cho bao nhiêu event.
- Score từng event.
- Đúng thứ tự hay không.
- Khoảng cách thời gian hợp lý.
- Có cùng context hay không.
- Có missing event hay không.

Sai video làm toàn bài nhận 0, vì vậy video selection phải được ưu tiên hơn tối ưu từng frame.

## 11.5. Temporal alignment

Với mỗi video candidate:

1. Lấy danh sách frame candidate cho từng event.
2. Sắp xếp theo frame.
3. Tìm chuỗi `f1 < f2 < ... < fN`.
4. Tối ưu tổng score.
5. Phạt khoảng cách bất hợp lý.
6. Phạt event bỏ sót.
7. Giữ nhiều chuỗi tốt nhất.
8. Cho người dùng xem timeline.

Có thể dùng tư duy dynamic programming hoặc beam search ở giai đoạn triển khai, nhưng tài liệu này không yêu cầu viết code.

## 11.6. Semantic keyframe refinement

Keyframe BTC có thể chưa nằm chính xác trong cửa sổ ground truth rất hẹp. Sau khi chọn đúng video và vùng thời gian:

- Mở video gốc.
- Duyệt frame dày quanh candidate.
- Xác định khoảnh khắc chuyển trạng thái.
- Chọn frame đầu tiên hoặc frame cực trị theo định nghĩa event.
- Giữ frame ID gốc.
- Cho người dùng xác nhận.

Đây là lý do TRAKE cần video nhiều hơn Textual KIS.

---

# 12. THIẾT KẾ GIAO DIỆN

## 12.1. Màn hình truy vấn

Thành phần:

- Ô nhập query.
- Chọn query type.
- Query parser output.
- Query variants.
- Modality toggles.
- Weight controls.
- Filter.
- Nút search.
- Nút reset.
- Lịch sử truy vấn.

## 12.2. Result grid

Mỗi card cần:

- Thumbnail.
- Video ID.
- Frame ID.
- Rank.
- Final score.
- Dense score.
- Object/OCR/ASR evidence.
- Batch.
- Nút mở timeline.
- Nút search similar.
- Relevant/irrelevant feedback.
- Pin.

## 12.3. Video timeline

Cần:

- Keyframe strip.
- Current frame.
- Jump theo frame ID.
- Previous/next.
- Neighborhood.
- Scene boundaries nếu có.
- OCR/ASR overlay.
- Event marker.
- TRAKE sequence marker.

## 12.4. TRAKE workspace

Hiển thị:

- Danh sách event.
- Candidate frame của từng event.
- Video-level score.
- Timeline chung.
- Cảnh báo sai thứ tự.
- Cảnh báo missing event.
- Nút chọn frame cuối.

## 12.5. Q&A workspace

Hiển thị:

- Candidate video.
- Evidence frames.
- Transcript.
- OCR.
- Proposed answer.
- Alternative answer.
- Ô sửa thủ công.
- Answer normalization preview.

## 12.6. Submission panel

Cần:

- Danh sách result hiện tại.
- Reorder.
- Remove.
- Duplicate warning.
- Format preview.
- Export.
- Validation.
- Log version.

---

# 13. CHIẾN LƯỢC LƯU TRỮ VÀ TÀI NGUYÊN

## 13.1. Phân tầng dữ liệu

### Hot data

Luôn local:

- CLIP feature.
- Mapping.
- FAISS index.
- Metadata.
- Objects.
- Registry.
- Thumbnail hoặc keyframe cache.

### Warm data

Nên local nếu đủ dung lượng:

- Toàn bộ keyframe ZIP.
- OCR index.
- ASR transcript.
- Additional feature.

### Cold data

Có thể đặt trên SSD ngoài hoặc server:

- Video gốc.
- Archive lớn.
- Model weight ít dùng.
- Intermediate processing output.

## 13.2. Giữ keyframe trong ZIP

Không cần giải nén toàn bộ keyframe nếu UI có thể đọc trực tiếp trong ZIP và cache ảnh đang xem.

Lợi ích:

- Tránh giữ cả ZIP và thư mục giải nén.
- Giảm dung lượng.
- Dễ quản lý batch.

Rủi ro:

- Random access ZIP chậm hơn.
- Cần cache.
- Archive hỏng ảnh hưởng cả nhóm.

## 13.3. Video

Trong giai đoạn phát triển:

- Không bắt buộc tải toàn bộ.
- Chỉ tải một số nhóm để phát triển Q&A và TRAKE.

Trước khi thi:

- Nên có coverage video tối đa nếu tài nguyên cho phép.
- Ưu tiên SSD ngoài 1–2 TB hoặc máy có ổ lớn.
- Không phụ thuộc vào tải video trong lúc thi nếu chưa biết Internet policy.

## 13.4. Chế độ tài nguyên

### Minimal mode

- CLIP feature.
- Mapping.
- Metadata.
- Objects.
- Một nhóm keyframe.
- FAISS.
- UI cơ bản.

### Standard mode

- Toàn bộ keyframe.
- Toàn bộ CLIP.
- Metadata/object.
- OCR.
- Một phần video.
- Reranker nhẹ.

### Full competition mode

- Toàn bộ keyframe.
- Toàn bộ video.
- OCR.
- ASR.
- Additional embedding.
- Reranker/LVLM.
- Temporal refinement.
- Backup storage.

---

# 14. ĐÁNH GIÁ NỘI BỘ

## 14.1. Vấn đề ground truth

BTC chưa cung cấp đầy đủ ground truth cho AIC 2026. Cần xây benchmark nội bộ từ:

- Task năm cũ.
- VBS Archive.
- LSC Archive.
- Truy vấn tự gán.
- Một tập video nhỏ do nhóm kiểm tra thủ công.
- Public examples từ BTC.

## 14.2. Bộ metric

### Retrieval

- Recall@1.
- Recall@5.
- Recall@20.
- Recall@50.
- Recall@100.
- MRR.
- Video Recall.
- Frame Recall.
- nDCG nếu có relevance graded.

### Ranking

- Tỷ lệ đúng video Top-1.
- Tỷ lệ đúng video Top-5.
- Tỷ lệ đúng scene.
- Diversity theo video.
- Duplicate rate.

### Q&A

- Answer exact match.
- Normalized exact match.
- Semantic match thủ công.
- Evidence accuracy.
- Video accuracy.
- Frame accuracy.

### TRAKE

- Video accuracy.
- Event frame accuracy.
- Sequence accuracy.
- Mean event score.
- Full-chain accuracy.
- Temporal order violation rate.

### Hệ thống

- Query latency.
- UI interaction latency.
- Index load time.
- Memory usage.
- Cache hit rate.
- Failure rate.
- Time-to-first-relevant-result.

## 14.3. Ablation

Mỗi module mới phải chứng minh giá trị:

- CLIP only.
- CLIP + object.
- CLIP + metadata.
- CLIP + OCR.
- CLIP + query expansion.
- CLIP + reranker.
- Fusion method A/B.
- Grouping on/off.
- Diversity on/off.
- LLM planner on/off.

Không thêm module chỉ vì “có vẻ mạnh”.

---

# 15. LỘ TRÌNH TRIỂN KHAI

## Phase 0 – Repository và data audit

Mục tiêu:

- Hiểu repository.
- Hiểu dữ liệu.
- Không viết feature.

Đầu ra:

- Repository inventory.
- Data inventory.
- Missing files.
- Dependency map.
- Risk list.
- Decision list.

Tiêu chí hoàn thành:

- Biết chính xác file nào có.
- Biết mapping format.
- Biết vector shape.
- Biết một keyframe ánh xạ tới frame nào.
- Biết cấu hình máy.

## Phase 1 – Data registry và baseline retrieval

Mục tiêu:

- Xây baseline CLIP search.

Đầu ra:

- Registry.
- Mapping validated.
- FAISS index.
- Text query.
- Top-K keyframe result.
- CLI hoặc test harness tối thiểu.

Tiêu chí:

- Query trả kết quả.
- Không sai mapping.
- Có thể mở đúng ảnh.
- Có benchmark latency.

## Phase 2 – UI retrieval cơ bản

Mục tiêu:

- Cho người dùng tìm và xem kết quả.

Đầu ra:

- Search box.
- Result grid.
- Video ID/frame ID.
- Paging.
- Search history.
- Open neighborhood.
- Pin result.

Tiêu chí:

- Một người mới có thể tìm query test.
- UI phản hồi nhanh.
- Không tải toàn bộ ảnh cùng lúc.

## Phase 3 – Video aggregation và diversity

Mục tiêu:

- Tối ưu ranking theo video.

Đầu ra:

- Grouping.
- Max frames per video.
- Video-level score.
- Diversity.
- Timeline.

Tiêu chí:

- Top result không bị chiếm bởi một scene.
- Có thể mở sâu trong video.

## Phase 4 – Structured retrieval

Mục tiêu:

- Kết hợp object và metadata.

Đầu ra:

- Object filter.
- Position filter.
- Count filter.
- Metadata search.
- Hybrid fusion.
- Evidence display.

Tiêu chí:

- Query object-specific tốt hơn CLIP-only trên benchmark.
- Có ablation report.

## Phase 5 – OCR và ASR

Mục tiêu:

- Tìm được text và speech.

Đầu ra:

- OCR pipeline/index.
- ASR pipeline/index.
- Search integration.
- Timeline evidence.

Tiêu chí:

- OCR query chính xác trên test set.
- ASR mapping đúng timestamp/frame.
- Module có thể tắt.

## Phase 6 – Reranking và query planner

Mục tiêu:

- Cải thiện Top-1/Top-5.

Đầu ra:

- Query decomposition.
- Query expansion.
- Fusion tuning.
- Reranker.
- Explanation.

Tiêu chí:

- Top-1 hoặc MRR tăng trên benchmark.
- Latency vẫn trong giới hạn.
- Có fallback không LLM.

## Phase 7 – Q&A

Mục tiêu:

- Trả lời trên đoạn video đã retrieve.

Đầu ra:

- Q&A workspace.
- Evidence selection.
- Answer generation.
- Normalization.
- Human confirmation.

Tiêu chí:

- Không trả lời khi retrieval confidence quá thấp.
- Evidence luôn đi kèm answer.
- Có logging.

## Phase 8 – TRAKE

Mục tiêu:

- Tìm chuỗi event.

Đầu ra:

- Event decomposition.
- Event-specific retrieval.
- Video-level scoring.
- Temporal sequence search.
- Timeline workspace.
- Frame refinement.

Tiêu chí:

- Đúng thứ tự.
- Không trộn video.
- Có nhiều chain candidate.
- Có manual correction.

## Phase 9 – Submission và hardening

Mục tiêu:

- Sẵn sàng vòng sơ tuyển.

Đầu ra:

- Submission adapter.
- Validation.
- Reproducible setup.
- Offline mode.
- Backup.
- Runbook.
- Report draft.

Tiêu chí:

- Xuất đúng format khi BTC công bố.
- Có test end-to-end.
- Không phụ thuộc đường dẫn máy cá nhân.
- Có recovery plan.

---

# 16. TIÊU CHÍ NGHIỆM THU CẤP HỆ THỐNG

Hệ thống chỉ được xem là sẵn sàng khi:

- Mapping vector → keyframe → video → frame được xác minh.
- Search baseline chạy ổn định.
- Top-100 có diversity.
- UI có timeline và frame neighborhood.
- Có export và validation.
- Có benchmark nội bộ.
- Có ablation.
- Có offline fallback.
- Có cấu hình resource mode.
- Có log mọi truy vấn và ranking.
- Có thể tái tạo index.
- Có thể thêm Batch 2 mà không sửa core logic.
- Có thể thay submission format bằng adapter.
- Có thể tắt LLM/LVLM/API.
- Có hướng dẫn vận hành trong ngày thi.
- Có backup dữ liệu và index.

---

# 17. QUẢN LÝ RỦI RO

## R1 – Batch 2 khác cấu trúc

Giảm thiểu:

- Adapter.
- Registry.
- Không hard-code.
- Data validation.

## R2 – Không có Internet

Giảm thiểu:

- Local text encoder.
- Local query parser fallback.
- Local reranker.
- Không phụ thuộc API.

## R3 – Không đủ dung lượng

Giảm thiểu:

- ZIP access.
- SSD ngoài.
- Tiered storage.
- Selective video.
- Cache.

## R4 – Keyframe không đủ chính xác cho TRAKE

Giảm thiểu:

- Video refinement.
- Dense frame extraction quanh candidate.
- Giữ frame mapping gốc.

## R5 – Query expansion làm lệch nghĩa

Giảm thiểu:

- Hiển thị query variants.
- Weight thấp.
- Người dùng tắt variant.
- Log contribution.

## R6 – Model reranker chậm

Giảm thiểu:

- Chỉ chạy Top-N.
- Batch inference.
- Cache.
- Model nhẹ fallback.

## R7 – Object labels không đủ

Giảm thiểu:

- Open-vocabulary detector trên candidate.
- Text embedding.
- Manual filtering.

## R8 – Q&A semantic normalization không rõ

Giảm thiểu:

- Lưu nhiều answer form.
- Cho sửa thủ công.
- Adapter theo luật BTC.

## R9 – Submission format thay đổi

Giảm thiểu:

- Tách exporter.
- Schema validation.
- Không gắn format vào domain model.

## R10 – UI tốt nhưng retrieval yếu

Giảm thiểu:

- Benchmark core.
- Ablation.
- Không đánh giá qua demo cảm tính.

## R11 – Retrieval tốt nhưng UI thao tác chậm

Giảm thiểu:

- User testing.
- Keyboard shortcuts.
- Lazy loading.
- Cache.
- Minimal clicks.

---

# 18. QUY TẮC LÀM VIỆC CỦA CODEX

Codex phải:

- Đọc file này trước mỗi thay đổi lớn.
- Kiểm tra repository hiện tại trước khi đề xuất.
- Không giả định file path.
- Không tự tải dữ liệu.
- Không tự chọn model lớn.
- Không xóa dữ liệu.
- Không đổi format mapping.
- Không re-encode video.
- Không viết code trước khi được phê duyệt phase.
- Mỗi thay đổi phải gắn với một requirement.
- Mỗi module phải có tiêu chí test.
- Mỗi dependency mới phải có lý do.
- Mỗi index phải có version.
- Mỗi model phải có model card nội bộ ngắn.
- Mỗi benchmark phải có dữ liệu và metric rõ.
- Mỗi giả định phải ghi trong Assumptions.
- Mỗi quyết định phải ghi trong Decision Log.
- Mỗi rủi ro mới phải cập nhật Risk Register.

Codex không được:

- Dùng Agent để che giấu retrieval yếu.
- Dùng LVLM quét toàn dataset online.
- Đánh đồng keyframe index với frame ID.
- Trộn score giữa modality mà không normalize hoặc rank-fusion.
- Tạo nhiều service không cần thiết.
- Tối ưu sớm.
- Thêm framework chỉ vì phổ biến.
- Viết một pipeline phụ thuộc GPU mạnh nếu máy không có.
- Tuyên bố performance khi chưa benchmark.
- Tuyên bố nội dung BTC chưa xác nhận.

---

# 19. CÁC QUYẾT ĐỊNH CẦN NGƯỜI DÙNG PHÊ DUYỆT

Trước Phase 1:

- Ngôn ngữ backend.
- Framework backend.
- Frontend framework.
- FAISS index type.
- SQLite hay DuckDB.
- Cách đọc keyframe ZIP.
- Vị trí lưu index.
- Máy chạy chính.
- Có SSD ngoài hay không.

Trước Phase 5:

- OCR model.
- ASR model.
- Phạm vi chạy OCR/ASR.
- Storage output.

Trước Phase 6:

- LLM local hay API.
- Reranker.
- Additional embedding.
- Latency budget.

Trước Phase 7:

- LVLM.
- Answer normalization.
- Human confirmation flow.

Trước Phase 8:

- Cách refine frame từ video.
- Số video candidate.
- Temporal search strategy.

Trước Phase 9:

- Deployment.
- Offline/online mode.
- Backup.
- Submission schema.

---

# 20. TÀI LIỆU NGUỒN

## Nguồn chính thức BTC

- “Thông tin vòng Sơ tuyển AIC2026.pdf”.
- “Tập huấn AIC 2026 – Buổi 1”.
- “Tập huấn AIC 2026 – Buổi 2”.
- “Tập huấn AIC 2026 – Buổi 3”.
- Google Sheet “[AIC 2026] TỔNG HỢP THẮC MẮC THÍ SINH”.
- Danh sách tải Batch 1 do BTC cung cấp.

## Nguồn tham khảo kiến trúc

- Các công trình AIC/SOICT về video retrieval.
- Lifelog Search Challenge.
- Video Browser Showdown.
- VISIONE.
- vitrivr, Cineast và vitrivr-ng.
- Các archive task/result công khai của LSC và VBS.

Các nguồn tham khảo kiến trúc chỉ dùng để học tư duy. Không được giả định chúng khớp hoàn toàn với luật AIC 2026.

---

# 21. CHECKLIST HÀNH ĐỘNG ĐẦU TIÊN CHO CODEX

Codex phải trả lời bằng một báo cáo planning, không viết code, gồm:

1. Tóm tắt lại bài toán bằng lời của Codex.
2. Liệt kê dữ kiện chính thức.
3. Liệt kê giả định.
4. Liệt kê điểm chưa rõ.
5. Kiểm kê repository.
6. Kiểm kê dữ liệu.
7. Vẽ module dependency.
8. Đề xuất MVP.
9. Đề xuất milestone.
10. Đề xuất metric.
11. Đề xuất test set.
12. Đề xuất resource mode.
13. Liệt kê decision cần duyệt.
14. Liệt kê rủi ro.
15. Dừng và chờ phê duyệt.

Codex không được triển khai sau báo cáo này cho đến khi người dùng xác nhận phase tiếp theo.

---

# 22. ĐỊNH NGHĨA MVP

MVP không phải hệ thống Q&A hoặc Agent.

MVP tối thiểu gồm:

- Đọc CLIP feature BTC.
- Đọc mapping.
- Xây FAISS index.
- Encode text query bằng đúng text encoder tương thích.
- Trả Top-K keyframe.
- Ánh xạ đúng về video/frame.
- Mở đúng ảnh.
- Hiển thị result grid.
- Group theo video.
- Xem frame lân cận.
- Ghim kết quả.
- Export danh sách logic.

MVP hoàn thành khi một người có thể nhập mô tả, nhìn kết quả và xác nhận một `video_id`, `frame_id` mà không cần thao tác trực tiếp với file hệ thống.

---

# 23. ĐỊNH HƯỚNG CUỐI CÙNG

Kiến trúc mục tiêu không phải:

> CLIP + chatbot.

Kiến trúc mục tiêu là:

> Nhiều chỉ mục chuyên biệt + candidate generation + fusion + video aggregation + reranking + temporal reasoning + giao diện tương tác nhanh.

Thứ tự ưu tiên:

1. Mapping đúng.
2. Retrieval nhanh.
3. Recall đủ.
4. Ranking tốt.
5. UI hiệu quả.
6. Structured search.
7. OCR/ASR.
8. Reranking.
9. Q&A.
10. TRAKE.
11. Agent.

Bất kỳ đề xuất nào đảo ngược thứ tự này phải có bằng chứng rõ ràng.

