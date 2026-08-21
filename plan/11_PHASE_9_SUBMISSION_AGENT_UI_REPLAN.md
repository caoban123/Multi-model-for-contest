# Phase 9 Replan - AIC Retrieval & Submission Agent UI

> UI redesign continuation: `plan/12_PHASE_9_AGENT_WORKSPACE_UI_REDESIGN.md`.
> Kế hoạch mới giữ nguyên retrieval/submission architecture trong file này, đồng thời bổ sung Agent result parity với trang `/`, Gemini raw/parsed/validated trace, responsive workspace và rollout theo feature flag.

## Mục tiêu

Nâng Phase 9 từ mức "internal submission hardening" thành một lớp **Submission Agent UI** dùng được trong ngày thi:

```text
Start session
-> nhập query
-> chọn task KIS / Q&A / TRAKE
-> chạy pipeline phù hợp
-> trả structured JSON
-> review / sửa / chọn nhiều prediction
-> confirm vào submission queue
-> done
-> validate toàn bộ
-> ghi submission/*.csv
-> zip submission/
-> tải submission.zip
```

Nguyên tắc quan trọng nhất: **retrieval/debug output và official BTC submission là hai lớp khác nhau**. UI benchmark, pin export, Q&A export nội bộ, TRAKE review export không được dùng trực tiếp để nộp BTC.

## Trạng thái hiện tại của repo

- KIS/visual retrieval đã có UI chính, CLIP NumPy index, structured retrieval nhẹ, pin/review/export debug.
- Q&A đã có workspace, evidence selection, answer review, Gemini option, SQLite store và export nội bộ.
- TRAKE Phase 8 vừa merge, đã có `/trake`, planner, search, align, review/export nội bộ.
- Phase 9 hiện có internal schema `aic-internal-submission-v1`, validator cơ bản và Q&A export adapter.
- Chưa có UI thống nhất để gom nhiều query thành một `submission.zip` đúng format BTC.
- Điểm rủi ro lớn nhất vẫn là mapping giữa `keyframe_id`, `frame_idx`, `timestamp` và **official frame_id**.

## BTC format cần hỗ trợ

Mỗi query tạo một file CSV riêng, tối đa 100 dòng, không header.

### KIS

```text
<video_id>,<frame_id>
```

Ví dụ:

```text
L21_V001,1234
```

### Q&A

```text
<video_id>,<frame_id>,<answer>
```

Ràng buộc:

- `answer` tối đa 100 ký tự.
- Có thể dùng tiếng Việt hoặc tiếng Anh.
- Nếu answer có dấu phẩy, dấu nháy kép hoặc xuống dòng thì phải để CSV writer chuẩn xử lý quoting.

### TRAKE

```text
<video_id>,<frame_event_1>,<frame_event_2>,...,<frame_event_N>
```

Ràng buộc:

- Tất cả frame trong một dòng phải thuộc cùng một video.
- Số frame phải đúng bằng số event.
- Thứ tự frame phải tuân theo thứ tự thời gian của event.
- Không thừa, không thiếu frame.

### ZIP

ZIP cuối cùng phải chứa thư mục `submission/`:

```text
submission.zip
└── submission/
    ├── query-1-kis.csv
    ├── query-2-kis.csv
    ├── query-3-qa.csv
    └── query-4-trake.csv
```

Không nén trực tiếp CSV ở root của ZIP.

## Kiến trúc đề xuất

Không viết lại retrieval. Thêm một lớp orchestration phía trên:

```text
User
  |
  v
Submission Session UI
  |
  v
Task Router
  |-------------|-------------|
  v             v             v
KIS Agent      Q&A Agent      TRAKE Agent
  |             |             |
  v             v             v
Structured JSON Result
  |
  v
Review/Edit UI
  |
  v
Official Prediction Builder
  |
  v
Submission Validator
  |
  v
Submission Queue
  |
  v
CSV/ZIP Exporter
```

Các module nên tách:

```text
src/aic_retrieval/submission_agent.py
src/aic_retrieval/submission_official.py
src/aic_retrieval/submission_session.py
src/aic_retrieval/submission_validator.py
web/submission_ui/
```

Tên file có thể đổi khi triển khai để khớp style repo.

## Session workflow

UI có nút:

```text
Start
```

Khi bấm Start, tạo một session:

```json
{
  "session_id": "sub-...",
  "status": "ACTIVE",
  "created_at": "...",
  "queries": [],
  "submission_queue": []
}
```

Trong session active, user xử lý nhiều query liên tiếp. Khi xong bấm:

```text
Done
```

Lúc đó hệ thống mới freeze session, validate toàn bộ và tạo ZIP.

## Query input UI

Form tối thiểu:

```text
Query ID:
[query-1-kis]

Query:
[...]

Task:
( ) KIS
( ) Q&A
( ) TRAKE

[Run Agent]
```

Task do user chọn là nguồn sự thật chính. Agent có thể cảnh báo query có vẻ không khớp task, nhưng không tự đổi task.

Với Q&A, P0/P2 dùng một ô query chung để agent tự phân tích:

```text
Q&A query:
[...]
```

Sau khi bấm Run Agent, hệ thống phải tách và hiển thị rõ:

```text
Detected event query:
[...]

Detected question:
[...]
```

User được sửa lại `event_query` và `question` trước khi chạy lại retrieval/answer nếu agent tách sai. Agent không được giấu bước tách này vì Q&A phụ thuộc mạnh vào việc event query dùng để tìm video, còn question dùng để sinh answer.

## Agent query analysis

Trước khi retrieval, Agent Router cần tạo một bản phân tích query có cấu trúc để UI hiển thị và audit:

```json
{
  "query_id": "query-3-qa",
  "task": "QA",
  "original_query": "...",
  "analysis": {
    "detected_task": "QA",
    "user_selected_task": "QA",
    "task_mismatch_warning": null,
    "language": "vi",
    "event_query": "người dẫn chương trình mặc áo đỏ",
    "question": "người đó đang làm gì",
    "clip_query": "a news presenter wearing a red shirt",
    "modalities": ["clip", "asr", "metadata"],
    "warnings": []
  }
}
```

Theo task:

- KIS: phân tích query chính, query rewrite tiếng Anh nếu có Gemini, modality gợi ý.
- Q&A: tách `event_query` và `question`, rewrite `event_query` thành CLIP-ready query, giữ `question` theo ngôn ngữ user.
- TRAKE: tách events, số event, thứ tự event, constraint thời gian nếu có.

Phân tích query là để user kiểm tra và sửa, không phải official output.

## Language routing for CLIP vs Vietnamese evidence

Hệ thống phải tách rõ hai loại query:

```text
visual_clip_query_en
```

và:

```text
text_trace_query_vi
```

Lý do:

- CLIP hiện tại hiểu tiếng Anh tốt hơn, nên query đưa vào CLIP phải được agent dịch/rewrite sang tiếng Anh, ngắn, cụ thể, giàu visual noun/action.
- ASR/OCR/metadata trong dữ liệu có thể là tiếng Việt, nên truy vết text evidence phải giữ tiếng Việt để match đúng nội dung gốc.

Ví dụ với query:

```text
Đoạn video về một chương trình từ thiện của câu lạc bộ FANA. Hỏi xã này tên là gì?
```

Agent analysis nên tạo:

```json
{
  "event_query_vi": "chương trình từ thiện của câu lạc bộ FANA đi trao quà tại một xã thuộc tỉnh Khánh Hòa",
  "visual_clip_query_en": "charity club members giving gifts at a local community event in Khanh Hoa",
  "text_trace_query_vi": "FANA trao quà xã Khánh Hòa tên xã",
  "question_vi": "xã này có tên là gì",
  "modalities": ["clip", "asr", "ocr", "metadata"]
}
```

Routing:

- `visual_clip_query_en` dùng cho CLIP/frame retrieval.
- `event_query_vi` dùng để hiển thị và audit.
- `text_trace_query_vi` dùng cho ASR/OCR/metadata/BM25/BGE tiếng Việt.
- `question_vi` dùng cho Q&A answer generation và final review.

Không được lấy bản dịch tiếng Anh rồi search ASR/OCR tiếng Việt một cách mặc định, vì sẽ mất các từ khóa quan trọng như tên riêng, địa danh, câu thơ, món ăn, bảng chữ, lời thoại.

Với KIS:

- CLIP dùng English visual query.
- Nếu query có tên riêng/địa danh/chữ trên màn hình, giữ thêm Vietnamese trace query để search metadata/OCR/ASR.

Với Q&A:

- Event retrieval dùng cả English CLIP query và Vietnamese ASR/OCR trace query.
- Answer extraction ưu tiên evidence tiếng Việt gốc nếu câu hỏi hỏi tên riêng, câu thơ, tiêu đề, địa danh hoặc nội dung lời nói.

Với TRAKE:

- Mỗi event có English visual query riêng cho CLIP.
- Mỗi event vẫn giữ Vietnamese trace query nếu event liên quan chữ/lời nói/tên riêng.

## Phase 5 OCR/ASR status and required behavior

Tình trạng kiểm tra hiện tại:

```text
artifacts/phase5/phase5_store.sqlite
OCR records: 0
OCR videos: 0
ASR segments: 563
ASR videos: 2
```

JSONL hiện tại:

```text
artifacts/phase5/ocr_l21.jsonl
  - có 569 frame records
  - nhưng 0 detections
  - nên OCR chưa usable trong search

artifacts/phase5/asr_l21.jsonl
  - có 2 transcript records
  - 563 speech segments
  - mới phủ 2 video
```

Ý nghĩa:

- ASR pipeline đã có và search được bằng SQLite FTS5, nhưng coverage quá ít cho thi thật.
- OCR pipeline/schema/store đã có, nhưng dữ liệu hiện rỗng nên các query cần chữ trên ảnh, bảng hiệu, câu thơ, tiêu đề, công thức nấu ăn chưa được hỗ trợ tốt.
- Q&A thật nhiều khả năng phụ thuộc mạnh vào OCR/ASR, nên Phase 9 UI phải hiển thị rõ khi OCR/ASR unavailable hoặc coverage thấp.

### ASR/OCR hoạt động như thế nào

ASR:

```text
video/audio
-> ASR transcript JSONL
-> build SQLite FTS5 store
-> map segment theo start_time/end_time về keyframe gần nhất
-> search_asr(text_trace_query_vi)
-> trả về video_id, keyframe_id, frame_idx, pts_time, matched_text
```

OCR:

```text
keyframe image
-> OCR engine detect text
-> OCR JSONL
-> build SQLite FTS5 store
-> search_ocr(text_trace_query_vi)
-> trả về video_id, keyframe_id, frame_idx, text_raw, bbox, confidence
```

### Yêu cầu khi triển khai Submission Agent

Không được dùng cùng một query text cho mọi kênh.

Hiện structured retrieval cũ có chỗ dùng `query.visual_text` cho OCR/ASR. Phase 9 agent phải sửa hướng đi này:

```text
CLIP/search_numpy_index:
  dùng visual_clip_query_en

ASR/search_asr:
  dùng text_trace_query_vi

OCR/search_ocr:
  dùng text_trace_query_vi

Metadata/BM25/BGE tiếng Việt:
  dùng text_trace_query_vi
```

Ví dụ Q&A:

```json
{
  "event_query_vi": "chương trình từ thiện của câu lạc bộ FANA đi trao quà tại một xã thuộc tỉnh Khánh Hòa",
  "visual_clip_query_en": "charity club members giving gifts at a local community event in Khanh Hoa",
  "text_trace_query_vi": "FANA trao quà xã Khánh Hòa tên xã",
  "question_vi": "xã này có tên là gì"
}
```

Routing đúng:

- `visual_clip_query_en` tìm candidate frame/video bằng CLIP.
- `text_trace_query_vi` truy FTS/BM25/BGE trên ASR/OCR/metadata.
- `question_vi` dùng để Q&A answer/review.

### UI cần hiển thị

Trong Submission Agent UI, mỗi query nên có capability panel:

```text
CLIP: available
ASR: available, 563 segments / 2 videos
OCR: unavailable or empty, 0 detections
Metadata: available
Object/Attribute: available or unavailable
```

Nếu query cần chữ/lời nói mà OCR/ASR thiếu dữ liệu, UI phải warning:

```text
OCR evidence unavailable for this dataset. Candidate may rely mostly on CLIP/metadata.
ASR coverage is partial. Text/audio answer should be manually verified.
```

### Việc cần làm trước thi thật

- Rebuild ASR cho toàn bộ video đã tải, không chỉ 2 video.
- Rebuild OCR bằng engine thật cho toàn bộ keyframes/video frames quan trọng.
- Build lại Phase 5 SQLite store sau khi có JSONL mới.
- Kiểm lại search tiếng Việt có dấu/không dấu.
- Sau này khi có BM25/BGE, dùng `text_trace_query_vi` làm input chính cho text retrieval.

## Hybrid Retrieval layer - CLIP + BGE + BM25

Phase 9 bổ sung một lớp retrieval mới theo hướng modular và opt-in. Lớp này không thay thế đường chạy CLIP hiện tại cho đến khi benchmark/ablation chứng minh có cải thiện và không gây regression.

```text
Agent query analysis
  |-- visual_clip_query_en ------> CLIPRetriever ------> visual hits
  |-- semantic_text_query_vi ----> BGERetriever -------> semantic text hits
  `-- text_trace_query_vi -------> BM25Retriever ------> lexical text hits
                                               |
                                               v
                                      Candidate normalizer
                                               |
                                               v
                                      RRF fusion (opt-in)
                                               |
                                               v
                                      video/frame candidates
```

### Nguyên tắc bắt buộc

- CLIP NumPy retrieval hiện tại tiếp tục là baseline mặc định và có đường chạy độc lập.
- BGE và BM25 được bật bằng config/feature flag; lỗi ở retriever mới phải fail-open về baseline CLIP, đồng thời ghi warning/provenance.
- Mỗi retriever có interface, index manifest, validator, test và benchmark riêng.
- Fusion thử RRF trước vì rank giữa CLIP, dense text và BM25 không cùng thang điểm. Weighted scoring chỉ được thử sau khi có score calibration và benchmark riêng.
- Index L21 trước. Dữ liệu nhóm mới phải đi qua cùng audit/manifest/build/validate contract, không hard-code `L21` trong retriever core.
- Retrieval output là candidate phục vụ review; official CSV/ZIP vẫn do deterministic submission builder tạo.
- Không tải model lớn, không cài FAISS hoặc thay dependency chính nếu chưa ghi decision/report và được người dùng chấp thuận theo protocol dự án.

### Retriever interfaces

Contract logic đề xuất:

```text
Retriever
  - name
  - index_version
  - capabilities
  - health()
  - search(RetrievalRequest) -> list[RetrievalHit]
```

`RetrievalRequest` tối thiểu gồm:

```json
{
  "query_id": "query-1-kis",
  "query_text": "...",
  "groups": ["L21"],
  "top_k": 100,
  "filters": {},
  "trace_id": "..."
}
```

`RetrievalHit` phải chuẩn hóa đủ để fusion và audit:

```json
{
  "retriever": "clip|bge|bm25",
  "rank": 1,
  "raw_score": 0.0,
  "video_id": "L21_V001",
  "keyframe_id": 15,
  "frame_idx": 1234,
  "pts_time": 41.1,
  "document_id": "...",
  "source_type": "keyframe|asr|ocr|metadata|object",
  "matched_text": "...",
  "provenance": {}
}
```

Không bắt buộc mọi text hit có `keyframe_id`. Hit cấp video/segment phải được resolver map về frame đại diện hoặc frame gần timestamp nhất; nếu mapping chưa chắc chắn thì giữ `null` và không tự bịa frame.

### CLIPRetriever

- Bọc đường chạy CLIP NumPy hiện có qua interface chung; không viết lại encoder/search trong milestone đầu.
- Input chính là `visual_clip_query_en`.
- Output giữ score cosine, rank, frame/video reference và đường dẫn keyframe.
- Benchmark baseline phải được snapshot trước khi thêm BGE/BM25 để phát hiện regression.

### BGERetriever + VectorStore

BGE chỉ embed corpus text, không thay thế CLIP image embedding. Corpus L21 dự kiến gồm:

- ASR segment text.
- OCR detection/line text.
- Metadata như title, author/channel, publish date và mô tả có sẵn.
- Object labels/attributes đã normalize thành text có provenance.
- Caption hoặc text mô tả frame nếu sau này có dữ liệu thật.

Các trường rỗng không được tạo document giả. Document phải có `document_id` ổn định, checksum nội dung, source type và mapping về video/timestamp/frame nếu có.

Vector backend đi qua `VectorStore` interface. Phương án đầu tiên trong PLAN là:

```text
FAISS local index
+ SQLite sidecar cho document metadata/provenance
```

Model BGE phải cấu hình qua `AIC_BGE_MODEL_ID`/CLI và có chế độ local-files-only tương tự CLIP. Ứng viên ưu tiên cho tiếng Việt là một BGE multilingual model, nhưng model ID, kích thước embedding, RAM/VRAM, dung lượng index và dependency chỉ được chốt sau Data/Index audit và decision log; chưa tự tải model trong bước replan.

Artifact dự kiến:

```text
artifacts/indexes/l21_bge/
  manifest.json
  documents.sqlite3
  vectors.faiss
  build_report.json
```

Manifest phải khóa model ID/revision, embedding dimension, normalization, corpus checksum, document count, source coverage, build time và tool version.

### BM25Retriever

- Dùng SQLite FTS5/BM25 làm backend đầu tiên vì repo đã có `phase5_store.py` và không cần thêm dependency ngoài.
- Index chung không chỉ giới hạn OCR/ASR; nó phải hỗ trợ document schema thống nhất cho metadata, OCR, ASR, object labels/attributes và text bổ sung.
- Input chính là `text_trace_query_vi`, giữ tên riêng, địa danh, số, cụm OCR/ASR và từ khóa tiếng Việt.
- Phải kiểm thử query có dấu/không dấu. Nếu bổ sung normalization không dấu, lưu cả original text và normalized search text để evidence vẫn hiển thị đúng nguyên bản.

Artifact dự kiến:

```text
artifacts/indexes/l21_bm25/
  manifest.json
  documents.sqlite3
  build_report.json
```

### Candidate normalization và RRF fusion

RRF đầu tiên dùng rank thay vì trộn raw score:

```text
rrf_score(d) = sum(weight_r / (rrf_k + rank_r(d)))
```

Mặc định thử `rrf_k=60`. Giai đoạn đầu dùng weight bằng nhau cho CLIP/BGE/BM25 để tạo baseline dễ giải thích. Weight chỉ được thay đổi bằng config và phải đi qua benchmark; không hard-code theo một vài query thủ công.

Khóa hợp nhất ưu tiên:

1. Frame-level: `video_id + official/frame_idx/keyframe mapping` khi mapping chắc chắn.
2. Segment-level: `video_id + timestamp range`, sau đó resolver chọn frame gần nhất.
3. Video-level: `video_id`, sau đó chọn representative frame có evidence tốt nhất.

Mỗi result phải hiển thị provenance theo từng retriever, rank riêng, raw score riêng và đóng góp RRF. Không được chỉ trả một `hybrid_score` không giải thích được.

### Feature flags và fallback

Config logic đề xuất:

```json
{
  "retrieval": {
    "enable_clip": true,
    "enable_bge": false,
    "enable_bm25": false,
    "fusion_method": "rrf",
    "rrf_k": 60
  }
}
```

Các profile benchmark tối thiểu:

- `clip_baseline`
- `bge_only`
- `bm25_only`
- `clip_bge_rrf`
- `clip_bm25_rrf`
- `bge_bm25_rrf`
- `clip_bge_bm25_rrf`

Nếu BGE model/index hoặc BM25 store unavailable, UI/API phải báo channel unavailable và tiếp tục chạy các channel còn khỏe. Chỉ hard fail khi user yêu cầu profile bắt buộc đủ channel để benchmark tái lập.

### Benchmark và ablation gate

Benchmark phải version hóa query set, judgement và config. Mỗi retriever đo độc lập trước khi đo fusion.

Metric tối thiểu:

- Video Recall@1, Recall@5, Recall@10.
- Frame/segment Recall@K khi judgement có frame/timestamp.
- MRR và nDCG@K nếu có graded judgement.
- Query coverage và số query không trả kết quả.
- Latency p50/p95, index load time, index size, peak RAM tương đối.
- Breakdown theo visual-heavy, lexical/name/OCR/ASR-heavy, semantic-text-heavy và temporal query.

Promotion gate từ opt-in sang default:

- CLIP baseline vẫn phải tái lập được bằng cùng benchmark command.
- Full hybrid không được thấp hơn CLIP baseline ở metric chính trên toàn bộ reviewed benchmark.
- Phải cải thiện ít nhất một subset text-heavy có judgement thật.
- Không được làm một nhóm query quan trọng chuyển từ hit thành miss mà không có phân tích và quyết định chấp nhận regression.
- Latency/resource tăng phải được ghi trong report và phù hợp cấu hình ngày thi.
- Test, benchmark artifact, ablation table và rollback config phải đầy đủ.

Nếu chưa đạt toàn bộ gate, BGE/BM25 giữ `opt-in`; không đổi default UI/API.

### Agent routing sau khi hybrid ổn định

Agent analysis cần bổ sung trường semantic riêng thay vì ép một query cho mọi kênh:

```json
{
  "visual_clip_query_en": "a news report about land subsidence",
  "semantic_text_query_vi": "phóng sự về tình trạng sụt lún đất và ảnh hưởng đến người dân",
  "text_trace_query_vi": "sụt lún đất người dân",
  "retrieval_profile": "clip_bge_bm25_rrf",
  "requested_retrievers": ["clip", "bge", "bm25"]
}
```

- CLIP nhận câu tiếng Anh giàu tín hiệu hình ảnh.
- BGE nhận câu semantic đầy đủ theo ngôn ngữ corpus/model hỗ trợ.
- BM25 nhận cụm từ khóa tiếng Việt ngắn, giữ entity/số/từ trên màn hình.
- Agent chỉ lập kế hoạch và chọn profile; retriever code mới là thành phần thực thi search và trả candidates.
- UI phải hiển thị cả query rewrite, channel được bật/tắt, health, latency, result contribution và fallback warning.

### Chuỗi milestone phụ thuộc cho Hybrid Retrieval

#### H0 - Data/Index audit

- Kiểm kê L21: CLIP vectors/refs, metadata, ASR, OCR, object/attribute và mapping frame.
- Đo coverage, null/duplicate, checksum, dung lượng và khả năng map text document về video/frame.
- Snapshot benchmark CLIP hiện tại và ghi command/metric.
- Xuất audit artifact có machine-readable JSON và report snapshot.

Gate: audit tái lập được, không sửa dữ liệu nguồn, biết rõ corpus nào đủ điều kiện build.

#### H1 - Retriever contracts và corpus builder

- Tạo interface chung, request/hit schema, health/manifest validator.
- Tạo document corpus L21 deterministic từ nguồn đã audit.
- Unit test document ID, deduplication, provenance và frame/timestamp mapping.

Gate: build corpus hai lần cho cùng checksum/record count; CLIP baseline test vẫn pass.

#### H2 - BGE index

- Chốt model/dependency trong report sau khi người dùng duyệt nếu cần tải/cài lớn.
- Build BGE embeddings cho L21 và VectorStore adapter.
- Validate dimension, normalization, document count, persistence/reload và corrupted manifest behavior.

Gate: BGE-only search test/benchmark độc lập pass; index có manifest và không ảnh hưởng CLIP path.

#### H3 - BM25 index

- Build SQLite FTS5/BM25 corpus L21.
- Test tiếng Việt có dấu/không dấu, entity, số, phrase và empty query.
- Benchmark BM25-only độc lập.

Gate: BM25-only ổn định, evidence/provenance đúng và không thêm dependency không cần thiết.

#### H4 - Individual retriever search tests

- Chạy cùng benchmark contract cho CLIP-only, BGE-only và BM25-only.
- Ghi latency, coverage, quality và failure modes riêng.
- Sửa lỗi mapping/index trước khi fusion.

Gate: ba kênh trả schema thống nhất; baseline CLIP không regression.

#### H5 - Hybrid/RRF

- Candidate normalization/resolver.
- RRF profiles và deterministic tie-break.
- Provenance/contribution per channel.
- Fail-open về channel khỏe khi chạy interactive; strict mode cho benchmark.

Gate: unit/integration tests pass và kết quả fusion tái lập được.

#### H6 - Benchmark/ablation

- Chạy đầy đủ bảy profile ablation.
- So sánh toàn bộ metric và subset.
- Ghi regression, latency/resource cost và recommendation.
- Chỉ promote default nếu đạt promotion gate.

Gate: có benchmark artifact và report đủ bằng chứng; nếu không đạt thì giữ opt-in.

#### H7 - Agent và UI

- Agent sinh ba query route và profile có cấu trúc.
- Tích hợp retriever registry vào KIS/Q&A/TRAKE.
- UI cho bật/tắt profile, xem provenance, health, latency và fallback warning.
- Giữ nút/đường chạy CLIP baseline để rollback tức thì.

Gate: API/UI integration tests và manual smoke test trên L21 pass.

#### H8 - Submission workflow

- Candidate hybrid đi qua review/neighbor viewer/confirm như candidate hiện tại.
- Không thay đổi deterministic official CSV/ZIP contract.
- Test không leak debug score/provenance vào official output.

Gate: end-to-end query -> hybrid candidates -> user confirm -> validated CSV/ZIP pass.

### Trạng thái triển khai Hybrid Retrieval ngày 20-08-2026

| Milestone | Trạng thái | Bằng chứng chính |
|---|---|---|
| H0 | `VERIFIED` | `artifacts/audits/l21_hybrid_retrieval_audit.json`, CLIP baseline 20 query deterministic |
| H1 | `VERIFIED` | Corpus L21 8.322 document, checksum `a0d1e734...`, build lặp lại cùng checksum |
| H2 | `VERIFIED_OPT_IN` | `BAAI/bge-m3` revision cố định, FAISS 8.322 x 1.024, BGE smoke Recall@10 = 1.0 |
| H3 | `VERIFIED_OPT_IN` | SQLite FTS5 8.322 row, integrity `ok`, BM25 smoke MRR = 0.9375 |
| H4 | `VERIFIED` | CLIP/BGE/BM25 cùng `RetrievalRequest`/`RetrievalHit` contract và benchmark độc lập |
| H5 | `VERIFIED_OPT_IN` | Video-level RRF, deterministic tie-break, per-channel evidence/contribution, fail-open/strict tests |
| H6 | `VERIFIED_KEEP_OPT_IN` | 7-profile ablation tại `artifacts/benchmarks/hybrid/hybrid_reviewed_pool_ablation_v1.json` |
| H7 | `VERIFIED_OPT_IN` | Agent sinh route CLIP/BGE/BM25; Visual, Q&A và TRAKE UI/API có opt-in, provenance/health/fallback; full suite hiện 337 test pass và L21 server smoke pass |
| H8 | `VERIFIED_GUARDED` | SQLite queue, KIS Agent candidates, reviewed Q&A/TRAKE adapters, official CSV/ZIP validator và `/submission`; official frame ID bắt buộc manual/BTC-certified mapping |

Quyết định sau H6:

- Không promote full hybrid thành mặc định.
- CLIP tiếp tục là baseline mặc định cho visual query.
- Trên 23 query từ pool chấm tay, CLIP MRR = 0.8913; BGE = 0.5317; BM25 = 0.5156.
- Equal-weight `CLIP + BM25 RRF` có MRR = 0.9217 nhưng nDCG@10 giảm từ 0.8732 xuống 0.7098; chưa đạt promotion gate.
- Equal-weight `CLIP + BGE + BM25 RRF` giảm MRR còn 0.8623.
- Benchmark 23 query là partial judgments sinh từ CLIP-pooled review CSV; video ngoài pool chưa được chấm, vì vậy không dùng làm ground truth độc lập để promote default.
- H7 phải route theo intent: visual-heavy ưu tiên CLIP; exact keyword/date/OCR/ASR ưu tiên BM25; semantic text ưu tiên BGE; chỉ fusion các kênh có bằng chứng phù hợp.
- BGE/BM25 vẫn modular và opt-in; mọi fallback/warning phải hiện trong trace/UI.

### Full-data rollout status - 21-08-2026

| Stage | Status | Evidence |
|---|---|---|
| L21-L30 registry/data audit | `VERIFIED` | 873 videos and 177,321 aligned frames across L21-L30 |
| Full CLIP NumPy index | `VERIFIED` | 177,321 normalized 512-dimensional vectors on E |
| Full object SQLite | `VERIFIED` | 177,321 available frames, 17,732,100 detections, SQLite quick check `ok` |
| Full deterministic text corpus | `VERIFIED` | 173,393 unique documents; SHA-256, JSONL, mapping and source-count audit pass |
| Full BM25 index | `VERIFIED` | 173,393 SQLite/FTS rows; checksum, integrity, group filter and multilingual search smokes pass |
| Full BGE/FAISS index | `NEXT_RESUMABLE` | 512-document checkpoints, atomic publish and batched FAISS construction are verified |

The full-data rollout does not change the retrieval default. CLIP remains the baseline and the L21-L30 BM25/BGE channels stay opt-in until new benchmark and ablation evidence passes the promotion gate.

## Structured JSON contract

Agent chỉ trả JSON có thông tin phục vụ review. CSV official phải được sinh bởi module deterministic riêng.

### KIS structured result

```json
{
  "task": "KIS",
  "query_id": "query-1-kis",
  "query": "a man holding a phone",
  "status": "READY_FOR_REVIEW",
  "candidates": [
    {
      "rank": 1,
      "video_id": "L21_V001",
      "frame_id": 1234,
      "keyframe_id": 15,
      "pts_time": 41.1,
      "score": 0.82,
      "thumbnail_url": "/api/keyframe?...",
      "evidence": {
        "clip": 0.82,
        "object": [],
        "metadata": null
      }
    }
  ]
}
```

Official CSV chỉ dùng:

```text
video_id,frame_id
```

### Q&A structured result

```json
{
  "task": "QA",
  "query_id": "query-3-qa",
  "event_query": "news anchor in red shirt",
  "question": "What is the person doing?",
  "status": "READY_FOR_REVIEW",
  "candidates": [
    {
      "rank": 1,
      "video_id": "L21_V011",
      "frame_id": 1450,
      "answer": "Đang dẫn chương trình",
      "confidence": 0.72,
      "thumbnail_url": "/api/keyframe?...",
      "evidence": []
    }
  ]
}
```

Official CSV chỉ dùng:

```text
video_id,frame_id,answer
```

### TRAKE structured result

```json
{
  "task": "TRAKE",
  "query_id": "query-4-trake",
  "query": "event 1 then event 2 then event 3",
  "event_count": 3,
  "status": "READY_FOR_REVIEW",
  "events": [
    {"event_id": "e1", "description": "event 1"},
    {"event_id": "e2", "description": "event 2"},
    {"event_id": "e3", "description": "event 3"}
  ],
  "candidates": [
    {
      "rank": 1,
      "video_id": "L21_V001",
      "events": [
        {"event_id": "e1", "frame_id": 1200, "keyframe_id": 12, "score": 0.81},
        {"event_id": "e2", "frame_id": 1850, "keyframe_id": 19, "score": 0.79},
        {"event_id": "e3", "frame_id": 2100, "keyframe_id": 24, "score": 0.83}
      ]
    }
  ]
}
```

Official CSV chỉ dùng:

```text
video_id,frame_event_1,frame_event_2,...,frame_event_N
```

## Review state machine

Không đưa kết quả agent thẳng vào submission.

```text
GENERATED
-> READY_FOR_REVIEW
-> USER_CONFIRMED
-> QUEUED
-> EXPORTED
```

UI cần có:

- Accept candidate.
- Edit `video_id`.
- Edit `frame_id`.
- Edit Q&A `answer`.
- Chọn nhiều prediction, tối đa 100 dòng/query.
- Reorder prediction ranking.
- Remove prediction.
- Reject query result.
- Open keyframe/video/neighborhood để kiểm.

Sau khi user chọn candidate, UI có thêm option:

```text
[Analyze selected candidate]
```

Đây là bước optional để agent giải thích lựa chọn của người dùng trước khi confirm. Kết quả phân tích chỉ dùng để hỗ trợ review, không tự động thay đổi submission nếu user chưa đồng ý.

Ví dụ output:

```json
{
  "candidate_id": "query-3-qa:r1",
  "decision_support": {
    "summary": "Candidate này khớp tốt phần người dẫn chương trình mặc áo đỏ.",
    "strengths": [
      "Visual frame có người mặc áo đỏ.",
      "Metadata cùng nhóm chương trình tin tức.",
      "ASR/OCR có dấu hiệu liên quan nội dung chương trình nếu available."
    ],
    "risks": [
      "Answer vẫn cần user xác nhận vì Gemini/VLM có thể suy luận sai.",
      "Nếu frame_id mapping chưa chắc, cần kiểm official frame_id."
    ],
    "suggested_action": "CONFIRM_OR_CHECK_NEIGHBOR_FRAMES"
  }
}
```

Với Q&A, option này có thể gọi Gemini/VLM nếu user bật, gửi selected evidence + ảnh frame + OCR/ASR/object/metadata để phân tích. Với KIS/TRAKE, có thể dùng phân tích local trước; Gemini/VLM chỉ là option.

## Submission queue

Mỗi query đã confirm được đưa vào queue:

```json
{
  "query_id": "query-1-kis",
  "task": "KIS",
  "status": "READY",
  "output_file": "query-1-kis.csv",
  "prediction_count": 10
}
```

Queue UI cần cho:

- Mở lại query đã confirm.
- Sửa prediction.
- Remove khỏi queue.
- Replace bằng kết quả mới.
- Xem validation status.
- Xem preview CSV không header.

## Official CSV builder

Module official builder phải dùng `csv.writer`, không nối chuỗi thủ công.

KIS:

```csv
L21_V001,1234
```

Q&A:

```csv
L21_V011,1450,Đang dẫn chương trình
```

TRAKE:

```csv
L21_V001,1200,1850,2100
```

Không được ghi các trường nội bộ như:

- score
- rank
- keyframe_id
- pts_time
- thumbnail
- CLIP/object/ASR/OCR evidence
- reasoning/debug

## Validator

Validator chạy ở 3 thời điểm:

1. Trước khi cho query vào queue.
2. Trước khi ghi CSV.
3. Sau khi tạo ZIP.

### Validate chung

- 1-100 predictions/query.
- Filename không trùng.
- Filename có hậu tố đúng: `-kis.csv`, `-qa.csv`, `-trake.csv`.
- Không header.
- UTF-8.
- Delimiter là `,`.
- `video_id` không có `.mp4`.
- `video_id` khớp pattern kiểu `L21_V001`.
- `frame_id` là integer.

### KIS

- Mỗi dòng đúng 2 cột.
- `frame_id` map được sang video/frame hợp lệ nếu có mapping.

### Q&A

- Mỗi dòng đúng 3 cột.
- `answer` không rỗng.
- `answer` <= 100 ký tự.
- CSV quoting phải do writer chuẩn sinh.

### TRAKE

- Mỗi dòng có đúng `1 + event_count` cột.
- Tất cả frame là integer.
- Tất cả frame trong một dòng thuộc cùng `video_id`.
- Frame theo thứ tự event.
- Nếu có `pts_time` nội bộ, validate monotonic bằng `pts_time`; nếu chỉ có official frame_id, validate tăng dần theo frame_id như fallback có cảnh báo.

## API endpoints đề xuất

Có thể đặt trong server UI hiện tại:

```text
POST /api/submission/session/start
POST /api/submission/agent/run
POST /api/submission/query/confirm
POST /api/submission/query/update
POST /api/submission/query/remove
GET  /api/submission/session/<session_id>
POST /api/submission/session/validate
POST /api/submission/session/done
GET  /api/submission/session/<session_id>/download
```

Nếu muốn giảm scope P0, có thể chưa làm download endpoint ngay, chỉ ghi ZIP ra `artifacts/submissions/<session_id>/submission.zip` và hiện path trong UI.

## UI đề xuất

Tạo một workspace riêng:

```text
/submission
```

Layout:

```text
Top bar:
Start / Done / Session status / Download ZIP

Left panel:
Query list + Submission queue

Center panel:
Query input + task selector + agent result candidates

Right panel:
Validation issues + CSV preview + export status
```

Không nên nhét workflow này vào Visual/Q&A/TRAKE cũ vì dễ lẫn debug export với official submission.

## Implementation priority

### P0 - KIS end-to-end submission

- Session state in memory hoặc SQLite đơn giản.
- Task selector.
- Agent router skeleton.
- KIS integration với retrieval hiện tại.
- Structured JSON result cho KIS.
- Review chọn nhiều candidate.
- Confirm vào queue.
- Official KIS CSV builder.
- Validator KIS.
- Ghi `submission/query-*-kis.csv`.

Mục tiêu P0: nhập một query KIS, chọn candidate, tạo CSV đúng format.

### P1 - ZIP và full session lifecycle

- Done workflow.
- Validate toàn bộ queue.
- Tạo thư mục `submission/`.
- Tạo `submission.zip` có đúng thư mục con.
- Final ZIP validation.
- UI download/path output.
- Tests cho CSV/ZIP.

Mục tiêu P1: tạo được ZIP đúng chuẩn BTC cho nhiều query KIS.

### P2 - Q&A submission

- Q&A agent bridge sang workflow hiện tại.
- Tách `event_query` và `question`.
- Review/edit `video_id`, `frame_id`, `answer`.
- Validate answer <= 100 ký tự.
- Official Q&A CSV builder.
- Tests cho CSV quoting.

Mục tiêu P2: Q&A vào queue và export official CSV đúng.

### P3 - TRAKE submission

- TRAKE agent bridge sang `/api/trake` workflow hiện tại.
- Render event timeline trong submission UI.
- Cho user chọn/sửa frame từng event.
- Validate đúng N event frames.
- Official TRAKE CSV builder.
- Tests cho cùng video và đúng số cột.

Mục tiêu P3: TRAKE tạo được dòng `video_id,frame_1,...,frame_N`.

### P4 - Agent improvements

- Query import batch nếu BTC cung cấp file danh sách query.
- Auto suggest task type nhưng không override user choice.
- Gemini planner optional cho query rewrite/planning.
- Better retrieval routing theo OCR/ASR/metadata/object.
- Confidence/evidence explanation.
- Recovery/autosave session.

## Test plan

Unit tests:

- Validate KIS row.
- Validate Q&A row và answer quoting.
- Validate TRAKE row đúng/sai số event.
- Validate no header.
- Validate max 100 rows.
- Validate ZIP contains `submission/`.
- Validate internal fields không leak ra CSV.

Integration tests:

- KIS query -> structured JSON -> confirm -> CSV.
- Q&A confirmed answer -> CSV.
- TRAKE reviewed chain -> CSV.
- Multi-query session -> `submission.zip`.

Manual tests:

- Mở `/submission`.
- Start session.
- Chạy 1 KIS, 1 Q&A, 1 TRAKE.
- Confirm từng query.
- Done.
- Mở ZIP kiểm cấu trúc.
- Mở CSV kiểm không header và đúng số cột.

## Rủi ro cần xử lý sớm

### 1. Official frame_id mapping

Đây là rủi ro lớn nhất. Phải xác nhận trường nào trong repo là frame ID BTC cần nộp:

- `frame_idx`
- `keyframe_id`
- `n` trong map-keyframes CSV
- một cột khác trong mapping BTC

Không được tự giả định `keyframe_id == frame_id`.

### 2. Q&A answer matching

Rules có điểm chưa rõ: answer được chấm theo ngữ nghĩa hay so sánh chuỗi chính xác. Vì vậy UI nên ưu tiên answer ngắn, canonical, dễ match:

```text
5
đỏ
nam
đang dẫn chương trình
```

### 3. TRAKE temporal validation

Nếu official frame_id không tăng tương ứng thời gian, validator không nên hard fail chỉ vì frame_id giảm. Tốt nhất dùng `pts_time` nội bộ để check thứ tự event, rồi export official frame_id.

### 4. Session persistence

Đã chốt: dùng SQLite ngay từ P0.

Lý do:

- Submission queue là dữ liệu quan trọng, không nên mất khi tắt server.
- User có thể xử lý nhiều query trong một phiên dài.
- Ngày thi cần khả năng mở lại session, sửa, validate và export lại.

Đường dẫn đề xuất:

```text
artifacts/submissions/submission_sessions.sqlite3
```

SQLite cần lưu tối thiểu:

- sessions
- query analyses
- structured agent results
- user selected predictions
- validation reports
- generated CSV/ZIP manifest
- optional candidate analysis records

## Quyết định đã chốt và câu hỏi còn lại

Đã chốt:

- Query ID nhập tay theo tên BTC như `query-1-kis`.
- Q&A dùng một ô query chung; agent tự tách `event_query` và `question`, sau đó UI hiển thị để user sửa.
- TRAKE số dòng export do user chọn.
- TRAKE/KIS/Q&A review phải có xem frame lân cận.
- Session dùng SQLite ngay từ P0.
- Sau khi chọn candidate, có option agent phân tích lựa chọn của user trước khi confirm.

Còn cần xác nhận:

1. Trong mapping hiện tại, trường nào chắc chắn là official `frame_id` để nộp BTC?
2. Candidate analysis optional nên mặc định dùng local explanation, hay nếu có Gemini key thì bật Gemini luôn?
3. Q&A parser nên ưu tiên rule-based trước, hay Gemini parser nếu có key?

## Definition of Done

Phase 9 Submission Agent UI chỉ được coi là xong khi làm được end-to-end:

```text
Start
-> nhập query
-> chọn task
-> Run Agent
-> nhận structured JSON
-> review/edit
-> confirm vào queue
-> xử lý nhiều query
-> Done
-> validate
-> tạo submission/
-> tạo submission.zip
-> kiểm ZIP lần cuối
```

Và ZIP cuối cùng phải đúng format BTC:

- Có thư mục `submission/`.
- Mỗi query là một file CSV riêng.
- CSV không header.
- Tối đa 100 dòng/file.
- KIS/Q&A/TRAKE đúng số cột.
- Không leak debug fields.
- Dùng official `frame_id`, không dùng nhầm `keyframe_id` hoặc timestamp.
