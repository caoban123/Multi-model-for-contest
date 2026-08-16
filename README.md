# Multi-model for AIC Video Retrieval

Repository này dùng để xây hệ thống truy xuất video cho AIC 2026. Dự án đang ở giai đoạn đầu: ưu tiên kiểm kê dữ liệu, xác minh mapping, dựng registry và baseline retrieval trước khi làm UI, Q&A, TRAKE hoặc agent.

## Đọc Gì Trước

Mọi thành viên và AI agent phải đọc theo thứ tự:

1. `.agent/codex.md` - protocol bắt buộc khi làm việc.
2. `reports/Phase_4_16-08-2026_12.md` - trạng thái mới nhất hiện tại.
3. `plan/PLAN_AIC2026_VIDEO_RETRIEVAL.md` - kế hoạch tổng thể.
4. File phase liên quan trong `plan/`, ví dụ `plan/02_PHASE_1_DATA_REGISTRY_BASELINE.md`.

Không bắt đầu sửa code chỉ dựa trên README. Report mới nhất là nguồn trạng thái vận hành chính.

## Trạng Thái Hiện Tại

Trạng thái hiện tại: **Phase 4 đã có structured/hybrid retrieval ở dạng opt-in experimental, nhưng chưa được promote làm mặc định vì còn thiếu manual relevance judgement**. L21 là scope KIS đã được chạy integration trên máy hiện tại; L22-L30 vẫn phụ thuộc data local của từng máy.

Đã có:

- Scanner tạo data registry.
- Validator kiểm tra CLIP feature, mapping, metadata, object, keyframe.
- Baseline NumPy cosine search bằng query vector `.npy` hoặc text query CLIP; L21 là scope benchmark KIS hiện tại.
- Test tối thiểu bằng pytest.
- Metadata lexical search độc lập ở cấp video, chỉ dùng cho thí nghiệm.
- Local retrieval UI với Video Ranking mặc định, Frame Ranking debug, matched-frame expansion và keyframe-neighborhood timeline.
- Max-score aggregation ở cấp video và benchmark diversity trước/sau.
- Structured retrieval experimental qua endpoint riêng `/api/structured-search`, có thể dùng CLIP, metadata, object evidence và attribute/color evidence opt-in.

Chưa có:

- FAISS index.
- Vector database.
- Q&A.
- TRAKE.
- Xử lý video gốc.

## Cấu Trúc Repo

```text
.agent/                 Protocol cho agent và template report
plan/                   Kế hoạch tổng và kế hoạch từng phase
reports/                Snapshot tiến độ theo phase
src/aic_retrieval/      Source code chính
tools/                  CLI scripts
tests/                  Test suite
data/                   Dữ liệu local, không commit
artifacts/              Output sinh ra, không commit
```

## Dữ Liệu Local

`data/` không được commit lên GitHub vì dung lượng lớn.

Mỗi thành viên nhận data theo cách riêng; `data/` và `artifacts/` không được commit. Không suy luận coverage của máy này cho máy khác. Với KIS baseline, bắt buộc là CLIP feature và mapping tương ứng; ảnh keyframe chỉ là dữ liệu hỗ trợ để xem cục bộ.

Coverage chuẩn đã được ghi nhận trong các report Phase 1 trước đó:

| Nhóm dữ liệu | Coverage chuẩn |
|---|---|
| `clip-features-32` | L21-L30, 873 file |
| `map-keyframes` | L21-L30, 873 file |
| `media-info` | L21-L30, 873 file |
| `objects` | L21-L30, 873 folder |
| `keyframes` | Chỉ L21, 29 folder trong bộ đầy đủ đã kiểm tra trước đây |
| `videos` | L21 local hiện có 29 file `.mp4`; video gốc vẫn là dữ liệu optional, không commit |

Các con số trên là coverage tham chiếu của bộ data chuẩn, không phải cam kết rằng mọi máy local đều đã cài đủ toàn bộ data.

## Scope Phase 1

- **L21** là baseline KIS và scope benchmark hiện tại. CLIP feature + mapping là bắt buộc; ảnh keyframe có thể chỉ là một tập cục bộ không đầy đủ.
- **L22-L30** thuộc scope `registry/mapping/metadata/index-capable` của Phase 1. Chúng không được coi là visual retrieval đã xác minh nếu máy local không có ảnh keyframe tương ứng.
- Baseline là **CLIP ViT-B/32 + NumPy cosine search**. FAISS là tùy chọn, không phải dependency bắt buộc.
- Metadata là kênh retrieval độc lập mang tính thử nghiệm. Object/attribute fusion cũng là thử nghiệm và bị tắt mặc định cho đường search chính.

## Setup Tối Thiểu

### Test-only: không cần dataset hoặc CLIP model

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python -m compileall -q src tools tests
```

Hai test integration dùng dataset sẽ được skip rõ ràng nếu `data/` không tồn tại.

### Phase-1 integration: registry, index và text retrieval

Yêu cầu hiện tại:

- Python 3.10+
- NumPy
- pytest
- torch, transformers cho text query CLIP

```powershell
python -m pip install -r requirements-phase1.txt
```

Không cần cài FAISS hoặc OpenCLIP ở thời điểm này. Text query dùng HuggingFace `openai/clip-vit-base-patch32`; lần chạy đầu trên máy mới có thể cần tải model khoảng 600 MB.

## Lệnh Hay Dùng

Kiểm tra máy có đủ dữ liệu Phase 1 hay không, không tải hoặc thay đổi dữ liệu:

```powershell
python tools\check_phase1_data.py --groups L21
```

`MISSING`/`INVALID` nghĩa là thiếu feature hoặc mapping trong scope đã chọn. `WARNING` cho keyframe image coverage là bình thường khi data được phân phối riêng. Manifest kỳ vọng nằm ở `configs\phase1_data_manifest.json`.

Tạo registry và validation report local:

```powershell
python tools\data_registry.py --data-root data --output artifacts\registry\data_registry.json --validation-output artifacts\registry\validation_report.json
```

Registry validation đầy đủ có thể nêu thiếu object hoặc ảnh keyframe cục bộ. Những mục đó không chặn baseline CLIP KIS; preflight L21 sẽ phân biệt lỗi blocking với warning hỗ trợ.

Chạy test:

```powershell
python -m pytest tests
```

Tạo query vector mẫu từ feature L21:

```powershell
python -c "from pathlib import Path; import numpy as np; Path('artifacts/query').mkdir(parents=True, exist_ok=True); a=np.load('data/clip-features-32/L21_V001.npy', mmap_mode='r'); np.save('artifacts/query/L21_V001_001.npy', a[0])"
```

Chạy search mẫu trên L21:

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --query-vector artifacts\query\L21_V001_001.npy --groups L21 --top-k 5 --output artifacts\registry\sample_search_results.json
```

Kỳ vọng Top-1:

```text
video_id=L21_V001
keyframe_id=1
frame_idx=0
keyframe_path có thể là null nếu ảnh keyframe tương ứng không nằm trong local data.
```

Build persistent NumPy index cho L21:

```powershell
python tools\build_numpy_index.py --registry artifacts\registry\data_registry.json --groups L21 --output-dir artifacts\indexes\l21_numpy
```

Search bằng index đã lưu (index mới được fingerprint theo registry, feature và mapping; tool sẽ từ chối index stale hoặc sai scope):

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-vector artifacts\query\L21_V001_001.npy --groups L21 --top-k 5 --output artifacts\registry\sample_search_results_from_index.json
```

Search debug trực tiếp bằng video/keyframe, không cần tự tạo query `.npy`:

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-from-video-id L21_V001 --query-keyframe-id 1 --groups L21 --top-k 5 --output artifacts\registry\sample_search_results_from_debug_query.json
```

Search có grouping/diversity theo video:

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-from-video-id L21_V001 --query-keyframe-id 1 --groups L21 --candidate-pool 20 --top-k 5 --max-frames-per-video 1 --group-by-video --output artifacts\registry\sample_search_results_grouped.json
```

Search bằng text query CLIP:

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-text "a person walking" --groups L21 --candidate-pool 20 --top-k 5 --max-frames-per-video 1 --group-by-video --output artifacts\registry\sample_search_results_text.json
```

Chạy local retrieval UI cho L21:

```powershell
$env:AIC_CLIP_MODEL_ID="D:\AIC\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
$env:AIC_TRANSLATION_API_KEY="<your-api-key>"
$env:AIC_TRANSLATION_PROVIDER="gemini"
$env:AIC_TRANSLATION_MODEL="gemini-3.5-flash"
python tools\retrieval_ui.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --groups L21 --clip-local-files-only
```

Mở `http://127.0.0.1:8765/` trong trình duyệt. Lệnh server này giữ terminal mở; dừng bằng `Ctrl+C`. Không cần dùng `Start-Process` hoặc redirect log dài dòng trừ khi thật sự muốn chạy nền.

Không commit API key. Mặc định translation provider là Gemini và dùng Gemini Generate Content API. Nếu key của bạn hỗ trợ model khác, có thể đổi `AIC_TRANSLATION_MODEL`, ví dụ `gemini-3.6-flash`. Nếu dùng nhà cung cấp OpenAI-compatible khác, đặt `AIC_TRANSLATION_PROVIDER="openai-compatible"` rồi đổi `AIC_TRANSLATION_API_URL` và `AIC_TRANSLATION_MODEL` theo nhà cung cấp đó. Nếu chưa cấu hình `AIC_TRANSLATION_API_KEY`, UI vẫn search bình thường nhưng nút `Translate` sẽ báo chưa cấu hình.

Nếu port `8765` đã được dùng, kiểm tra server đang chạy:

```powershell
curl.exe -s http://127.0.0.1:8765/api/health
netstat -ano | Select-String ":8765"
```

Nếu cần dừng server chạy nền, lấy PID từ `netstat` rồi chạy:

```powershell
Stop-Process -Id <PID>
```

UI dùng index local đã build, encode text query bằng CLIP, hiển thị thumbnail L21, `video_id`, `keyframe_id`, timestamp, score và metadata. Nếu index cũ báo thiếu provenance, rebuild bằng `tools\build_numpy_index.py`.

Review kết quả ngay trong UI:

1. Chạy query.
2. Chọn `Good`, `Partial` hoặc `Bad` trên từng result.
3. Ghi note nếu cần.
4. Bấm `Pin` nếu muốn giữ candidate đáng chú ý.
5. Bấm `Export CSV`.

CSV export có các cột tương thích với benchmark review như `query_id`, `query_text`, `clip_query`, `rank`, `video_id`, `keyframe_id`, `score`, `manual_judgement`, `manual_notes`.

UI cũng lưu search history và pinned results trong browser `localStorage`. Có thể bấm lại query trong `History`, hoặc xuất riêng danh sách pinned bằng `Export Pins`.

Pin review workflow:

- `Pin` lưu snapshot của candidate đang xem, kèm query, ranking mode, rank, score, metadata và video URL nếu có raw video.
- Nếu đã pin rồi mới chọn `Good`/`Partial`/`Bad` hoặc ghi note, pin sẽ tự động cập nhật judgement/note.
- Trong panel `Pins`, bấm `Open video` để mở lại raw video ở timestamp của candidate đã pin.
- `Export CSV` xuất toàn bộ result đang hiển thị, có thêm `is_pinned`, `pinned_at`, `video_url`, `watch_url` và `description_preview`.
- `Export Pins` xuất shortlist riêng, có thêm `manual_judgement`, `manual_notes`, `submission_video_id`, `submission_keyframe_id` và `submission_pts_time`.

Trong mỗi visual result, bấm `Neighbors` để xem các keyframe lân cận trong cùng video. Viewer này dùng mapping/index hiện có, đánh dấu keyframe trung tâm và hữu ích để kiểm tra ngữ cảnh trước/sau một kết quả.

### Phase 3: Video Ranking Và Diversity

Visual search mặc định hiển thị `Video Ranking`: một video chỉ có một card, score video bằng score cao nhất của các keyframe trong candidate pool. `Frame Ranking` giữ raw Top-K keyframe để debug và so sánh duplicate.

Default UI:

```text
Top videos = 12
Frame pool = 40
aggregation = max
matched frames shown per video = 5
neighbor radius = 3
```

Trong video card, bấm `Explore video` để xem các retrieval-matched frames theo score giảm dần. Bấm `Timeline` trên một matched frame để gọi lại Neighbors API và xem ngữ cảnh theo thời gian. Matched frames là kết quả retrieval; timeline neighbors chỉ là frame lân cận và không có retrieval score nếu chúng không nằm trong candidate pool.

API `/api/search` giữ `results` cho tương thích Phase 2 và trả thêm:

- `raw_results`: candidate frames nguyên thứ hạng.
- `video_results`: Top-K unique videos sau aggregation.
- `aggregation_method`, `retrieval_ms`, `aggregation_ms` và các config áp dụng.

Benchmark Phase 3 từ output của text benchmark:

```powershell
python tools\phase3_benchmark.py --input artifacts\benchmarks\l21_phase3_source_pool50.json --judgements-csv artifacts\benchmarks\l21_phase3_source_pool50.csv --top-k 12 --candidate-pool 50 --matched-frames-per-video 5 --aggregation-method max --output artifacts\benchmarks\l21_phase3_comparison_pool50.json
```

Máy hiện tại chỉ có 18 ảnh keyframe L21 thuộc một video. Retrieval/video ranking vẫn hoạt động với toàn bộ 7.800 vector; card thiếu ảnh hiển thị `No image`. Cần bổ sung full ảnh L21 và chấm manual judgement trước khi đánh dấu Phase 3 `DONE`.

UI có hai chế độ:

- `Visual`: tìm bằng CLIP image features theo keyframe.
- `Metadata`: tìm theo title, author, publish date, description và keywords trong `data\media-info`.

Metadata search hữu ích cho query như tên chương trình, ngày, kênh hoặc tác giả. Chế độ này tách riêng với visual search; chưa trộn điểm hoặc rerank mặc định.

Nếu đã có CLIP cache local, chạy offline/local-only bằng cách đặt biến môi trường:

```powershell
$env:AIC_CLIP_CACHE_DIR=(Resolve-Path "artifacts\models")
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-text "a person walking" --clip-local-files-only --groups L21 --candidate-pool 20 --top-k 5 --max-frames-per-video 1 --group-by-video
```

Không hard-code đường dẫn `D:\AIC` vào code. Thành viên khác khi pull repo có thể dùng model id mặc định `openai/clip-vit-base-patch32`; nếu chưa có cache local thì HuggingFace sẽ tải model khi chạy text query.

Chạy benchmark text query L21:

```powershell
python tools\text_query_benchmark.py --queries benchmarks\text_queries_l21.json --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --groups L21 --candidate-pool 25 --top-k 5 --max-frames-per-video 1 --output artifacts\benchmarks\l21_text_benchmark.json --csv-output artifacts\benchmarks\l21_text_benchmark.csv
```

File CSV có cột `manual_judgement` và `manual_notes` để thành viên chấm thủ công Top-K. Nếu dùng snapshot local trên máy hiện tại, đặt `AIC_CLIP_MODEL_ID` như ví dụ phía trên và thêm `--clip-local-files-only`.

Sinh HTML review để chấm nhanh bằng hình ảnh:

```powershell
python tools\benchmark_review_html.py --input artifacts\benchmarks\l21_text_benchmark.json --output artifacts\benchmarks\l21_text_review.html
```

Mở `artifacts\benchmarks\l21_text_review.html` trong trình duyệt, chọn `good`, `partial` hoặc `bad`, ghi notes rồi bấm `Export CSV`.

Tóm tắt CSV đã chấm:

```powershell
python tools\summarize_benchmark_judgements.py --input artifacts\benchmarks\l21_text_benchmark_judged.csv --output artifacts\benchmarks\l21_text_judgement_summary.json
```

Inspect object detections của các dòng `bad`/`partial` trước khi thiết kế fusion:

```powershell
python tools\inspect_judged_objects.py --input artifacts\benchmarks\l21_text_benchmark_judged.csv --object-root data\objects --judgements bad,partial --top-n 8 --output artifacts\benchmarks\l21_judged_object_inspection.json
```

Chạy object-fusion experiment offline, chỉ re-rank benchmark đã chấm và không thay đổi search mặc định:

```powershell
python tools\fusion_experiment.py --benchmark artifacts\benchmarks\l21_text_benchmark.json --judgements artifacts\benchmarks\l21_text_benchmark_judged.csv --object-root data\objects --object-weight 0.05 --top-k 5 --output artifacts\benchmarks\l21_fusion_experiment.json
```

Với benchmark pool50, dùng `raw_results` để đánh giá đúng toàn bộ candidate đã chấm:

```powershell
python tools\fusion_experiment.py --benchmark artifacts\benchmarks\l21_weak_text_benchmark_pool50.json --judgements artifacts\benchmarks\l21_text_benchmark_judged.csv --object-root data\objects --result-set raw_results --object-weight 0.05 --interaction-weight 0.50 --top-k 50 --output artifacts\benchmarks\l21_weak_pool50_fusion_interaction.json
```

Kết quả hiện tại: object-only và phone-hand interaction heuristic đều không cải thiện pool50; chưa bật fusion vào search mặc định.

Audit encoding và coverage metadata:

```powershell
python tools\audit_metadata.py --media-dir data\media-info --output artifacts\metadata\media_info_audit.json --sample-limit 10
```

Kết quả hiện tại: 873 JSON đọc được, 0 lỗi JSON, 0 marker mojibake, 0 control character. Nếu PowerShell hiển thị chữ Việt bị sai, đó là vấn đề console encoding chứ không phải file metadata hỏng.

Search metadata độc lập, chưa trộn vào CLIP visual search:

```powershell
python tools\metadata_search.py --media-dir data\media-info --groups L21 --query "60 giay sang 01082024" --top-k 5 --output artifacts\metadata\search_60giay_01082024.json --csv-output artifacts\metadata\search_60giay_01082024.csv
python tools\metadata_search.py --media-dir data\media-info --query "Bao Tuoi Tre" --top-k 5 --output artifacts\metadata\search_all_bao_tuoi_tre.json --csv-output artifacts\metadata\search_all_bao_tuoi_tre.csv
```

Metadata search hiện hữu ích cho query theo title, ngày, tác giả, kênh, keywords. Đây vẫn là kênh thử nghiệm riêng; chưa dùng để rerank kết quả hình ảnh mặc định.

Chạy metadata benchmark fixture (chỉ kiểm tra kỹ thuật CI, **không phải ground truth hoặc điểm AIC chính thức**):

```powershell
python tools\benchmark_metadata.py --output artifacts\benchmarks\metadata_fixture_benchmark.json
```

Khi có `data\media-info` thật, thay `--media-dir` bằng đường dẫn đó và chỉ dùng một query set có expected video IDs đã được nhóm kiểm tra thủ công.

Tạo benchmark mở rộng cho các query yếu q008/q009/q010 với raw candidate pool 50:

```powershell
python tools\text_query_benchmark.py --queries benchmarks\text_queries_l21_weak.json --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --groups L21 --candidate-pool 50 --top-k 5 --max-frames-per-video 1 --csv-result-set raw_results --output artifacts\benchmarks\l21_weak_text_benchmark_pool50.json --csv-output artifacts\benchmarks\l21_weak_text_benchmark_pool50.csv
python tools\benchmark_review_html.py --input artifacts\benchmarks\l21_weak_text_benchmark_pool50.json --result-set raw_results --output artifacts\benchmarks\l21_weak_text_review_pool50.html
```

Mở `artifacts\benchmarks\l21_weak_text_review_pool50.html` để chấm 150 candidates rồi export CSV. Fusion experiment chỉ nên chạy lại sau khi có file judged pool50.

## Quy Tắc Làm Việc

- Luôn đọc report mới nhất trước khi làm.
- Sau task đáng kể, tạo report snapshot mới trong `reports/`.
- Không commit `data/` hoặc `artifacts/`.
- Không tải model/dữ liệu lớn nếu chưa được duyệt.
- Không cài dependency nặng như FAISS hoặc OpenCLIP nếu chưa được duyệt.
- Không decode/re-encode hoặc xử lý hàng loạt video gốc nếu chưa có yêu cầu rõ ràng. `data/videos` hiện có L21 local để audit/mở đúng video về sau.
- Không đánh dấu phase/module là DONE nếu chưa có bằng chứng test/validation.

## Bước Tiếp Theo Đề Xuất

Phase 3 implementation đã có max-score video aggregation, unique-video ranking, representative frame, matched-frame expansion, Frame Ranking debug và L21 neighbor timeline. Bước tiếp theo là bổ sung đầy đủ ảnh keyframe L21 và chấm manual relevance cho benchmark before/after; metadata fusion, object fusion và FAISS vẫn chưa bật mặc định.

## Phase 4 — Structured Retrieval (Hybrid Opt-In)

Phase 4 giữ `/api/search` và UI CLIP Phase 3 làm mặc định. Structured retrieval chỉ chạy khi gọi `/api/structured-search` hoặc bật **Use experimental structured retrieval** trong panel **Structured Filters**.

Build/audit object store L21:

```powershell
python tools\audit_structured_data.py
python tools\build_object_store.py
python tools\object_search.py --labels "điện thoại" --min-confidence 0.3 --limit 10
```

Chạy UI (object store được nạp từ đường dẫn mặc định dưới `artifacts/structured`):

```powershell
python tools\retrieval_ui.py `
  --clip-cache-dir artifacts\models `
  --clip-local-files-only
```

Ví dụ endpoint opt-in:

```text
/api/structured-search?q=a+person+holding+a+phone&top_k=12&candidate_pool=100&enable_clip=true&enable_objects=true&object_label=phone&object_min_count=1&object_position=any&object_min_confidence=0.3&enable_metadata=false&fusion_method=rrf
```

Attribute/color evidence có thể bật bằng UI `Attributes` hoặc API:

```text
/api/structured-search?q=person+wearing+a+red+shirt&top_k=12&candidate_pool=100&enable_clip=true&enable_attributes=true&attribute_filter_mode=soft&fusion_method=rrf
```

Nếu `attribute_color` để trống, backend cố gắng parse màu từ query như `red`, `blue`, `white`, `black` hoặc tiếng Việt tương ứng. Đây là color evidence nhẹ trên keyframe image, hữu ích để giảm nhiễu cho query màu sắc như áo đỏ, nhưng chưa phải detector trang phục/person-crop chuẩn.

Evidence phân biệt `matched`, `not_matched`, `unknown` và `disabled`. `unknown` nghĩa là thiếu object/attribute data, không phải “không phát hiện object”. Metadata luôn là evidence cấp video; representative frame phải đến từ CLIP/object/attribute evidence thật.

Chạy baseline và ablation:

```powershell
python tools\phase4_baseline.py --clip-cache-dir artifacts\models --clip-local-files-only
python tools\phase4_benchmark.py --clip-cache-dir artifacts\models --clip-local-files-only
```

Output được tạo dưới `artifacts/benchmarks/phase4/` và không commit. `ablation_v1_review.csv` có cột chấm tay và evidence CLIP/object/metadata/RRF; UI export mới có thêm attribute fields khi bật Attributes. Khi chưa có manual ground truth, Recall/MRR/NDCG được ghi `null/unavailable`; không được diễn giải là 0.

Final evaluation query set cho manual review nằm ở `benchmarks/phase4_final_eval_queries_v1.json`: 30 query, chia 15 development / 15 holdout, có nhóm attribute-color để kiểm tra các query như `person wearing a red shirt`. Chỉ dùng development để tuning; holdout chỉ dùng sau khi đã chốt cấu hình.

Quyết định P4.10: **HYBRID OPT-IN**. Implementation, API, UI và ablation pipeline đã hoàn thành; quality verification/default promotion vẫn chờ manual judgement. Không đổi framework, model, vector database, OCR/ASR hoặc detector.
