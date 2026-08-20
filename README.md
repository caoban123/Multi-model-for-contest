# Multi-model for AIC Video Retrieval

## Current Snapshot - 2026-08-20

> Read this section first. Older sections below are retained as historical phase notes and may describe earlier limitations.

The repository now contains a working L21-focused video retrieval system with CLIP, opt-in BGE/FAISS and BM25 retrieval, Agent routing, evidence-grounded Q&A, TRAKE temporal alignment, and a guarded official submission workspace for KIS/Q&A/TRAKE. CLIP remains the default because the current partial benchmark does not justify promoting always-on Hybrid Retrieval.

Latest comprehensive report:

```text
reports/Phase_9_20-08-2026_12.md
```

High-level status:

| Area | Status | Notes |
|---|---|---|
| Data registry | Implemented | `artifacts/registry/data_registry.json` describes local assets. |
| CLIP NumPy index | Implemented | L21 index is expected at `artifacts/indexes/l21_numpy`. |
| Retrieval UI | Implemented | Visual search, metadata search, Agent workspace, pins, history, raw video preview, keyframe neighborhood viewer. |
| Agent query routing | Implemented, opt-in | Separate CLIP-English, BGE semantic and BM25 lexical queries with raw/parsed/validated Gemini trace and local fallback. |
| Structured retrieval | Implemented, opt-in | CLIP + objects + metadata + attributes + OCR/ASR via `/api/structured-search`. |
| Object store | Built for L21 | `artifacts/structured/l21_objects.sqlite`, 7,800 frames, 780,000 detections. Raw labels live in `detection_class_entities`. |
| Attribute/color evidence | Implemented | Useful for constraints such as red shirt or white car. |
| Phase 5 ASR/OCR | Partially useful | ASR has 563 transcript segments. OCR JSONL exists but current detections are empty. |
| Phase 6 reranker/planner | Implemented | Local query planner/reranker config exists at `configs/phase6_reranker_v1.json`. |
| Phase 7 Q&A | Implemented | Evidence-first Q&A, manual review, SQLite session store, export record flow. |
| Gemini Q&A/VLM | Implemented as option | `Draft with Gemini` sends selected evidence plus related keyframe images to Gemini. |
| Phase 8 TRAKE | Implementation complete; quality pending | Local plan/retrieve/temporal-align, `/trake` workspace, SQLite review, internal export and selected-event refinement exist; manual development/holdout labels are pending. |
| Phase 9 submission workflow | Implemented, guarded | `/submission` has rich Agent candidate cards, Gemini trace, SQLite queue, official KIS/Q&A/TRAKE CSV, validation and `submission.zip`. Official frame IDs require manual/BTC-certified mapping. |
| FAISS/vector DB | Implemented, opt-in | BGE-M3 FAISS index covers 8,322 L21 text documents; it does not replace the CLIP NumPy baseline. |
| BM25 | Implemented, opt-in | SQLite FTS5 index covers the same 8,322-document L21 corpus. |

Current important local artifacts:

```text
artifacts/registry/data_registry.json
artifacts/indexes/l21_numpy/
artifacts/indexes/l21_bge/
artifacts/indexes/l21_bm25/
artifacts/structured/l21_objects.sqlite
artifacts/structured/l21_objects_manifest.json
artifacts/phase5/ocr_l21.jsonl
artifacts/phase5/asr_l21.jsonl
artifacts/phase5/phase5_store.sqlite
artifacts/phase5/stores/phase5_l21.sqlite3
artifacts/qa/phase7_qa.sqlite3
artifacts/submissions/
```

Run the UI:

```powershell
$env:AIC_CLIP_MODEL_ID="D:\AIC\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
$env:AIC_BGE_MODEL_PATH="D:\AIC\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
$env:AIC_TRANSLATION_API_KEY="<your-gemini-api-key>"
$env:AIC_TRANSLATION_PROVIDER="gemini"
$env:AIC_TRANSLATION_MODEL="gemini-3.5-flash"

python tools\retrieval_ui.py `
  --registry artifacts\registry\data_registry.json `
  --index-dir artifacts\indexes\l21_numpy `
  --groups L21 `
  --clip-local-files-only `
  --enable-hybrid-retrieval `
  --bge-index-dir artifacts\indexes\l21_bge `
  --bge-model-path $env:AIC_BGE_MODEL_PATH `
  --bm25-index-dir artifacts\indexes\l21_bm25 `
  --phase5-store artifacts\phase5\phase5_store.sqlite `
  --qa-store artifacts\qa\phase7_qa.sqlite3
```

Open:

```text
http://127.0.0.1:8765
http://127.0.0.1:8765/trake
http://127.0.0.1:8765/submission
```

Current test status:

```text
python -m pytest -q
340 passed
```

Known limitations:

- OCR is not yet useful on current L21 artifact because all OCR detections are empty.
- Hybrid Retrieval remains opt-in; CLIP remains the verified default visual baseline.
- Gemini VLM answer drafting requires Internet and a configured Gemini API key.
- Gemini structured-filter output is a reviewed suggestion: use `Use in Structured Search` to load it, then review and run Visual Structured Search. It is not silently applied to the Agent result ranking.
- OCR currently has zero usable documents and ASR covers only 2/29 L21 videos.
- Phase 8 TRAKE core is implemented for L21; quality/holdout verification is pending manual labels. See `docs/PHASE8_TESTING.md`.
- Automatic official `frame_id` mapping is intentionally blocked. The submission UI requires a manually verified official frame ID or a future BTC-certified mapping.

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

## Phase 5 — OCR/ASR Opt-In

Phase 5 bổ sung schema, temporal mapping, SQLite FTS5 store, lexical search, OCR/ASR evidence trong structured RRF và timeline. Hai channel bị tắt mặc định và chỉ khả dụng khi truyền `--phase5-store` trỏ tới store đã build.

### 1. Trích xuất OCR & ASR Evidence (JSONL):

Để trích xuất dữ liệu OCR và ASR chuẩn schema `Phase5`:

```powershell
# Trích xuất OCR & ASR cho tập L21 (Whisper model local)
python tools\extract_phase5_evidence.py --groups L21 --whisper-model-name models--Systran--faster-whisper-base

# Tùy chọn bỏ qua OCR hoặc ASR khi cần test từng kênh:
python tools\extract_phase5_evidence.py --groups L21 --skip-ocr
python tools\extract_phase5_evidence.py --groups L21 --skip-asr
```

Output sẽ sinh ra 2 file JSONL chuẩn schema tại `artifacts/phase5/`:
- `artifacts/phase5/ocr_l21.jsonl`
- `artifacts/phase5/asr_l21.jsonl`

### 2. Build SQLite FTS5 Evidence Store:

Sau khi đã có file OCR & ASR JSONL, ingest dữ liệu vào SQLite FTS5 store:

```powershell
python tools\build_phase5_store.py `
  --ocr-jsonl artifacts\phase5\ocr_l21.jsonl `
  --asr-jsonl artifacts\phase5\asr_l21.jsonl `
  --index-dir artifacts\indexes\l21_numpy `
  --output artifacts\phase5\phase5_store.sqlite `
  --overwrite
```

### 3. Workflow & Testing:

Hướng dẫn audit, pilot, build store, API/UI và benchmark nằm tại [`docs/PHASE5_TESTING.md`](docs/PHASE5_TESTING.md). Snapshot tiến độ nằm tại [`reports/Phase_5_17-08-2026_1.md`](reports/Phase_5_17-08-2026_1.md).

Nếu `artifacts/` bị xóa, dùng `python tools/prepare_phase5.py --data-root data --groups L21 --skip-store` để rebuild registry và NumPy index bằng pipeline thật. khi có OCR/ASR JSONL đã review, bỏ `--skip-store` và truyền `--ocr-jsonl`/`--asr-jsonl` để build lại store; workflow không tạo evidence giả.

## Phase 6 — Local Query Planner Và Top-N Reranker

Phase 6 bổ sung rule-based query decomposition chạy offline, query variants có thể xem trong UI và lightweight reranker chỉ áp dụng trên Top-N candidate. Cả planner và reranker đều opt-in; khi tắt, structured retrieval giữ đường baseline Phase 5. Hướng dẫn test tự động, test tay và benchmark before/after nằm tại [`docs/PHASE6_TESTING.md`](docs/PHASE6_TESTING.md).

Reranker seed config nằm tại `configs/phase6_reranker_v1.json`. Dùng `tools/phase6_tune.py` chỉ trên development labels để sinh config đã tune, sau đó khóa cùng config cho holdout và UI bằng `--phase6-config`.

## Phase 7 — Evidence-Grounded Q&A

Phase 7 tạo workflow Q&A có kiểm soát: event query + question → retrieval → evidence pack → answer draft → normalization preview → human review → append-only local audit log. Đây không phải chatbot tự do: không evidence thì không confirm; không OCR/ASR store thì không tạo claim OCR/speech; không có human confirmation thì không có internal export record. Gemini VLM draft là tùy chọn trong UI và chỉ dùng selected evidence cùng ảnh keyframe liên quan.

Khởi động UI cùng local Q&A store:

```powershell
python tools\retrieval_ui.py --qa-store artifacts\qa\phase7_qa.sqlite3
```

Mở UI, dùng phần **Q&A workspace**, nhập `Event query` và `Question`, chọn evidence, tạo draft và xác nhận thủ công. Q&A tự rewrite `Event query` thành `Auto CLIP event query` bằng Gemini để retrieve candidate, còn `Question` được giữ nguyên để trả lời theo ngôn ngữ người dùng. Bật **Draft with Gemini** nếu muốn Gemini đọc selected evidence và ảnh keyframe để đề xuất answer. Internal export chỉ khả dụng sau review `confirmed` hoặc `edited` với evidence frame-level được chọn. Chi tiết test và benchmark: [`docs/PHASE7_TESTING.md`](docs/PHASE7_TESTING.md).

```powershell
python -m pytest tests\test_qa_schema.py tests\test_qa_evidence.py tests\test_qa_question_router.py tests\test_qa_answering.py tests\test_qa_normalization.py tests\test_qa_confidence.py tests\test_qa_store.py tests\test_qa_workflow.py tests\test_qa_api.py tests\test_qa_ui_static.py tests\test_qa_benchmark.py -q
python tools\phase7_benchmark.py --queries benchmarks\phase7_qa_queries_v1.json --split development --qa-store artifacts\qa\phase7_qa.sqlite3 --output artifacts\benchmarks\phase7\development.json
```

Khi benchmark template chưa được chấm tay, quality metrics sẽ là `null/unavailable`; đây là behavior đúng, không phải metric 0.

## Phase 9 — Submission Hardening

Confirmed or edited Q&A exports can now be converted into an internal submission schema before the final organizer format is known. This keeps retrieval/Q&A separate from contest-specific CSV or JSON adapters.

```powershell
python tools\export_qa_submissions.py --qa-store artifacts\qa\phase7_qa.sqlite3
```

Default outputs are written under `artifacts\submissions\`:

- `qa_submission_candidates.jsonl`
- `qa_submission_candidates.csv`
- `qa_submission_validation.json`

See [`docs/PHASE9_SUBMISSION_HARDENING.md`](docs/PHASE9_SUBMISSION_HARDENING.md) for schema, validation rules, and the offline checklist.

## Phase 9 Current - Hybrid Retrieval and Submission Agent

Phase 9 adds modular BGE-M3/FAISS and BM25 retrievers without replacing CLIP. The Agent chooses a visible opt-in route per query and falls back to deterministic local planning when Gemini is unavailable. Visual/KIS and Q&A use video-level RRF; TRAKE keeps frame-level event fusion so temporal alignment still receives moments rather than video-only hits.

The separate `/submission` workspace stores an ACTIVE session and confirmed query queue in SQLite. KIS can run Agent retrieval directly. Q&A and TRAKE import only reviewed workflow outputs. All three tasks require a manually verified official frame ID until a BTC mapping is certified.

Official output rules enforced by code:

- KIS: `video_id,frame_id`.
- Q&A: `video_id,frame_id,answer`, answer at most 100 characters.
- TRAKE: `video_id,frame_event_1,...,frame_event_N` with exactly N frames.
- One CSV per query, no header, UTF-8, at most 100 rows.
- ZIP entries are always under `submission/`.
- Score, rank, provenance, keyframe ID and timestamp cannot enter the official writer.

Use `/submission` to Start, retrieve/import, review nearby frames, enter verified official frame IDs, Confirm, Validate, Done and download `submission.zip`. Session files are written under `artifacts/submissions/` and are not committed.

Latest implementation plan and report:

- [`plan/11_PHASE_9_SUBMISSION_AGENT_UI_REPLAN.md`](plan/11_PHASE_9_SUBMISSION_AGENT_UI_REPLAN.md)
- [`reports/Phase_9_20-08-2026_9.md`](reports/Phase_9_20-08-2026_9.md)
