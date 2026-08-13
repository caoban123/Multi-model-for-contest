# Multi-model for AIC Video Retrieval

Repository này dùng để xây hệ thống truy xuất video cho AIC 2026. Dự án đang ở giai đoạn đầu: ưu tiên kiểm kê dữ liệu, xác minh mapping, dựng registry và baseline retrieval trước khi làm UI, Q&A, TRAKE hoặc agent.

## Đọc Gì Trước

Mọi thành viên và AI agent phải đọc theo thứ tự:

1. `.agent/codex.md` - protocol bắt buộc khi làm việc.
2. `reports/Phase_1_13-08-2026_2.md` - trạng thái mới nhất hiện tại.
3. `plan/PLAN_AIC2026_VIDEO_RETRIEVAL.md` - kế hoạch tổng thể.
4. File phase liên quan trong `plan/`, ví dụ `plan/02_PHASE_1_DATA_REGISTRY_BASELINE.md`.

Không bắt đầu sửa code chỉ dựa trên README. Report mới nhất là nguồn trạng thái vận hành chính.

## Trạng Thái Hiện Tại

Phase hiện tại: **Phase 1 - Data registry và baseline retrieval**.

Đã có:

- Scanner tạo data registry.
- Validator kiểm tra CLIP feature, mapping, metadata, object, keyframe.
- Baseline NumPy cosine search cho L21 bằng query vector `.npy` hoặc text query CLIP.
- Test tối thiểu bằng pytest.

Chưa có:

- FAISS index.
- UI.
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

Trạng thái dữ liệu trên máy hiện tại:

| Nhóm dữ liệu | Coverage |
|---|---|
| `clip-features-32` | L21-L30, 873 file |
| `map-keyframes` | L21-L30, 873 file |
| `media-info` | L21-L30, 873 file |
| `objects` | L21-L30, 873 folder |
| `keyframes` | Chỉ L21, 29 folder |
| `videos` | Chưa có |

Keyframes hiện chỉ có L21, nên mọi kiểm tra hình ảnh hoặc viewer chỉ được coi là đúng với L21.

## Setup Tối Thiểu

Yêu cầu hiện tại:

- Python 3.10+
- NumPy
- pytest
- torch, transformers cho text query CLIP

Không cần cài FAISS hoặc OpenCLIP ở thời điểm này. Text query dùng HuggingFace `openai/clip-vit-base-patch32`; lần chạy đầu trên máy mới có thể cần tải model khoảng 600 MB.

## Lệnh Hay Dùng

Tạo registry và validation report local:

```powershell
python tools\data_registry.py --data-root data --output artifacts\registry\data_registry.json --validation-output artifacts\registry\validation_report.json
```

Kỳ vọng hiện tại:

```text
total_videos=873
errors=0
warnings=844
keyframe_group_counts={'L21': 29}
```

844 warning là bình thường vì L22-L30 chưa có keyframe images.

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
python tools\vector_search.py --registry artifacts\registry\data_registry.json --query-vector artifacts\query\L21_V001_001.npy --groups L21 --require-keyframes --top-k 5 --output artifacts\registry\sample_search_results.json
```

Kỳ vọng Top-1:

```text
video_id=L21_V001
keyframe_id=1
frame_idx=0
keyframe_path=data/keyframes/L21_V001/001.jpg
```

Build persistent NumPy index cho L21:

```powershell
python tools\build_numpy_index.py --registry artifacts\registry\data_registry.json --groups L21 --require-keyframes --output-dir artifacts\indexes\l21_numpy
```

Search bằng index đã lưu:

```powershell
python tools\vector_search.py --index-dir artifacts\indexes\l21_numpy --query-vector artifacts\query\L21_V001_001.npy --groups L21 --require-keyframes --top-k 5 --output artifacts\registry\sample_search_results_from_index.json
```

Search debug trực tiếp bằng video/keyframe, không cần tự tạo query `.npy`:

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-from-video-id L21_V001 --query-keyframe-id 1 --groups L21 --require-keyframes --top-k 5 --output artifacts\registry\sample_search_results_from_debug_query.json
```

Search có grouping/diversity theo video:

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-from-video-id L21_V001 --query-keyframe-id 1 --groups L21 --require-keyframes --candidate-pool 20 --top-k 5 --max-frames-per-video 1 --group-by-video --output artifacts\registry\sample_search_results_grouped.json
```

Search bằng text query CLIP:

```powershell
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-text "a person walking" --groups L21 --require-keyframes --candidate-pool 20 --top-k 5 --max-frames-per-video 1 --group-by-video --output artifacts\registry\sample_search_results_text.json
```

Máy hiện tại có sẵn snapshot CLIP ở `D:\AIC`. Có thể chạy offline/local-only bằng cách đặt biến môi trường:

```powershell
$env:AIC_CLIP_MODEL_ID="D:\AIC\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
python tools\vector_search.py --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --query-text "a person walking" --clip-local-files-only --groups L21 --require-keyframes --candidate-pool 20 --top-k 5 --max-frames-per-video 1 --group-by-video
```

Không hard-code đường dẫn `D:\AIC` vào code. Thành viên khác khi pull repo có thể dùng model id mặc định `openai/clip-vit-base-patch32`; nếu chưa có cache local thì HuggingFace sẽ tải model khi chạy text query.

Chạy benchmark text query L21:

```powershell
python tools\text_query_benchmark.py --queries benchmarks\text_queries_l21.json --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --groups L21 --require-keyframes --candidate-pool 25 --top-k 5 --max-frames-per-video 1 --output artifacts\benchmarks\l21_text_benchmark.json --csv-output artifacts\benchmarks\l21_text_benchmark.csv
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

Tạo benchmark mở rộng cho các query yếu q008/q009/q010 với raw candidate pool 50:

```powershell
python tools\text_query_benchmark.py --queries benchmarks\text_queries_l21_weak.json --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --groups L21 --require-keyframes --candidate-pool 50 --top-k 5 --max-frames-per-video 1 --csv-result-set raw_results --output artifacts\benchmarks\l21_weak_text_benchmark_pool50.json --csv-output artifacts\benchmarks\l21_weak_text_benchmark_pool50.csv
python tools\benchmark_review_html.py --input artifacts\benchmarks\l21_weak_text_benchmark_pool50.json --result-set raw_results --output artifacts\benchmarks\l21_weak_text_review_pool50.html
```

Mở `artifacts\benchmarks\l21_weak_text_review_pool50.html` để chấm 150 candidates rồi export CSV. Fusion experiment chỉ nên chạy lại sau khi có file judged pool50.

## Quy Tắc Làm Việc

- Luôn đọc report mới nhất trước khi làm.
- Sau task đáng kể, tạo report snapshot mới trong `reports/`.
- Không commit `data/` hoặc `artifacts/`.
- Không tải model/dữ liệu lớn nếu chưa được duyệt.
- Không cài dependency nặng như FAISS hoặc OpenCLIP nếu chưa được duyệt.
- Không xử lý video gốc vì `data/videos` chưa có.
- Không đánh dấu phase/module là DONE nếu chưa có bằng chứng test/validation.

## Bước Tiếp Theo Đề Xuất

Tiếp tục Phase 1 bằng chuẩn bị text query:

1. Chấm thủ công `artifacts\benchmarks\l21_text_benchmark.csv`.
2. Cải thiện schema/manifest dựa trên lỗi quan sát được.
3. Sau khi có thêm keyframes, mở search ra ngoài L21.

Chỉ sau đó mới quyết định có cần FAISS hay không.
