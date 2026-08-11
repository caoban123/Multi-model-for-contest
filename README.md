# Multi-model for AIC Video Retrieval

Repository này dùng để xây hệ thống truy xuất video cho AIC 2026. Dự án đang ở giai đoạn đầu: ưu tiên kiểm kê dữ liệu, xác minh mapping, dựng registry và baseline retrieval trước khi làm UI, Q&A, TRAKE hoặc agent.

## Đọc Gì Trước

Mọi thành viên và AI agent phải đọc theo thứ tự:

1. `.agent/codex.md` - protocol bắt buộc khi làm việc.
2. `reports/Phase_1_11-08-2026_2.md` - trạng thái mới nhất hiện tại.
3. `plan/PLAN_AIC2026_VIDEO_RETRIEVAL.md` - kế hoạch tổng thể.
4. File phase liên quan trong `plan/`, ví dụ `plan/02_PHASE_1_DATA_REGISTRY_BASELINE.md`.

Không bắt đầu sửa code chỉ dựa trên README. Report mới nhất là nguồn trạng thái vận hành chính.

## Trạng Thái Hiện Tại

Phase hiện tại: **Phase 1 - Data registry và baseline retrieval**.

Đã có:

- Scanner tạo data registry.
- Validator kiểm tra CLIP feature, mapping, metadata, object, keyframe.
- Baseline NumPy cosine search cho L21 bằng query vector `.npy`.
- Test tối thiểu bằng pytest.

Chưa có:

- Text encoder cho query ngôn ngữ tự nhiên.
- FAISS index.
- Persistent vector index.
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

Không cần cài FAISS, OpenCLIP, CLIP hoặc tải model ở thời điểm này.

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

## Quy Tắc Làm Việc

- Luôn đọc report mới nhất trước khi làm.
- Sau task đáng kể, tạo report snapshot mới trong `reports/`.
- Không commit `data/` hoặc `artifacts/`.
- Không tải model/dữ liệu lớn nếu chưa được duyệt.
- Không cài dependency nặng như FAISS hoặc OpenCLIP nếu chưa được duyệt.
- Không xử lý video gốc vì `data/videos` chưa có.
- Không đánh dấu phase/module là DONE nếu chưa có bằng chứng test/validation.

## Bước Tiếp Theo Đề Xuất

Tiếp tục Phase 1 bằng grouping/diversity và chuẩn bị text query:

1. Thêm grouping/diversity cơ bản theo video để Top-K không bị nhiều frame gần nhau chiếm hết.
2. Thêm tùy chọn export kết quả grouped theo video.
3. Quyết định text encoder local trước khi làm Textual KIS thật.

Chỉ sau đó mới quyết định có cần FAISS và text encoder hay không.
