# Phase 0 Audit Report

Ngày audit: 2026-08-11

## Tóm tắt

Repository hiện là repo planning-first cho hệ thống truy xuất video AIC 2026. Chưa có mã nguồn ứng dụng. Nội dung đang được commit lên GitHub gồm `.gitignore` và các tài liệu trong `plan`.

Thư mục `data` tồn tại local nhưng đã được ignore khỏi Git vì dung lượng lớn. Dữ liệu hiện có đủ 5 nhóm hỗ trợ chính: CLIP features, keyframe mapping, media info, objects và keyframes. Chưa có thư mục `videos`. Keyframes hiện chỉ có nhóm L21.

## Repository Inventory

Các mục tracked chính:

- `.gitignore`
- `plan/PLAN_AIC2026_VIDEO_RETRIEVAL.md`
- `plan/00_MUC_LUC_GIAI_DOAN.md`
- `plan/01_PHASE_0_AUDIT.md`
- `plan/02_PHASE_1_DATA_REGISTRY_BASELINE.md`
- `plan/03_PHASE_2_UI_RETRIEVAL_CO_BAN.md`
- `plan/04_PHASE_3_VIDEO_AGGREGATION_DIVERSITY.md`
- `plan/05_PHASE_4_STRUCTURED_RETRIEVAL.md`
- `plan/06_PHASE_5_OCR_ASR.md`
- `plan/07_PHASE_6_RERANKING_QUERY_PLANNER.md`
- `plan/08_PHASE_7_QA.md`
- `plan/09_PHASE_8_TRAKE.md`
- `plan/10_PHASE_9_SUBMISSION_HARDENING.md`

Chưa có:

- Backend source code.
- Frontend source code.
- Scripts tiền xử lý.
- Test suite.
- Config runtime.
- Index sinh ra từ feature.

## Data Inventory

Local `data` hiện có 5 thư mục:

- `data/clip-features-32`
- `data/map-keyframes`
- `data/media-info`
- `data/objects`
- `data/keyframes`

Chưa có:

- `data/videos`

Tổng số video/file-level item:

| Data type | Tổng item | Ghi chú |
|---|---:|---|
| CLIP features `.npy` | 873 | Có L21-L30 |
| Mapping `.csv` | 873 | Có L21-L30 |
| Media info `.json` | 873 | Có L21-L30 |
| Object folders | 873 | Có L21-L30 |
| Keyframe video folders | 29 | Chỉ có L21 |
| Keyframe images `.jpg` | 7800 | Chỉ có L21 |

Coverage theo nhóm:

| Nhóm | CLIP | Mapping | Metadata | Objects | Keyframes |
|---|---:|---:|---:|---:|---:|
| L21 | 29 | 29 | 29 | 29 | 29 |
| L22 | 31 | 31 | 31 | 31 | 0 |
| L23 | 25 | 25 | 25 | 25 | 0 |
| L24 | 43 | 43 | 43 | 43 | 0 |
| L25 | 88 | 88 | 88 | 88 | 0 |
| L26 | 498 | 498 | 498 | 498 | 0 |
| L27 | 16 | 16 | 16 | 16 | 0 |
| L28 | 24 | 24 | 24 | 24 | 0 |
| L29 | 23 | 23 | 23 | 23 | 0 |
| L30 | 96 | 96 | 96 | 96 | 0 |

## Format Mẫu

Mẫu kiểm tra: `L21_V001`.

CLIP feature:

- File: `data/clip-features-32/L21_V001.npy`
- Shape: `(307, 512)`
- Dtype: `float16`

Mapping:

- File: `data/map-keyframes/L21_V001.csv`
- Header: `n,pts_time,fps,frame_idx`
- Số dòng gồm header: 308
- Số dòng dữ liệu: 307

Keyframes:

- Folder: `data/keyframes/L21_V001`
- Số ảnh `.jpg`: 307
- Tên ảnh dạng `001.jpg`, `002.jpg`, ...

Objects:

- Folder: `data/objects/L21_V001`
- File mỗi keyframe dạng `001.json`, `002.json`, ...
- JSON chứa các trường như `detection_scores`, `detection_class_names`, `detection_class_entities`, `detection_boxes`, `detection_class_labels`.

Kết luận mẫu: với `L21_V001`, số vector CLIP, số dòng mapping và số ảnh keyframe khớp nhau. Quy tắc ánh xạ dự kiến là `n` trong mapping tương ứng ảnh keyframe zero-padded, ví dụ `n = 1` tương ứng `001.jpg`, và `frame_idx` là frame gốc trong video.

## Dependency Map Dự Kiến

```text
Raw local data
  -> Data registry
    -> Mapping validator
      -> CLIP index
        -> Baseline retrieval
          -> Video aggregation
            -> UI result grid
              -> Submission candidate list

Objects + metadata
  -> Structured indexes
    -> Hybrid fusion
      -> Evidence display

Keyframes L21
  -> Thumbnail/keyframe viewer
    -> Timeline/neighborhood UI

Videos L21, khi có
  -> ASR
  -> dense frame refinement
  -> Q&A/TRAKE nâng cao
```

## Assumptions

- `data/clip-features-32`, `data/map-keyframes`, `data/media-info`, `data/objects` dùng cùng danh sách video_id L21-L30.
- CLIP feature là 512 chiều, phù hợp CLIP ViT-B/32 theo tài liệu kế hoạch.
- Với keyframes đã giải nén, tên ảnh zero-padded theo cột `n` trong mapping.
- MVP nên ưu tiên L21 end-to-end vì chỉ L21 có ảnh keyframe local.
- L22-L30 có thể được index ở mức feature/mapping/metadata/object, nhưng chưa thể hiển thị ảnh keyframe nếu chưa tải thêm.

## Risks

| ID | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| R1 | Chưa có `videos`, không thể làm ASR/refine frame dày thật sự | Cao | Hoãn video-heavy work, chỉ thiết kế adapter trước |
| R2 | Keyframes chỉ có L21, UI visual ngoài L21 thiếu ảnh | Cao | Registry có cờ `has_keyframe_image`; hiển thị trạng thái thiếu asset |
| R3 | Metadata tiếng Việt có dấu hiệu mojibake khi đọc mẫu | Trung bình | Phase 1 kiểm tra encoding, chuẩn hóa đọc JSON |
| R4 | Chưa validate toàn bộ 873 cặp feature/mapping | Trung bình | Phase 1 viết validator bắt buộc trước khi index |
| R5 | Data local bị ignore khỏi Git, người khác clone repo sẽ thiếu data | Trung bình | Viết hướng dẫn layout data và checksum/manifest sau |
| R6 | Object labels dùng cả entity id và text label, cần normalize | Trung bình | Tạo object schema chuẩn trong Phase 4 |

## Decisions Cần Duyệt Trước Phase 1

- Backend dùng Python hay stack khác.
- Vector search dùng FAISS hay baseline NumPy trước.
- Registry lưu bằng JSON, SQLite hay DuckDB.
- Có index tất cả 873 video ngay hay chỉ index L21 trước.
- Text encoder CLIP chạy local bằng thư viện nào.
- Index sinh ra lưu ở đâu, ví dụ `artifacts/indexes` hoặc ngoài repo.
- Có cho phép tạo script validation trong Phase 1 hay vẫn chỉ planning.

## Tiêu Chí Phase 0

Đã đạt:

- Biết repo hiện chỉ có planning docs, chưa có code.
- Biết data hiện có 5 thư mục và thiếu `videos`.
- Biết coverage L21-L30 cho feature/mapping/metadata/object.
- Biết keyframes chỉ có L21.
- Kiểm tra mẫu `L21_V001` cho thấy feature, mapping và ảnh keyframe khớp số lượng.
- Xác định các rủi ro chính trước khi viết code.

Chưa làm trong Phase 0:

- Chưa validate toàn bộ 873 video.
- Chưa tạo registry.
- Chưa tạo index.
- Chưa chạy retrieval.
- Chưa xử lý encoding metadata.

## Khuyến Nghị Chuyển Phase 1

Phase 1 nên bắt đầu bằng data registry và validator, chưa vội làm UI. Thứ tự đề xuất:

1. Tạo cấu trúc project tối thiểu.
2. Tạo script scan local data thành registry.
3. Validate toàn bộ feature shape với mapping row count.
4. Đánh dấu `has_keyframes` chỉ cho L21.
5. Tạo baseline index nhỏ cho L21 trước.
6. Sau khi L21 chạy đúng, mở rộng index sang toàn bộ 873 video nhưng giữ trạng thái thiếu ảnh cho L22-L30.

