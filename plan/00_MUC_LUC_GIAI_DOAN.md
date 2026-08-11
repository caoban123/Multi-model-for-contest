# Mục Lục Giai Đoạn AIC 2026 Video Retrieval

Tài liệu này tách lộ trình trong `PLAN_AIC2026_VIDEO_RETRIEVAL.md` thành các file giai đoạn nhỏ để dễ triển khai, theo dõi và nghiệm thu.

## Trạng thái dữ liệu hiện tại

Trong `data` hiện có 5 thư mục local:

- `clip-features-32`: feature CLIP theo video, có các nhóm L21 đến L30.
- `map-keyframes`: mapping keyframe sang frame/video, có các nhóm L21 đến L30.
- `media-info`: metadata JSON, có các nhóm L21 đến L30.
- `objects`: object detection theo video, có các nhóm L21 đến L30.
- `keyframes`: hiện chỉ có các thư mục `L21_V...`.

Thư mục thứ 6 dự kiến là `videos`, nhưng chưa đưa vào phạm vi ngay vì dung lượng lớn. Khi cần phát triển các phần phụ thuộc video gốc, ưu tiên làm với video `L21` trước, tương ứng với keyframes `L21` đã tải.

## Nguyên tắc chia phase

- Phase 0 đến Phase 4 có thể làm chủ yếu trên 5 thư mục hiện có.
- Các phần cần hiển thị ảnh keyframe phải giới hạn chắc chắn ở `L21` cho tới khi tải thêm keyframes.
- Các phần cần video gốc như ASR, refine frame dày, Q&A/TRAKE nâng cao chỉ thiết kế trước hoặc chạy thử khi có `videos/L21`.
- Không giả định keyframes đầy đủ cho L22-L30 dù feature, mapping, metadata và objects đã có.
- Không tải, giải nén hàng loạt, re-encode video, hoặc thêm model nặng nếu chưa có xác nhận riêng.

## Danh sách file phase

- [Phase 0 - Audit repository và data](./01_PHASE_0_AUDIT.md)
- [Phase 1 - Data registry và baseline retrieval](./02_PHASE_1_DATA_REGISTRY_BASELINE.md)
- [Phase 2 - UI retrieval cơ bản](./03_PHASE_2_UI_RETRIEVAL_CO_BAN.md)
- [Phase 3 - Video aggregation và diversity](./04_PHASE_3_VIDEO_AGGREGATION_DIVERSITY.md)
- [Phase 4 - Structured retrieval](./05_PHASE_4_STRUCTURED_RETRIEVAL.md)
- [Phase 5 - OCR và ASR](./06_PHASE_5_OCR_ASR.md)
- [Phase 6 - Reranking và query planner](./07_PHASE_6_RERANKING_QUERY_PLANNER.md)
- [Phase 7 - Q&A](./08_PHASE_7_QA.md)
- [Phase 8 - TRAKE](./09_PHASE_8_TRAKE.md)
- [Phase 9 - Submission và hardening](./10_PHASE_9_SUBMISSION_HARDENING.md)

