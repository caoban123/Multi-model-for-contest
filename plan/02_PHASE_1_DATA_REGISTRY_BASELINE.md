# Phase 1 - Data Registry Và Baseline Retrieval

## Mục tiêu

Xây baseline tìm kiếm bằng CLIP feature, đặt nền cho toàn bộ hệ thống retrieval.

## Phạm vi khuyến nghị

- Index feature cho tất cả nhóm đang có trong `clip-features-32` nếu tài nguyên cho phép.
- Registry phải đánh dấu riêng `has_keyframe_image = true` chỉ cho `L21`.
- UI hoặc công cụ mở ảnh chỉ được coi là hoàn chỉnh với L21 cho tới khi tải thêm keyframes.

## Việc cần làm

- Thiết kế data registry mô tả video_id, feature path, mapping path, metadata path, object path, keyframe path.
- Validate shape của `.npy` so với số dòng mapping.
- Xây FAISS index hoặc baseline vector search tương đương.
- Encode text query bằng text encoder tương thích với CLIP ViT-B/32.
- Trả Top-K keyframe candidate kèm video_id, keyframe_id, frame_id và score.
- Có CLI hoặc test harness tối thiểu để chạy một query thử.

## Đầu ra

- Data registry có version.
- Mapping validation report.
- Baseline CLIP index.
- Search result schema.
- Latency benchmark thô.

## Tiêu chí hoàn thành

- Query trả kết quả Top-K ổn định.
- Mapping không lệch số dòng.
- Với kết quả L21, mở được đúng ảnh keyframe local.
- Với kết quả ngoài L21, hệ thống báo rõ thiếu ảnh keyframe thay vì lỗi im lặng.

