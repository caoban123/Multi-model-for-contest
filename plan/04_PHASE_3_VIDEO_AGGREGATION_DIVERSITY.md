# Phase 3 - Video Aggregation Và Diversity

## Mục tiêu

Giảm trùng lặp kết quả, nâng ranking ở cấp video và giúp người dùng duyệt sâu trong một video.

## Phạm vi khuyến nghị

- Có thể tính aggregation trên toàn bộ feature/mapping hiện có.
- Timeline bằng ảnh chỉ hoàn chỉnh với L21.
- Không yêu cầu video gốc.

## Việc cần làm

- Group result theo video_id.
- Tính video-level score từ nhiều candidate frame.
- Giới hạn số frame tối đa mỗi video trong Top-K.
- Thêm diversity để tránh một scene chiếm toàn bộ kết quả đầu.
- Làm timeline/keyframe strip cho L21.
- Cho phép mở sâu vào video candidate qua các keyframe lân cận.

## Đầu ra

- Ranking cấp video.
- Result diversification.
- Timeline/keyframe strip cho L21.
- Cấu hình `max_frames_per_video`.

## Tiêu chí hoàn thành

- Top result không bị lấp bởi nhiều frame gần nhau của cùng một scene.
- Có thể xem các keyframe lân cận trong L21.
- Candidate ngoài L21 vẫn giữ thông tin video/frame nhưng báo rõ thiếu ảnh.

