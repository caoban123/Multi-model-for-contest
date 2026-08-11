# Phase 2 - UI Retrieval Cơ Bản

## Mục tiêu

Cho người dùng nhập mô tả, xem kết quả và xác nhận `video_id/frame_id` mà không phải thao tác trực tiếp với file.

## Phạm vi khuyến nghị

- UI end-to-end trước mắt nên ưu tiên L21 vì đã có keyframes.
- Kết quả ngoài L21 có thể hiển thị metadata và thông tin frame, nhưng thumbnail cần trạng thái "chưa tải keyframe".
- Không phụ thuộc video gốc trong phase này.

## Việc cần làm

- Làm search box và chọn query type cơ bản.
- Hiển thị result grid gồm thumbnail, video_id, frame_id, score và nguồn dữ liệu.
- Hỗ trợ paging/lazy loading.
- Mở neighborhood quanh keyframe nếu ảnh L21 tồn tại.
- Cho phép pin result.
- Lưu search history trong phạm vi local.

## Đầu ra

- UI search cơ bản.
- Result grid.
- Keyframe viewer cho L21.
- Pin/search history.
- Trạng thái thiếu asset rõ ràng cho non-L21.

## Tiêu chí hoàn thành

- Người dùng mới có thể chạy query test và xem ảnh L21.
- UI không tải toàn bộ ảnh cùng lúc.
- Không crash khi candidate thuộc L22-L30 mà chưa có keyframe image.
- Có đường dẫn logic từ result tới submission candidate.

