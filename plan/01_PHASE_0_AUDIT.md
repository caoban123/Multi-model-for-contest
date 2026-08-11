# Phase 0 - Repository Và Data Audit

## Mục tiêu

Hiểu chính xác repository và dữ liệu đang có trước khi viết bất kỳ feature nào.

## Phạm vi dữ liệu hiện tại

- Có 5 thư mục trong `data`: `clip-features-32`, `map-keyframes`, `media-info`, `objects`, `keyframes`.
- Chưa tính thư mục `videos`; đây là thư mục thứ 6 dự kiến nhưng để sau vì dung lượng lớn.
- `keyframes` hiện chỉ có `L21`, tương ứng các video `L21_V...`.
- Feature, mapping, metadata và objects có nhiều nhóm L21-L30, nhưng phần hiển thị ảnh thật chỉ chắc chắn làm được với L21.

## Việc cần làm

- Kiểm kê cây thư mục repository.
- Kiểm kê số lượng file theo từng nhóm L21-L30 trong mỗi loại dữ liệu.
- Kiểm tra format của một file `.npy`, một file mapping `.csv`, một file metadata `.json` và một thư mục object mẫu.
- Xác minh quan hệ `clip feature -> keyframe index -> image file -> video_id/frame_id`.
- Ghi rõ video nào có đủ bộ feature/mapping/metadata/object/keyframe.
- Tạo danh sách thiếu hụt, đặc biệt là keyframes ngoài L21 và toàn bộ video gốc.

## Đầu ra

- Repository inventory.
- Data inventory.
- Missing-data matrix theo video group.
- Dependency map cấp cao.
- Risk list.
- Decision list trước Phase 1.

## Tiêu chí hoàn thành

- Biết chính xác dữ liệu nào đang có, dữ liệu nào chưa có.
- Không nhầm keyframe index với frame ID.
- Biết các nhóm ngoài L21 có thể index nhưng chưa thể mở ảnh keyframe local nếu chưa tải thêm ảnh.
- Có quyết định rõ về phạm vi MVP: trước mắt ưu tiên L21 end-to-end.

