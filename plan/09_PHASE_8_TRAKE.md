# Phase 8 - TRAKE

## Mục tiêu

Tìm đúng video chứa chuỗi event và chọn semantic keyframe cho từng event theo đúng thứ tự thời gian.

## Phạm vi khuyến nghị

- Prototype trên L21 trước vì có keyframes.
- Frame refinement dày cần video gốc, nên chỉ chạy khi có `videos/L21`.
- Không xem keyframes L22-L30 là sẵn sàng cho TRAKE nếu chưa tải ảnh.

## Việc cần làm

- Tách truy vấn TRAKE thành các event con.
- Retrieve candidate cho từng event.
- Gom candidate theo video_id.
- Tìm chuỗi frame tăng dần theo thời gian.
- Tính video-level sequence score.
- Hiển thị TRAKE workspace với event list, candidate frame, timeline và cảnh báo sai thứ tự.
- Khi có video gốc, refine quanh candidate để chọn semantic frame tốt hơn.

## Đầu ra

- Event decomposition.
- Event-specific retrieval.
- Temporal sequence search.
- TRAKE workspace.
- Manual correction flow.

## Tiêu chí hoàn thành

- Không trộn frame từ nhiều video trong một chain.
- Chain candidate giữ đúng thứ tự thời gian.
- Người dùng có thể thay frame cho từng event.
- Nếu thiếu video gốc, hệ thống ghi rõ chỉ refine ở mức keyframe.

