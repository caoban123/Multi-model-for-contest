# Phase 9 - Submission Và Hardening

## Mục tiêu

Chuẩn bị hệ thống sẵn sàng cho vòng sơ tuyển, có thể export, validate, chạy offline và phục hồi khi lỗi.

## Phạm vi khuyến nghị

- Submission adapter phải tách khỏi retrieval core vì format BTC có thể thay đổi.
- Offline mode là mặc định an toàn.
- Video full dataset chỉ đưa vào khi đã có quyết định lưu trữ rõ ràng.

## Việc cần làm

- Thiết kế submission schema nội bộ độc lập format BTC.
- Viết exporter adapter khi BTC công bố format cuối.
- Validate duplicate, missing field, frame_id và video_id.
- Ghi run log và version của index/model/config.
- Tạo runbook vận hành ngày thi.
- Kiểm tra backup dữ liệu/index.
- Kiểm tra chế độ chạy khi thiếu Internet.

## Đầu ra

- Submission adapter.
- Validation report.
- Reproducible setup notes.
- Offline mode checklist.
- Backup/recovery plan.
- Report draft.

## Tiêu chí hoàn thành

- Export đúng format sau khi BTC chốt schema.
- Có test end-to-end từ query tới submission candidate.
- Không phụ thuộc đường dẫn cá nhân.
- Có recovery plan cho index, registry và config.
- Có danh sách rõ dữ liệu chưa tải, đặc biệt là video và keyframes ngoài L21.

