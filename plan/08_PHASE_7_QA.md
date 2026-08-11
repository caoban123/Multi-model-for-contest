# Phase 7 - Q&A

## Mục tiêu

Trả lời câu hỏi dựa trên đoạn video/keyframe đã retrieve, luôn kèm evidence để người dùng xác nhận.

## Phạm vi khuyến nghị

- Làm workspace và luồng evidence trước với L21.
- Nếu chưa có video gốc, Q&A chỉ dựa trên keyframe, OCR, metadata và object; không tuyên bố có transcript đầy đủ.
- Khi có `videos/L21`, bổ sung ASR và frame refinement cho nhóm này trước.

## Việc cần làm

- Thiết kế Q&A workspace: candidate video, evidence frames, OCR/ASR nếu có, proposed answer, ô sửa tay.
- Chọn answer normalization strategy.
- Chỉ sinh answer khi retrieval confidence đủ cao hoặc người dùng chọn evidence.
- Lưu log query, evidence, answer và phiên bản.
- Export candidate theo schema submission dự kiến.

## Đầu ra

- Q&A workspace.
- Evidence selection.
- Proposed answer + manual edit.
- Answer normalization preview.
- Logging.

## Tiêu chí hoàn thành

- Mọi answer đều có evidence đi kèm.
- Hệ thống biết từ chối hoặc yêu cầu xác nhận khi confidence thấp.
- Không bịa transcript nếu chưa có ASR/video.
- Có thể chỉnh answer thủ công trước khi export.

