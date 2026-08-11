# Phase 5 - OCR Và ASR

## Mục tiêu

Tìm kiếm được chữ trong khung hình và lời nói trong video.

## Phạm vi khuyến nghị

- OCR có thể bắt đầu với keyframes L21 đã tải.
- ASR cần video gốc, nên chỉ thiết kế adapter trước hoặc chạy thử khi có `videos/L21`.
- Không chạy OCR/ASR toàn bộ L22-L30 khi chưa có keyframes/video tương ứng.

## Việc cần làm

- Chọn OCR model và phạm vi chạy OCR.
- Tạo OCR output schema gắn với video_id, keyframe_id, frame_id, text, bbox, confidence.
- Index OCR text để search.
- Thiết kế ASR schema gắn timestamp với frame_id.
- Khi có `videos/L21`, chạy ASR thử trên một số video L21.
- Tích hợp OCR/ASR evidence vào result và timeline.

## Đầu ra

- OCR index cho L21.
- ASR design hoặc ASR pilot cho `videos/L21` nếu đã có.
- Search integration cho text/speech.
- Cấu hình bật/tắt module.

## Tiêu chí hoàn thành

- OCR query trả đúng keyframe L21 trên test set nhỏ.
- ASR không được coi là hoàn chỉnh nếu chưa có video gốc.
- Timestamp/frame mapping được ghi rõ, không suy đoán mơ hồ.

