# Phase 4 - Structured Retrieval

## Mục tiêu

Kết hợp object và metadata với CLIP để cải thiện truy vấn có điều kiện cụ thể.

## Phạm vi khuyến nghị

- Dùng `objects` và `media-info` hiện có cho L21-L30.
- Evidence ảnh chỉ hiển thị đầy đủ với L21.
- Chưa cần OCR/ASR hoặc video gốc.

## Việc cần làm

- Chuẩn hóa object labels và confidence.
- Tạo object filter theo label, số lượng và vị trí nếu dữ liệu hỗ trợ.
- Tạo metadata search theo title/channel/description/time nếu có.
- Kết hợp CLIP, object và metadata bằng rank fusion hoặc weighted fusion.
- Hiển thị evidence góp điểm cho từng result.
- Làm ablation CLIP-only so với CLIP + object/metadata.

## Đầu ra

- Object index/filter.
- Metadata index/filter.
- Hybrid retrieval.
- Evidence display.
- Ablation report.

## Tiêu chí hoàn thành

- Query object-specific tốt hơn CLIP-only trên test set nội bộ.
- Người dùng thấy vì sao result được xếp hạng cao.
- Có thể tắt từng modality để debug.

## Implementation Snapshot — 16-08-2026

- P4.0–P4.8: implemented and verified with deterministic baseline, audited/normalized object store, structured filters, RRF, evidence API and opt-in UI.
- P4.9: ablation implementation complete; 120 A–F runs and 48 development-only threshold runs produced machine-readable/reviewable artifacts.
- Relevance metrics remain unavailable because manual ground truth has not been completed.
- Default decision: **HYBRID OPT-IN**; Phase-3 CLIP `/api/search` remains unchanged/default.
- Final status: **IMPLEMENTATION COMPLETE — BENCHMARK / VERIFICATION PENDING**.

