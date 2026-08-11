# Phase 6 - Reranking Và Query Planner

## Mục tiêu

Cải thiện Top-1 và Top-5 bằng query decomposition, fusion tuning và reranking.

## Phạm vi khuyến nghị

- Rerank trên candidate nhỏ, không chạy model nặng trên toàn bộ dataset.
- Query planner phải có fallback không cần Internet/API.
- Kết quả giải thích tốt nhất với L21 vì có ảnh để kiểm tra.

## Việc cần làm

- Tách query thành event, object, text, metadata và temporal hints.
- Sinh query variants nhưng cho phép người dùng xem và tắt.
- Tune trọng số fusion.
- Thêm reranker nhẹ trên Top-N candidate.
- Log contribution của từng modality.
- Benchmark Top-1/MRR trước và sau reranking.

## Đầu ra

- Query planner.
- Fusion config.
- Reranker Top-N.
- Explanation/logging.
- Benchmark comparison.

## Tiêu chí hoàn thành

- Top-1 hoặc MRR tăng trên benchmark nội bộ.
- Latency vẫn nằm trong mức dùng được.
- Tắt planner/reranker vẫn chạy được baseline.
- Không dùng LLM/API như điểm phụ thuộc bắt buộc.

