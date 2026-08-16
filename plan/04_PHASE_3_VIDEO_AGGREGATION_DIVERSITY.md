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

## Thiết Kế Phase 3 Đã Chốt

Pipeline mặc định:

```text
text query
  -> CLIP frame retrieval
  -> candidate frame pool
  -> raw_results
  -> group by video_id
  -> max-score aggregation
  -> Top-K unique video_results
  -> representative frame
  -> matched frames
  -> existing Neighbors API / timeline
```

API visual search tách rõ:

- `raw_results`: candidate keyframe nguyên thứ hạng và score, dùng cho benchmark/debug.
- `results`: frame results đã diversity-filter, giữ để tương thích với Phase 2.
- `video_results`: kết quả chính của Phase 3, mỗi `video_id` xuất hiện một lần.
- `frames`: retrieval-matched frames thuộc video trong candidate pool, sắp theo score giảm dần.
- Neighbor frames: ngữ cảnh timeline theo `pts_time` tăng dần, không được coi là retrieval match.

`VideoResult` gồm `video_score`, best/representative frame, `frame_count` là tổng số match trong pool, `matched_frame_count` là số frame đính kèm và `frames`.

Default Phase 3:

```text
top_k_videos = 12
candidate_pool_size = 40
aggregation_method = max
max_frames_per_video = 1        # đường Frame Ranking tương thích
matched_frames_per_video = 5    # frame đính kèm trong Explore
neighbor_radius = 3
mean_top_n = 3                  # chỉ dùng khi chủ động chọn mean_top_n
```

Tie-break deterministic:

1. `video_score` giảm dần;
2. raw candidate rank của representative frame tăng dần;
3. `video_id` tăng dần.

Representative frame là frame score cao nhất; nếu bằng score thì giữ raw candidate rank sớm hơn. `mean_top_n` có extension point cho benchmark offline nhưng `max` vẫn là mặc định.

## UI Phase 3

- Visual search mặc định mở `Video Ranking`.
- `Frame Ranking` giữ raw Top-K keyframes để debug và so sánh duplicate.
- Video card hiển thị video score, representative keyframe, timestamp và matched-frame count.
- `Explore video` hiển thị matched frames; mỗi frame có thể mở timeline qua API `/api/neighborhood` hiện có.
- Matched frame và timeline neighbor được label riêng để tránh hiểu nhầm.
- Thiếu ảnh hiển thị `No image`; metadata/frame vẫn dùng được.

## Benchmark Phase 3

Chạy benchmark retrieval nguồn:

```powershell
$env:AIC_CLIP_MODEL_ID=(Resolve-Path "artifacts\models\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268").Path
python tools\text_query_benchmark.py --queries benchmarks\text_queries_l21.json --registry artifacts\registry\data_registry.json --index-dir artifacts\indexes\l21_numpy --groups L21 --candidate-pool 50 --top-k 12 --max-frames-per-video 1 --matched-frames-per-video 5 --aggregation-method max --clip-local-files-only --output artifacts\benchmarks\l21_phase3_source_pool50.json --csv-output artifacts\benchmarks\l21_phase3_source_pool50.csv
```

So sánh Frame Ranking và Video Ranking mà không encode lại query:

```powershell
python tools\phase3_benchmark.py --input artifacts\benchmarks\l21_phase3_source_pool50.json --judgements-csv artifacts\benchmarks\l21_phase3_source_pool50.csv --top-k 12 --candidate-pool 50 --matched-frames-per-video 5 --aggregation-method max --output artifacts\benchmarks\l21_phase3_comparison_pool50.json
```

Snapshot local ngày 16-08-2026, 10 query L21:

| Metric | Frame Ranking | Video Ranking |
|---|---:|---:|
| Mean Unique Video@12 | 0.65 | 1.00 |
| Mean duplicate count | 4.2 | 0.0 |
| Mean max results / same video | 3.4 | 1.0 |

Top frame video được giữ ở video rank 1 trên 100% query. Aggregation offline trung bình khoảng `0.07 ms/query`. Manual quality chưa có evidence vì CSV hiện có `0` judgement; không được suy diễn chất lượng relevance từ diversity metric.

## Trạng Thái Xác Minh Local

- Aggregation/API/tests/browser workflow: `VERIFIED`.
- L21 feature/mapping/index: 7.800 frame, 29 video, `VERIFIED`.
- L21 image timeline: `PARTIAL`; máy hiện chỉ có 18 ảnh thuộc một video, chưa đủ xác minh full L21 image experience.
- Non-L21 no-image behavior: `VERIFIED` bằng fixture; integration dữ liệu thật chưa đầy đủ trên máy.
- Manual relevance before/after: `NOT_STARTED`; cần chấm `Good/Partial/Bad` trước khi kết luận quality.

Phase 3 hiện ở trạng thái: **IMPLEMENTATION COMPLETE — INTEGRATION VERIFICATION PENDING**.

