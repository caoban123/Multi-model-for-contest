# Phase 10 - Nâng cấp chất lượng và workspace Q&A / TRAKE

## 1. Mục tiêu

Phase 10 nâng Q&A và TRAKE từ mức "workflow chạy được" thành hai chế độ có thể đo chất lượng, sửa kết quả nhanh trong lúc thi và chỉ promote thay đổi khi benchmark chứng minh có lợi.

Mục tiêu chính:

1. Tăng xác suất đưa đúng video và đúng vùng thời gian vào Top-K.
2. Tăng độ đúng của answer Q&A và chuỗi semantic keyframe TRAKE.
3. Đưa các khả năng review quan trọng ra `/submission` với trải nghiệm tương đương KIS.

Phase này không thay đổi format CSV/ZIP và không tự động submit. Người dùng vẫn xác nhận video, frame và answer cuối cùng.

## 2. Baseline đã audit

### 2.1 Q&A hiện tại

Đã có:

- Tách `event_query` và `question`.
- CLIP baseline và Agent Hybrid opt-in.
- Evidence typed theo keyframe, CLIP, object, attribute, metadata, OCR và ASR.
- Answerer local evidence-first và Gemini VLM option.
- Review, SQLite audit, export và import vào submission queue.
- Frame inspector, neighborhood và time-to-frame.

Điểm yếu xác nhận:

- `benchmarks/phase7_qa_queries_v1.json` chỉ có 4 template chưa gán video/frame/answer thật; chưa có quality baseline.
- Evidence bị giới hạn toàn cục theo modality (`max_evidence_per_modality=8`), dễ thiên về các candidate đầu.
- Có thể chọn evidence từ nhiều video cho một answer; submission lại lấy frame-level evidence đầu tiên. Đây là nguy cơ cross-video contamination.
- Q&A chưa render candidate card đầy đủ như KIS; metadata, representative frame, matched frames và provenance khó kiểm tra cùng lúc.
- Metadata evidence phụ thuộc cache `metadata_by_video` đã được nạp, chưa enrich chủ động cho mọi candidate.
- Answerer local chủ yếu xử lý metadata/object/OCR/ASR; visual, count và temporal thường chỉ yêu cầu review.
- Audit full-data ngày 28-08-2026 cho thấy OCR đã có 845.478 detection trên 873/873 video; ASR vẫn chỉ phủ 2/873 video.
- Confidence chưa được calibration bằng correctness thật.

### 2.2 TRAKE hiện tại

Đã có TRAKE V2:

- Rule-first planner, query variants và modality routing.
- Multi-channel event retrieval, adaptive Top-K và same-video grouping.
- Temporal feasibility, candidate windows và dense coarse-to-fine refinement.
- DANTE-inspired alignment, context/linkage và multimodal reranking.
- Backend update plan, replace candidate, lock event và refine frame.
- Review, SQLite store, export và submission adapter.
- Benchmark/ablation tool với metric theo timestamp.

Điểm yếu xác nhận:

- Development/holdout vẫn là template trống; metric quality phải là `null/unavailable`.
- `/submission` chưa đưa update plan, alternate candidate, replace/lock và refine ra UI.
- Scope quality vẫn chủ yếu L21; chưa audit khả năng refinement L21-L30.
- Dense refinement phụ thuộc video, FPS, decoder và scorer; impact chưa được đo.
- VLM verifier đang tắt và chưa có benchmark OFF/ON.
- Score weights là seed config, chưa tune development và freeze holdout.

## 3. Nguyên tắc

1. Không thay baseline nếu chưa có benchmark.
2. Tách candidate generation, reranking, answering/alignment và UI để ablation độc lập.
3. Evidence của một Q&A answer phải thuộc cùng video đã chọn.
4. Metadata là evidence cấp video; frame nộp phải đến từ frame-level evidence thật.
5. Missing OCR/ASR là `UNAVAILABLE/UNKNOWN`, không phải `NO_MATCH`.
6. Gemini/VLM chỉ rerank hoặc draft trên candidate đã retrieve, không tự tạo video/frame.
7. TRAKE giữ same-video và temporal order như hard invariant.
8. Tuning chỉ dùng development; holdout khóa đến lần đánh giá cuối.
9. Mỗi milestone có test, benchmark, report và rollback config.
10. KIS và official submission contract không được regression.

## 4. Kiến trúc mục tiêu

```text
Q&A
  event/question parser
    -> Stage A event/video retrieval
    -> Stage B question-aware frame/evidence retrieval
    -> per-video evidence pack
    -> local/VLM answer + citations + confidence
    -> human review -> CSV

TRAKE
  ordered event planner
    -> per-event multi-channel retrieval
    -> video/windows shortlist
    -> temporal feasibility + dense refinement
    -> DANTE alignment + context/linkage/VLM rerank
    -> human repair -> CSV
```

Shared layer:

- `ReviewCandidate` contract cho video/frame/metadata/provenance.
- Frame inspector, raw video, neighborhood, timeline và time-to-frame.
- Health/latency/fallback trace.
- Pin/compare/selection state theo Query ID.

## 5. Dependency và thứ tự

```text
M0 Audit + labels
  -> M1 Shared candidate/evidence contract
     -> M2 Q&A retrieval/evidence -> M3 Q&A answer quality
     -> M4 TRAKE planner/retrieval -> M5 TRAKE alignment/refinement
  -> M6 Competition UI
  -> M7 Agent + submission integration
  -> M8 Holdout + promotion decision
```

Không tune model/weight trước khi M0 có labels tối thiểu.

## 6. M0 - Data và annotation audit

### 6.1 Audit capability

- Số video/keyframe/video gốc usable theo L21-L30.
- Mapping FPS, frame index và timestamp monotonicity.
- Metadata title, author/channel, date, description, keywords.
- Object label coverage và confidence distribution.
- OCR/ASR coverage theo video, timestamp, ngôn ngữ và confidence.
- Video có ảnh nhưng thiếu video gốc hoặc mapping không hợp lệ.
- Sửa/regen benchmark JSON bị mojibake trước annotation.

Artifacts:

```text
artifacts/audits/phase10_data_capability.json
artifacts/audits/phase10_qa_modality_coverage.json
artifacts/audits/phase10_trake_refinement_coverage.json
```

### 6.2 Q&A annotation v2

- Development: tối thiểu 40 query.
- Holdout: tối thiểu 20 query.
- Có `OBJECT`, `VISUAL`, `COUNT`, `OCR`, `ASR`, `METADATA`, `TEMPORAL`.
- Mỗi query có expected video, accepted PTS/frame ranges, accepted answers và accepted modalities.
- Cho phép nhiều paraphrase đúng.
- Ghi `unanswerable_from_available_evidence` nếu dữ liệu local không đủ.

```json
{
  "query_id": "qa-dev-001",
  "event_query_vi": "...",
  "question_vi": "...",
  "question_type": "OBJECT",
  "expected_video_ids": ["L21_V001"],
  "accepted_pts_ranges": [[12.0, 16.5]],
  "accepted_frame_ranges": [[360, 495]],
  "expected_answers": ["điện thoại", "một chiếc điện thoại"],
  "required_modalities": ["keyframe", "object"],
  "unanswerable_from_available_evidence": false
}
```

### 6.3 TRAKE annotation v2

- Development: tối thiểu 30 query.
- Holdout: tối thiểu 15 query.
- Chuỗi 2-5 events; có visual, object/attribute, speech/text và mixed.
- Mỗi event có PTS interval; keyframe ID chỉ là debug.
- Ghi required/optional và gap constraint nếu query có nêu.

Gate M0:

- Annotation validator pass 100%.
- Ít nhất 80% development query answerable/retrievable từ data hiện có.
- Holdout khóa checksum và không dùng tuning.

## 7. M1 - Shared candidate/evidence contract

Contract additive:

```json
{
  "candidate_id": "...",
  "video_id": "L21_V001",
  "rank": 1,
  "representative_frame": {
    "keyframe_id": 12,
    "frame_idx": 360,
    "pts_time": 12.0,
    "image_url": "/keyframe?..."
  },
  "matched_frames": [],
  "metadata": {
    "title": "...",
    "author": "...",
    "publish_date": "...",
    "keywords": []
  },
  "retrieval": {
    "score": 0.0,
    "retriever_ranks": {},
    "contributions": {},
    "matched_text": []
  },
  "evidence": [],
  "warnings": []
}
```

Tasks:

- Enrich metadata deterministic cho mọi candidate.
- Chuẩn hóa `best_*` và frame-level field thành `representative_frame`.
- Trả nhiều matched frames theo video.
- Gắn provenance CLIP/BGE/BM25/object/attribute/OCR/ASR/metadata.
- Adapter tương thích KIS/Q&A/TRAKE và response v1.

Gate M1:

- Cùng candidate component render được ở cả ba task.
- Không còn metadata phụ thuộc cache truy cập trước.
- API contract/snapshot tests pass.

## 8. M2 - Q&A question-aware retrieval

### 8.1 Hai tầng retrieval

Tầng A tìm event/video:

- CLIP query tiếng Anh cho hình ảnh.
- BGE query giữ ngữ nghĩa tiếng Việt.
- BM25 giữ từ khóa gốc cho metadata/ASR/OCR.
- Structured object/attribute/metadata chỉ bật khi có constraint rõ.
- Video-level RRF sau các channel thành công.

Tầng B tìm evidence trả lời:

- `OBJECT/VISUAL/COUNT`: frame và neighborhood trong candidate video.
- `OCR`: OCR lexical + temporal frame mapping.
- `ASR`: transcript segment + timestamp.
- `METADATA`: field video-level, không tạo frame giả.
- `TEMPORAL`: multi-frame trước/trong/sau event.

### 8.2 Evidence pack theo video

- Group evidence theo `video_id`.
- Limit theo `(video_id, modality)`, không limit modality toàn cục.
- Chỉ answer trên một active video.
- Đổi active video phải reset selection/draft có chủ đích.
- Cấm confirm nếu evidence chứa nhiều video.
- Deduplicate cùng document/frame nhưng hợp nhất provenance.
- Thêm relevance score và temporal distance.

### 8.3 Multi-frame context

- Representative frame + matched frames + neighborhood nhỏ.
- Scene-aware sampling để tránh ảnh gần trùng.
- Count/action/temporal dùng before/current/after.
- Ghi rõ ảnh nào được gửi cho VLM.

Ablation:

1. CLIP-only.
2. CLIP + BGE.
3. CLIP + BM25.
4. CLIP + BGE + BM25 RRF.
5. Hybrid + structured.
6. Hybrid + question-aware Stage B.

Gate M2:

- Video Recall@10 không thấp hơn baseline.
- Evidence Recall@20 tăng trên development.
- Cross-video evidence rate bằng 0.
- Missing modality fail-open về channel khỏe.

## 9. M3 - Q&A answer quality

### 9.1 Typed answer plan

Planner trả schema ngắn:

- question type;
- expected answer type: number, color, object, name, phrase, metadata field;
- language;
- required modalities;
- multi-frame required hay không.

Rule-first parse; Gemini chỉ bổ sung và phải qua deterministic validation.

### 9.2 Answer pipeline

```text
selected same-video evidence
  -> deterministic extractors
  -> optional Gemini VLM
  -> citation allowlist validator
  -> type/length normalization
  -> confidence + abstain
  -> human review
```

Local upgrades:

- Metadata field aliases Việt/Anh.
- OCR/ASR span extraction có timestamp.
- Object/attribute vote trên nhiều frame.
- Count theo scene-aware frames, không cộng detection giữa frame.
- Synonym map cho color/object.
- Temporal answer dựa trên before/current/after.

Gemini VLM:

- Chỉ nhận active-video evidence.
- Nhận question, answer type, structured evidence và ảnh đã chọn.
- Trả `answer`, `evidence_ids`, `uncertainty`, `warnings`.
- Evidence IDs phải thuộc allowlist.
- Không đủ evidence thì answer rỗng.
- Timeout, retry tối đa 1, cache và local fallback.

Confidence features:

- rank/margin;
- modality availability;
- evidence agreement;
- local/Gemini agreement;
- answer type validity;
- citation coverage.

States: `SUPPORTED_HIGH`, `SUPPORTED_REVIEW`, `CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE`.

Metrics:

- Video Recall@1/5/10/20.
- Frame/PTS accuracy.
- Evidence Precision/Recall/F1.
- Exact/normalized exact/token F1/manual semantic accuracy.
- Unsupported/hallucinated answer rate.
- Review edit/reject rate.
- Latency và Gemini calls/query.

Gate M3:

- Không tăng hallucinated answer rate.
- Metric answer chính tăng trên development, không regression theo question type đủ mẫu.
- Answer confirm có same-video frame evidence, trừ metadata-only rule rõ ràng.
- Answer tối đa 100 ký tự trước submission.

## 10. M4 - TRAKE planner và candidate recall

### 10.1 Planner v3

Mỗi event có:

- text gốc;
- visual CLIP query tiếng Anh;
- Vietnamese lexical/semantic query;
- modalities;
- required/optional;
- object/attribute constraints;
- min/max/preferred gap nếu query nêu;
- anchor/entity continuity hints.

UI cho sửa, reorder, add/remove và required/optional trước search.

### 10.2 Candidate generation

- Multi-query variants theo event.
- Adaptive Top-K theo margin/distinctiveness.
- Group video sớm theo event coverage.
- Diversity theo scene/time bucket.
- BGE/BM25 cho event lời nói/OCR/metadata; CLIP là visual baseline.
- Object/attribute support/rerank, không hard-filter khi unknown.
- Cache theo event plan fingerprint.

### 10.3 Video shortlist

Rerank bằng:

- required event coverage;
- per-event best rank;
- modality support;
- temporal feasibility;
- entity/context continuity;
- penalty nếu mọi event dồn cùng timestamp.

Gate M4:

- Event Recall@100 và Video Recall@20 không thấp hơn V2.
- Correct Video Top-5 tăng trên development.
- Không bỏ video đúng do modality unavailable.

## 11. M5 - TRAKE alignment/refinement/verification

### 11.1 Alignment

- Giữ DANTE V2 baseline.
- Beam giữ chain đa dạng.
- Hard same-video và monotonic PTS.
- Optional skip có penalty rõ.
- Gap penalty dùng query constraint khi có.
- Linkage chỉ hard-gate khi evidence available.

### 11.2 Coarse-to-fine refinement

- Refine Top-N chain/video, không toàn corpus.
- Dùng raw video trong candidate windows.
- Seed config coarse 1 FPS, fine 8 FPS; chỉ tune development.
- Ghi decoded/requested PTS, seek error và FPS.
- Cache dense assets/embeddings.
- UI refine từng event quanh ±1/3/5/10 giây.

### 11.3 Conditional VLM verifier

- Chỉ thử sau khi baseline đã đo.
- Chạy Top-3 chain hoặc margin thấp.
- Nhận event text + event images + adjacent context.
- Trả support/consistency, không tạo candidate/frame mới.
- Không được vượt hard temporal gate.
- Ablation OFF/ON có cost, latency và failure.

Metrics dùng tool hiện có:

- Event Recall@20/50/100.
- Video Recall@5/10/20.
- Correct Video Top-1/Top-5.
- Chain success rate và temporal order accuracy.
- Mean/median timestamp error.
- Event within 1s/3s/5s.
- Stage latency p50/p95.

Gate M5:

- Chain success và Event within 3s tăng development.
- Temporal order pass 100% cho exported chain.
- Holdout chỉ chạy sau config freeze.
- VLM chỉ promote nếu tăng quality đủ bù latency/cost.

## 12. M6 - Competition UI cho Q&A

Q&A dùng result card cùng chuẩn KIS:

- ảnh lớn, rank, score;
- video ID, keyframe ID, frame index, timestamp;
- title, author/channel, date, keywords;
- retriever ranks/contributions và matched text;
- Open video, zoom, copy;
- neighborhood 3/6/12/20;
- time-to-frame;
- matched frames/timeline;
- pin và compare candidate.

Ba tab:

1. `Candidates`: Top video cards và active video.
2. `Evidence`: evidence active video, group modality/time.
3. `Answer`: local/Gemini draft, citations, confidence, warning và final answer.

Hành vi bắt buộc:

- Đổi candidate video không giữ âm thầm evidence/answer cũ.
- Metadata hiển thị cấp video cạnh representative frame.
- Click evidence mở frame/timeline.
- Có `Use this frame` cho official frame.
- Compare 2-3 candidate video.
- Hiển thị OCR/ASR coverage/unavailable.

## 13. M6 - Competition UI cho TRAKE

Workspace:

- Editable event plan.
- Video shortlist.
- Timeline lanes theo event.
- Chain score/diagnostics.
- Sticky selected-chain submission dock.

Tính năng:

- Sửa/reorder/add/remove event.
- Xem query variants/modalities.
- So sánh Top-N chain.
- Alternate candidate drawer từng event.
- Replace, lock/unlock event.
- Refine quanh timestamp với radius chọn được.
- Zoom, neighborhood, raw video.
- Gap/order warnings và video metadata.
- Copy frame/time, time-to-frame và undo.

Người thi phải sửa được một event sai mà không chạy lại toàn bộ query.

## 14. M7 - Agent và submission integration

- Agent chọn profile; retriever thực thi.
- Q&A/TRAKE dùng shared candidate contract.
- Hiển thị Gemini raw/parsed/validated trace.
- Persist corrections, locks, refinements và evidence selection.
- CSV chỉ lấy human-confirmed video/frame/answer.
- Giữ multi-query queue và multi-CSV workflow.
- VLM không tự đi thẳng vào queue.

Gate M7:

- End-to-end Q&A/TRAKE đến CSV pass.
- Reopen server khôi phục review state.
- Không regression KIS, CSV/ZIP và manual official frame gate.

## 15. M8 - Ablation và promotion

### Q&A matrix

| ID | Retrieval | Answerer | Multi-frame |
|---|---|---|---|
| Q0 | CLIP | local | no |
| Q1 | Hybrid | local | no |
| Q2 | Hybrid question-aware | local | yes |
| Q3 | Hybrid question-aware | Gemini | yes |
| Q4 | Q3 + confidence gate | Gemini/local | yes |

### TRAKE matrix

- T0: V2 baseline.
- T1: planner v3.
- T2: diversity/video shortlist.
- T3: query-aware gaps/linkage.
- T4: expanded dense refinement.
- T5: conditional VLM verifier.

Promotion rule:

- Tune development, freeze model/config/prompt/checksum, rồi mới chạy holdout.
- Promote khi metric chính tăng hoặc non-inferior và safety không giảm.
- Nếu chỉ tốt cho một query type thì giữ route-specific opt-in.
- Report quality, latency, memory, API calls và failure rate.

## 16. Test strategy

Sau mỗi task:

- Unit test schema/parser/scoring.
- API additive contract test.
- Fixture test không dùng làm quality claim.
- Missing-modality regression.
- Persistence/reopen test.
- Browser screenshot desktop/mobile.
- Full suite trước khi chuyển milestone.

Q&A bắt buộc test:

- Evidence không lẫn video.
- Metadata luôn enrich.
- Unicode/answer tối đa 100 ký tự.
- Gemini citation ngoài allowlist bị loại.
- OCR/ASR unavailable không sinh claim.
- Đổi active video reset draft.

TRAKE bắt buộc test:

- Same-video và monotonic order.
- Replace/lock/refine giữ đúng chain identity.
- Optional skip đúng penalty.
- Dense fallback không xóa coarse chain.
- Config fingerprint/cache invalidation.
- VLM không tạo candidate/frame mới.

## 17. Resource modes

### Thi local/offline

- CLIP + BGE + BM25 + structured stores local.
- Q&A local answerer luôn chạy.
- TRAKE V2/dense refinement local.
- Gemini tắt vẫn dùng được workflow.

### Có Gemini

- Planner/answer/verifier bounded và cached.
- Timeout ngắn, retry tối đa 1.
- Hiển thị model/latency/fallback.
- Không log API key.

### Benchmark

- Seed/config/model fingerprint cố định.
- Strict availability.
- Không thao tác tay giữa run.
- Ghi JSON, Markdown summary và stage latency.

## 18. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Không có labels thật | M0 bắt buộc trước tuning |
| OCR/ASR quá thấp | Bổ sung data hoặc route `UNAVAILABLE` |
| Evidence lẫn video | Active-video hard gate |
| Gemini hallucination | Citation allowlist + abstain + review |
| Dense refine chậm | Top-N windows + cache + refine thủ công |
| TRAKE overfit | Freeze config, holdout checksum |
| Network/API fail | Optional, timeout, local fallback |
| UI quá nặng | Tabs/drawer, keyboard, stable layout |
| Mở L22-L30 quá sớm | Per-group capability audit |

## 19. Thay đổi lớn cần phê duyệt riêng

- Tải/fine-tune OCR, ASR, second visual encoder hoặc VLM mới.
- Bật Gemini/VLM verifier mặc định.
- Trích dense frames toàn bộ video.
- Promote Hybrid/VLM toàn cục.
- Mở TRAKE ngoài group đã pass audit.
- Thay SQLite/FAISS bằng vector DB khác.

## 20. Milestone ưu tiên

### 10A - Foundation và UX parity

1. Audit data/OCR/ASR/refinement.
2. Sửa benchmark Unicode và annotation v2.
3. Metadata enrichment deterministic.
4. Shared candidate contract.
5. Q&A cards giống KIS.
6. Active-video evidence gate.
7. TRAKE plan editor + replace/lock/refine UI.

10A chưa tuyên bố accuracy tăng; nó loại lỗi kiến trúc và giúp người dùng sửa kết quả đúng cách.

### 10B - Measured retrieval

1. Hoàn thành development labels.
2. Q&A two-stage retrieval.
3. TRAKE planner/candidate recall.
4. Ablation và tune development.

### 10C - Answer/alignment

1. Typed Q&A answerer + multi-frame VLM option.
2. TRAKE alignment/refinement tuning.
3. Conditional VLM experiment.
4. Freeze config và chạy holdout.

### 10D - Competition hardening

1. Browser acceptance.
2. Latency/memory/API budget.
3. Failure drills.
4. End-to-end CSV/ZIP rehearsal.

## 21. Definition of Done

- Có Q&A và TRAKE development/holdout labels thật.
- Baseline/ablation có artifact tái lập.
- Q&A không dùng cross-video evidence.
- Q&A candidate/evidence/metadata/frame đạt feature parity với KIS.
- TRAKE sửa được plan và từng event candidate trong `/submission`.
- Holdout quyết định quality gate, không dựa vài query thủ công.
- Gemini/VLM optional và có fallback.
- KIS, official frame gate, CSV/ZIP và full suite không regression.
- Mọi issue/config/benchmark/promote/rollback được ghi vào report Phase 10.

## 22. Trạng thái triển khai 10A - 22-08-2026

Đã hoàn thành và có test:

- Audit L21-L30 tạo ba artifact capability cho 873 video; mapping monotonic pass.
- Benchmark Q&A v1 được sửa Unicode; annotation Q&A v2 có template và validator.
- Q&A candidate được enrich metadata, media URL và provenance chủ động.
- Evidence quota đổi từ toàn cục sang từng `(video_id, modality)`.
- Draft, review và submission import đều chặn evidence trộn nhiều video.
- `/submission` Q&A có candidate card, zoom, raw video, metadata, neighborhood và active-video review.
- `/submission` TRAKE có sửa text/required, thêm/xóa/đổi thứ tự event, rerun plan, alternate candidate, replace + lock và bounded refine.
- KIS/CSV/ZIP baseline giữ nguyên; full suite `376 passed`.

Chưa thể đánh dấu quality DONE:

- Q&A development/holdout và TRAKE development/holdout chưa có nhãn thật.
- OCR consolidated đã được map vào 146.036 source frame và tạo 845.478 detection searchable trên 873/873 video; issue thiếu OCR đã được giải quyết ở mức data/runtime.
- ASR vẫn chỉ phủ 2/873 video vì gói consolidated L21-L30 không chứa ASR; route ASR phải giữ trạng thái partial/unavailable theo coverage.
- Q&A two-stage Stage B, tuning TRAKE và VLM ablation thuộc 10B/10C, chỉ bắt đầu sau khi development labels hợp lệ.
- Không promote Hybrid/VLM hoặc thay default từ kết quả fixture; current quality metrics vẫn `null/unavailable`.

## 23. Trạng thái tích hợp OCR consolidated - 28-08-2026

Đã hoàn thành và có kiểm chứng:

- Adapter nhận cả schema timestamp L21-L26 và schema frame-number L27-L30.
- Source OCR được map sang retrieval refs; bbox được normalize và provenance nguồn được giữ trong SQLite.
- Build atomic chỉ promote store sau `integrity_check`, OCR/ASR FTS parity và source-count validation.
- Store full-data tại `E:\AIC2026\artifacts\phase5\l21_l30_phase5.sqlite3` có 845.478 OCR detection, 563 ASR segment, integrity `ok`.
- Corpus text v2 có 292.488 document; BM25 v2 có cùng 292.488 FTS row và integrity `ok`.
- Agent, structured retrieval, Q&A và TRAKE kiểm tra availability riêng cho OCR/ASR; lexical OCR giữ query tiếng Việt, không dùng CLIP query tiếng Anh.

Giới hạn còn lại:

- 85/146.121 source OCR frame không có retrieval ref trong ngưỡng 5 giây và được giữ là `UNAVAILABLE`.
- Logo/channel text lặp lại có thể lấn át BM25; cần benchmark temporal/video dedup hoặc logo penalty trước khi đổi ranking mặc định.
- BGE hiện tại vẫn index corpus cũ 173.393 document; chỉ rebuild BGE v2 bằng GPU sau khi có benchmark/ablation rõ ràng.
- P10-003 `OCR contributes no usable evidence`: `RESOLVED`.
- P10-004 `ASR coverage is 2/873 videos`: `BLOCKED`.

## 24. Competition runtime profile - 28-08-2026

OCR/BM25 v2 không còn chỉ là artifact thử nghiệm. Profile `configs/competition_l21_l30.json` và launcher `tools/competition_ui.py` áp dụng trực tiếp cấu hình thi:

- CLIP full L21-L30 là visual baseline.
- BGE corpus hiện tại phục vụ semantic text.
- BM25 v2 và Phase 5 FTS phục vụ lexical/OCR tiếng Việt.
- Object, metadata, Q&A, TRAKE và submission SQLite cùng được nạp.
- Hybrid/RRF được bật nhưng vẫn route theo intent; không ép mọi query qua tất cả retriever.
- Preflight chặn startup nếu thiếu model local, registry/index/store hoặc manifest không đạt minimum/integrity.
- ASR partial, Gemini chưa cấu hình và BGE pre-OCR là warning có hiển thị, không bị che giấu.

Preflight thật trên máy thi hiện tại: `READY`, không có blocking error.

## 25. Multi-query visual retrieval and OCR noise control - 31-08-2026

Implemented and verified:

- Hybrid query-plan v5 emits one to four bounded English CLIP visual queries.
- Long narrative scenes are split by temporal events; trailing answer clauses are excluded from CLIP prompts.
- CLIP query variants are batch-encoded and combined by video-level RRF before the optional outer CLIP/BGE/BM25 RRF.
- Agent and Submission inspectors show Q1-Q4 and preserve per-query candidate provenance.
- BGE/BM25 candidate generation collapses identical OCR text per video and caps broad global OCR noise.
- Candidate-local refinement keeps distinct OCR hits inside the selected video instead of applying the global cap.
- External-SSD SHA-256 reads use bounded retry for transient Windows device errors.
- Baseline `/api/search`, index formats and official submission rules are unchanged.

Verification:

- Full suite: `427 passed`.
- Real L21 Agent smoke test: four visual queries, inner multi-query RRF, 29 review candidates.
- Report: `reports/Phase_10_31-08-2026_2.md`.

Remaining limits:

- No labelled accuracy benchmark was run by operator decision.
- OCR suppression reduces duplicate candidate noise; it does not repair OCR recognition errors.
- ASR coverage remains 2/873 videos.
