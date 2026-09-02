# Phase 10B - Frame Localization, VLM Review and Index Upgrade

## Status - 01-09-2026

This plan extends `13_PHASE_10_QA_TRAKE_QUALITY_UPGRADE.md`. All new ranking behavior remains modular and opt-in until labelled development and holdout evidence supports promotion.

## Implemented

- Shared `FrameLocalizer` performs bounded Stage-B CLIP rescoring only inside Stage-A video candidates. It preserves video rank, searches around seed timestamps and returns a temporally diverse frame shortlist.
- Q&A `/submission` exposes Deep frame search, radius/limit controls, a per-video frame contact strip, full-size preview, nearby frames and a one-click frame-field helper.
- Optional Gemini frame review receives at most 12 system-generated images, validates an allowlist of candidate IDs and may reorder frames only within a video. It cannot add a video or frame.
- TRAKE has an opt-in Gemini multi-image verifier behind low-confidence, network and decision-log gates. Same-video and monotonic-order invariants remain server-owned; Gemini output is advisory only.
- `configs/phase10_trake_gemini_opt_in.json` extends the frozen Phase-8 V2 config. The default config remains offline and VLM-disabled.
- Object-store builder supports optional confidence threshold, per-label NMS and max detections per frame. Defaults preserve the current store exactly.
- `tools/phase10_upgrade_audit.py` compares Phase 5, corpus, BGE, BM25 and object manifests and emits reproducible long-running commands.

## Dense Frame Localization v2 - Verified Technical Milestone

The keyframe-only Stage-B path now has an opt-in decoded-video extension shared by KIS and Q&A, while TRAKE refinement uses the same registry-owned video resolver.

- Resolve source video only through the registry allowlist; external SSD paths and repository junctions are supported without broadening arbitrary file access.
- Start from retrieval timestamps, sample a bounded coarse window, score decoded frames with the existing CLIP encoder, resample small fine windows around the strongest anchors, and return a temporally diverse shortlist.
- Preserve exact decoder frame positions and timestamps in every result. Associate each decoded frame with its nearest known keyframe only as review context.
- Cache decoded shortlist images and metadata using a fingerprint over video stat, query, seeds, sampling config and model identity.
- Fall back to the existing keyframe result whenever the source video, cache identity, decoder, scorer or cache writer is unavailable.
- Expose decoded frames as a contact sheet with zoom, frame/timestamp details and existing nearby-frame review controls. Gemini remains optional and may only reorder the supplied allowlist.
- Keep the feature opt-in in the generic UI CLI. The checked competition profile enables it explicitly and places the cache on `E:/AIC2026/artifacts/cache/dense_frames`.

Technical verification on 01-09-2026:

- Full suite: `432 passed in 18.46s`.
- Static checks: `node --check web/submission_ui/app.js` and `python -m compileall -q src tools` passed.
- Real L21 smoke: `L21_V010`, query `a television news presenter`, decoded result frame `13630` at `545.2s`.
- The percent-encoded keyframe endpoint returned HTTP `200` and `163024` bytes.
- Repeated refinement reported `cache_hit=true`; the cached dense stage completed in about `1.024 ms`, with total candidate refinement about `112.147 ms`.

This verifies operation and fallback behavior, not retrieval accuracy. Default promotion still requires labelled development/holdout comparison against the keyframe-only baseline.

## Long Visual Narrative Routing v6 - Implemented 01-09-2026

Contest-style KIS descriptions frequently contain several visible events and incidental counts. The previous guardrail treated any digit and the standalone Vietnamese word `số` as lexical evidence. This incorrectly routed visual queries such as `5 people` through BM25, where unrelated OCR news captions could dominate RRF.

- Numeric counts do not enable BM25 by themselves.
- Exact OCR cues remain lexical: quoted text, signs, tables, addresses, countdowns, codes, explicit number/quantity questions and metadata fields.
- Sentence boundaries and temporal connectors now produce individual visual moments.
- Gemini may return at most six bounded CLIP prompts, enough for a primary prompt plus up to five event moments.
- Generic `person` is not auto-applied as an object constraint.
- Color constraints require an explicit visual context such as `màu vàng`, `áo đỏ` or `white car`; lexical compounds such as `bí đỏ` are not interpreted as red attributes.
- Gemini route suggestions remain visible in the trace, but deterministic intent guardrails remove unsupported BM25/BGE routes before execution.

Observed pre-fix evidence for `query-p2-1-kis`: the plan used `clip_bm25_rrf`; rank 1 `L22_V003` received BM25 rank 4 from unrelated OCR text containing `5 người thương vong`, while its CLIP rank was only 37. After the fix, the same local intent is CLIP-only with three event prompts plus the primary prompt. This is a routing regression fix, not a claim that the correct video is now rank 1.

Multi-query review also exposes event lanes without replacing RRF consensus. The first three fused videos remain first; up to two unique leaders from each Q1-Q6 lane are then round-robined into the bounded review window. The UI can filter existing candidate cards by lane without clearing operator selections or official-frame input. This is a recall/review feature, and its trace is returned under `channel_postprocessing.clip.variant_rescue`.

## Temporal Sequence Review Lane - Implemented 01-09-2026

Long visual narratives now receive a bounded temporal-coherence pass over the raw per-query CLIP hits. The broad Q1 scene prompt remains retrieval context; when at least three CLIP lanes exist, Q2-Q6 are treated as concrete ordered events.

- Group frame hits by video and visual event channel.
- Retain at most six frame hits per event/video to bound alignment work.
- Use dynamic programming to find the strongest monotonic chain with non-decreasing timestamps.
- Require at least two matched events; report coverage, span, rank evidence and the exact frame chosen for each event.
- Preserve the first three RRF consensus videos and inject at most two Sequence candidates before individual event-lane rescue.
- Expose `Sequence` filtering and `SEQ matched/total - span` evidence in `/submission`.
- Keep the feature advisory. It changes the bounded operator review order but does not replace CLIP, RRF, structured retrieval or official-frame verification.

Trace location: `channel_postprocessing.clip.variant_rescue.temporal_sequence`.

## Verified Evidence

- Full test suite: `407 passed in 9.91s`.
- TRAKE fixture regression: expected `[1, 8]`, actual `[1, 8]`, pipeline pass.
- Q&A development benchmark remains `unavailable`: zero labelled development queries. No accuracy improvement is claimed.
- Full-data audit: OCR `873/873`, BM25 v2 matches the OCR corpus, ASR `2/873`, BGE v2 missing, object baseline `100 detections/frame` with no filtering.

## Long-running Tasks

1. Label dense-v2 KIS/Q&A frame-localization development cases and compare hit-at-window against keyframe-only Stage B.
2. Run CLIP/BGE/BM25 independent benchmarks and RRF ablation before changing global retrieval defaults.
3. Build `l21_l30_objects_v2.sqlite` with `confidence=0.30`, per-label `NMS=0.50` and at most 30 detections/frame.
4. Benchmark structured retrieval before promoting object v2.
5. Acquire or extract full ASR. Current coverage is only 2/873 videos and must remain partial/unavailable elsewhere.

Run the audit to get the current commands:

```powershell
python tools\phase10_upgrade_audit.py --output artifacts\audits\phase10_upgrade_audit.json
```

## Promotion Gate

Do not promote BGE v2, object v2, Gemini VLM or a ranking default without labelled development evidence, ablation against the current baseline and holdout protection.
