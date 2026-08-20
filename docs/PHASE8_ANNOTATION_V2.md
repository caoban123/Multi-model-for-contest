# Phase 8 TRAKE V2 annotation guide

This guide creates development/holdout ground truth without inferring labels from system output. Synthetic fixture labels must never be merged into the quality dataset.

## Required split policy

1. Annotate and tune only the development split.
2. Freeze the complete V2 config and record its SHA-256 fingerprint.
3. Only then annotate/evaluate holdout.
4. Never change retrieval, gap, DANTE, context, linkage, or rerank parameters from holdout results.

## Query categories

Use one or more of:

- `simple_visual`
- `2_event`, `3_event`, `4_event`, `5_event`
- `generic_to_distinctive`, `distinctive_to_generic`
- `OCR`, `ASR`, `object`, `attribute_color`
- `short_gap`, `medium_gap`, `long_gap`
- `same_scene`, `cross_scene`
- `ambiguous`, `OOD_rare_entity`

## Annotation procedure

1. Write the natural-language query independently of TRAKE output.
2. Identify the expected L21 video by manual video inspection.
3. Split the query into 1–5 ordered events and mark each as required or optional.
4. For every required event, seek the source video and record an inclusive PTS interval (`start_pts`, `end_pts`) in seconds. A single instant may use `representative_pts`.
5. Check that intervals are temporally monotonic. Overlap is allowed only when the intended query permits the same frame.
6. Record annotator, timestamp, category, ambiguity, and any boundary uncertainty in notes.
7. Have a second reviewer resolve ambiguous video/event boundaries before setting `annotation_status` to `LABELLED`.

Do not use keyframe ID distance as the primary timestamp label. Do not copy the algorithm’s selected timestamp into the ground truth without independent inspection.

## Validation and evaluation

Start from `benchmarks/trake_annotation_template_v2.json`, then validate:

```powershell
python tools\phase8_validate_annotations.py --annotations benchmarks\trake_development_v2.json --trake-config configs\phase8_trake_v2.json
```

Run the evaluator only after validation passes:

```powershell
python tools\phase8_benchmark.py --queries benchmarks\trake_development_v2.json --trake-store artifacts\trake\phase8_trake.sqlite3
```

An unlabelled template correctly produces `null` quality metrics. This is absence of evidence, not a score of zero.
