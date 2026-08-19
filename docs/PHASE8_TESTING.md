# Phase 8 TRAKE — local testing and review

Phase 8 is scoped to L21 and works without external API/VLM. The core uses existing CLIP keyframe retrieval, same-video temporal alignment, manual review and an optional selected-event dense refinement. `configs/phase8_trake_v1.json` keeps P8.7 disabled.

## Start

```powershell
.\.venv-ocr\Scripts\python.exe tools\retrieval_ui.py `
  --registry artifacts\registry\data_registry_rebuild.json `
  --index-dir artifacts\indexes\l21_numpy_rebuild `
  --groups L21 --require-keyframes `
  --clip-cache-dir artifacts\models --clip-local-files-only `
  --phase5-store artifacts\phase5\phase5_store_full.sqlite `
  --trake-store artifacts\trake\phase8_trake.sqlite3
```

Open `http://127.0.0.1:8765/trake`. Do not use the older `l21_numpy` index because it has no keyframe paths.

## Required review flow

1. Enter a one-to-five-event query and review/edit the decomposition.
2. Retrieve per event, then align.
3. Verify all selected events belong to one video and `pts_time` is strictly increasing.
4. Optionally replace/lock a candidate or refine that selected event only.
5. Confirm manually; internal export is unavailable before a confirmed review.

## Tests

```powershell
python -m pytest tests\test_trake_schema.py tests\test_trake_candidates.py tests\test_trake_alignment.py tests\test_trake_workflow.py tests\test_trake_api.py tests\test_trake_store.py tests\test_trake_refinement.py tests\test_trake_benchmark.py tests\test_trake_ui_static.py -q
python -m pytest tests -q
node --check web\trake_ui\app.js
python -m compileall -q src tools tests
```

## Benchmark

The development and holdout files are deliberately unlabeled templates. Fill them by manual annotation. Freeze development configuration before evaluating holdout.

```powershell
python tools\phase8_benchmark.py --queries benchmarks\trake_development_v1.json --trake-store artifacts\trake\phase8_trake.sqlite3 --output artifacts\benchmarks\trake\development.json
python tools\phase8_benchmark.py --queries benchmarks\trake_holdout_v1.json --trake-store artifacts\trake\phase8_trake.sqlite3 --output artifacts\benchmarks\trake\holdout.json
```

Missing labels produce `null`/`unavailable`, never zero. The fixture benchmark validates plumbing only and is not a quality claim.
