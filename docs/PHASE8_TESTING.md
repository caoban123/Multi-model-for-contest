# Phase 8 TRAKE V2 — local testing and review

Phase 8 is scoped to L21. TRAKE V2 uses coarse-to-fine retrieval, temporal feasibility filtering, bounded dense window expansion, DANTE-inspired ordered alignment, explicit reranking components, and manual review. VLM verification is disabled by default and requires an explicit network permission plus decision log before it can be enabled.

The production V2 configuration is `configs/phase8_trake_v2.json`. The legacy configuration remains available only for controlled comparison and store/session compatibility.

## Start

```powershell
.\.venv-ocr\Scripts\python.exe tools\retrieval_ui.py `
  --registry artifacts\registry\data_registry_rebuild.json `
  --index-dir artifacts\indexes\l21_numpy_rebuild `
  --groups L21 --require-keyframes `
  --clip-cache-dir artifacts\models --clip-local-files-only `
  --phase5-store artifacts\phase5\phase5_store_full.sqlite `
  --trake-store artifacts\trake\phase8_trake.sqlite3 `
  --trake-config configs\phase8_trake_v2.json
```

Open `http://127.0.0.1:8765/trake`. Do not use the older `l21_numpy` index because it has no keyframe paths. When `--clip-local-files-only` is active, the configured CLIP model must already exist in the local cache; otherwise the API returns the controlled `RETRIEVAL_UNAVAILABLE` domain error.

## Required review flow

1. Enter a one-to-five-event query and review/edit the rule-first V2 plan.
2. Retrieve per-event coarse candidates and inspect modality/query-variant provenance.
3. Align the coarse timeline, apply feasibility/window selection, and run bounded fine expansion where enabled.
4. Verify required events use one video and strictly increasing PTS; inspect score breakdown, warnings, and latency diagnostics.
5. Apply manual corrections if needed, then confirm or reject. Internal export remains unavailable before a confirmed review.

## Tests

```powershell
$trakeTests = Get-ChildItem tests -Filter 'test_trake*.py' | ForEach-Object FullName
python -m pytest $trakeTests -q
python -m pytest tests -q
node --check web\trake_ui\app.js
python -m compileall -q src tools tests
```

`tests/test_trake_e2e_v2.py` covers the controlled plan → search → feasibility/windows → coarse/fine expansion → DANTE alignment → reranking → validation → review → export → store reopen route. Static UI tests cover safe rendering and the V2 diagnostics surface. Browser verification of the real local server is still required before release; an unavailable local model is a controlled environment result, not a retrieval-quality result.

## Annotation and evaluation

The development and holdout files are deliberately unlabeled templates. Author and validate annotations using `docs/PHASE8_ANNOTATION_V2.md`; freeze the development configuration before evaluating holdout.

```powershell
python tools\phase8_validate_annotations.py `
  --annotations benchmarks\trake_development_v1.json `
  --trake-config configs\phase8_trake_v2.json

python tools\phase8_benchmark.py `
  --queries benchmarks\trake_development_v1.json `
  --trake-store artifacts\trake\phase8_trake.sqlite3 `
  --output artifacts\benchmarks\trake\development.json

python tools\phase8_profile.py `
  --trake-store artifacts\trake\phase8_trake.sqlite3 `
  --algorithm trake_v2 `
  --json-output artifacts\benchmarks\trake\profile_v2.json `
  --markdown-output artifacts\benchmarks\trake\profile_v2.md

python tools\phase8_ablation.py `
  --queries benchmarks\trake_development_v1.json `
  --trake-store artifacts\trake\phase8_trake.sqlite3 `
  --trake-config configs\phase8_trake_v2.json `
  --json-output artifacts\benchmarks\trake\ablation_v2.json `
  --markdown-output artifacts\benchmarks\trake\ablation_v2.md
```

Missing labels produce `null`/`unavailable`, never zero. The fixture benchmark validates pipeline plumbing only and is not a quality claim. Do not report quality improvements without labeled development/holdout evidence.
