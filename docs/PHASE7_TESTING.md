# Phase 7 Testing

## Safety contract

- No evidence: no confirmed answer.
- No verified OCR/ASR source: no OCR/speech claim.
- Human review is mandatory before an internal export record is created.
- Raw answers are retained alongside normalization outputs.
- LVLM is deferred and must not be downloaded or enabled by this workflow.

## Milestone commands

Run the completed Phase 7 suite:

```powershell
python -m pytest tests/test_qa_schema.py tests/test_qa_evidence.py tests/test_qa_question_router.py tests/test_qa_answering.py tests/test_qa_normalization.py tests/test_qa_confidence.py tests/test_qa_store.py tests/test_qa_workflow.py tests/test_qa_api.py tests/test_qa_ui_static.py tests/test_qa_benchmark.py -q
```

The command is expanded as milestones are implemented. It must not be interpreted as a quality benchmark without manually annotated development and holdout labels.

Run a benchmark split without fabricating labels:

```powershell
python tools/phase7_benchmark.py --queries benchmarks/phase7_qa_queries_v1.json --split development --qa-store artifacts/qa/phase7_qa.sqlite3 --output artifacts/benchmarks/phase7/development.json
```

Use the same command with `--split holdout` only after development decisions are frozen. Empty expected fields correctly yield `availability: unavailable` and null quality metrics.
