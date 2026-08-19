# Phase 9 - Submission Hardening

Phase 9 keeps submission output independent from the final contest format. Until the organizer publishes the official schema, the project exports confirmed answers into an internal, validated schema.

## Internal Candidate Schema

Each candidate contains:

- `query_id`
- `video_id`
- `frame_id`
- `answer`
- `session_id`
- `review_id`
- `evidence_ids`
- `source`
- `schema_version`
- `metadata`

The current schema version is `aic-internal-submission-v1`.

## Export Confirmed Q&A Reviews

Confirmed or edited Q&A reviews can be exported from the local QA SQLite store:

```powershell
python tools\export_qa_submissions.py `
  --qa-store artifacts\qa\phase7_qa.sqlite3 `
  --output-jsonl artifacts\submissions\qa_submission_candidates.jsonl `
  --output-csv artifacts\submissions\qa_submission_candidates.csv `
  --report artifacts\submissions\qa_submission_validation.json
```

The tool writes:

- JSONL: machine-readable internal candidates.
- CSV: review-friendly table.
- Validation report: error/warning counts and issue list.

If there are no confirmed exports yet, the validation report contains `EMPTY_SUBMISSION`. This is expected until at least one Q&A review is confirmed and exported.

## Validation Rules

Current validation checks:

- duplicate `(query_id, video_id, frame_id)` rows are errors;
- missing evidence ids are warnings;
- missing answer text is a warning;
- an empty submission is an error.

These checks are intentionally stricter than a plain CSV export because contest-day mistakes are usually caused by duplicate rows, missing frame ids, or untraceable answers.

## Contest Format Adapter

When the official submission format is announced, add a separate adapter that reads `SubmissionCandidate` records and writes the organizer format. Do not couple BTC-specific columns directly to retrieval, Q&A, or UI code.

Suggested next adapter:

```text
internal SubmissionCandidate
-> official BTC CSV/JSON adapter
-> official validator smoke test
```

## Offline Checklist

- Keep `artifacts\registry\data_registry.json` available.
- Keep index directory available, for example `artifacts\indexes\l21_numpy`.
- Keep QA store if using Q&A exports: `artifacts\qa\phase7_qa.sqlite3`.
- Keep Gemini/API features optional; confirmed manual exports must remain readable without Internet.
- Run `python -m pytest -q` before packaging.
