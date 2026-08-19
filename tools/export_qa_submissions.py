from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_retrieval.qa_store import QaStore
from aic_retrieval.submission import (
    SubmissionCandidate,
    validate_candidates,
    validation_report,
    write_candidates_csv,
    write_candidates_jsonl,
)


def candidates_from_qa_store(store: QaStore, limit: int = 1000) -> list[SubmissionCandidate]:
    candidates: list[SubmissionCandidate] = []
    for row in store.list_exports(limit=limit):
        candidates.append(
            SubmissionCandidate(
                query_id=str(row["query_id"]),
                video_id=str(row["video_id"]),
                frame_id=int(row["frame_id"]),
                answer=str(row["answer"]),
                session_id=str(row["session_id"]),
                review_id=str(row["review_id"]),
                evidence_ids=tuple(str(item) for item in row["evidence_ids"]),
                source="qa_confirmed_export",
                metadata={
                    "export_id": row["export_id"],
                    "event_query": row["event_query"],
                    "question": row["question"],
                    "qa_export_schema_version": row["schema_version"],
                    **dict(row["metadata"]),
                },
            )
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Export confirmed Q&A reviews to the internal submission schema.")
    parser.add_argument("--qa-store", type=Path, default=ROOT / "artifacts" / "qa" / "phase7_qa.sqlite3")
    parser.add_argument("--output-jsonl", type=Path, default=ROOT / "artifacts" / "submissions" / "qa_submission_candidates.jsonl")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "artifacts" / "submissions" / "qa_submission_candidates.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "submissions" / "qa_submission_validation.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if not args.qa_store.is_file():
        raise SystemExit(f"Q&A store not found: {args.qa_store}")

    candidates = candidates_from_qa_store(QaStore(args.qa_store), limit=args.limit)
    issues = validate_candidates(candidates)
    report = validation_report(candidates, issues)
    write_candidates_jsonl(candidates, args.output_jsonl)
    write_candidates_csv(candidates, args.output_csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "candidates": len(candidates),
        "errors": report["error_count"],
        "warnings": report["warning_count"],
        "jsonl": str(args.output_jsonl),
        "csv": str(args.output_csv),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_error and report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
