"""Internal submission schema and validation independent of final contest format."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


SUBMISSION_SCHEMA_VERSION = "aic-internal-submission-v1"


@dataclass(frozen=True)
class SubmissionCandidate:
    query_id: str
    video_id: str
    frame_id: int
    answer: str | None = None
    session_id: str | None = None
    review_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    source: str = "manual_review"
    schema_version: str = SUBMISSION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("query_id", "video_id", "source", "schema_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value.strip())
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.answer is not None:
            answer = self.answer.strip()
            if not answer:
                raise ValueError("answer must not be empty when provided")
            object.__setattr__(self, "answer", answer)
        if self.schema_version != SUBMISSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported submission schema version: {self.schema_version}")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    row_index: int | None = None


def validate_candidates(candidates: Iterable[SubmissionCandidate]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen: dict[tuple[str, str, int], int] = {}
    rows = list(candidates)
    for index, candidate in enumerate(rows):
        key = (candidate.query_id, candidate.video_id, candidate.frame_id)
        if key in seen:
            issues.append(ValidationIssue("ERROR", "DUPLICATE_CANDIDATE", f"duplicate query/video/frame also appears at row {seen[key]}", index))
        else:
            seen[key] = index
        if not candidate.evidence_ids:
            issues.append(ValidationIssue("WARNING", "NO_EVIDENCE_IDS", "candidate has no linked evidence ids", index))
        if candidate.answer is None:
            issues.append(ValidationIssue("WARNING", "NO_ANSWER", "candidate has no answer text", index))
    if not rows:
        issues.append(ValidationIssue("ERROR", "EMPTY_SUBMISSION", "no submission candidates found", None))
    return issues


def write_candidates_jsonl(candidates: Iterable[SubmissionCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")


def write_candidates_csv(candidates: Iterable[SubmissionCandidate], path: Path) -> None:
    rows = [asdict(candidate) for candidate in candidates]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("query_id", "video_id", "frame_id", "answer", "session_id", "review_id", "evidence_ids", "source", "schema_version", "metadata")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row["evidence_ids"] = "||".join(row["evidence_ids"])
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True, default=str)
            writer.writerow(row)


def validation_report(candidates: list[SubmissionCandidate], issues: list[ValidationIssue]) -> dict[str, Any]:
    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "candidate_count": len(candidates),
        "error_count": sum(1 for issue in issues if issue.severity == "ERROR"),
        "warning_count": sum(1 for issue in issues if issue.severity == "WARNING"),
        "issues": [asdict(issue) for issue in issues],
    }
