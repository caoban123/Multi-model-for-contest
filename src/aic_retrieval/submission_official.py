from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TASK_SUFFIX = {"KIS": "kis", "QA": "qa", "TRAKE": "trake"}
VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{3}$")
QUERY_ID_PATTERN = re.compile(r"^query-\d+-(kis|qa|trake)$")
MAX_PREDICTIONS = 100


@dataclass(frozen=True)
class OfficialPrediction:
    video_id: str
    frame_ids: tuple[int, ...]
    answer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", self.video_id.strip())
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))
        if self.answer is not None:
            object.__setattr__(self, "answer", self.answer.strip())


@dataclass(frozen=True)
class OfficialQuery:
    query_id: str
    task: str
    predictions: tuple[OfficialPrediction, ...]
    event_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", self.query_id.strip())
        object.__setattr__(self, "task", self.task.strip().upper())
        object.__setattr__(self, "predictions", tuple(self.predictions))

    @property
    def filename(self) -> str:
        suffix = TASK_SUFFIX.get(self.task, self.task.lower())
        return f"{self.query_id}.csv" if self.query_id.endswith(f"-{suffix}") else f"{self.query_id}-{suffix}.csv"


@dataclass(frozen=True)
class OfficialValidationIssue:
    severity: str
    code: str
    message: str
    row_index: int | None = None


def validate_official_query(query: OfficialQuery) -> tuple[OfficialValidationIssue, ...]:
    issues: list[OfficialValidationIssue] = []
    if query.task not in TASK_SUFFIX:
        issues.append(OfficialValidationIssue("ERROR", "UNSUPPORTED_TASK", f"unsupported task: {query.task}"))
        return tuple(issues)
    expected_suffix = TASK_SUFFIX[query.task]
    if not QUERY_ID_PATTERN.fullmatch(query.query_id) or not query.query_id.endswith(f"-{expected_suffix}"):
        issues.append(OfficialValidationIssue("ERROR", "INVALID_QUERY_ID", f"query_id must match query-N-{expected_suffix}"))
    if not 1 <= len(query.predictions) <= MAX_PREDICTIONS:
        issues.append(OfficialValidationIssue("ERROR", "INVALID_PREDICTION_COUNT", "each query requires 1-100 predictions"))
    if query.task == "TRAKE" and (query.event_count is None or query.event_count < 1):
        issues.append(OfficialValidationIssue("ERROR", "INVALID_EVENT_COUNT", "TRAKE requires a positive event_count"))

    seen: set[tuple[str, tuple[int, ...], str | None]] = set()
    for index, prediction in enumerate(query.predictions):
        if not VIDEO_ID_PATTERN.fullmatch(prediction.video_id) or prediction.video_id.endswith(".mp4"):
            issues.append(OfficialValidationIssue("ERROR", "INVALID_VIDEO_ID", f"invalid video_id: {prediction.video_id}", index))
        if not prediction.frame_ids or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in prediction.frame_ids):
            issues.append(OfficialValidationIssue("ERROR", "INVALID_FRAME_ID", "frame_id values must be non-negative integers", index))
        if query.task in {"KIS", "QA"} and len(prediction.frame_ids) != 1:
            issues.append(OfficialValidationIssue("ERROR", "INVALID_COLUMN_COUNT", f"{query.task} requires exactly one frame_id", index))
        if query.task == "TRAKE" and query.event_count is not None and len(prediction.frame_ids) != query.event_count:
            issues.append(OfficialValidationIssue("ERROR", "INVALID_COLUMN_COUNT", f"TRAKE requires exactly {query.event_count} frame_ids", index))
        if query.task == "TRAKE" and len(prediction.frame_ids) > 1 and any(left >= right for left, right in zip(prediction.frame_ids, prediction.frame_ids[1:])):
            issues.append(OfficialValidationIssue("WARNING", "FRAME_ORDER_FALLBACK", "official frame_ids are not strictly increasing; verify order using source timestamps", index))
        if query.task == "QA":
            if not prediction.answer:
                issues.append(OfficialValidationIssue("ERROR", "ANSWER_REQUIRED", "Q&A answer must not be empty", index))
            elif len(prediction.answer) > 100:
                issues.append(OfficialValidationIssue("ERROR", "ANSWER_TOO_LONG", "Q&A answer must contain at most 100 characters", index))
        elif prediction.answer is not None:
            issues.append(OfficialValidationIssue("ERROR", "UNEXPECTED_ANSWER", f"{query.task} does not accept an answer column", index))
        key = (prediction.video_id, prediction.frame_ids, prediction.answer)
        if key in seen:
            issues.append(OfficialValidationIssue("ERROR", "DUPLICATE_PREDICTION", "duplicate prediction row", index))
        seen.add(key)
    return tuple(issues)


def has_errors(issues: Iterable[OfficialValidationIssue]) -> bool:
    return any(issue.severity == "ERROR" for issue in issues)


def csv_row(task: str, prediction: OfficialPrediction) -> tuple[str | int, ...]:
    if task == "QA":
        return (prediction.video_id, prediction.frame_ids[0], prediction.answer or "")
    return (prediction.video_id, *prediction.frame_ids)


def write_official_csv(query: OfficialQuery, path: Path) -> None:
    issues = validate_official_query(query)
    if has_errors(issues):
        codes = ", ".join(issue.code for issue in issues if issue.severity == "ERROR")
        raise ValueError(f"official query validation failed: {codes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for prediction in query.predictions:
            writer.writerow(csv_row(query.task, prediction))


def render_official_csv(query: OfficialQuery) -> str:
    issues = validate_official_query(query)
    if has_errors(issues):
        codes = ", ".join(issue.code for issue in issues if issue.severity == "ERROR")
        raise ValueError(f"official query validation failed: {codes}")
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    for prediction in query.predictions:
        writer.writerow(csv_row(query.task, prediction))
    return handle.getvalue()


def build_submission_zip(queries: Iterable[OfficialQuery], output_dir: Path) -> Path:
    rows = tuple(queries)
    if not rows:
        raise ValueError("submission requires at least one query")
    filenames = [query.filename for query in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("submission filenames must be unique")
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for query in rows:
        write_official_csv(query, submission_dir / query.filename)
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in sorted(filenames):
            archive.write(submission_dir / filename, arcname=f"submission/{filename}")
    return zip_path


def validate_submission_zip(path: Path, expected_filenames: Iterable[str]) -> tuple[OfficialValidationIssue, ...]:
    expected = {f"submission/{name}" for name in expected_filenames}
    try:
        with zipfile.ZipFile(path) as archive:
            actual = {name for name in archive.namelist() if not name.endswith("/")}
    except (OSError, zipfile.BadZipFile) as exc:
        return (OfficialValidationIssue("ERROR", "INVALID_ZIP", str(exc)),)
    if actual != expected:
        return (OfficialValidationIssue("ERROR", "INVALID_ZIP_LAYOUT", f"expected {sorted(expected)}, got {sorted(actual)}"),)
    return ()
