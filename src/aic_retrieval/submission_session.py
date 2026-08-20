from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from aic_retrieval.submission_official import (
    OfficialPrediction,
    OfficialQuery,
    OfficialValidationIssue,
    build_submission_zip,
    has_errors,
    render_official_csv,
    validate_official_query,
    validate_submission_zip,
)


SESSION_SCHEMA_VERSION = "submission-session-v1"
CERTIFIED_MAPPING_SOURCES = {"manual_official", "btc_mapping"}


class SubmissionSessionStore:
    def __init__(self, path: Path, output_root: Path) -> None:
        self.path = path
        self.output_root = output_root
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS submission_sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    zip_path TEXT,
                    schema_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS submission_queries (
                    session_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    event_count INTEGER,
                    predictions_json TEXT NOT NULL,
                    mapping_sources_json TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, query_id),
                    FOREIGN KEY (session_id) REFERENCES submission_sessions(session_id) ON DELETE CASCADE
                );
                """
            )

    def start(self) -> dict[str, Any]:
        now = time.time()
        session_id = f"submission-{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO submission_sessions VALUES (?, 'ACTIVE', ?, ?, NULL, ?)",
                (session_id, now, now, SESSION_SCHEMA_VERSION),
            )
        return self.get(session_id)

    def confirm_query(
        self,
        session_id: str,
        query: OfficialQuery,
        *,
        mapping_sources: tuple[str, ...],
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.get(session_id)
        if session["status"] != "ACTIVE":
            raise ValueError("submission session is not ACTIVE")
        if len(mapping_sources) != len(query.predictions):
            raise ValueError("each prediction requires one mapping source")
        invalid_sources = sorted(set(mapping_sources) - CERTIFIED_MAPPING_SOURCES)
        if invalid_sources:
            raise ValueError(f"uncertified frame mapping sources: {invalid_sources}")
        issues = validate_official_query(query)
        if has_errors(issues):
            codes = ", ".join(issue.code for issue in issues if issue.severity == "ERROR")
            raise ValueError(f"official query validation failed: {codes}")
        now = time.time()
        payload = _query_payload(query)
        validation = [asdict(issue) for issue in issues]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO submission_queries (
                    session_id, query_id, task, status, event_count, predictions_json,
                    mapping_sources_json, source_json, validation_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'READY', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, query_id) DO UPDATE SET
                    task=excluded.task, status='READY', event_count=excluded.event_count,
                    predictions_json=excluded.predictions_json,
                    mapping_sources_json=excluded.mapping_sources_json,
                    source_json=excluded.source_json, validation_json=excluded.validation_json,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    query.query_id,
                    query.task,
                    query.event_count,
                    json.dumps(payload["predictions"], ensure_ascii=False, separators=(",", ":")),
                    json.dumps(mapping_sources, separators=(",", ":")),
                    json.dumps(source or {}, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")),
                    json.dumps(validation, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )
            connection.execute("UPDATE submission_sessions SET updated_at=? WHERE session_id=?", (now, session_id))
        return self.get(session_id)

    def remove_query(self, session_id: str, query_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if session["status"] != "ACTIVE":
            raise ValueError("submission session is not ACTIVE")
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM submission_queries WHERE session_id=? AND query_id=?",
                (session_id, query_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown submission query: {query_id}")
            connection.execute("UPDATE submission_sessions SET updated_at=? WHERE session_id=?", (time.time(), session_id))
        return self.get(session_id)

    def validate(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        issues: list[dict[str, Any]] = []
        filenames: set[str] = set()
        if not session["queries"]:
            issues.append(asdict(OfficialValidationIssue("ERROR", "EMPTY_SUBMISSION", "submission queue is empty")))
        for item in session["queries"]:
            query = _official_query(item)
            for issue in validate_official_query(query):
                issues.append({**asdict(issue), "query_id": query.query_id})
            if item["output_file"] in filenames:
                issues.append(asdict(OfficialValidationIssue("ERROR", "DUPLICATE_FILENAME", item["output_file"])))
            filenames.add(item["output_file"])
        return {
            "session_id": session_id,
            "valid": not any(issue["severity"] == "ERROR" for issue in issues),
            "error_count": sum(issue["severity"] == "ERROR" for issue in issues),
            "warning_count": sum(issue["severity"] == "WARNING" for issue in issues),
            "issues": issues,
        }

    def done(self, session_id: str) -> dict[str, Any]:
        validation = self.validate(session_id)
        if not validation["valid"]:
            raise ValueError("submission validation failed")
        session = self.get(session_id)
        queries = tuple(_official_query(item) for item in session["queries"])
        output_dir = self.output_root / session_id
        zip_path = build_submission_zip(queries, output_dir)
        zip_issues = validate_submission_zip(zip_path, [query.filename for query in queries])
        if has_errors(zip_issues):
            raise ValueError("submission ZIP validation failed")
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "UPDATE submission_sessions SET status='DONE', zip_path=?, updated_at=? WHERE session_id=?",
                (str(zip_path), now, session_id),
            )
        return {**self.get(session_id), "validation": validation}

    def get(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM submission_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"unknown submission session: {session_id}")
            rows = connection.execute(
                "SELECT * FROM submission_queries WHERE session_id=? ORDER BY created_at, query_id",
                (session_id,),
            ).fetchall()
        queries = [_row_payload(row) for row in rows]
        return {
            "session_id": session["session_id"],
            "status": session["status"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "zip_path": session["zip_path"],
            "schema_version": session["schema_version"],
            "queries": queries,
            "submission_queue": [
                {
                    "query_id": item["query_id"],
                    "task": item["task"],
                    "status": item["status"],
                    "output_file": item["output_file"],
                    "prediction_count": len(item["predictions"]),
                }
                for item in queries
            ],
        }

    def zip_path(self, session_id: str) -> Path:
        session = self.get(session_id)
        if session["status"] != "DONE" or not session["zip_path"]:
            raise ValueError("submission ZIP is not ready")
        path = Path(session["zip_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def _query_payload(query: OfficialQuery) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "task": query.task,
        "event_count": query.event_count,
        "predictions": [
            {"video_id": item.video_id, "frame_ids": list(item.frame_ids), "answer": item.answer}
            for item in query.predictions
        ],
    }


def _official_query(payload: dict[str, Any]) -> OfficialQuery:
    return OfficialQuery(
        str(payload["query_id"]),
        str(payload["task"]),
        tuple(
            OfficialPrediction(
                str(item["video_id"]),
                tuple(int(value) for value in item["frame_ids"]),
                str(item["answer"]) if item.get("answer") is not None else None,
            )
            for item in payload["predictions"]
        ),
        int(payload["event_count"]) if payload.get("event_count") is not None else None,
    )


def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = {
        "query_id": row["query_id"],
        "task": row["task"],
        "event_count": row["event_count"],
        "predictions": json.loads(row["predictions_json"]),
    }
    query = _official_query(payload)
    return {
        **payload,
        "status": row["status"],
        "mapping_sources": json.loads(row["mapping_sources_json"]),
        "source": json.loads(row["source_json"]),
        "validation": json.loads(row["validation_json"]),
        "output_file": query.filename,
        "csv_preview": render_official_csv(query),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
