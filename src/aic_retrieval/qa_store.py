"""Append-only local SQLite audit store for Phase-7 Q&A sessions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from aic_retrieval.qa_question_router import QuestionRoute
from aic_retrieval.qa_schema import (
    AnswerDraft,
    AvailabilityStatus,
    ConfidenceState,
    EvidenceModality,
    EvidencePack,
    EvidenceRef,
    QaExportRecord,
    QaRequest,
    QaReview,
    QaSession,
    QuestionType,
    ReviewDecision,
)


QA_STORE_VERSION = "phase7-qa-store-v1"


@dataclass(frozen=True)
class StoredQaState:
    session: QaSession
    pack: EvidencePack
    route: QuestionRoute
    drafts: tuple[AnswerDraft, ...]
    reviews: tuple[QaReview, ...]


class QaStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS qa_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS qa_sessions(
                  session_id TEXT PRIMARY KEY, query_id TEXT NOT NULL, event_query TEXT NOT NULL, question TEXT NOT NULL,
                  selected_video_id TEXT, selected_frame_id INTEGER, retrieval_json TEXT NOT NULL, availability_json TEXT NOT NULL,
                  route_json TEXT NOT NULL, created_at TEXT NOT NULL, schema_version TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS qa_candidates(
                  session_id TEXT NOT NULL, ordinal INTEGER NOT NULL, video_id TEXT NOT NULL, candidate_json TEXT NOT NULL,
                  PRIMARY KEY(session_id, ordinal));
                CREATE TABLE IF NOT EXISTS qa_evidence(
                  session_id TEXT NOT NULL, evidence_id TEXT NOT NULL, modality TEXT NOT NULL, video_id TEXT NOT NULL,
                  keyframe_id INTEGER, frame_idx INTEGER, timestamp REAL, payload_json TEXT NOT NULL, source TEXT NOT NULL,
                  source_version TEXT, availability_status TEXT NOT NULL, PRIMARY KEY(session_id, evidence_id));
                CREATE TABLE IF NOT EXISTS qa_answer_drafts(
                  draft_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, raw_answer TEXT, normalized_answer TEXT,
                  alternatives_json TEXT NOT NULL, question_type TEXT NOT NULL, evidence_ids_json TEXT NOT NULL,
                  confidence_state TEXT NOT NULL, strategy_version TEXT NOT NULL, answer_source TEXT,
                  generation_method TEXT NOT NULL, warnings_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS qa_reviews(
                  review_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, draft_id TEXT NOT NULL, decision TEXT NOT NULL,
                  final_answer TEXT, normalized_final_answer TEXT, alternatives_json TEXT NOT NULL,
                  selected_evidence_json TEXT NOT NULL, reviewer TEXT NOT NULL, timestamp TEXT NOT NULL,
                  normalization_rules_version TEXT);
                CREATE TABLE IF NOT EXISTS qa_exports(
                  export_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, review_id TEXT NOT NULL, video_id TEXT NOT NULL,
                  frame_id INTEGER NOT NULL, answer TEXT NOT NULL, evidence_ids_json TEXT NOT NULL,
                  confirmed INTEGER NOT NULL, created_at TEXT NOT NULL, schema_version TEXT NOT NULL, metadata_json TEXT NOT NULL);
            """)
            row = connection.execute("SELECT value FROM qa_metadata WHERE key='store_version'").fetchone()
            if row is None:
                connection.execute("INSERT INTO qa_metadata(key,value) VALUES('store_version',?)", (QA_STORE_VERSION,))
            elif row["value"] != QA_STORE_VERSION:
                raise ValueError(f"unsupported Q&A store version: {row['value']}")

    def save_session(self, session: QaSession, pack: EvidencePack, route: QuestionRoute) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO qa_sessions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (session.session_id, session.request.query_id, session.request.event_query, session.request.question,
                 session.selected_video_id, session.selected_frame_id, _dump(session.retrieval_context),
                 _dump({key.value: value.value for key, value in pack.modality_availability.items()}),
                 _dump({"question_type": route.question_type.value, "required_modalities": [item.value for item in route.required_modalities], "reason": route.reason, "version": route.version}),
                 session.created_at, session.schema_version),
            )
            connection.executemany(
                "INSERT INTO qa_candidates VALUES(?,?,?,?)",
                [(session.session_id, index, str(item.get("video_id", "")), _dump(item)) for index, item in enumerate(pack.candidates)],
            )
            connection.executemany(
                "INSERT INTO qa_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [(session.session_id, ref.evidence_id, ref.modality.value, ref.video_id, ref.keyframe_id, ref.frame_idx,
                  ref.timestamp, _dump(ref.payload), ref.source, ref.source_version, ref.availability_status.value)
                 for ref in pack.evidence_refs],
            )

    def save_draft(self, session_id: str, draft: AnswerDraft) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO qa_answer_drafts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (draft.draft_id, session_id, draft.raw_answer, draft.normalized_answer, _dump(draft.alternative_answers),
                 draft.question_type.value, _dump(draft.evidence_refs), draft.confidence_state.value, draft.strategy_version,
                 draft.answer_source, draft.generation_method, _dump(draft.warnings)),
            )

    def save_review(self, session_id: str, review: QaReview) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO qa_reviews VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (review.review_id, session_id, review.draft_id, review.decision.value, review.final_answer,
                 review.normalized_final_answer, _dump(review.alternative_answers), _dump(review.selected_evidence_refs),
                 review.reviewer, review.timestamp, review.normalization_rules_version),
            )

    def load_state(self, session_id: str) -> StoredQaState:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM qa_sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                raise ValueError("Q&A session was not found")
            request = QaRequest(row["query_id"], row["event_query"], row["question"], row["selected_video_id"], row["selected_frame_id"], row["schema_version"])
            session = QaSession(row["session_id"], request, _load(row["retrieval_json"]), row["created_at"], row["schema_version"], row["selected_video_id"], row["selected_frame_id"])
            availability = {EvidenceModality(key): AvailabilityStatus(value) for key, value in _load(row["availability_json"]).items()}
            candidates = tuple(_load(item["candidate_json"]) for item in connection.execute("SELECT candidate_json FROM qa_candidates WHERE session_id=? ORDER BY ordinal", (session_id,)))
            refs = tuple(EvidenceRef(
                evidence_id=item["evidence_id"], modality=EvidenceModality(item["modality"]), video_id=item["video_id"],
                keyframe_id=item["keyframe_id"], frame_idx=item["frame_idx"], timestamp=item["timestamp"],
                payload=_load(item["payload_json"]), source=item["source"], source_version=item["source_version"],
                availability_status=AvailabilityStatus(item["availability_status"]),
            ) for item in connection.execute("SELECT * FROM qa_evidence WHERE session_id=? ORDER BY evidence_id", (session_id,)))
            pack = EvidencePack(request, session.retrieval_context, candidates, refs, availability, row["schema_version"])
            route_data = _load(row["route_json"])
            route = QuestionRoute(QuestionType(route_data["question_type"]), tuple(EvidenceModality(item) for item in route_data["required_modalities"]), route_data["reason"], route_data["version"], route_data.get("answer_field"))
            drafts = tuple(AnswerDraft(
                item["draft_id"], item["raw_answer"], item["normalized_answer"], tuple(_load(item["alternatives_json"])),
                QuestionType(item["question_type"]), tuple(_load(item["evidence_ids_json"])), ConfidenceState(item["confidence_state"]),
                item["strategy_version"], item["answer_source"], item["generation_method"], tuple(_load(item["warnings_json"])),
            ) for item in connection.execute("SELECT * FROM qa_answer_drafts WHERE session_id=? ORDER BY rowid", (session_id,)))
            reviews = tuple(QaReview(
                item["review_id"], ReviewDecision(item["decision"]), item["final_answer"], tuple(_load(item["selected_evidence_json"])),
                item["reviewer"], item["timestamp"], item["draft_id"], item["normalized_final_answer"],
                tuple(_load(item["alternatives_json"])), item["normalization_rules_version"],
            ) for item in connection.execute("SELECT * FROM qa_reviews WHERE session_id=? ORDER BY rowid", (session_id,)))
        return StoredQaState(session, pack, route, drafts, reviews)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT session_id,query_id,event_query,question,created_at FROM qa_sessions ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
            return [dict(row) for row in rows]

    def list_exports(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.export_id,e.session_id,e.review_id,e.video_id,e.frame_id,e.answer,e.evidence_ids_json,e.confirmed,
                       e.created_at,e.schema_version,e.metadata_json,s.query_id,s.event_query,s.question
                FROM qa_exports e
                JOIN qa_sessions s ON s.session_id=e.session_id
                ORDER BY e.created_at ASC
                LIMIT ?
                """,
                (max(1, min(limit, 10000)),),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "evidence_ids": tuple(_load(row["evidence_ids_json"])),
                    "metadata": _load(row["metadata_json"]),
                }
                for row in rows
            ]

    def create_export(self, session_id: str, review_id: str) -> QaExportRecord:
        state = self.load_state(session_id)
        review = next((item for item in state.reviews if item.review_id == review_id), None)
        if review is None:
            raise ValueError("review_id was not created in this session")
        if review.decision not in {ReviewDecision.CONFIRMED, ReviewDecision.EDITED}:
            raise ValueError("only confirmed or edited reviews may create an internal export record")
        if not review.final_answer or not review.selected_evidence_refs:
            raise ValueError("confirmed review requires an answer and selected evidence")
        by_id = {item.evidence_id: item for item in state.pack.evidence_refs}
        frame_ref = next((by_id[item] for item in review.selected_evidence_refs if item in by_id and by_id[item].keyframe_id is not None), None)
        if frame_ref is None or frame_ref.keyframe_id is None:
            raise ValueError("internal export requires selected frame-level evidence")
        record = QaExportRecord(
            export_id=f"export-{uuid4().hex}", session_id=session_id, review_id=review_id,
            video_id=frame_ref.video_id, frame_id=frame_ref.keyframe_id, answer=review.final_answer,
            evidence_ids=review.selected_evidence_refs, confirmed=True,
            metadata={"store_version": QA_STORE_VERSION},
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO qa_exports VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (record.export_id, record.session_id, record.review_id, record.video_id, record.frame_id, record.answer,
                 _dump(record.evidence_ids), int(record.confirmed), _now(), record.schema_version, _dump(record.metadata)),
            )
        return record


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load(value: str) -> Any:
    return json.loads(value)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
