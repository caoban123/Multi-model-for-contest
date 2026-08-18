"""Versioned domain contracts for the evidence-grounded Q&A workflow.

This module intentionally contains validation and data shapes only.  Retrieval,
answering, review guards, persistence, and normalization live in dedicated
Phase-7 modules so a Q&A record can always be audited independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


QA_SCHEMA_VERSION = "phase7-qa-v1"


class EvidenceModality(str, Enum):
    KEYFRAME = "keyframe"
    CLIP = "clip"
    OBJECT = "object"
    METADATA = "metadata"
    OCR = "ocr"
    ASR = "asr"
    ATTRIBUTE = "attribute"
    TEMPORAL = "temporal"


class AvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class QuestionType(str, Enum):
    OCR = "OCR"
    ASR = "ASR"
    METADATA = "METADATA"
    VISUAL = "VISUAL"
    OBJECT = "OBJECT"
    ATTRIBUTE_COLOR = "ATTRIBUTE_COLOR"
    COUNT = "COUNT"
    TEMPORAL = "TEMPORAL"
    UNKNOWN = "UNKNOWN"


class ConfidenceState(str, Enum):
    SUPPORTED = "SUPPORTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"


class ReviewDecision(str, Enum):
    CONFIRMED = "confirmed"
    EDITED = "edited"
    REJECTED = "rejected"


def _required_text(field_name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True)
class QaRequest:
    query_id: str
    event_query: str
    question: str
    selected_video_id: str | None = None
    selected_frame_id: int | None = None
    schema_version: str = QA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _required_text("query_id", self.query_id))
        object.__setattr__(self, "event_query", _required_text("event_query", self.event_query))
        object.__setattr__(self, "question", _required_text("question", self.question))
        if self.selected_video_id is not None:
            object.__setattr__(self, "selected_video_id", _required_text("selected_video_id", self.selected_video_id))
        if self.selected_frame_id is not None and self.selected_frame_id < 0:
            raise ValueError("selected_frame_id must be non-negative")
        if self.schema_version != QA_SCHEMA_VERSION:
            raise ValueError(f"unsupported Q&A schema version: {self.schema_version}")


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    modality: EvidenceModality
    video_id: str
    keyframe_id: int | None
    frame_idx: int | None
    timestamp: float | None
    payload: dict[str, Any]
    source: str
    source_version: str | None
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _required_text("evidence_id", self.evidence_id))
        object.__setattr__(self, "video_id", _required_text("video_id", self.video_id))
        object.__setattr__(self, "source", _required_text("source", self.source))
        if self.keyframe_id is not None and self.keyframe_id < 0:
            raise ValueError("keyframe_id must be non-negative")
        if self.frame_idx is not None and self.frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if self.timestamp is not None and self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


@dataclass(frozen=True)
class EvidencePack:
    request: QaRequest
    retrieval_context: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    evidence_refs: tuple[EvidenceRef, ...]
    modality_availability: dict[EvidenceModality, AvailabilityStatus]
    schema_version: str = QA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != QA_SCHEMA_VERSION:
            raise ValueError(f"unsupported Q&A schema version: {self.schema_version}")


@dataclass(frozen=True)
class AnswerDraft:
    draft_id: str
    raw_answer: str | None
    normalized_answer: str | None
    alternative_answers: tuple[str, ...]
    question_type: QuestionType
    evidence_refs: tuple[str, ...]
    confidence_state: ConfidenceState
    strategy_version: str
    answer_source: str | None = None
    generation_method: str = "evidence_first"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "draft_id", _required_text("draft_id", self.draft_id))
        object.__setattr__(self, "strategy_version", _required_text("strategy_version", self.strategy_version))
        if self.raw_answer is not None:
            object.__setattr__(self, "raw_answer", _required_text("raw_answer", self.raw_answer))
        if self.normalized_answer is not None:
            object.__setattr__(self, "normalized_answer", _required_text("normalized_answer", self.normalized_answer))


@dataclass(frozen=True)
class QaReview:
    review_id: str
    decision: ReviewDecision
    final_answer: str | None
    selected_evidence_refs: tuple[str, ...]
    reviewer: str
    timestamp: str
    draft_id: str
    normalized_final_answer: str | None = None
    alternative_answers: tuple[str, ...] = ()
    normalization_rules_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("review_id", "reviewer", "timestamp", "draft_id"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if self.final_answer is not None:
            object.__setattr__(self, "final_answer", _required_text("final_answer", self.final_answer))


@dataclass(frozen=True)
class QaSession:
    session_id: str
    request: QaRequest
    retrieval_context: dict[str, Any]
    created_at: str
    schema_version: str = QA_SCHEMA_VERSION
    selected_video_id: str | None = None
    selected_frame_id: int | None = None
    selected_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_text("session_id", self.session_id))
        object.__setattr__(self, "created_at", _required_text("created_at", self.created_at))
        if self.schema_version != QA_SCHEMA_VERSION:
            raise ValueError(f"unsupported Q&A schema version: {self.schema_version}")


@dataclass(frozen=True)
class QaExportRecord:
    export_id: str
    session_id: str
    review_id: str
    video_id: str
    frame_id: int
    answer: str
    evidence_ids: tuple[str, ...]
    confirmed: bool
    schema_version: str = QA_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("export_id", "session_id", "review_id", "video_id", "answer"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.schema_version != QA_SCHEMA_VERSION:
            raise ValueError(f"unsupported Q&A schema version: {self.schema_version}")
