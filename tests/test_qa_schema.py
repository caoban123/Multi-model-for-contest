from __future__ import annotations

import pytest

from aic_retrieval.qa_schema import (
    AnswerDraft,
    AvailabilityStatus,
    ConfidenceState,
    EvidenceModality,
    EvidenceRef,
    QaRequest,
    QuestionType,
)


def test_request_requires_event_query_and_question() -> None:
    with pytest.raises(ValueError, match="event_query"):
        QaRequest("q1", "", "What is shown?")
    with pytest.raises(ValueError, match="question"):
        QaRequest("q1", "a person", " ")


def test_evidence_ref_keeps_video_level_metadata_without_frame() -> None:
    ref = EvidenceRef(
        evidence_id="metadata:L21_V001:channel",
        modality=EvidenceModality.METADATA,
        video_id="L21_V001",
        keyframe_id=None,
        frame_idx=None,
        timestamp=None,
        payload={"field": "channel", "value": "News"},
        source="metadata_documents",
        source_version="metadata-v1",
    )
    assert ref.availability_status is AvailabilityStatus.AVAILABLE
    assert ref.keyframe_id is None


def test_answer_draft_allows_guarded_no_answer_state() -> None:
    draft = AnswerDraft(
        draft_id="draft-1",
        raw_answer=None,
        normalized_answer=None,
        alternative_answers=(),
        question_type=QuestionType.ASR,
        evidence_refs=(),
        confidence_state=ConfidenceState.NEEDS_EVIDENCE,
        strategy_version="phase7-evidence-first-v1",
        warnings=("ASR_UNAVAILABLE",),
    )
    assert draft.raw_answer is None
