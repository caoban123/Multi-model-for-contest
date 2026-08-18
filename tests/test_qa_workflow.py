from __future__ import annotations

import pytest

from aic_retrieval.qa_schema import AvailabilityStatus, EvidenceModality, QaRequest, ReviewDecision
from aic_retrieval.qa_workflow import QaWorkflow, state_payload


def _response() -> dict:
    return {"video_results":[{"video_id":"L21_V001","rank":1,"frames":[{"video_id":"L21_V001","keyframe_id":5,"frame_idx":125,"pts_time":5.0,"evidence":{"ocr":[{"matched_text":"Samsung","confidence":.9}],"asr":[],"object":[],"attribute":[]}}]}]}


def test_workflow_prepares_drafts_reviews_and_reopens_in_memory_session() -> None:
    workflow = QaWorkflow()
    state = workflow.prepare(
        QaRequest("q1", "a sign", "Biển hiệu ghi gì?"), _response(),
        modality_availability={EvidenceModality.OCR: AvailabilityStatus.AVAILABLE},
    )
    _, draft = workflow.draft(state.session.session_id)
    _, review = workflow.review(state.session.session_id, draft.draft_id, ReviewDecision.CONFIRMED, None, draft.evidence_refs, "tester")
    reopened = state_payload(workflow.get(state.session.session_id))
    assert review.final_answer == "Samsung"
    assert review.normalized_final_answer == "samsung"
    assert reopened["reviews"][0]["decision"] == "confirmed"
    assert reopened["evidence_pack"]["evidence_refs"]


def test_workflow_rejects_confirmation_without_selected_evidence() -> None:
    workflow = QaWorkflow()
    state = workflow.prepare(
        QaRequest("q2", "a sign", "Biển hiệu ghi gì?"), _response(),
        modality_availability={EvidenceModality.OCR: AvailabilityStatus.AVAILABLE},
    )
    _, draft = workflow.draft(state.session.session_id)
    with pytest.raises(ValueError, match="selected evidence"):
        workflow.review(state.session.session_id, draft.draft_id, ReviewDecision.CONFIRMED, None, (), "tester")
