from __future__ import annotations

from aic_retrieval.qa_schema import AvailabilityStatus, EvidenceModality, QaRequest, ReviewDecision
from aic_retrieval.qa_store import QaStore
from aic_retrieval.qa_workflow import QaWorkflow


def _response() -> dict:
    return {"video_results":[{"video_id":"L21_V001","rank":1,"frames":[{"video_id":"L21_V001","keyframe_id":5,"frame_idx":125,"pts_time":5.0,"evidence":{"ocr":[{"matched_text":"Samsung"}],"asr":[],"object":[],"attribute":[]}}]}]}


def test_sqlite_store_reopens_append_only_reviews_and_exports_confirmed_record(tmp_path) -> None:
    store = QaStore(tmp_path / "qa.sqlite3")
    workflow = QaWorkflow(store=store)
    state = workflow.prepare(
        QaRequest("q1", "a sign", "Biển hiệu ghi gì?"), _response(),
        modality_availability={EvidenceModality.OCR: AvailabilityStatus.AVAILABLE},
    )
    _, draft = workflow.draft(state.session.session_id)
    _, confirmed = workflow.review(state.session.session_id, draft.draft_id, ReviewDecision.CONFIRMED, None, draft.evidence_refs, "tester")
    _, edited = workflow.review(state.session.session_id, draft.draft_id, ReviewDecision.EDITED, "Samsung Electronics", draft.evidence_refs, "tester")
    reopened = QaWorkflow(store=store).get(state.session.session_id)
    export = workflow.create_export(state.session.session_id, confirmed.review_id)
    assert len(reopened.reviews) == 2
    assert reopened.reviews[0].final_answer == "Samsung"
    assert reopened.reviews[1].final_answer == "Samsung Electronics"
    assert export.video_id == "L21_V001"
    assert export.frame_id == 5
    assert export.confirmed is True


def test_sqlite_store_refuses_export_for_rejected_review(tmp_path) -> None:
    store = QaStore(tmp_path / "qa.sqlite3")
    workflow = QaWorkflow(store=store)
    state = workflow.prepare(QaRequest("q2", "a sign", "Biển hiệu ghi gì?"), _response(), modality_availability={EvidenceModality.OCR: AvailabilityStatus.AVAILABLE})
    _, draft = workflow.draft(state.session.session_id)
    _, rejected = workflow.review(state.session.session_id, draft.draft_id, ReviewDecision.REJECTED, None, (), "tester")
    try:
        workflow.create_export(state.session.session_id, rejected.review_id)
    except ValueError as exc:
        assert "confirmed or edited" in str(exc)
    else:
        raise AssertionError("rejected review must not export")


def test_sqlite_store_allows_repeated_evidence_across_sessions(tmp_path) -> None:
    store = QaStore(tmp_path / "qa.sqlite3")
    workflow = QaWorkflow(store=store)
    availability = {EvidenceModality.OCR: AvailabilityStatus.AVAILABLE}
    first = workflow.prepare(QaRequest("q-repeat-1", "a sign", "sign text"), _response(), modality_availability=availability)
    second = workflow.prepare(QaRequest("q-repeat-2", "a sign", "sign text"), _response(), modality_availability=availability)

    _, first_draft = workflow.draft(first.session.session_id)
    _, second_draft = workflow.draft(second.session.session_id)

    assert first_draft.draft_id != second_draft.draft_id
    assert len(QaWorkflow(store=store).get(first.session.session_id).drafts) == 1
    assert len(QaWorkflow(store=store).get(second.session.session_id).drafts) == 1
