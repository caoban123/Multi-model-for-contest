from __future__ import annotations

import csv
import json

from aic_retrieval.qa_schema import AvailabilityStatus, EvidenceModality, QaRequest, ReviewDecision
from aic_retrieval.qa_store import QaStore
from aic_retrieval.qa_workflow import QaWorkflow
from aic_retrieval.submission import (
    SubmissionCandidate,
    validate_candidates,
    validation_report,
    write_candidates_csv,
    write_candidates_jsonl,
)
from tools.export_qa_submissions import candidates_from_qa_store


def test_submission_validator_flags_duplicates_and_empty_payloads() -> None:
    candidates = [
        SubmissionCandidate("q1", "L21_V001", 5, answer="Samsung", evidence_ids=("e1",)),
        SubmissionCandidate("q1", "L21_V001", 5),
    ]
    issues = validate_candidates(candidates)
    assert any(issue.code == "DUPLICATE_CANDIDATE" and issue.severity == "ERROR" for issue in issues)
    assert any(issue.code == "NO_EVIDENCE_IDS" and issue.severity == "WARNING" for issue in issues)
    assert validation_report(candidates, issues)["error_count"] == 1


def test_submission_writers_create_jsonl_and_csv(tmp_path) -> None:
    candidate = SubmissionCandidate("q1", "L21_V001", 5, answer="Samsung", evidence_ids=("e1", "e2"), metadata={"source": "test"})
    jsonl_path = tmp_path / "submission.jsonl"
    csv_path = tmp_path / "submission.csv"
    write_candidates_jsonl([candidate], jsonl_path)
    write_candidates_csv([candidate], csv_path)

    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["video_id"] == "L21_V001"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["evidence_ids"] == "e1||e2"
    assert json.loads(row["metadata"]) == {"source": "test"}


def test_confirmed_qa_exports_become_submission_candidates(tmp_path) -> None:
    response = {
        "video_results": [
            {
                "video_id": "L21_V001",
                "rank": 1,
                "frames": [
                    {
                        "video_id": "L21_V001",
                        "keyframe_id": 5,
                        "frame_idx": 125,
                        "pts_time": 5.0,
                        "evidence": {"ocr": [{"matched_text": "Samsung"}], "asr": [], "object": [], "attribute": []},
                    }
                ],
            }
        ]
    }
    store = QaStore(tmp_path / "qa.sqlite3")
    workflow = QaWorkflow(store=store)
    state = workflow.prepare(
        QaRequest("q-submission", "a sign", "sign text"),
        response,
        modality_availability={EvidenceModality.OCR: AvailabilityStatus.AVAILABLE},
    )
    _, draft = workflow.draft(state.session.session_id)
    _, review = workflow.review(state.session.session_id, draft.draft_id, ReviewDecision.CONFIRMED, None, draft.evidence_refs, "tester")
    workflow.create_export(state.session.session_id, review.review_id)

    candidates = candidates_from_qa_store(store)
    assert len(candidates) == 1
    assert candidates[0].query_id == "q-submission"
    assert candidates[0].video_id == "L21_V001"
    assert candidates[0].frame_id == 5
    assert candidates[0].source == "qa_confirmed_export"
