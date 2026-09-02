from __future__ import annotations

import json
from pathlib import Path

from tools.phase10_validate_qa_annotations import validate_annotations


ROOT = Path(__file__).parents[1]


def test_phase10_qa_template_is_valid_without_fake_labels() -> None:
    path = ROOT / "benchmarks" / "phase10_qa_annotation_template_v2.json"
    result = validate_annotations(path)

    assert result["valid"] is True
    assert result["labelled_query_count"] == 0
    assert result["ready_for_quality_benchmark"] is False


def test_labelled_qa_requires_video_answer_range_and_manual_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "split": "development",
        "annotation_status": "LABELLED",
        "queries": [{
            "query_id": "q1",
            "event_query_vi": "event",
            "question_vi": "question",
            "question_type": "OBJECT",
            "expected_video_ids": [],
            "accepted_pts_ranges": [],
            "accepted_frame_ranges": [],
            "expected_answers": [],
            "required_modalities": ["object"],
            "annotator": "",
            "annotated_at": None,
        }],
    }), encoding="utf-8")

    result = validate_annotations(path)
    messages = " ".join(item["message"] for item in result["errors"])

    assert result["valid"] is False
    assert "expected video" in messages
    assert "accepted answer" in messages
    assert "keyframe evidence" in messages
    assert "PTS or official frame range" in messages
    assert "annotator" in messages


def test_labelled_qa_accepts_a_complete_manual_record(tmp_path: Path) -> None:
    path = tmp_path / "good.json"
    path.write_text(json.dumps({
        "split": "development",
        "annotation_status": "LABELLED",
        "queries": [{
            "query_id": "qa-dev-001",
            "event_query_vi": "nguoi cam dien thoai",
            "question_vi": "Nguoi do cam gi?",
            "question_type": "OBJECT",
            "expected_video_ids": ["L21_V001"],
            "accepted_pts_ranges": [[12.0, 16.5]],
            "accepted_frame_ranges": [[360, 495]],
            "expected_answers": ["dien thoai"],
            "required_modalities": ["keyframe", "object"],
            "unanswerable_from_available_evidence": False,
            "annotator": "reviewer",
            "annotated_at": "2026-08-22T00:00:00Z",
        }],
    }), encoding="utf-8")

    result = validate_annotations(path)

    assert result["valid"] is True
    assert result["ready_for_quality_benchmark"] is True
