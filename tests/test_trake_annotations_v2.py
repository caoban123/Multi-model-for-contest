from __future__ import annotations

import json
from pathlib import Path

from tools.phase8_validate_annotations import validate_annotations


ROOT = Path(__file__).parents[1]


def test_unlabelled_template_is_valid_and_has_no_fake_labels() -> None:
    path = ROOT / "benchmarks" / "trake_annotation_template_v2.json"
    result = validate_annotations(path, ROOT / "configs" / "phase8_trake_v2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert result["valid"] is True and result["labelled_query_count"] == 0
    assert payload["queries"][0]["expected_video_id"] is None
    assert payload["queries"][0]["events"][0]["start_pts"] is None


def test_labelled_annotations_require_manual_metadata_categories_and_pts(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "split": "development",
        "annotation_status": "LABELLED",
        "queries": [{"query_id": "q", "query": "", "categories": [], "expected_video_id": "wrong", "events": [{"event_id": "e1", "required": True}], "annotator": "", "annotated_at": None}],
    }), encoding="utf-8")

    result = validate_annotations(path)

    assert result["valid"] is False
    messages = " ".join(error["message"] for error in result["errors"])
    assert "L21_V###" in messages and "PTS interval" in messages and "annotator" in messages


def test_holdout_is_blocked_until_development_config_is_frozen(tmp_path: Path) -> None:
    annotations = tmp_path / "holdout.json"
    annotations.write_text(json.dumps({
        "split": "holdout",
        "annotation_status": "LABELLED",
        "queries": [{
            "query_id": "q", "query": "one event", "categories": ["simple_visual"], "expected_video_id": "L21_V001",
            "events": [{"event_id": "e1", "required": True, "representative_pts": 1.0}], "annotator": "reviewer", "annotated_at": "2026-08-19T00:00:00Z",
        }],
    }), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"benchmark": {"development_config_frozen": False}}), encoding="utf-8")

    result = validate_annotations(annotations, config)

    assert result["valid"] is False
    assert any("frozen development config" in error["message"] for error in result["errors"])
