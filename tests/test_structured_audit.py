import json
from pathlib import Path

from aic_retrieval.structured_audit import FrameIdentity, audit_metadata, audit_objects, valid_bbox


def write_object(path: Path, scores=None, entities=None, boxes=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = scores or ["0.8"]
    entities = entities or ["Person"]
    boxes = boxes or [["0.1", "0.2", "0.8", "0.9"]]
    path.write_text(json.dumps({"detection_scores": scores, "detection_class_entities": entities, "detection_boxes": boxes,
                                "detection_class_names": ["/m/person"] * len(scores), "detection_class_labels": ["1"] * len(scores)}), encoding="utf-8")


def test_bbox_validation_has_inclusive_normalized_boundaries() -> None:
    assert valid_bbox([0, 0, 1, 1])
    assert not valid_bbox([0.5, 0, 0.4, 1])
    assert not valid_bbox([-0.1, 0, 1, 1])


def test_object_audit_distinguishes_missing_unknown_from_no_match(tmp_path: Path) -> None:
    write_object(tmp_path / "objects" / "L21_V001" / "001.json")
    frames = [FrameIdentity("L21_V001", 1), FrameIdentity("L21_V001", 2), FrameIdentity("L21_V002", 1)]
    audit = audit_objects(tmp_path / "objects", frames)
    assert audit["frames_with_object_data"] == 1
    assert audit["frames_missing_object_data"] == 2
    assert audit["per_video"]["L21_V001"]["status_when_queried"] == "PARTIAL_UNKNOWN"
    assert audit["per_video"]["L21_V002"]["status_when_queried"] == "UNKNOWN"
    assert "does not match" in audit["semantics"]["NO_MATCH"]
    assert "missing" in audit["semantics"]["UNKNOWN"]


def test_object_audit_reports_mismatch_and_invalid_box(tmp_path: Path) -> None:
    write_object(tmp_path / "objects" / "L21_V001" / "001.json", scores=["0.8"], entities=["Person", "Car"])
    write_object(tmp_path / "objects" / "L21_V001" / "002.json", boxes=[[0.8, 0, 0.2, 1]])
    audit = audit_objects(tmp_path / "objects", [FrameIdentity("L21_V001", 1), FrameIdentity("L21_V001", 2)])
    assert audit["length_mismatch_count"] == 1
    assert audit["invalid_bbox_count"] == 1


def test_metadata_audit_maps_length_to_duration_and_derives_group(tmp_path: Path) -> None:
    (tmp_path / "L21_V001.json").write_text(json.dumps({"title":"News", "author":"Channel", "publish_date":"01/08/2024", "keywords":["news"], "description":"Text", "length":12}), encoding="utf-8")
    audit = audit_metadata(tmp_path, {"L21"})
    assert audit["valid_json_count"] == 1
    assert audit["field_coverage"]["duration"]["ratio"] == 1.0
    assert audit["field_coverage"]["group"]["ratio"] == 1.0
    assert audit["granularity"] == "video"
