import json
import sqlite3
from pathlib import Path

from aic_retrieval.object_search import Detection, ObjectPredicate, ObjectSearchConfig, ObjectSearchService, count_matches, horizontal_region, nms, position_matches, vertical_region
from aic_retrieval.object_store import create_schema, load_alias_dictionary


def detection(x1, y1, x2, y2, confidence=0.9, label="person"):
    return Detection(label.title(), label, confidence, y1, x1, y2, x2, (x1+x2)/2, (y1+y2)/2, (x2-x1)*(y2-y1))


def test_nms_count_and_low_confidence_pipeline() -> None:
    overlapping = [detection(0.1,0.1,0.4,0.4,0.9), detection(0.11,0.11,0.41,0.41,0.8)]
    assert len(nms(overlapping, 0.5)) == 1
    separate = [detection(0.1,0.1,0.2,0.2), detection(0.7,0.7,0.8,0.8)]
    assert len(nms(separate, 0.5)) == 2
    assert count_matches(2, "=", 2) and count_matches(2, ">=", 2) and count_matches(2, "<=", 2)


def test_position_boundaries_are_disjoint_and_any_accepts_large_box() -> None:
    assert horizontal_region(0.329999) == "left"
    assert horizontal_region(0.33) == "center"
    assert horizontal_region(0.659999) == "center"
    assert horizontal_region(0.66) == "right"
    assert vertical_region(0.33) == "middle" and vertical_region(0.66) == "bottom"
    assert position_matches(detection(0, 0, 1, 1), "any", "any")
    missing = Detection("Person", "person", 0.9, 0, 0, 1, 1, None, None, 1)  # type: ignore[arg-type]
    assert not position_matches(missing, "left", "top")


def make_store(tmp_path: Path) -> tuple[Path, object]:
    db = tmp_path / "objects.sqlite"
    connection = sqlite3.connect(db); create_schema(connection)
    connection.executemany("INSERT INTO frames VALUES (?,?,?,?,?,?)", [("L21_V001",1,1,1.0,1,"AVAILABLE"),("L21_V001",2,2,2.0,0,"UNKNOWN")])
    rows = [
        ("L21_V001",1,"1","Person","person",0.9,0.1,0.05,0.3,0.25,0.15,0.2,0.04,"test","v1"),
        ("L21_V001",1,"1","Person","person",0.8,0.11,0.06,0.31,0.26,0.16,0.21,0.04,"test","v1"),
        ("L21_V001",1,"1","Person","person",0.95,0.6,0.7,0.8,0.9,0.8,0.7,0.04,"test","v1"),
        ("L21_V001",1,"2","Mobile phone","phone",0.1,0.1,0.1,0.2,0.2,0.15,0.15,0.01,"test","v1"),
    ]
    connection.executemany("INSERT INTO detections(video_id,keyframe_id,label_id,label_raw,label_normalized,confidence,bbox_y1,bbox_x1,bbox_y2,bbox_x2,center_x,center_y,area,source,schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.commit(); connection.close()
    aliases = load_alias_dictionary(Path(__file__).resolve().parents[1] / "config" / "object_aliases_v1.json")
    return db, aliases


def test_service_alias_count_position_confidence_unknown_and_determinism(tmp_path: Path) -> None:
    db, aliases = make_store(tmp_path); service = ObjectSearchService(db, aliases)
    query = ObjectPredicate(("người",), ">=", 1, "left", "any")
    first = service.search(query, ObjectSearchConfig(0.3, 0.5)); second = service.search(query, ObjectSearchConfig(0.3, 0.5))
    assert first == second
    assert first["results"] == service.search(ObjectPredicate(("person",), ">=", 1, "left", "any"), ObjectSearchConfig(0.3, 0.5))["results"]
    assert first["results"][0]["matched_count"] == 1
    assert first["results"][0]["matched_labels"] == ["person"]
    assert first["unknown_frames"] == [{"video_id":"L21_V001","keyframe_id":2,"data_status":"UNKNOWN"}]
    assert first["unknown_count"] == 1
    assert service.search(ObjectPredicate(("phone",)), ObjectSearchConfig(0.3,0.5))["results"] == []
    at_most_zero = service.search(ObjectPredicate(("phone",), "<=", 0), ObjectSearchConfig(0.3,0.5))
    assert at_most_zero["results"][0]["matched_count"] == 0


def test_multiple_same_label_count_after_nms_and_position(tmp_path: Path) -> None:
    db, aliases = make_store(tmp_path); service = ObjectSearchService(db, aliases)
    result = service.search(ObjectPredicate(("person",), "=", 2), ObjectSearchConfig(0.3,0.5))
    assert result["results"][0]["matched_count"] == 2


def test_phone_near_hand_is_labeled_as_heuristic_with_evidence(tmp_path: Path) -> None:
    db, aliases = make_store(tmp_path)
    connection = sqlite3.connect(db)
    connection.executemany("INSERT INTO detections(video_id,keyframe_id,label_id,label_raw,label_normalized,confidence,bbox_y1,bbox_x1,bbox_y2,bbox_x2,center_x,center_y,area,source,schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("L21_V001",1,"2","Mobile phone","phone",0.9,0.2,0.2,0.3,0.3,0.25,0.25,0.01,"test","v1"),
        ("L21_V001",1,"3","Human hand","hand",0.8,0.22,0.3,0.32,0.4,0.35,0.27,0.01,"test","v1"),
    ])
    connection.commit(); connection.close()
    payload = ObjectSearchService(db, aliases).phone_near_hand(ObjectSearchConfig(0.3,0.5), 0.35)
    assert payload["experimental"] is True
    assert payload["results"][0]["claim"] == "heuristic_proximity_not_holding_fact"
    assert set(payload["results"][0]["evidence"]) >= {"phone_confidence", "hand_confidence", "center_distance", "proximity_score"}
