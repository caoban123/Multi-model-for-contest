import json
import sqlite3
from pathlib import Path

import pytest

from aic_retrieval.object_store import build_object_store, load_alias_dictionary


def test_alias_dictionary_is_bilingual_and_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    aliases = load_alias_dictionary(root / "config" / "object_aliases_v1.json")
    assert aliases.version == "object-aliases-v1.0"
    assert aliases.normalize("Mobile phone") == "phone"
    assert aliases.normalize("điện thoại") == "phone"
    assert aliases.normalize("Human arm") == "hand"
    assert aliases.normalize("Unmapped Label") == "unmapped_label"


def test_build_store_preserves_missing_frame_as_unknown(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    refs = [
        {"video_id":"L21_V001","group":"L21","keyframe_id":1,"frame_idx":10,"pts_time":1.0},
        {"video_id":"L21_V001","group":"L21","keyframe_id":2,"frame_idx":20,"pts_time":2.0},
    ]
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps(refs), encoding="utf-8")
    object_path = tmp_path / "objects" / "L21_V001" / "001.json"
    object_path.parent.mkdir(parents=True)
    object_path.write_text(json.dumps({"detection_class_labels":["1"], "detection_class_entities":["Mobile phone"], "detection_scores":["0.8"], "detection_boxes":[["0.1","0.2","0.5","0.6"]]}), encoding="utf-8")
    db_path = tmp_path / "objects.sqlite"
    manifest = build_object_store(refs_path, tmp_path / "objects", root / "config" / "object_aliases_v1.json", db_path, {"L21"})
    assert manifest["frame_count"] == 2
    assert manifest["detection_count"] == 1
    assert manifest["missing_data"]["frames_unknown"] == 1
    assert manifest["build_configuration"]["detection_threshold"] is None
    connection = sqlite3.connect(db_path)
    frames = connection.execute("SELECT keyframe_id,object_status FROM frames ORDER BY keyframe_id").fetchall()
    detection = connection.execute("SELECT label_raw,label_normalized,center_x,center_y,area FROM detections").fetchone()
    connection.close()
    assert frames == [(1, "AVAILABLE"), (2, "UNKNOWN")]
    assert detection[:2] == ("Mobile phone", "phone")
    assert detection[2:] == pytest.approx((0.4, 0.3, 0.16))
