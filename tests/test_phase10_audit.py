from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools.phase10_audit import audit


def test_phase10_audit_reports_group_and_phase5_coverage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for directory in ("clip-features-32", "map-keyframes", "media-info", "objects/L21_V001", "keyframes/L21_V001", "videos"):
        (data / directory).mkdir(parents=True, exist_ok=True)
    (data / "clip-features-32/L21_V001.npy").write_bytes(b"npy")
    (data / "map-keyframes/L21_V001.csv").write_text("keyframe_id,frame_idx,pts_time\n1,0,0.0\n2,30,1.0\n", encoding="utf-8")
    (data / "media-info/L21_V001.json").write_text(json.dumps({"title": "News", "author": "Channel"}), encoding="utf-8")
    (data / "keyframes/L21_V001/001.jpg").write_bytes(b"jpg")
    (data / "videos/L21_V001.mp4").write_bytes(b"video")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"data_root": str(data), "videos": [{
        "video_id": "L21_V001", "group": "L21", "clip_feature_path": "data/clip-features-32/L21_V001.npy",
        "mapping_path": "data/map-keyframes/L21_V001.csv", "media_info_path": "data/media-info/L21_V001.json",
        "object_path": "data/objects/L21_V001", "keyframe_path": "data/keyframes/L21_V001",
        "video_path": "data/videos/L21_V001.mp4", "mapping_rows": 2, "object_file_count": 1,
    }]}), encoding="utf-8")
    phase5 = tmp_path / "phase5.sqlite"
    with sqlite3.connect(phase5) as connection:
        connection.execute("CREATE TABLE ocr(video_id TEXT)")
        connection.execute("CREATE TABLE asr(video_id TEXT)")
        connection.execute("INSERT INTO ocr VALUES('L21_V001')")
        connection.executemany("INSERT INTO asr VALUES(?)", [("L21_V001",), ("L21_V001",)])

    outputs = audit(registry, tmp_path / "audits", phase5_store=phase5, check_mappings=True)
    capability = json.loads(outputs["capability"].read_text(encoding="utf-8"))
    qa = json.loads(outputs["qa"].read_text(encoding="utf-8"))
    trake = json.loads(outputs["trake"].read_text(encoding="utf-8"))

    assert capability["status"] == "READY"
    assert capability["groups"]["L21"]["videos"] == 1
    assert qa["ocr"]["record_count"] == 1
    assert qa["asr"]["record_count"] == 2
    assert trake["refinement_ready_video_count"] == 1
