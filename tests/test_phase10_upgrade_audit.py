from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tools.phase10_upgrade_audit import audit


def test_upgrade_audit_detects_current_indexes_and_partial_asr(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"corpus_sha256": "abc", "source_counts": {"ocr": 10, "asr": 2}}), encoding="utf-8")
    bge = tmp_path / "bge.json"
    bge.write_text(json.dumps({"corpus": {"sha256": "abc"}}), encoding="utf-8")
    bm25 = tmp_path / "bm25.json"
    bm25.write_text(json.dumps({"corpus": {"sha256": "abc"}}), encoding="utf-8")
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps({"frame_count": 2, "detection_count": 10, "build_configuration": {"all_valid_detections_preserved": False}}), encoding="utf-8")
    phase5 = tmp_path / "phase5.sqlite"
    connection = sqlite3.connect(phase5)
    connection.executescript("CREATE TABLE ocr(video_id TEXT); CREATE TABLE asr(video_id TEXT); INSERT INTO ocr VALUES ('v1'); INSERT INTO asr VALUES ('v1');")
    connection.commit(); connection.close()
    args = argparse.Namespace(
        corpus_manifest=corpus, corpus_documents=tmp_path / "documents.jsonl", phase5_store=phase5,
        bge_manifest=bge, bm25_manifest=bm25, object_manifest=objects,
        object_manifest_v2=tmp_path / "objects-v2.json", refs=tmp_path / "refs.json",
        object_root=tmp_path / "objects", bge_output=tmp_path / "bge", object_output=tmp_path / "objects.sqlite",
        groups="L21", expected_videos=2,
    )
    result = audit(args)
    statuses = {item["name"]: item["status"] for item in result["checks"]}
    assert statuses["bge_index"] == "READY"
    assert statuses["bm25_index"] == "READY"
    assert statuses["object_store_v2"] == "READY"
    assert statuses["asr_coverage"] == "PARTIAL"
    assert result["quality_claim"] == "NOT_EVALUATED_WITHOUT_LABELED_DEV_SET"
