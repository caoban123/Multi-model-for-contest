from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from aic_retrieval.phase5_store import create_store
from aic_retrieval.text_corpus import CorpusBuildConfig, build_text_corpus, write_text_corpus


def _write_registry(root: Path) -> None:
    path = root / "artifacts" / "registry" / "data_registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"videos": [{"video_id": "L21_V001", "group": "L21"}]}), encoding="utf-8")


def _write_metadata(root: Path) -> None:
    path = root / "data" / "media-info" / "L21_V001.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"title": "Bản tin sáng", "author": "HTV", "keywords": ["tin tức", "Việt Nam"]}), encoding="utf-8")


def _write_phase5_store(root: Path) -> None:
    path = root / "artifacts" / "phase5" / "phase5_store.sqlite"
    connection = create_store(path)
    connection.execute(
        "INSERT INTO asr(video_id,segment_id,start_time,end_time,text_raw,text_normalized,text_folded,confidence,keyframe_id,frame_idx,pts_time,mapping_kind,distance_seconds,run_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("L21_V001", 0, 1.0, 2.0, "Xin chào quý vị", "xin chào quý vị", "xin chao quy vi", 0.8, 1, 25, 1.0, "inside_segment", 0.0, "v1"),
    )
    connection.commit()
    connection.close()


def _write_object_store(root: Path) -> None:
    path = root / "artifacts" / "structured" / "l21_objects.sqlite"
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE frames(video_id TEXT,keyframe_id INTEGER,frame_idx INTEGER,pts_time REAL);"
        "CREATE TABLE detections(video_id TEXT,keyframe_id INTEGER,label_normalized TEXT,confidence REAL);"
    )
    connection.execute("INSERT INTO frames VALUES('L21_V001',1,25,1.0)")
    connection.execute("INSERT INTO frames VALUES('L21_V001',2,50,2.0)")
    connection.executemany(
        "INSERT INTO detections VALUES(?,?,?,?)",
        [
            ("L21_V001", 1, "person", 0.9),
            ("L21_V001", 1, "person", 0.7),
            ("L21_V001", 1, "phone", 0.8),
            ("L21_V001", 1, "noise", 0.1),
            ("L21_V001", 2, "vehicle", 0.85),
        ],
    )
    connection.commit()
    connection.close()


def test_text_corpus_is_deterministic_and_aggregates_object_labels(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    _write_metadata(tmp_path)
    _write_phase5_store(tmp_path)
    _write_object_store(tmp_path)
    audit_path = tmp_path / "artifacts" / "audits" / "l21_hybrid_retrieval_audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps({"source_fingerprint": "source-v1", "generated_at": "first"}), encoding="utf-8")

    first, first_manifest = build_text_corpus(tmp_path, CorpusBuildConfig(object_min_confidence=0.3))
    audit_path.write_text(json.dumps({"source_fingerprint": "source-v1", "generated_at": "second"}), encoding="utf-8")
    second, second_manifest = build_text_corpus(tmp_path, CorpusBuildConfig(object_min_confidence=0.3))

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert first_manifest["corpus_sha256"] == second_manifest["corpus_sha256"]
    assert first_manifest["input_fingerprint"] == second_manifest["input_fingerprint"]
    assert second_manifest["audit_source_fingerprint"] == "source-v1"
    assert first_manifest["source_counts"] == {"metadata": 3, "asr": 1, "ocr": 0, "object": 2}
    assert first_manifest["warnings"] == ["OCR contributed 0 documents"]
    object_document = next(item for item in first if item.source_type == "object")
    assert object_document.text == "person, phone"
    assert object_document.provenance["labels"] == [{"label": "person", "confidence": 0.9}, {"label": "phone", "confidence": 0.8}]
    assert [item.text for item in first if item.source_type == "object"] == ["person, phone", "vehicle"]

    result = write_text_corpus(first, first_manifest, tmp_path / "output")
    assert result["document_count"] == 6
    assert result["corpus_sha256"] == first_manifest["corpus_sha256"]


def test_text_corpus_skips_missing_optional_stores_with_warning(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    _write_metadata(tmp_path)

    documents, manifest = build_text_corpus(tmp_path)

    assert len(documents) == 3
    assert manifest["status"] == "READY_WITH_WARNINGS"
    assert len(manifest["warnings"]) == 2


def test_text_corpus_combines_multiple_groups(tmp_path: Path) -> None:
    registry = tmp_path / "artifacts" / "registry" / "data_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "videos": [
                    {"video_id": "L21_V001", "group": "L21"},
                    {"video_id": "L22_V001", "group": "L22"},
                ]
            }
        ),
        encoding="utf-8",
    )
    for video_id in ("L21_V001", "L22_V001"):
        path = tmp_path / "data" / "media-info" / f"{video_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"title": f"Title {video_id}"}), encoding="utf-8")

    documents, manifest = build_text_corpus(
        tmp_path,
        CorpusBuildConfig(groups=("L22", "L21")),
        phase5_store_path=tmp_path / "missing-phase5.sqlite",
        object_store_path=tmp_path / "missing-objects.sqlite",
    )

    assert manifest["group"] is None
    assert manifest["groups"] == ["L21", "L22"]
    assert {item.video_id for item in documents} == {"L21_V001", "L22_V001"}
