from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aic_retrieval.phase5_consolidated import (
    ConsolidatedImportStats,
    build_consolidated_phase5_store,
    consolidated_group_files,
    iter_consolidated_ocr,
    refs_by_video,
)
from aic_retrieval.phase5_schema import AsrSegment, AsrTranscript
from aic_retrieval.phase5_store import Phase5SearchService, create_store, ingest_asr
from aic_retrieval.search import FrameRef


def ref(keyframe_id: int, frame_idx: int, pts_time: float) -> FrameRef:
    return FrameRef("L21_V001", "L21", keyframe_id, frame_idx, pts_time, 30.0, f"{keyframe_id:03}.jpg")


def source_row(*, timestamp_ms: int = 900, text: str = "THỜI SỰ 19H", detections: bool = True) -> dict:
    row = {
        "schema_version": "ocr-keyframe-v2",
        "collection_id": "L21",
        "video_id": "L21_V001",
        "keyframe_id": f"L21_V001_F{timestamp_ms // 10:09d}",
        "timestamp_ms": timestamp_ms,
        "status": "visually_corrected",
        "image": {"width": 200, "height": 100},
        "retrieval_text": text,
        "processing_config_sha256": "abc123",
    }
    row["retrieval_detections"] = (
        [{"text": text, "score": 0.91, "bbox": [20, 10, 180, 90]}] if detections else []
    )
    return row


def kaggle_source_row() -> dict:
    row = source_row(timestamp_ms=900, text="HTV Online")
    row["schema_version"] = "ocr-kaggle-paddle-v1"
    row["timestamp_ms"] = None
    row["frame_number"] = 28
    row["image"] = {}
    row["retrieval_detections"] = [
        {"text": "HTY", "confidence": 0.8, "bbox": [0, 0, 100, 50]},
        {"text": "Online", "confidence": 0.9, "bbox": [0, 50, 100, 100]},
    ]
    return row


def write_source(root: Path, rows: list[dict]) -> Path:
    ocr = root / "consolidated_L21_L30" / "ocr"
    group = ocr / "L21"
    group.mkdir(parents=True)
    (ocr / "manifest.json").write_text(
        json.dumps({"schema": "ocr-consolidated-l21-l30-v1", "collections": ["L21"]}),
        encoding="utf-8",
    )
    (group / "ocr.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def write_refs(path: Path) -> list[FrameRef]:
    refs = [ref(1, 0, 0.0), ref(2, 30, 1.0), ref(3, 90, 3.0)]
    path.write_text(json.dumps([item.__dict__ for item in refs]), encoding="utf-8")
    return refs


def test_consolidated_ocr_maps_timestamp_and_normalizes_bbox(tmp_path: Path) -> None:
    root = write_source(tmp_path / "source", [source_row()])
    refs = refs_by_video([ref(1, 0, 0.0), ref(2, 30, 1.0)])
    stats = ConsolidatedImportStats()

    rows = list(iter_consolidated_ocr(
        consolidated_group_files(root / "consolidated_L21_L30" / "ocr", ["L21"]),
        refs,
        max_mapping_distance=1.0,
        stats=stats,
    ))

    assert len(rows) == 1
    assert (rows[0].keyframe_id, rows[0].frame_idx) == (2, 30)
    assert rows[0].mapping_distance_seconds == pytest.approx(0.1)
    assert rows[0].bbox == ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
    assert rows[0].text == "THỜI SỰ 19H"
    assert stats.to_dict()["mapping_distance_buckets"] == {"le_0.25s": 1}


def test_consolidated_ocr_skips_distant_rows_and_uses_reviewed_text_fallback(tmp_path: Path) -> None:
    root = write_source(
        tmp_path / "source",
        [source_row(timestamp_ms=900, text="ĐÃ SỬA", detections=False), source_row(timestamp_ms=9000)],
    )
    stats = ConsolidatedImportStats()
    rows = list(iter_consolidated_ocr(
        consolidated_group_files(root / "consolidated_L21_L30" / "ocr", ["L21"]),
        refs_by_video([ref(1, 0, 0.0), ref(2, 30, 1.0)]),
        max_mapping_distance=1.0,
        stats=stats,
    ))

    assert [row.text for row in rows] == ["ĐÃ SỬA"]
    assert rows[0].bbox == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert stats.fallback_text_detections == 1
    assert stats.unmapped_frames == 1


def test_kaggle_schema_maps_frame_number_and_indexes_visual_correction(tmp_path: Path) -> None:
    root = write_source(tmp_path / "source", [kaggle_source_row()])
    stats = ConsolidatedImportStats()
    rows = list(iter_consolidated_ocr(
        consolidated_group_files(root / "consolidated_L21_L30" / "ocr", ["L21"]),
        refs_by_video([ref(1, 0, 0.0), ref(2, 30, 1.0)]),
        max_mapping_distance=1.0,
        stats=stats,
    ))

    assert rows[-1].text == "HTV Online"
    assert rows[-1].keyframe_id == 2
    assert rows[0].bbox[2] == pytest.approx((100 / 1280, 50 / 720), abs=1e-6)
    assert stats.invalid_rows == 0
    assert stats.fallback_text_detections == 1


def test_atomic_builder_preserves_asr_and_exposes_searchable_ocr_provenance(tmp_path: Path) -> None:
    source = write_source(tmp_path / "source", [source_row()])
    refs_path = tmp_path / "refs.json"
    refs = write_refs(refs_path)
    base = tmp_path / "base.sqlite3"
    connection = create_store(base)
    ingest_asr(
        connection,
        [AsrTranscript(
            "L21_V001",
            "AVAILABLE",
            (AsrSegment(0, 0.8, 1.2, "Xin chào quý vị", 0.8),),
            duration=2.0,
            run_id="asr-base",
        )],
        refs,
    )
    connection.close()
    output = tmp_path / "phase5.sqlite3"
    manifest = tmp_path / "manifest.json"

    result = build_consolidated_phase5_store(
        ocr_root=source,
        refs_path=refs_path,
        groups=["L21"],
        output=output,
        base_store=base,
        manifest_output=manifest,
        max_mapping_distance=1.0,
    )

    assert result["status"] == "READY"
    assert result["ocr_detections"] == 1
    assert result["asr_segments"] == 1
    assert manifest.is_file()
    service = Phase5SearchService(output, refs)
    assert service.search_ocr("thoi su")[0]["source_keyframe_id"].startswith("L21_V001_F")
    assert service.search_asr("xin chao")[0]["text_raw"] == "Xin chào quý vị"
    with sqlite3.connect(output) as check:
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert check.execute("SELECT COUNT(*) FROM ocr_fts").fetchone()[0] == 1


def test_existing_store_schema_is_migrated_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
          CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE ocr(
            id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, keyframe_id INTEGER NOT NULL,
            frame_idx INTEGER NOT NULL, pts_time REAL NOT NULL, text_raw TEXT NOT NULL,
            text_normalized TEXT NOT NULL, text_folded TEXT NOT NULL, confidence REAL NOT NULL,
            bbox_json TEXT NOT NULL, run_id TEXT NOT NULL);
        """)
    connection = create_store(path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(ocr)")}
    connection.close()
    assert {"source_keyframe_id", "mapping_distance_seconds", "source_status"} <= columns
