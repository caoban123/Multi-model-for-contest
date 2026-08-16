from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from aic_retrieval.provenance import fingerprint_paths, sha256_file

OBJECT_SCHEMA_VERSION = "phase4-object-store-1.0"
OBJECT_SOURCE = "open_images_object_detection"


@dataclass(frozen=True)
class AliasDictionary:
    version: str
    aliases: dict[str, str]

    def normalize(self, label: str) -> str:
        folded = fold_label(label)
        return self.aliases.get(folded, folded.replace(" ", "_"))


def fold_label(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", without_marks.lower()).strip()


def load_alias_dictionary(path: Path) -> AliasDictionary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("dictionary_version", "")).strip()
    aliases = payload.get("aliases")
    if not version or not isinstance(aliases, dict):
        raise ValueError("alias dictionary requires dictionary_version and aliases")
    normalized: dict[str, str] = {}
    for raw, canonical in aliases.items():
        key = fold_label(str(raw))
        value = fold_label(str(canonical)).replace(" ", "_")
        if not key or not value:
            raise ValueError("alias keys and values must not be empty")
        if key in normalized and normalized[key] != value:
            raise ValueError(f"conflicting alias after normalization: {raw}")
        normalized[key] = value
    return AliasDictionary(version, normalized)


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE frames (
            video_id TEXT NOT NULL,
            keyframe_id INTEGER NOT NULL,
            frame_idx INTEGER NOT NULL,
            pts_time REAL NOT NULL,
            has_object_data INTEGER NOT NULL CHECK (has_object_data IN (0, 1)),
            object_status TEXT NOT NULL CHECK (object_status IN ('AVAILABLE', 'UNKNOWN')),
            PRIMARY KEY (video_id, keyframe_id)
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            video_id TEXT NOT NULL,
            keyframe_id INTEGER NOT NULL,
            label_id TEXT NOT NULL,
            label_raw TEXT NOT NULL,
            label_normalized TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_y1 REAL NOT NULL,
            bbox_x1 REAL NOT NULL,
            bbox_y2 REAL NOT NULL,
            bbox_x2 REAL NOT NULL,
            center_x REAL NOT NULL,
            center_y REAL NOT NULL,
            area REAL NOT NULL,
            source TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            FOREIGN KEY (video_id, keyframe_id) REFERENCES frames(video_id, keyframe_id)
        );
        CREATE INDEX idx_detections_label_confidence ON detections(label_normalized, confidence DESC);
        CREATE INDEX idx_detections_frame ON detections(video_id, keyframe_id);
        CREATE INDEX idx_detections_center_x ON detections(center_x);
        CREATE INDEX idx_detections_center_y ON detections(center_y);
        """
    )


def detection_rows(video_id: str, keyframe_id: int, payload: dict[str, Any], aliases: AliasDictionary) -> Iterable[tuple[Any, ...]]:
    fields = [payload.get(name) for name in ("detection_class_labels", "detection_class_entities", "detection_scores", "detection_boxes")]
    if any(not isinstance(value, list) for value in fields) or len({len(value) for value in fields}) != 1:
        raise ValueError("object arrays are missing or have different lengths")
    label_ids, entities, scores, boxes = fields
    for label_id, raw_label, raw_score, raw_box in zip(label_ids, entities, scores, boxes):
        confidence = float(raw_score)
        y1, x1, y2, x2 = (float(value) for value in raw_box)
        if not (0 <= confidence <= 1 and 0 <= y1 <= y2 <= 1 and 0 <= x1 <= x2 <= 1):
            raise ValueError("invalid confidence or bbox")
        yield (
            video_id, keyframe_id, str(label_id), str(raw_label), aliases.normalize(str(raw_label)), confidence,
            y1, x1, y2, x2, (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (y2 - y1),
            OBJECT_SOURCE, OBJECT_SCHEMA_VERSION,
        )


def build_object_store(
    refs_path: Path,
    object_root: Path,
    aliases_path: Path,
    output_path: Path,
    groups: set[str] | None = None,
) -> dict[str, Any]:
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    if groups:
        refs = [item for item in refs if str(item["group"]) in groups]
    aliases = load_alias_dictionary(aliases_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    connection = sqlite3.connect(output_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    create_schema(connection)
    detection_count = available = invalid = 0
    object_paths: list[Path] = []
    insert_sql = """INSERT INTO detections(video_id,keyframe_id,label_id,label_raw,label_normalized,confidence,bbox_y1,bbox_x1,bbox_y2,bbox_x2,center_x,center_y,area,source,schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    for ref in refs:
        video_id, keyframe_id = str(ref["video_id"]), int(ref["keyframe_id"])
        path = object_root / video_id / f"{keyframe_id:03d}.json"
        rows: list[tuple[Any, ...]] = []
        source_valid = False
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = list(detection_rows(video_id, keyframe_id, payload, aliases))
                object_paths.append(path)
                available += 1
                source_valid = True
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                invalid += 1
        # An empty but schema-valid file is still AVAILABLE and means no detections.
        has_data = int(source_valid)
        connection.execute("INSERT INTO frames VALUES (?,?,?,?,?,?)", (video_id, keyframe_id, int(ref["frame_idx"]), float(ref["pts_time"]), has_data, "AVAILABLE" if has_data else "UNKNOWN"))
        if rows:
            connection.executemany(insert_sql, rows)
            detection_count += len(rows)
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    video_count = len({str(item["video_id"]) for item in refs})
    source_fingerprint = fingerprint_paths([refs_path, aliases_path, *object_paths])
    return {
        "schema_version": OBJECT_SCHEMA_VERSION,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": source_fingerprint,
        "refs_fingerprint": sha256_file(refs_path),
        "label_dictionary_version": aliases.version,
        "label_dictionary_fingerprint": sha256_file(aliases_path),
        "video_count": video_count,
        "frame_count": len(refs),
        "detection_count": detection_count,
        "coverage": {"frames_with_object_data": available, "ratio": available / len(refs) if refs else 0.0},
        "missing_data": {"frames_unknown": len(refs) - available, "invalid_source_files": invalid, "semantics": "UNKNOWN, never NO_MATCH"},
        "build_configuration": {"groups": sorted(groups) if groups else "all", "detection_threshold": None, "nms_threshold": None, "all_valid_detections_preserved": True},
        "sqlite_integrity_check": integrity,
        "store_size_bytes": output_path.stat().st_size,
    }
