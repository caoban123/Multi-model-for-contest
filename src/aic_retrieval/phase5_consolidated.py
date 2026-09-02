from __future__ import annotations

import bisect
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from aic_retrieval.phase5_schema import normalize_text
from aic_retrieval.phase5_store import create_store
from aic_retrieval.provenance import sha256_file
from aic_retrieval.search import FrameRef


CONSOLIDATED_IMPORT_VERSION = "phase5-consolidated-import-v1"
_SOURCE_FRAME_PATTERN = re.compile(r"_F(\d+)$")


@dataclass(frozen=True)
class ConsolidatedOcrDetection:
    video_id: str
    keyframe_id: int
    frame_idx: int
    pts_time: float
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...]
    run_id: str
    source_keyframe_id: str
    source_frame_idx: int | None
    source_pts_time: float
    mapping_distance_seconds: float
    source_status: str


@dataclass(frozen=True)
class RefTimeline:
    refs: tuple[FrameRef, ...]
    pts_times: tuple[float, ...]
    frame_refs: tuple[FrameRef, ...]
    frame_indices: tuple[int, ...]


@dataclass
class ConsolidatedImportStats:
    source_frames: int = 0
    mapped_frames: int = 0
    unmapped_frames: int = 0
    empty_frames: int = 0
    imported_detections: int = 0
    fallback_text_detections: int = 0
    invalid_rows: int = 0
    max_mapping_distance_seconds: float = 0.0
    groups: Counter[str] = field(default_factory=Counter)
    statuses: Counter[str] = field(default_factory=Counter)
    mapping_distance_buckets: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["groups"] = dict(sorted(self.groups.items()))
        payload["statuses"] = dict(sorted(self.statuses.items()))
        payload["mapping_distance_buckets"] = dict(self.mapping_distance_buckets)
        payload["max_mapping_distance_seconds"] = round(self.max_mapping_distance_seconds, 6)
        return payload


def resolve_ocr_root(path: Path) -> Path:
    candidates = (
        path,
        path / "ocr",
        path / "consolidated_L21_L30" / "ocr",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find consolidated OCR manifest below {path}. "
        "Expected manifest.json in the path itself, ocr/, or consolidated_L21_L30/ocr/."
    )


def load_refs(path: Path) -> list[FrameRef]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"refs file must contain a JSON array: {path}")
    return [FrameRef(**item) for item in payload]


def refs_by_video(refs: Iterable[FrameRef]) -> dict[str, RefTimeline]:
    grouped: dict[str, list[FrameRef]] = defaultdict(list)
    for ref in refs:
        grouped[ref.video_id].append(ref)
    timelines = {}
    for video_id, items in grouped.items():
        ordered = tuple(sorted(items, key=lambda item: (item.pts_time, item.keyframe_id)))
        frame_ordered = tuple(sorted(items, key=lambda item: (item.frame_idx, item.keyframe_id)))
        timelines[video_id] = RefTimeline(
            ordered,
            tuple(item.pts_time for item in ordered),
            frame_ordered,
            tuple(item.frame_idx for item in frame_ordered),
        )
    return timelines


def nearest_ref(timeline: RefTimeline, pts_time: float) -> tuple[FrameRef, float] | None:
    if not timeline.refs:
        return None
    position = bisect.bisect_left(timeline.pts_times, pts_time)
    candidates = timeline.refs[max(0, position - 1):min(len(timeline.refs), position + 1)]
    selected = min(candidates, key=lambda item: (abs(item.pts_time - pts_time), item.keyframe_id))
    return selected, abs(selected.pts_time - pts_time)


def nearest_ref_by_frame(timeline: RefTimeline, frame_idx: int) -> tuple[FrameRef, float] | None:
    if not timeline.frame_refs:
        return None
    position = bisect.bisect_left(timeline.frame_indices, frame_idx)
    candidates = timeline.frame_refs[
        max(0, position - 1):min(len(timeline.frame_refs), position + 1)
    ]
    selected = min(candidates, key=lambda item: (abs(item.frame_idx - frame_idx), item.keyframe_id))
    return selected, abs(selected.frame_idx - frame_idx) / selected.fps


def consolidated_group_files(ocr_root: Path, groups: Iterable[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for group in sorted(set(groups)):
        path = ocr_root / group / "ocr.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing consolidated OCR group file: {path}")
        files.append((group, path))
    return files


def iter_consolidated_ocr(
    files: Iterable[tuple[str, Path]],
    refs: Mapping[str, RefTimeline],
    *,
    max_mapping_distance: float,
    stats: ConsolidatedImportStats,
) -> Iterator[ConsolidatedOcrDetection]:
    if max_mapping_distance < 0:
        raise ValueError("max_mapping_distance must be non-negative")
    for expected_group, path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                stats.source_frames += 1
                try:
                    raw = json.loads(line)
                    yield from _map_source_frame(
                        raw,
                        expected_group,
                        refs,
                        max_mapping_distance=max_mapping_distance,
                        stats=stats,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    stats.invalid_rows += 1


def _map_source_frame(
    raw: Mapping[str, Any],
    expected_group: str,
    refs: Mapping[str, RefTimeline],
    *,
    max_mapping_distance: float,
    stats: ConsolidatedImportStats,
) -> Iterator[ConsolidatedOcrDetection]:
    group = str(raw["collection_id"])
    video_id = str(raw["video_id"])
    if group != expected_group or not video_id.startswith(f"{group}_"):
        raise ValueError("consolidated OCR collection/video identity mismatch")
    source_keyframe_id = str(raw["keyframe_id"])
    source_frame_match = _SOURCE_FRAME_PATTERN.search(source_keyframe_id)
    source_frame_value = raw.get("frame_number")
    source_frame_idx = (
        int(source_frame_value)
        if source_frame_value is not None
        else int(source_frame_match.group(1)) if source_frame_match else None
    )
    status = str(raw.get("status") or "unknown")
    stats.groups[group] += 1
    stats.statuses[status] += 1

    timeline = refs.get(video_id)
    timestamp_ms = raw.get("timestamp_ms")
    if timestamp_ms is not None:
        source_pts_time = float(timestamp_ms) / 1000.0
        mapped = nearest_ref(timeline, source_pts_time) if timeline else None
    elif timeline and source_frame_idx is not None:
        mapped = nearest_ref_by_frame(timeline, source_frame_idx)
        source_pts_time = source_frame_idx / mapped[0].fps if mapped else 0.0
    else:
        mapped = None
        source_pts_time = 0.0
    if mapped is None or mapped[1] > max_mapping_distance:
        stats.unmapped_frames += 1
        return
    ref, distance = mapped
    stats.mapped_frames += 1
    stats.max_mapping_distance_seconds = max(stats.max_mapping_distance_seconds, distance)
    stats.mapping_distance_buckets[_distance_bucket(distance)] += 1

    detections = raw.get("retrieval_detections")
    if not isinstance(detections, list):
        detections = []
    reviewed_text = str(raw.get("retrieval_text") or "").strip()
    detected_text = " ".join(
        str(item.get("text") or "").strip()
        for item in detections
        if isinstance(item, Mapping) and str(item.get("text") or "").strip()
    )
    if reviewed_text and (not detections or normalize_text(reviewed_text) != normalize_text(detected_text)):
        detections = [
            *detections,
            {
                "text": reviewed_text,
                "score": 1.0,
                "polygon": _full_image_polygon(raw),
                "_reviewed_fallback": True,
            },
        ]
    if not detections:
        stats.empty_frames += 1
        return

    image = raw.get("image") if isinstance(raw.get("image"), Mapping) else {}
    width = float(image.get("width") or 1280)
    height = float(image.get("height") or 720)
    run_id = ":".join(
        part for part in (
            str(raw.get("schema_version") or "ocr-keyframe-v2"),
            str(raw.get("processing_config_sha256") or "unknown")[:12],
        ) if part
    )
    seen: set[tuple[str, tuple[tuple[float, float], ...]]] = set()
    for item in detections:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        bbox = _normalized_polygon(item, width, height)
        identity = (normalize_text(text), bbox)
        if identity in seen:
            continue
        seen.add(identity)
        confidence = _confidence(item)
        stats.imported_detections += 1
        if item.get("_reviewed_fallback") is True:
            stats.fallback_text_detections += 1
        yield ConsolidatedOcrDetection(
            video_id=video_id,
            keyframe_id=ref.keyframe_id,
            frame_idx=ref.frame_idx,
            pts_time=ref.pts_time,
            text=text,
            confidence=confidence,
            bbox=bbox,
            run_id=run_id,
            source_keyframe_id=source_keyframe_id,
            source_frame_idx=source_frame_idx,
            source_pts_time=source_pts_time,
            mapping_distance_seconds=distance,
            source_status=status,
        )


def _confidence(item: Mapping[str, Any]) -> float:
    for key in ("score", "recognizer_confidence", "detector_confidence"):
        value = item.get(key)
        if value is not None:
            return max(0.0, min(1.0, float(value)))
    return 1.0


def _full_image_polygon(raw: Mapping[str, Any]) -> list[list[float]]:
    image = raw.get("image") if isinstance(raw.get("image"), Mapping) else {}
    width = float(image.get("width") or 1280)
    height = float(image.get("height") or 720)
    return [[0, 0], [width, 0], [width, height], [0, height]]


def _normalized_polygon(item: Mapping[str, Any], width: float, height: float) -> tuple[tuple[float, float], ...]:
    points = item.get("polygon")
    if not isinstance(points, list) or len(points) != 4:
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if not isinstance(points, list) or len(points) != 4 or width <= 0 or height <= 0:
        return ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        normalized.append((
            round(max(0.0, min(1.0, float(point[0]) / width)), 6),
            round(max(0.0, min(1.0, float(point[1]) / height)), 6),
        ))
    return tuple(normalized)


def _distance_bucket(distance: float) -> str:
    for limit in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        if distance <= limit:
            return f"le_{limit:g}s"
    return "gt_5s"


def build_consolidated_phase5_store(
    *,
    ocr_root: Path,
    refs_path: Path,
    groups: Iterable[str],
    output: Path,
    base_store: Path | None = None,
    manifest_output: Path | None = None,
    max_mapping_distance: float = 5.0,
    max_invalid_rows: int = 0,
    overwrite: bool = False,
    progress: Callable[[ConsolidatedImportStats], None] | None = None,
) -> dict[str, Any]:
    ocr_root = resolve_ocr_root(ocr_root)
    source_manifest_path = ocr_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected_groups = tuple(sorted(set(groups)))
    if not selected_groups:
        raise ValueError("at least one group is required")
    files = consolidated_group_files(ocr_root, selected_groups)
    all_refs = load_refs(refs_path)
    grouped_refs = refs_by_video(ref for ref in all_refs if ref.group in selected_groups)
    missing_videos = sorted(
        video_id
        for group in selected_groups
        for video_id in _manifest_video_ids(source_manifest, group)
        if video_id not in grouped_refs
    )
    if missing_videos:
        raise ValueError(f"OCR source contains videos absent from refs: {missing_videos[:10]}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"store already exists: {output}; pass --overwrite to rebuild")
    if base_store is not None and not base_store.is_file():
        raise FileNotFoundError(f"base Phase 5 store does not exist: {base_store}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building")
    temporary.unlink(missing_ok=True)
    stats = ConsolidatedImportStats()
    connection = create_store(temporary)
    try:
        connection.commit()
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("BEGIN")
        batch: list[tuple[Any, ...]] = []
        for detection in iter_consolidated_ocr(
            files,
            grouped_refs,
            max_mapping_distance=max_mapping_distance,
            stats=stats,
        ):
            batch.append(_ocr_insert_row(detection))
            if len(batch) >= 5_000:
                _insert_ocr_batch(connection, batch)
                batch.clear()
                if progress:
                    progress(stats)
        if batch:
            _insert_ocr_batch(connection, batch)
        if stats.invalid_rows > max_invalid_rows:
            raise ValueError(
                f"consolidated OCR contains {stats.invalid_rows} invalid rows; "
                f"configured maximum is {max_invalid_rows}"
            )
        expected_rows = sum(
            int(source_manifest.get("per_collection", {}).get(group, {}).get("rows", 0))
            for group in selected_groups
        )
        if expected_rows and stats.source_frames != expected_rows:
            raise ValueError(
                f"consolidated OCR row count mismatch: read {stats.source_frames}, expected {expected_rows}"
            )
        connection.commit()
        asr_segments = _copy_asr(connection, base_store, selected_groups) if base_store else 0
        metadata = {
            "consolidated_import_version": CONSOLIDATED_IMPORT_VERSION,
            "consolidated_source_schema": str(source_manifest.get("schema") or "unknown"),
            "consolidated_source_manifest": str(source_manifest_path),
            "consolidated_source_manifest_sha256": sha256_file(source_manifest_path),
            "consolidated_groups": json.dumps(selected_groups),
            "consolidated_max_mapping_distance_seconds": str(max_mapping_distance),
            "consolidated_max_invalid_rows": str(max_invalid_rows),
            "consolidated_import_stats": json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True),
            "asr_source_store": str(base_store) if base_store else "",
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        connection.executemany("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", metadata.items())
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        ocr_count = int(connection.execute("SELECT COUNT(*) FROM ocr").fetchone()[0])
        ocr_fts_count = int(connection.execute("SELECT COUNT(*) FROM ocr_fts").fetchone()[0])
        asr_count = int(connection.execute("SELECT COUNT(*) FROM asr").fetchone()[0])
        asr_fts_count = int(connection.execute("SELECT COUNT(*) FROM asr_fts").fetchone()[0])
        if integrity != "ok" or ocr_count != ocr_fts_count or asr_count != asr_fts_count:
            raise RuntimeError(
                "Phase 5 store validation failed: "
                f"integrity={integrity}, ocr={ocr_count}/{ocr_fts_count}, asr={asr_count}/{asr_fts_count}"
            )
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(temporary, output)
    result = {
        "status": "READY",
        "output": str(output),
        "groups": list(selected_groups),
        "ocr_detections": ocr_count,
        "asr_segments": asr_count,
        "asr_segments_copied": asr_segments,
        "sqlite_integrity_check": integrity,
        "store_size_bytes": output.stat().st_size,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": metadata["consolidated_source_manifest_sha256"],
        "stats": stats.to_dict(),
    }
    if manifest_output:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _manifest_video_ids(manifest: Mapping[str, Any], group: str) -> tuple[str, ...]:
    # Current source manifest records ranges/counts, not an explicit video list.
    # Identity validation therefore happens row-by-row against refs.
    return ()


def _ocr_insert_row(item: ConsolidatedOcrDetection) -> tuple[Any, ...]:
    return (
        item.video_id,
        item.keyframe_id,
        item.frame_idx,
        item.pts_time,
        item.text,
        normalize_text(item.text),
        normalize_text(item.text, fold_accents=True),
        item.confidence,
        json.dumps(item.bbox, separators=(",", ":")),
        item.run_id,
        item.source_keyframe_id,
        item.source_frame_idx,
        item.source_pts_time,
        item.mapping_distance_seconds,
        item.source_status,
    )


def _insert_ocr_batch(connection: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> None:
    connection.executemany(
        """
        INSERT INTO ocr(
          video_id,keyframe_id,frame_idx,pts_time,text_raw,text_normalized,text_folded,
          confidence,bbox_json,run_id,source_keyframe_id,source_frame_idx,source_pts_time,
          mapping_distance_seconds,source_status
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def _copy_asr(connection: sqlite3.Connection, base_store: Path, groups: Sequence[str]) -> int:
    connection.execute("ATTACH DATABASE ? AS phase5_base", (str(base_store),))
    try:
        table = connection.execute(
            "SELECT 1 FROM phase5_base.sqlite_master WHERE type='table' AND name='asr'"
        ).fetchone()
        if table is None:
            return 0
        connection.execute("BEGIN")
        group_clause = " OR ".join("video_id LIKE ?" for _ in groups)
        connection.execute(
            f"""
            INSERT INTO asr(
              video_id,segment_id,start_time,end_time,text_raw,text_normalized,text_folded,
              confidence,keyframe_id,frame_idx,pts_time,mapping_kind,distance_seconds,run_id
            )
            SELECT video_id,segment_id,start_time,end_time,text_raw,text_normalized,text_folded,
                   confidence,keyframe_id,frame_idx,pts_time,mapping_kind,distance_seconds,run_id
            FROM phase5_base.asr
            WHERE {group_clause}
            """,
            tuple(f"{group}_%" for group in groups),
        )
        connection.commit()
        return int(connection.execute("SELECT COUNT(*) FROM asr").fetchone()[0])
    finally:
        connection.execute("DETACH DATABASE phase5_base")
