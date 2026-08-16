from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


OBJECT_ARRAY_FIELDS = (
    "detection_class_entities",
    "detection_scores",
    "detection_boxes",
    "detection_class_names",
    "detection_class_labels",
)
METADATA_FIELDS = ("video_id", "title", "author", "publish_date", "keywords", "description", "duration", "group")


@dataclass(frozen=True)
class FrameIdentity:
    video_id: str
    keyframe_id: int


def load_frame_identities(refs_path: Path, groups: set[str] | None = None) -> list[FrameIdentity]:
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    return [
        FrameIdentity(str(item["video_id"]), int(item["keyframe_id"]))
        for item in refs
        if not groups or str(item["group"]) in groups
    ]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def valid_bbox(raw: Any) -> bool:
    try:
        ymin, xmin, ymax, xmax = (float(value) for value in raw)
    except (TypeError, ValueError):
        return False
    return 0.0 <= ymin <= ymax <= 1.0 and 0.0 <= xmin <= xmax <= 1.0


def object_json_path(object_root: Path, frame: FrameIdentity) -> Path:
    return object_root / frame.video_id / f"{frame.keyframe_id:03d}.json"


def audit_objects(object_root: Path, frames: Iterable[FrameIdentity]) -> dict[str, Any]:
    frames = list(frames)
    videos = sorted({frame.video_id for frame in frames})
    frames_by_video: dict[str, list[FrameIdentity]] = {video_id: [] for video_id in videos}
    for frame in frames:
        frames_by_video[frame.video_id].append(frame)
    existing_paths = {path.resolve() for path in object_root.glob("*/*.json")}
    expected_paths = {object_json_path(object_root, frame).resolve() for frame in frames}
    scores: list[float] = []
    labels: Counter[str] = Counter()
    invalid_json = length_mismatch = invalid_bbox = invalid_score = missing_fields = 0
    frames_with_data = 0
    detections = 0
    per_video: dict[str, dict[str, Any]] = {}

    for video_id, video_frames in frames_by_video.items():
        available = 0
        for frame in video_frames:
            path = object_json_path(object_root, frame)
            if not path.exists():
                continue
            available += 1
            frames_with_data += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_json += 1
                continue
            if any(not isinstance(payload.get(field), list) for field in OBJECT_ARRAY_FIELDS):
                missing_fields += 1
                continue
            lengths = {len(payload[field]) for field in OBJECT_ARRAY_FIELDS}
            if len(lengths) != 1:
                length_mismatch += 1
                continue
            entities = payload["detection_class_entities"]
            raw_scores = payload["detection_scores"]
            boxes = payload["detection_boxes"]
            detections += len(entities)
            for entity, raw_score, box in zip(entities, raw_scores, boxes):
                labels[str(entity).strip() or "<empty>"] += 1
                try:
                    score = float(raw_score)
                    if not 0.0 <= score <= 1.0:
                        raise ValueError
                    scores.append(score)
                except (TypeError, ValueError):
                    invalid_score += 1
                if not valid_bbox(box):
                    invalid_bbox += 1
        per_video[video_id] = {
            "frames_total": len(video_frames),
            "frames_with_object_data": available,
            "frames_missing_object_data": len(video_frames) - available,
            "status_when_queried": "AVAILABLE" if available == len(video_frames) else ("UNKNOWN" if available == 0 else "PARTIAL_UNKNOWN"),
        }

    videos_with = [video_id for video_id, item in per_video.items() if item["frames_with_object_data"] > 0]
    videos_missing = [video_id for video_id, item in per_video.items() if item["frames_with_object_data"] == 0]
    thresholds = (0.1, 0.2, 0.3, 0.5)
    return {
        "semantics": {
            "MATCH": "object data exists and the query predicate matches",
            "NO_MATCH": "object data exists and the query predicate does not match",
            "UNKNOWN": "object data is missing or invalid; it is never treated as NO_MATCH",
        },
        "videos_total": len(videos),
        "videos_with_object_data": len(videos_with),
        "video_ids_with_object_data": videos_with,
        "videos_missing_object_data": len(videos_missing),
        "video_ids_missing_object_data": videos_missing,
        "frames_total": len(frames),
        "frames_with_object_data": frames_with_data,
        "frames_missing_object_data": len(frames) - frames_with_data,
        "object_files_outside_frame_registry": len(existing_paths - expected_paths),
        "invalid_json_count": invalid_json,
        "missing_array_fields_count": missing_fields,
        "length_mismatch_count": length_mismatch,
        "invalid_bbox_count": invalid_bbox,
        "invalid_score_count": invalid_score,
        "detection_count_valid_records": detections,
        "confidence_distribution": {
            "count": len(scores), "min": min(scores) if scores else None,
            "p25": percentile(scores, 0.25), "p50": percentile(scores, 0.50),
            "p75": percentile(scores, 0.75), "p95": percentile(scores, 0.95), "max": max(scores) if scores else None,
            "counts_at_or_above_threshold": {str(value): sum(score >= value for score in scores) for value in thresholds},
        },
        "unique_label_count": len(labels),
        "label_frequency": [{"label": label, "count": count} for label, count in labels.most_common()],
        "per_video": per_video,
    }


def audit_metadata(media_dir: Path, groups: set[str] | None = None) -> dict[str, Any]:
    paths = sorted(media_dir.glob("*.json"))
    if groups:
        paths = [path for path in paths if path.stem.split("_", 1)[0] in groups]
    invalid_json = 0
    coverage: Counter[str] = Counter()
    valid = 0
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid_json += 1
            continue
        valid += 1
        video_id = str(payload.get("video_id") or path.stem)
        if video_id in seen:
            duplicate_ids.append(video_id)
        seen.add(video_id)
        values = {
            "video_id": video_id,
            "title": payload.get("title"),
            "author": payload.get("author") or payload.get("channel"),
            "publish_date": payload.get("publish_date"),
            "keywords": payload.get("keywords"),
            "description": payload.get("description"),
            "duration": payload.get("duration") if payload.get("duration") is not None else payload.get("length"),
            "group": video_id.split("_", 1)[0] if "_" in video_id else "",
        }
        for field, value in values.items():
            if value not in (None, "", []):
                coverage[field] += 1
    return {
        "files_total": len(paths), "valid_json_count": valid, "invalid_json_count": invalid_json,
        "duplicate_video_ids": sorted(set(duplicate_ids)),
        "field_coverage": {
            field: {"present": coverage[field], "missing": valid - coverage[field], "ratio": coverage[field] / valid if valid else 0.0}
            for field in METADATA_FIELDS
        },
        "scope_groups": sorted(groups) if groups else "all",
        "granularity": "video",
    }
