from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_VERSION = "phase10-data-capability-v1"


def audit(
    registry_path: Path,
    output_dir: Path,
    *,
    data_root: Path | None = None,
    phase5_store: Path | None = None,
    check_mappings: bool = False,
) -> dict[str, Path]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    root = _data_root(registry, registry_path, data_root)
    videos = list(registry.get("videos") or ())
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    metadata_fields = Counter()
    mapping_failures: list[dict[str, str]] = []
    refinement_ready: list[str] = []

    for asset in videos:
        group = str(asset.get("group") or str(asset.get("video_id") or "").split("_", 1)[0])
        stats = groups[group]
        stats["videos"] += 1
        paths = {
            "clip_features": _asset_path(asset.get("clip_feature_path"), root, registry_path.parent),
            "mapping": _asset_path(asset.get("mapping_path"), root, registry_path.parent),
            "metadata": _asset_path(asset.get("media_info_path"), root, registry_path.parent),
            "objects": _asset_path(asset.get("object_path"), root, registry_path.parent),
            "keyframes": _asset_path(asset.get("keyframe_path"), root, registry_path.parent),
            "video": _asset_path(asset.get("video_path"), root, registry_path.parent),
        }
        for name, path in paths.items():
            if path is not None and path.exists():
                stats[f"{name}_available"] += 1
        stats["keyframe_rows"] += int(asset.get("mapping_rows") or asset.get("keyframe_image_count") or 0)
        stats["object_files"] += int(asset.get("object_file_count") or 0)
        if paths["video"] and paths["video"].is_file() and paths["mapping"] and paths["mapping"].is_file():
            refinement_ready.append(str(asset.get("video_id")))
        if paths["metadata"] and paths["metadata"].is_file():
            try:
                metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
                for field in ("title", "author", "channel_id", "publish_date", "description", "keywords", "duration"):
                    if metadata.get(field) not in (None, "", []):
                        metadata_fields[field] += 1
            except (OSError, json.JSONDecodeError) as exc:
                mapping_failures.append({"video_id": str(asset.get("video_id")), "kind": "metadata", "error": str(exc)})
        if check_mappings and paths["mapping"] and paths["mapping"].is_file():
            failure = _mapping_failure(paths["mapping"])
            if failure:
                mapping_failures.append({"video_id": str(asset.get("video_id")), "kind": "mapping", "error": failure})

    phase5 = _phase5_coverage(phase5_store)
    generated_at = datetime.now(timezone.utc).isoformat()
    capability = {
        "audit_version": AUDIT_VERSION,
        "generated_at": generated_at,
        "registry": str(registry_path),
        "data_root": str(root),
        "video_count": len(videos),
        "groups": {group: dict(sorted(stats.items())) for group, stats in sorted(groups.items())},
        "metadata_field_video_counts": dict(metadata_fields),
        "mapping_check_enabled": check_mappings,
        "mapping_or_metadata_failures": mapping_failures,
        "status": "READY" if videos and not mapping_failures else ("PARTIAL" if videos else "EMPTY"),
    }
    qa = {
        "audit_version": AUDIT_VERSION,
        "generated_at": generated_at,
        "video_count": len(videos),
        "keyframe_video_count": sum(stats["keyframes_available"] for stats in groups.values()),
        "object_video_count": sum(stats["objects_available"] for stats in groups.values()),
        "metadata_video_count": sum(stats["metadata_available"] for stats in groups.values()),
        "ocr": phase5["ocr"],
        "asr": phase5["asr"],
        "semantics": "Missing OCR/ASR is UNAVAILABLE, never NO_MATCH.",
    }
    trake = {
        "audit_version": AUDIT_VERSION,
        "generated_at": generated_at,
        "video_count": len(videos),
        "refinement_ready_video_count": len(refinement_ready),
        "refinement_ready_video_ids": refinement_ready,
        "requirements": ["raw video", "mapping", "decoder", "CLIP scorer"],
        "mapping_check_enabled": check_mappings,
        "mapping_failure_count": sum(item["kind"] == "mapping" for item in mapping_failures),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "capability": output_dir / "phase10_data_capability.json",
        "qa": output_dir / "phase10_qa_modality_coverage.json",
        "trake": output_dir / "phase10_trake_refinement_coverage.json",
    }
    for name, payload in (("capability", capability), ("qa", qa), ("trake", trake)):
        outputs[name].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def _data_root(registry: dict[str, Any], registry_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    value = Path(str(registry.get("data_root") or "data"))
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _asset_path(value: Any, data_root: Path, registry_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0].lower() == "data":
        return data_root.joinpath(*parts[1:])
    candidate = Path.cwd() / path
    return candidate if candidate.exists() else registry_dir / path


def _mapping_failure(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return "mapping is empty"
        frame_key = next((key for key in ("frame_idx", "frame_id", "frame") if key in rows[0]), None)
        pts_key = next((key for key in ("pts_time", "timestamp", "time") if key in rows[0]), None)
        for key in (frame_key, pts_key):
            if key is None:
                continue
            values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
            if any(right < left for left, right in zip(values, values[1:])):
                return f"{key} is not monotonic"
        return None
    except (OSError, ValueError, csv.Error) as exc:
        return str(exc)


def _phase5_coverage(path: Path | None) -> dict[str, dict[str, Any]]:
    result = {name: {"available": False, "record_count": 0, "video_count": 0} for name in ("ocr", "asr")}
    if path is None or not path.is_file():
        return result
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in result:
            if name in tables:
                count, videos = connection.execute(f"SELECT COUNT(*), COUNT(DISTINCT video_id) FROM {name}").fetchone()
                result[name] = {"available": True, "record_count": int(count), "video_count": int(videos)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Phase-10 Q&A/TRAKE data capabilities without building indexes.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--phase5-store", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audits"))
    parser.add_argument("--check-mappings", action="store_true")
    args = parser.parse_args()
    outputs = audit(args.registry, args.output_dir, data_root=args.data_root, phase5_store=args.phase5_store, check_mappings=args.check_mappings)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
