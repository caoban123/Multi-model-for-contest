from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aic_retrieval.resilient_io import sha256_file_with_retry


AUDIT_SCHEMA_VERSION = "hybrid-retrieval-audit-v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    return sha256_file_with_retry(path, chunk_size=chunk_size)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def _path_info(path: Path, checksum: bool = False) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        info["size_bytes"] = path.stat().st_size
        if checksum:
            info["sha256"] = sha256_file(path)
    return info


def audit_registry(registry_path: Path, root: Path, groups: set[str]) -> dict[str, Any]:
    if not registry_path.is_file():
        return {"status": "MISSING", "path": str(registry_path), "videos": 0}
    payload = _load_json(registry_path)
    videos = [item for item in payload.get("videos", []) if str(item.get("group")) in groups]
    path_fields = (
        "clip_feature_path",
        "mapping_path",
        "media_info_path",
        "object_path",
        "keyframe_path",
        "video_path",
    )
    availability: dict[str, dict[str, Any]] = {}
    for field in path_fields:
        present = 0
        size_bytes = 0
        for item in videos:
            raw = item.get(field)
            path = root / str(raw) if raw else None
            if path and path.exists():
                present += 1
                if path.is_file():
                    size_bytes += path.stat().st_size
        availability[field] = {
            "present": present,
            "missing": len(videos) - present,
            "coverage": _ratio(present, len(videos)),
            "file_size_bytes": size_bytes,
        }
    ids = [str(item.get("video_id", "")) for item in videos]
    return {
        "status": "READY" if videos else "EMPTY",
        "path": str(registry_path),
        "sha256": sha256_file(registry_path),
        "registry_version": payload.get("version"),
        "groups": sorted(groups),
        "videos": len(videos),
        "unique_video_ids": len(set(ids)),
        "duplicate_video_ids": len(ids) - len(set(ids)),
        "video_ids": sorted(set(ids)),
        "source_availability": availability,
    }


def _norm_stats(vectors: np.ndarray, chunk_size: int = 4096) -> dict[str, Any]:
    norm_parts: list[np.ndarray] = []
    all_finite = True
    for start in range(0, len(vectors), chunk_size):
        block = np.asarray(vectors[start : start + chunk_size], dtype=np.float32)
        all_finite = all_finite and bool(np.isfinite(block).all())
        norm_parts.append(np.linalg.norm(block, axis=1))
    norms = np.concatenate(norm_parts) if norm_parts else np.asarray([], dtype=np.float32)
    return {
        "all_finite": all_finite,
        "zero_norm_vectors": int(np.count_nonzero(norms == 0)),
        "norm_min": round(float(norms.min()), 6) if len(norms) else None,
        "norm_max": round(float(norms.max()), 6) if len(norms) else None,
        "norm_mean": round(float(norms.mean()), 6) if len(norms) else None,
    }


def audit_clip_index(index_dir: Path, root: Path, groups: set[str]) -> dict[str, Any]:
    vectors_path = index_dir / "vectors.npy"
    refs_path = index_dir / "refs.json"
    metadata_path = index_dir / "metadata.json"
    files = {
        "vectors": _path_info(vectors_path, checksum=vectors_path.is_file()),
        "refs": _path_info(refs_path, checksum=refs_path.is_file()),
        "metadata": _path_info(metadata_path, checksum=metadata_path.is_file()),
    }
    if not vectors_path.is_file() or not refs_path.is_file():
        return {"status": "MISSING", "index_dir": str(index_dir), "files": files}

    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    refs = _load_json(refs_path)
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    scoped_positions = [index for index, ref in enumerate(refs) if str(ref.get("group")) in groups]
    keys = [(str(ref.get("video_id")), int(ref.get("keyframe_id", -1))) for ref in refs]
    scoped_refs = [refs[index] for index in scoped_positions]
    scoped_keys = [keys[index] for index in scoped_positions]
    keyframes_present = 0
    for ref in scoped_refs:
        raw_path = ref.get("keyframe_path")
        if raw_path and (root / str(raw_path)).is_file():
            keyframes_present += 1

    checks = {
        "vectors_are_2d": vectors.ndim == 2,
        "vector_ref_count_match": vectors.ndim >= 1 and int(vectors.shape[0]) == len(refs),
        "metadata_vector_count_match": metadata.get("vector_count", metadata.get("vectors")) in (None, int(vectors.shape[0])),
        "metadata_refs_count_match": metadata.get("refs_count") in (None, len(refs)),
        "unique_reference_keys": len(scoped_keys) == len(set(scoped_keys)),
    }
    status = "READY" if all(checks.values()) and scoped_refs else "INVALID"
    dimension = int(vectors.shape[1]) if vectors.ndim == 2 else None
    return {
        "status": status,
        "index_dir": str(index_dir),
        "files": files,
        "shape": [int(value) for value in vectors.shape],
        "dtype": str(vectors.dtype),
        "dimension": dimension,
        "vector_count": int(vectors.shape[0]) if vectors.ndim else 0,
        "refs_count": len(refs),
        "scoped_refs_count": len(scoped_refs),
        "scoped_video_count": len({str(ref.get("video_id")) for ref in scoped_refs}),
        "group_counts": dict(sorted(Counter(str(ref.get("group")) for ref in refs).items())),
        "duplicate_reference_keys": len(scoped_keys) - len(set(scoped_keys)),
        "keyframe_images": {
            "present": keyframes_present,
            "missing": len(scoped_refs) - keyframes_present,
            "coverage": _ratio(keyframes_present, len(scoped_refs)),
        },
        "mapping_fields": {
            field: sum(ref.get(field) is not None for ref in scoped_refs)
            for field in ("keyframe_id", "frame_idx", "pts_time", "fps")
        },
        "checks": checks,
        "vector_stats": _norm_stats(vectors),
        "manifest": metadata,
    }


def audit_mapping_files(data_root: Path, video_ids: Iterable[str]) -> dict[str, Any]:
    header_counts: Counter[tuple[str, ...]] = Counter()
    files_present = rows = invalid_rows = 0
    for video_id in sorted(set(video_ids)):
        path = data_root / "map-keyframes" / f"{video_id}.csv"
        if not path.is_file():
            continue
        files_present += 1
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header_counts[tuple(reader.fieldnames or ())] += 1
            for row in reader:
                rows += 1
                try:
                    int(row["n"])
                    int(row["frame_idx"])
                    float(row["pts_time"])
                    float(row["fps"])
                except (KeyError, TypeError, ValueError):
                    invalid_rows += 1
    headers = [
        {"fields": list(fields), "file_count": count}
        for fields, count in sorted(header_counts.items(), key=lambda item: item[0])
    ]
    return {
        "files_present": files_present,
        "files_missing": len(set(video_ids)) - files_present,
        "rows": rows,
        "invalid_rows": invalid_rows,
        "headers": headers,
        "official_frame_id_status": "UNRESOLVED",
        "official_frame_id_candidates": ["frame_idx", "n/keyframe_id"],
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None
                continue
            yield line_number, payload if isinstance(payload, dict) else None


def audit_ocr_jsonl(path: Path, ref_keys: set[tuple[str, int]]) -> dict[str, Any]:
    records = malformed = detections = nonempty_text = mapped_records = 0
    videos: set[str] = set()
    for _, payload in _iter_jsonl(path):
        records += 1
        if payload is None:
            malformed += 1
            continue
        video_id = str(payload.get("video_id", ""))
        videos.add(video_id)
        try:
            key = (video_id, int(payload.get("keyframe_id")))
        except (TypeError, ValueError):
            key = (video_id, -1)
        mapped_records += key in ref_keys
        rows = payload.get("detections", [])
        if not isinstance(rows, list):
            malformed += 1
            continue
        detections += len(rows)
        nonempty_text += sum(bool(str(item.get("text", "")).strip()) for item in rows if isinstance(item, dict))
    return {
        **_path_info(path, checksum=path.is_file()),
        "records": records,
        "malformed_records": malformed,
        "video_count": len(videos - {""}),
        "detections": detections,
        "nonempty_text_detections": nonempty_text,
        "records_mapped_to_clip_refs": mapped_records,
        "mapping_coverage": _ratio(mapped_records, records - malformed),
        "usable_for_text_retrieval": nonempty_text > 0,
    }


def audit_asr_jsonl(path: Path, known_video_ids: set[str]) -> dict[str, Any]:
    records = malformed = available_records = segments = nonempty_segments = 0
    videos: set[str] = set()
    mapped_videos: set[str] = set()
    for _, payload in _iter_jsonl(path):
        records += 1
        if payload is None:
            malformed += 1
            continue
        video_id = str(payload.get("video_id", ""))
        if video_id:
            videos.add(video_id)
        if video_id in known_video_ids:
            mapped_videos.add(video_id)
        if payload.get("status") == "AVAILABLE":
            available_records += 1
        rows = payload.get("segments", [])
        if not isinstance(rows, list):
            malformed += 1
            continue
        segments += len(rows)
        nonempty_segments += sum(bool(str(item.get("text", "")).strip()) for item in rows if isinstance(item, dict))
    return {
        **_path_info(path, checksum=path.is_file()),
        "records": records,
        "malformed_records": malformed,
        "available_records": available_records,
        "video_count": len(videos),
        "videos_known_to_clip_index": len(mapped_videos),
        "segments": segments,
        "nonempty_text_segments": nonempty_segments,
        "usable_for_text_retrieval": nonempty_segments > 0,
    }


def _open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def audit_phase5_store(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", **_path_info(path)}
    with _open_sqlite_readonly(path) as connection:
        table_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        ocr_records = connection.execute("SELECT COUNT(*) FROM ocr").fetchone()[0] if "ocr" in table_names else 0
        asr_segments = connection.execute("SELECT COUNT(*) FROM asr").fetchone()[0] if "asr" in table_names else 0
        ocr_videos = connection.execute("SELECT COUNT(DISTINCT video_id) FROM ocr").fetchone()[0] if "ocr" in table_names else 0
        asr_videos = connection.execute("SELECT COUNT(DISTINCT video_id) FROM asr").fetchone()[0] if "asr" in table_names else 0
        mapped_asr = connection.execute("SELECT COUNT(*) FROM asr WHERE keyframe_id IS NOT NULL").fetchone()[0] if "asr" in table_names else 0
    return {
        "status": "READY" if integrity == "ok" else "INVALID",
        **_path_info(path, checksum=True),
        "sqlite_integrity_check": integrity,
        "ocr_records": ocr_records,
        "ocr_videos": ocr_videos,
        "asr_segments": asr_segments,
        "asr_videos": asr_videos,
        "asr_segments_mapped_to_frame": mapped_asr,
        "asr_mapping_coverage": _ratio(mapped_asr, asr_segments),
    }


def audit_metadata(data_root: Path, video_ids: Iterable[str]) -> dict[str, Any]:
    fields = ("title", "description", "author", "keywords", "publish_date")
    presence: Counter[str] = Counter()
    files_present = valid = malformed = 0
    total_text_chars = 0
    for video_id in sorted(set(video_ids)):
        path = data_root / "media-info" / f"{video_id}.json"
        if not path.is_file():
            continue
        files_present += 1
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            malformed += 1
            continue
        valid += 1
        for field in fields:
            value = payload.get(field)
            if isinstance(value, list):
                text = " ".join(str(item) for item in value if str(item).strip())
            else:
                text = str(value or "").strip()
            if text:
                presence[field] += 1
                total_text_chars += len(text)
    return {
        "files_present": files_present,
        "valid_files": valid,
        "malformed_files": malformed,
        "field_presence": {
            field: {"present": presence[field], "coverage": _ratio(presence[field], valid)}
            for field in fields
        },
        "total_text_characters": total_text_chars,
        "usable_for_text_retrieval": valid > 0 and total_text_chars > 0,
    }


def audit_object_store(path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    if not path.is_file():
        return {"status": "MISSING", **_path_info(path), "manifest": manifest}
    with _open_sqlite_readonly(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        frames = connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        videos = connection.execute("SELECT COUNT(DISTINCT video_id) FROM frames").fetchone()[0]
        detections = connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        labels = connection.execute("SELECT COUNT(DISTINCT label_normalized) FROM detections").fetchone()[0]
    return {
        "status": "READY" if integrity == "ok" else "INVALID",
        **_path_info(path, checksum=False),
        "sqlite_integrity_check": integrity,
        "videos": videos,
        "frames": frames,
        "detections": detections,
        "unique_normalized_labels": labels,
        "manifest": manifest,
        "usable_for_text_retrieval": labels > 0,
    }


def _query_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("queries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def audit_benchmarks(benchmark_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    definitions: list[dict[str, Any]] = []
    total_queries = judged_queries = 0
    hybrid_judged_queries = 0
    for path in sorted(benchmark_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            definitions.append({"path": str(path), "status": "INVALID_JSON"})
            continue
        queries = _query_list(payload)
        judged = 0
        for query in queries:
            expected = query.get("expected_video_ids", query.get("expected"))
            if isinstance(expected, list) and expected:
                judged += 1
        total_queries += len(queries)
        judged_queries += judged
        if "hybrid_retrieval" in path.stem:
            hybrid_judged_queries += judged
        definitions.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "query_count": len(queries),
                "judged_query_count": judged,
                "version": payload.get("version", payload.get("query_set_version")) if isinstance(payload, dict) else None,
            }
        )

    snapshots: list[dict[str, Any]] = []
    candidates = (
        artifact_dir / "phase4" / "clip_baseline_v1.json",
        artifact_dir / "l21_text_judgement_summary.json",
        artifact_dir / "l21_weak_pool50_judgement_summary.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = _load_json(path)
        item: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
        for key in ("benchmark_version", "query_count", "rows", "totals", "top1", "quality_metrics", "latency_ms"):
            if key in payload:
                item[key] = payload[key]
        snapshots.append(item)
    return {
        "definitions": definitions,
        "definition_count": len(definitions),
        "query_count": total_queries,
        "judged_query_count": judged_queries,
        "hybrid_retrieval_judged_query_count": hybrid_judged_queries,
        "existing_snapshots": snapshots,
        "hybrid_promotion_benchmark_ready": hybrid_judged_queries > 0,
        "note": "Existing manual result judgements are useful but are not yet a frozen cross-retriever relevance set.",
    }


def audit_dependencies() -> dict[str, Any]:
    packages = {
        "faiss": ("faiss-cpu", "faiss-gpu", "faiss"),
        "sentence_transformers": ("sentence-transformers",),
        "torch": ("torch",),
        "transformers": ("transformers",),
    }
    result: dict[str, Any] = {}
    for module_name, distribution_names in packages.items():
        installed = importlib.util.find_spec(module_name) is not None
        version = distribution = None
        if installed:
            for candidate in distribution_names:
                try:
                    version = importlib_metadata.version(candidate)
                    distribution = candidate
                    break
                except importlib_metadata.PackageNotFoundError:
                    continue
        result[module_name] = {"installed": installed, "distribution": distribution, "version": version}
    return result


def build_hybrid_audit(
    root: Path,
    group: str = "L21",
    registry_path: Path | None = None,
    index_dir: Path | None = None,
    phase5_dir: Path | None = None,
    phase5_store_path: Path | None = None,
    object_store_path: Path | None = None,
    object_manifest_path: Path | None = None,
    benchmark_dir: Path | None = None,
    benchmark_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    groups = {group}
    data_root = root / "data"
    registry_path = registry_path or root / "artifacts" / "registry" / "data_registry.json"
    index_dir = index_dir or root / "artifacts" / "indexes" / "l21_numpy"
    phase5_dir = phase5_dir or root / "artifacts" / "phase5"
    phase5_store_path = phase5_store_path or phase5_dir / "phase5_store.sqlite"
    object_store_path = object_store_path or root / "artifacts" / "structured" / "l21_objects.sqlite"
    object_manifest_path = object_manifest_path or root / "artifacts" / "structured" / "l21_objects_manifest.json"
    benchmark_dir = benchmark_dir or root / "benchmarks"
    benchmark_artifact_dir = benchmark_artifact_dir or root / "artifacts" / "benchmarks"

    registry = audit_registry(registry_path, root, groups)
    clip = audit_clip_index(index_dir, root, groups)
    refs_path = index_dir / "refs.json"
    refs = _load_json(refs_path) if refs_path.is_file() else []
    scoped_refs = [ref for ref in refs if str(ref.get("group")) in groups]
    ref_keys = {(str(ref.get("video_id")), int(ref.get("keyframe_id", -1))) for ref in scoped_refs}
    video_ids = set(registry.get("video_ids", [])) or {key[0] for key in ref_keys}

    ocr = audit_ocr_jsonl(phase5_dir / f"ocr_{group.lower()}.jsonl", ref_keys)
    asr = audit_asr_jsonl(phase5_dir / f"asr_{group.lower()}.jsonl", video_ids)
    phase5_store = audit_phase5_store(phase5_store_path)
    metadata = audit_metadata(data_root, video_ids)
    objects = audit_object_store(object_store_path, object_manifest_path)
    mappings = audit_mapping_files(data_root, video_ids)
    benchmarks = audit_benchmarks(benchmark_dir, benchmark_artifact_dir)

    blockers: list[str] = []
    warnings: list[str] = []
    if registry.get("status") != "READY":
        blockers.append("L21 registry is missing or empty")
    if clip.get("status") != "READY":
        blockers.append("CLIP index failed structural validation")
    if not ocr.get("usable_for_text_retrieval"):
        warnings.append("OCR JSONL has no non-empty text detections")
    if asr.get("video_count", 0) < len(video_ids):
        warnings.append(f"ASR covers {asr.get('video_count', 0)}/{len(video_ids)} L21 videos")
    if mappings.get("official_frame_id_status") != "RESOLVED":
        warnings.append("Official submission frame_id mapping is unresolved")
    if not benchmarks.get("hybrid_promotion_benchmark_ready"):
        warnings.append("No judged benchmark definition is ready for hybrid promotion")

    fingerprint_inputs = [registry_path, refs_path, index_dir / "metadata.json", phase5_dir / f"ocr_{group.lower()}.jsonl", phase5_dir / f"asr_{group.lower()}.jsonl"]
    fingerprint = hashlib.sha256()
    for path in fingerprint_inputs:
        if path.is_file():
            fingerprint.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
            fingerprint.update(bytes.fromhex(sha256_file(path)))

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"group": group, "root": str(root), "read_only_source_audit": True},
        "source_fingerprint": fingerprint.hexdigest(),
        "status": "BLOCKED" if blockers else "READY_WITH_WARNINGS" if warnings else "READY",
        "registry": registry,
        "clip_index": clip,
        "frame_mapping": mappings,
        "text_sources": {"metadata": metadata, "ocr": ocr, "asr": asr, "phase5_store": phase5_store, "objects": objects},
        "benchmarks": benchmarks,
        "dependencies": audit_dependencies(),
        "blockers": blockers,
        "warnings": warnings,
        "next_gate": "H1_RETRIEVER_CONTRACTS_AND_CORPUS" if not blockers else "FIX_H0_BLOCKERS",
    }
