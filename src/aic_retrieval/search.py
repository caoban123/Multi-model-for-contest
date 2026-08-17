from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aic_retrieval.provenance import fingerprint_paths, sha256_file


INDEX_SCHEMA_VERSION = "2.0"


class IndexValidationError(ValueError):
    """Raised when a persistent index cannot safely serve the requested search."""


@dataclass(frozen=True)
class FrameRef:
    video_id: str
    group: str
    keyframe_id: int
    frame_idx: int
    pts_time: float
    fps: float
    keyframe_path: str | None


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    video_id: str
    group: str
    keyframe_id: int
    frame_idx: int
    pts_time: float
    fps: float
    keyframe_path: str | None


@dataclass(frozen=True)
class VideoResult:
    rank: int
    video_id: str
    group: str
    video_score: float
    best_score: float
    best_keyframe_id: int
    best_frame_idx: int
    best_pts_time: float
    best_keyframe_path: str | None
    frame_count: int
    matched_frame_count: int
    aggregation_method: str
    frames: list[SearchResult]


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_numpy_index(
    registry: dict[str, Any],
    repo_root: Path,
    groups: set[str] | None = None,
    require_keyframes: bool = False,
) -> tuple[np.ndarray, list[FrameRef]]:
    vectors: list[np.ndarray] = []
    refs: list[FrameRef] = []

    for asset in registry["videos"]:
        if groups is not None and asset["group"] not in groups:
            continue
        if require_keyframes and not asset["has_keyframe_images"]:
            continue

        feature_path = _repo_path(repo_root, asset["clip_feature_path"])
        mapping_path = _repo_path(repo_root, asset["mapping_path"])
        if feature_path is None or mapping_path is None:
            continue

        feature = np.load(feature_path, mmap_mode="r")
        mapping_rows = read_mapping_rows(mapping_path)
        if len(feature) != len(mapping_rows):
            raise ValueError(
                f"{asset['video_id']} feature rows ({len(feature)}) "
                f"do not match mapping rows ({len(mapping_rows)})"
            )

        keyframe_root = _repo_path(repo_root, asset["keyframe_path"])
        vectors.append(np.asarray(feature, dtype=np.float32))
        refs.extend(_frame_refs(asset, mapping_rows, keyframe_root, repo_root))

    if not vectors:
        raise ValueError("no vectors matched the requested registry filters")

    matrix = np.concatenate(vectors, axis=0)
    return normalize_rows(matrix), refs


def save_numpy_index(index_dir: Path, index: np.ndarray, refs: list[FrameRef], metadata: dict[str, Any]) -> None:
    if len(index) != len(refs):
        raise ValueError(f"index vector count ({len(index)}) does not match refs count ({len(refs)})")
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", index.astype(np.float32, copy=False))
    (index_dir / "refs.json").write_text(
        json.dumps([asdict(ref) for ref in refs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_numpy_index(index_dir: Path) -> tuple[np.ndarray, list[FrameRef], dict[str, Any]]:
    vectors_path = index_dir / "vectors.npy"
    refs_path = index_dir / "refs.json"
    metadata_path = index_dir / "metadata.json"
    missing = [path for path in (vectors_path, refs_path) if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"missing NumPy index files:\n{formatted}")

    index = np.asarray(np.load(vectors_path, mmap_mode="r"), dtype=np.float32)
    refs_payload = json.loads(refs_path.read_text(encoding="utf-8"))
    refs = [FrameRef(**item) for item in refs_payload]
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, refs, metadata


def build_index_metadata(
    registry_path: Path,
    registry: dict[str, Any],
    repo_root: Path,
    groups: set[str],
    require_keyframes: bool,
    index: np.ndarray,
    refs: list[FrameRef],
    build_elapsed_ms: float,
) -> dict[str, Any]:
    feature_paths, mapping_paths = _source_paths(registry, repo_root, groups, require_keyframes)
    return {
        "format": "aic_numpy_index",
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "registry_version": registry.get("version"),
        "registry_path": str(registry_path),
        "registry_fingerprint": sha256_file(registry_path),
        "groups": sorted(groups),
        "require_keyframes": require_keyframes,
        "vector_count": int(index.shape[0]),
        "vectors": int(index.shape[0]),
        "dimension": int(index.shape[1]),
        "dim": int(index.shape[1]),
        "dtype": str(index.dtype),
        "refs_count": len(refs),
        "feature_source_fingerprint": fingerprint_paths(feature_paths, repo_root),
        "mapping_source_fingerprint": fingerprint_paths(mapping_paths, repo_root),
        "model_name": "unknown",
        "feature_provenance": "BTC-provided CLIP image features; exact extraction provenance is not recorded in the local registry.",
        "build_elapsed_ms": build_elapsed_ms,
    }


def validate_numpy_index(
    index: np.ndarray,
    refs: list[FrameRef],
    metadata: dict[str, Any],
    requested_groups: set[str],
    require_keyframes: bool,
    registry_path: Path | None,
    repo_root: Path,
    allow_stale_index: bool = False,
) -> None:
    """Fail closed for scope/provenance mismatches before a persistent index is searched."""
    if len(index) != len(refs):
        raise IndexValidationError(f"index has {len(index)} vectors but {len(refs)} refs; rebuild the index")
    if index.ndim != 2:
        raise IndexValidationError(f"index vectors must be 2-D, got shape {index.shape}")
    missing = {"index_schema_version", "groups", "require_keyframes", "registry_fingerprint", "feature_source_fingerprint", "mapping_source_fingerprint"} - set(metadata)
    if missing and not allow_stale_index:
        raise IndexValidationError(
            "index provenance is incomplete (missing " + ", ".join(sorted(missing)) + "). Rebuild it, or pass --allow-stale-index only for an explicit debug run."
        )
    indexed_groups = set(metadata.get("groups", []))
    if indexed_groups and indexed_groups != requested_groups:
        raise IndexValidationError(
            f"index groups are {sorted(indexed_groups)}, but the request is {sorted(requested_groups)}. Rebuild/select an index with the same groups."
        )
    indexed_keyframes = metadata.get("require_keyframes")
    if indexed_keyframes is not None and bool(indexed_keyframes) != require_keyframes:
        raise IndexValidationError(
            f"index require_keyframes={indexed_keyframes}, but request require_keyframes={require_keyframes}. Rebuild/select a matching index."
        )
    if registry_path is None or not registry_path.exists():
        if allow_stale_index:
            return
        raise IndexValidationError(
            "cannot verify index freshness because the registry is unavailable. Provide --registry or pass --allow-stale-index for an explicit debug run."
        )
    registry = load_registry(registry_path)
    if metadata.get("registry_fingerprint") != sha256_file(registry_path):
        if not allow_stale_index:
            raise IndexValidationError("STALE INDEX: registry fingerprint changed. Rebuild the index from the current registry.")
        return
    feature_paths, mapping_paths = _source_paths(registry, repo_root, requested_groups, require_keyframes)
    if metadata.get("feature_source_fingerprint") != fingerprint_paths(feature_paths, repo_root):
        if not allow_stale_index:
            raise IndexValidationError("STALE INDEX: CLIP feature source changed. Rebuild the index.")
    if metadata.get("mapping_source_fingerprint") != fingerprint_paths(mapping_paths, repo_root):
        if not allow_stale_index:
            raise IndexValidationError("STALE INDEX: mapping source changed. Rebuild the index.")


def search_numpy_index(index: np.ndarray, refs: list[FrameRef], query: np.ndarray, top_k: int) -> list[SearchResult]:
    query_vector = normalize_query(query, expected_dim=index.shape[1])
    scores = index @ query_vector
    top_k = min(top_k, len(scores))
    if top_k <= 0:
        return []

    if top_k == len(scores):
        candidate_indices = np.arange(len(scores))
    else:
        cutoff = float(np.partition(scores, len(scores) - top_k)[len(scores) - top_k])
        above_cutoff = np.flatnonzero(scores > cutoff)
        cutoff_ties = np.flatnonzero(scores == cutoff)
        candidate_indices = np.concatenate(
            [above_cutoff, cutoff_ties[: top_k - len(above_cutoff)]]
        )
    ranked_indices = candidate_indices[
        np.lexsort((candidate_indices, -scores[candidate_indices]))
    ]

    results: list[SearchResult] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        ref = refs[int(idx)]
        results.append(
            SearchResult(
                rank=rank,
                score=float(scores[int(idx)]),
                video_id=ref.video_id,
                group=ref.group,
                keyframe_id=ref.keyframe_id,
                frame_idx=ref.frame_idx,
                pts_time=ref.pts_time,
                fps=ref.fps,
                keyframe_path=ref.keyframe_path,
            )
        )
    return results


def diversify_results_by_video(results: list[SearchResult], max_frames_per_video: int) -> list[SearchResult]:
    if max_frames_per_video <= 0:
        raise ValueError("max_frames_per_video must be positive")

    counts: dict[str, int] = {}
    diversified: list[SearchResult] = []
    for result in results:
        count = counts.get(result.video_id, 0)
        if count >= max_frames_per_video:
            continue
        counts[result.video_id] = count + 1
        diversified.append(_rerank_frame(result, len(diversified) + 1))
    return diversified


def aggregate_results_by_video(
    results: list[SearchResult],
    max_frames_per_video: int,
    aggregation_method: str = "max",
    mean_top_n: int = 3,
) -> list[VideoResult]:
    if max_frames_per_video <= 0:
        raise ValueError("max_frames_per_video must be positive")
    if mean_top_n <= 0:
        raise ValueError("mean_top_n must be positive")
    if aggregation_method not in {"max", "mean_top_n"}:
        raise ValueError(f"unsupported aggregation method: {aggregation_method}")

    grouped: dict[str, list[SearchResult]] = {}
    for result in results:
        if result.video_id not in grouped:
            grouped[result.video_id] = []
        grouped[result.video_id].append(result)

    unranked: list[tuple[float, int, str, SearchResult, list[SearchResult], int]] = []
    for video_id, raw_frames in grouped.items():
        ranked_frames = sorted(
            raw_frames,
            key=lambda frame: (-frame.score, frame.rank, frame.keyframe_id),
        )
        best = ranked_frames[0]
        if aggregation_method == "max":
            video_score = best.score
        else:
            top_scores = [frame.score for frame in ranked_frames[:mean_top_n]]
            video_score = sum(top_scores) / len(top_scores)
        unranked.append(
            (
                video_score,
                best.rank,
                video_id,
                best,
                ranked_frames[:max_frames_per_video],
                len(raw_frames),
            )
        )

    unranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    video_results: list[VideoResult] = []
    for rank, (video_score, _best_rank, video_id, best, frames, frame_count) in enumerate(
        unranked, start=1
    ):
        video_results.append(
            VideoResult(
                rank=rank,
                video_id=video_id,
                group=best.group,
                video_score=video_score,
                best_score=best.score,
                best_keyframe_id=best.keyframe_id,
                best_frame_idx=best.frame_idx,
                best_pts_time=best.pts_time,
                best_keyframe_path=best.keyframe_path,
                frame_count=frame_count,
                matched_frame_count=len(frames),
                aggregation_method=aggregation_method,
                frames=frames,
            )
        )
    return video_results


def group_results_by_video(
    results: list[SearchResult],
    max_frames_per_video: int,
) -> list[VideoResult]:
    """Backward-compatible max-score video grouping used by Phase-1/2 callers."""
    return aggregate_results_by_video(
        results,
        max_frames_per_video=max_frames_per_video,
        aggregation_method="max",
    )


def load_query_vector(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


def load_query_vector_from_asset(
    registry: dict[str, Any],
    repo_root: Path,
    video_id: str,
    keyframe_id: int,
) -> np.ndarray:
    asset = find_asset(registry, video_id)
    feature_path = _repo_path(repo_root, asset["clip_feature_path"])
    mapping_path = _repo_path(repo_root, asset["mapping_path"])
    if feature_path is None:
        raise ValueError(f"{video_id} has no CLIP feature path")
    if mapping_path is None:
        raise ValueError(f"{video_id} has no mapping path")

    mapping_rows = read_mapping_rows(mapping_path)
    row_index = None
    for index, row in enumerate(mapping_rows):
        if int(row["n"]) == keyframe_id:
            row_index = index
            break
    if row_index is None:
        raise ValueError(f"{video_id} has no keyframe_id {keyframe_id}")

    feature = np.load(feature_path, mmap_mode="r")
    if row_index >= len(feature):
        raise ValueError(f"{video_id} keyframe_id {keyframe_id} maps outside feature rows")
    return np.asarray(feature[row_index], dtype=np.float32)


def find_asset(registry: dict[str, Any], video_id: str) -> dict[str, Any]:
    for asset in registry["videos"]:
        if asset["video_id"] == video_id:
            return asset
    raise ValueError(f"video_id not found in registry: {video_id}")


def write_results(path: Path, results: list[SearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": [asdict(result) for result in results]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_mapping_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_query(query: np.ndarray, expected_dim: int) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1:
        raise ValueError(f"query vector must be 1-D or shape (1, dim), got {query.shape}")
    if query.shape[0] != expected_dim:
        raise ValueError(f"query dim {query.shape[0]} does not match index dim {expected_dim}")
    norm = float(np.linalg.norm(query))
    if norm == 0:
        raise ValueError("query vector must not be all zeros")
    return query / norm


def _frame_refs(
    asset: dict[str, Any],
    mapping_rows: list[dict[str, str]],
    keyframe_root: Path | None,
    repo_root: Path,
) -> list[FrameRef]:
    refs: list[FrameRef] = []
    for row in mapping_rows:
        keyframe_id = int(row["n"])
        keyframe_path = None
        if keyframe_root is not None:
            image_path = keyframe_root / f"{keyframe_id:03d}.jpg"
            if image_path.exists():
                keyframe_path = _relative(image_path, repo_root)

        refs.append(
            FrameRef(
                video_id=asset["video_id"],
                group=asset["group"],
                keyframe_id=keyframe_id,
                frame_idx=int(row["frame_idx"]),
                pts_time=float(row["pts_time"]),
                fps=float(row["fps"]),
                keyframe_path=keyframe_path,
            )
        )
    return refs


def _repo_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _rerank_frame(result: SearchResult, rank: int) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=result.score,
        video_id=result.video_id,
        group=result.group,
        keyframe_id=result.keyframe_id,
        frame_idx=result.frame_idx,
        pts_time=result.pts_time,
        fps=result.fps,
        keyframe_path=result.keyframe_path,
    )


def _source_paths(
    registry: dict[str, Any], repo_root: Path, groups: set[str], require_keyframes: bool
) -> tuple[list[Path], list[Path]]:
    feature_paths: list[Path] = []
    mapping_paths: list[Path] = []
    for asset in registry.get("videos", []):
        if asset.get("group") not in groups:
            continue
        if require_keyframes and not asset.get("has_keyframe_images"):
            continue
        feature_path = _repo_path(repo_root, asset.get("clip_feature_path"))
        mapping_path = _repo_path(repo_root, asset.get("mapping_path"))
        if feature_path is None or mapping_path is None:
            raise IndexValidationError(f"{asset.get('video_id', '<unknown>')} is missing feature or mapping source path")
        if not feature_path.exists() or not mapping_path.exists():
            raise IndexValidationError(
                f"source data for {asset.get('video_id', '<unknown>')} is unavailable. Run tools/check_phase1_data.py or provide the configured data root."
            )
        feature_paths.append(feature_path)
        mapping_paths.append(mapping_path)
    if not feature_paths:
        raise IndexValidationError("no feature/mapping sources matched the requested index scope")
    return feature_paths, mapping_paths
