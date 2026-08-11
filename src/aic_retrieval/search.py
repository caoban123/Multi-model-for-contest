from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    if not vectors_path.exists() or not refs_path.exists():
        raise FileNotFoundError(f"missing NumPy index files in {index_dir}")

    index = np.asarray(np.load(vectors_path, mmap_mode="r"), dtype=np.float32)
    refs_payload = json.loads(refs_path.read_text(encoding="utf-8"))
    refs = [FrameRef(**item) for item in refs_payload]
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, refs, metadata


def search_numpy_index(index: np.ndarray, refs: list[FrameRef], query: np.ndarray, top_k: int) -> list[SearchResult]:
    query_vector = normalize_query(query, expected_dim=index.shape[1])
    scores = index @ query_vector
    top_k = min(top_k, len(scores))
    if top_k <= 0:
        return []

    candidate_indices = np.argpartition(-scores, top_k - 1)[:top_k]
    ranked_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

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


def load_query_vector(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


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
