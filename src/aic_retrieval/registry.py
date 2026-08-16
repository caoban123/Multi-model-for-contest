from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REGISTRY_VERSION = "0.1"


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    group: str
    clip_feature_path: str | None
    mapping_path: str | None
    media_info_path: str | None
    object_path: str | None
    keyframe_path: str | None
    video_path: str | None
    has_keyframe_images: bool
    feature_rows: int | None = None
    feature_dim: int | None = None
    feature_dtype: str | None = None
    mapping_rows: int | None = None
    keyframe_image_count: int | None = None
    object_file_count: int | None = None
    video_file_size_bytes: int | None = None


@dataclass(frozen=True)
class RegistryValidation:
    total_videos: int
    error_count: int
    warning_count: int
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]
    group_counts: dict[str, int]
    keyframe_group_counts: dict[str, int]
    raw_video_group_counts: dict[str, int]


def scan_data_root(data_root: Path) -> list[VideoAsset]:
    data_root = data_root.resolve()
    feature_dir = data_root / "clip-features-32"
    mapping_dir = data_root / "map-keyframes"
    media_dir = data_root / "media-info"
    objects_dir = data_root / "objects"
    keyframes_dir = data_root / "keyframes"
    videos_dir = data_root / "videos"

    video_ids = set()
    video_ids.update(_ids_from_files(feature_dir, "*.npy"))
    video_ids.update(_ids_from_files(mapping_dir, "*.csv"))
    video_ids.update(_ids_from_files(media_dir, "*.json"))
    video_ids.update(_ids_from_dirs(objects_dir))
    video_ids.update(_ids_from_dirs(keyframes_dir))
    video_ids.update(_ids_from_video_files(videos_dir))

    assets: list[VideoAsset] = []
    for video_id in sorted(video_ids):
        group = video_id.split("_", 1)[0]
        feature_path = _existing(feature_dir / f"{video_id}.npy")
        mapping_path = _existing(mapping_dir / f"{video_id}.csv")
        media_path = _existing(media_dir / f"{video_id}.json")
        object_path = _existing(objects_dir / video_id)
        keyframe_path = _existing(keyframes_dir / video_id)
        video_path = _find_video_file(videos_dir, video_id)

        feature_rows: int | None = None
        feature_dim: int | None = None
        feature_dtype: str | None = None
        if feature_path:
            shape, dtype = read_feature_info(Path(feature_path))
            feature_rows = shape[0] if len(shape) >= 1 else None
            feature_dim = shape[1] if len(shape) >= 2 else None
            feature_dtype = dtype

        mapping_rows = count_mapping_rows(Path(mapping_path)) if mapping_path else None
        keyframe_image_count = count_files(Path(keyframe_path), "*.jpg") if keyframe_path else None
        object_file_count = count_files(Path(object_path), "*.json") if object_path else None
        video_file_size_bytes = Path(video_path).stat().st_size if video_path else None

        assets.append(
            VideoAsset(
                video_id=video_id,
                group=group,
                clip_feature_path=_rel(feature_path, data_root.parent),
                mapping_path=_rel(mapping_path, data_root.parent),
                media_info_path=_rel(media_path, data_root.parent),
                object_path=_rel(object_path, data_root.parent),
                keyframe_path=_rel(keyframe_path, data_root.parent),
                video_path=_rel(video_path, data_root.parent),
                has_keyframe_images=keyframe_path is not None,
                feature_rows=feature_rows,
                feature_dim=feature_dim,
                feature_dtype=feature_dtype,
                mapping_rows=mapping_rows,
                keyframe_image_count=keyframe_image_count,
                object_file_count=object_file_count,
                video_file_size_bytes=video_file_size_bytes,
            )
        )

    return assets


def validate_assets(assets: list[VideoAsset]) -> RegistryValidation:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    group_counts: dict[str, int] = {}
    keyframe_group_counts: dict[str, int] = {}
    raw_video_group_counts: dict[str, int] = {}

    for asset in assets:
        group_counts[asset.group] = group_counts.get(asset.group, 0) + 1
        if asset.has_keyframe_images:
            keyframe_group_counts[asset.group] = keyframe_group_counts.get(asset.group, 0) + 1
        if asset.video_path:
            raw_video_group_counts[asset.group] = raw_video_group_counts.get(asset.group, 0) + 1

        for field_name in (
            "clip_feature_path",
            "mapping_path",
            "media_info_path",
            "object_path",
        ):
            if getattr(asset, field_name) is None:
                errors.append(_issue(asset.video_id, f"missing_{field_name}", field_name))

        if asset.feature_rows is not None and asset.mapping_rows is not None:
            if asset.feature_rows != asset.mapping_rows:
                errors.append(
                    _issue(
                        asset.video_id,
                        "feature_mapping_row_mismatch",
                        f"feature_rows={asset.feature_rows}, mapping_rows={asset.mapping_rows}",
                    )
                )

        if asset.feature_dim is not None and asset.feature_dim != 512:
            errors.append(
                _issue(asset.video_id, "unexpected_feature_dim", f"feature_dim={asset.feature_dim}")
            )

        if asset.feature_dtype is not None and asset.feature_dtype != "float16":
            warnings.append(
                _issue(asset.video_id, "unexpected_feature_dtype", f"feature_dtype={asset.feature_dtype}")
            )

        if asset.has_keyframe_images:
            if asset.mapping_rows != asset.keyframe_image_count:
                errors.append(
                    _issue(
                        asset.video_id,
                        "mapping_keyframe_count_mismatch",
                        f"mapping_rows={asset.mapping_rows}, keyframe_images={asset.keyframe_image_count}",
                    )
                )
        else:
            warnings.append(_issue(asset.video_id, "missing_keyframe_images", "visual viewer unavailable"))

    return RegistryValidation(
        total_videos=len(assets),
        error_count=len(errors),
        warning_count=len(warnings),
        errors=errors,
        warnings=warnings,
        group_counts=dict(sorted(group_counts.items())),
        keyframe_group_counts=dict(sorted(keyframe_group_counts.items())),
        raw_video_group_counts=dict(sorted(raw_video_group_counts.items())),
    )


def write_registry(path: Path, data_root: Path, assets: list[VideoAsset], validation: RegistryValidation) -> None:
    payload: dict[str, Any] = {
        "version": REGISTRY_VERSION,
        "data_root": str(data_root),
        "total_videos": len(assets),
        "validation": asdict(validation),
        "videos": [asdict(asset) for asset in assets],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_feature_info(path: Path) -> tuple[tuple[int, ...], str]:
    array = np.load(path, mmap_mode="r")
    return tuple(int(dim) for dim in array.shape), str(array.dtype)


def count_mapping_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file())


def _ids_from_files(directory: Path, pattern: str) -> set[str]:
    if not directory.exists():
        return set()
    return {path.stem for path in directory.glob(pattern) if path.is_file()}


def _ids_from_dirs(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {path.name for path in directory.iterdir() if path.is_dir()}


def _ids_from_video_files(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {path.stem for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}}


def _find_video_file(directory: Path, video_id: str) -> str | None:
    if not directory.exists():
        return None
    matches = sorted(
        path
        for path in directory.rglob(f"{video_id}.*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    )
    return str(matches[0]) if matches else None


def _existing(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _rel(path: str | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _issue(video_id: str, code: str, detail: str) -> dict[str, str]:
    return {"video_id": video_id, "code": code, "detail": detail}
