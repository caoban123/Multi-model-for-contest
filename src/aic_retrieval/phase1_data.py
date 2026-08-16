from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aic_retrieval.registry import REGISTRY_VERSION, scan_data_root, validate_assets


VALID_STATUSES = {"OK", "WARNING", "MISSING", "INVALID"}
BASELINE_BLOCKING_CODES = {
    "missing_clip_feature_path",
    "missing_mapping_path",
    "feature_mapping_row_mismatch",
    "unexpected_feature_dim",
}


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str


def load_phase1_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"manifest_schema_version", "registry_schema_version", "required_directories", "groups"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Phase-1 manifest is missing required fields: {', '.join(missing)}")
    return payload


def preflight_phase1_data(
    data_root: Path,
    manifest: dict[str, Any],
    registry_path: Path | None = None,
    groups: set[str] | None = None,
) -> dict[str, Any]:
    checks: list[PreflightCheck] = []
    data_root = data_root.resolve()
    for directory in manifest["required_directories"]:
        path = data_root / directory
        checks.append(
            PreflightCheck(
                name=f"directory:{directory}",
                status="OK" if path.is_dir() else "MISSING",
                detail=str(path) if path.is_dir() else f"Required Phase-1 directory is absent: {path}",
            )
        )

    keyframes_dir = data_root / "keyframes"
    checks.append(
        PreflightCheck(
            name="directory:keyframes",
            status="OK" if keyframes_dir.is_dir() else "WARNING",
            detail=str(keyframes_dir) if keyframes_dir.is_dir() else "No keyframe directory. L21 visual verification is unavailable.",
        )
    )
    videos_dir = data_root / "videos"
    checks.append(
        PreflightCheck(
            name="directory:videos",
            status="OK" if videos_dir.is_dir() else "WARNING",
            detail=str(videos_dir) if videos_dir.is_dir() else "Raw videos are not required for the Phase-1 baseline.",
        )
    )

    configured_groups = set(manifest["groups"])
    selected_groups = configured_groups if groups is None else groups
    unknown_groups = sorted(selected_groups - configured_groups)
    if unknown_groups:
        raise ValueError(f"Groups are absent from the Phase-1 manifest: {', '.join(unknown_groups)}")

    assets = scan_data_root(data_root)
    validation = validate_assets(assets)
    observed_groups = Counter(asset.group for asset in assets)
    selected_assets = [asset for asset in assets if asset.group in selected_groups]
    for group, expectation in sorted(manifest["groups"].items()):
        if group not in selected_groups:
            continue
        observed = observed_groups.get(group, 0)
        expected = expectation.get("expected_videos")
        status = "OK" if observed == expected else ("MISSING" if observed == 0 else "WARNING")
        checks.append(
            PreflightCheck(
                name=f"group:{group}:videos",
                status=status,
                detail=f"observed={observed}; expected={expected}",
            )
        )
        if expectation.get("has_keyframe_images"):
            image_count = sum(
                asset.keyframe_image_count or 0
                for asset in assets
                if asset.group == group and asset.has_keyframe_images
            )
            expected_images = expectation.get("expected_keyframe_images")
            checks.append(
                PreflightCheck(
                    name=f"group:{group}:keyframe_images",
                    status="OK" if image_count == expected_images else "WARNING",
                    detail=(
                        f"observed={image_count}; expected_full_local_copy={expected_images}. "
                        "Keyframe images are optional local visual aids; baseline KIS uses CLIP features and mappings."
                    ),
                )
            )
        else:
            image_folders = sum(1 for asset in assets if asset.group == group and asset.has_keyframe_images)
            checks.append(
                PreflightCheck(
                    name=f"group:{group}:keyframe_images",
                    status="WARNING",
                    detail=(
                        "EXPECTED: no local keyframe images; this group remains registry/mapping/metadata/index-capable only."
                        if image_folders == 0
                        else "Images are present although manifest records this group as non-visual scope; verify and update the manifest if intentional."
                    ),
                )
            )

    if selected_assets:
        blocking_errors = [
            issue
            for issue in validation.errors
            if issue["video_id"].split("_", 1)[0] in selected_groups
            and issue["code"] in BASELINE_BLOCKING_CODES
        ]
        checks.append(
            PreflightCheck(
                name="registry_validation",
                status="OK" if not blocking_errors else "INVALID",
                detail=(
                    f"selected_videos={len(selected_assets)}; baseline_blocking_errors={len(blocking_errors)}; "
                    f"full_registry_errors={validation.error_count}; full_registry_warnings={validation.warning_count}. "
                    "Object coverage and partial keyframe-image coverage do not block the CLIP KIS baseline."
                ),
            )
        )
    else:
        checks.append(PreflightCheck("registry_validation", "MISSING", "No local video assets were discovered; provide --data-root with Phase-1 data."))

    if registry_path is not None:
        if not registry_path.exists():
            checks.append(PreflightCheck("registry_artifact", "WARNING", f"Registry artifact not found: {registry_path}. Build it with tools/data_registry.py."))
        else:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            version = registry.get("version")
            checks.append(
                PreflightCheck(
                    name="registry_artifact",
                    status="OK" if version == REGISTRY_VERSION else "INVALID",
                    detail=f"registry_version={version!r}; expected={REGISTRY_VERSION!r}",
                )
            )

    return {
        "phase": "1",
        "data_root": str(data_root),
        "manifest_schema_version": manifest["manifest_schema_version"],
        "registry_schema_version": manifest["registry_schema_version"],
        "checks": [asdict(check) for check in checks],
        "observed": {
            "total_assets": len(assets),
            "selected_asset_count": len(selected_assets),
            "selected_groups": sorted(selected_groups),
            "group_counts": dict(sorted(observed_groups.items())),
            "raw_video_group_counts": validation.raw_video_group_counts,
            "validation": asdict(validation),
        },
        "overall_status": overall_status(checks),
    }


def overall_status(checks: list[PreflightCheck]) -> str:
    statuses = {check.status for check in checks}
    if not statuses <= VALID_STATUSES:
        raise ValueError("preflight contains an unknown status")
    if "INVALID" in statuses:
        return "INVALID"
    if "MISSING" in statuses:
        return "MISSING"
    if "WARNING" in statuses:
        return "WARNING"
    return "OK"
