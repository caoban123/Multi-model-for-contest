from __future__ import annotations

from pathlib import Path

NUMPY_INDEX_REQUIRED_FILES = ("vectors.npy", "refs.json")


class Phase5PreparationError(RuntimeError):
    """Actionable error for a missing generated Phase 5 prerequisite."""


def require_index_source_data(data_root: Path, groups: str) -> None:
    requested = [item.strip() for item in groups.split(",") if item.strip()]
    feature_dir = data_root / "clip-features-32"
    mapping_dir = data_root / "map-keyframes"
    missing: list[str] = []
    for group in requested:
        if not any(feature_dir.glob(f"{group}_*.npy")):
            missing.append(f"{feature_dir}/{group}_*.npy")
        if not any(mapping_dir.glob(f"{group}_*.csv")):
            missing.append(f"{mapping_dir}/{group}_*.csv")
    if missing:
        formatted = "\n".join(f"  {item}" for item in missing)
        raise Phase5PreparationError(
            f"Cannot build a real NumPy index; required source data is missing:\n{formatted}\n"
            "The index builder requires BTC-provided CLIP feature arrays and matching keyframe mapping CSV files."
        )


def missing_numpy_index_files(index_dir: Path) -> list[Path]:
    return [index_dir / name for name in NUMPY_INDEX_REQUIRED_FILES if not (index_dir / name).is_file()]


def require_numpy_index(index_dir: Path, registry: Path, groups: str = "L21") -> None:
    missing = missing_numpy_index_files(index_dir)
    if not missing:
        return
    paths = "\n".join(f"  {path}" for path in missing)
    raise Phase5PreparationError(
        f"Missing Phase 5 NumPy index files:\n{paths}\n\n"
        "Build the registry and index with:\n"
        "  python tools/data_registry.py --data-root data "
        f"--output {registry} --validation-output artifacts/registry/validation_report.json\n"
        f"  python tools/build_numpy_index.py --registry {registry} "
        f"--groups {groups} --output-dir {index_dir}\n\n"
        "Or run the orchestration command:\n"
        f"  python tools/prepare_phase5.py --data-root data --groups {groups} --skip-store"
    )


def require_phase5_inputs(ocr_jsonl: Path | None, asr_jsonl: Path | None) -> None:
    if ocr_jsonl is None and asr_jsonl is None:
        raise Phase5PreparationError(
            "No Phase 5 evidence input was selected. Pass --ocr-jsonl and/or --asr-jsonl.\n"
            "These files are real OCR/ASR outputs and are not fabricated by the store builder."
        )
    missing = [path for path in (ocr_jsonl, asr_jsonl) if path is not None and not path.is_file()]
    if not missing:
        return
    paths = "\n".join(f"  {path}" for path in missing)
    raise Phase5PreparationError(
        f"Missing Phase 5 OCR/ASR JSONL inputs:\n{paths}\n\n"
        "The current repository has only a bounded raw model pilot; it does not fabricate or "
        "automatically normalize a full L21 OCR/ASR corpus. Regenerate the reviewed, schema-compatible "
        "JSONL from the source keyframes/videos, or restore it from your generated-artifact backup.\n"
        "Use --skip-store with tools/prepare_phase5.py to rebuild only the registry and NumPy index."
    )
