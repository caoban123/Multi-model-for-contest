from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROFILE_SCHEMA_VERSION = "aic-competition-profile-v1"


@dataclass(frozen=True)
class CompetitionPreflight:
    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    facts: Mapping[str, Any]
    paths: Mapping[str, Path]

    @property
    def ready(self) -> bool:
        return self.status == "READY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "facts": dict(self.facts),
            "paths": {key: str(value) for key, value in self.paths.items()},
        }


def load_competition_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported competition profile schema")
    if not payload.get("groups") or not isinstance(payload.get("paths"), dict):
        raise ValueError("competition profile requires groups and paths")
    return payload


def resolve_profile_paths(profile: Mapping[str, Any], repo_root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key, raw_value in dict(profile["paths"]).items():
        path = Path(str(raw_value))
        resolved[key] = path if path.is_absolute() else repo_root / path
    return resolved


def preflight_competition_profile(
    profile: Mapping[str, Any],
    repo_root: Path,
    *,
    environment: Mapping[str, str] | None = None,
    allow_online_models: bool = False,
) -> CompetitionPreflight:
    paths = resolve_profile_paths(profile, repo_root)
    minimums = dict(profile.get("minimums", {}))
    env = os.environ if environment is None else environment
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {"groups": list(profile["groups"])}

    required_files = (
        "registry",
        "object_store",
        "object_aliases",
        "phase5_store",
        "phase5_manifest",
    )
    required_directories = ("index_dir", "metadata_dir", "bge_index_dir", "bm25_index_dir")
    for key in required_files:
        if not paths.get(key) or not paths[key].is_file():
            errors.append(f"missing required file: {key}={paths.get(key)}")
    for key in required_directories:
        if not paths.get(key) or not paths[key].is_dir():
            errors.append(f"missing required directory: {key}={paths.get(key)}")

    clip_model = env.get("AIC_CLIP_MODEL_ID", "").strip()
    bge_model = env.get("AIC_BGE_MODEL_PATH", "").strip()
    if not allow_online_models:
        _require_local_model("AIC_CLIP_MODEL_ID", clip_model, errors)
        _require_local_model("AIC_BGE_MODEL_PATH", bge_model, errors)
    elif not clip_model or not bge_model:
        warnings.append("one or more model paths are unset; startup may require Internet access")
    if not env.get("AIC_TRANSLATION_API_KEY"):
        warnings.append("Gemini API key is not configured; local planner fallback will be used")

    if not errors:
        _audit_manifests(paths, minimums, facts, errors, warnings)
        metadata_count = sum(1 for _ in paths["metadata_dir"].glob("*.json"))
        facts["metadata_videos"] = metadata_count
        _require_minimum("metadata_videos", metadata_count, minimums, errors)

    return CompetitionPreflight(
        status="READY" if not errors else "BLOCKED",
        errors=tuple(errors),
        warnings=tuple(warnings),
        facts=facts,
        paths=paths,
    )


def build_retrieval_ui_command(
    profile: Mapping[str, Any],
    preflight: CompetitionPreflight,
    *,
    python_executable: str,
    repo_root: Path,
    host: str,
    port: int,
) -> list[str]:
    runtime = dict(profile.get("runtime", {}))
    paths = preflight.paths
    command = [
        python_executable,
        str(repo_root / "tools" / "retrieval_ui.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--registry",
        str(paths["registry"]),
        "--index-dir",
        str(paths["index_dir"]),
        "--metadata-dir",
        str(paths["metadata_dir"]),
        "--object-store",
        str(paths["object_store"]),
        "--object-aliases",
        str(paths["object_aliases"]),
        "--phase5-store",
        str(paths["phase5_store"]),
        "--groups",
        ",".join(str(item) for item in profile["groups"]),
        "--bge-index-dir",
        str(paths["bge_index_dir"]),
        "--bge-model-path",
        os.environ.get("AIC_BGE_MODEL_PATH", ""),
        "--bge-device",
        str(runtime.get("bge_device", "cpu")),
        "--bm25-index-dir",
        str(paths["bm25_index_dir"]),
        "--qa-store",
        str(paths["qa_store"]),
        "--trake-store",
        str(paths["trake_store"]),
    ]
    if runtime.get("require_keyframes", True):
        command.append("--require-keyframes")
    if runtime.get("enable_hybrid_retrieval", True):
        command.append("--enable-hybrid-retrieval")
    if runtime.get("clip_local_files_only", True):
        command.append("--clip-local-files-only")
    if runtime.get("enable_dense_frame_localization", False):
        command.append("--enable-dense-frame-localization")
        if paths.get("dense_frame_cache"):
            command.extend(("--dense-frame-cache-dir", str(paths["dense_frame_cache"])))
    return command


def _audit_manifests(
    paths: Mapping[str, Path],
    minimums: Mapping[str, Any],
    facts: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    clip_manifest = paths["index_dir"] / "manifest.json"
    if not clip_manifest.is_file():
        clip_manifest = paths["index_dir"] / "metadata.json"
    clip = _read_json(clip_manifest, "CLIP", errors)
    bge = _read_json(paths["bge_index_dir"] / "manifest.json", "BGE", errors)
    bm25 = _read_json(paths["bm25_index_dir"] / "manifest.json", "BM25", errors)
    phase5 = _read_json(paths["phase5_manifest"], "Phase 5", errors)
    if errors:
        return

    clip_vectors = int(clip.get("vector_count", clip.get("vectors", 0)))
    bge_documents = int(bge.get("corpus", {}).get("document_count", 0))
    bm25_documents = int(bm25.get("corpus", {}).get("document_count", 0))
    ocr_detections = int(phase5.get("ocr_detections", 0))
    asr_segments = int(phase5.get("asr_segments", 0))
    facts.update(
        {
            "clip_vectors": clip_vectors,
            "bge_documents": bge_documents,
            "bm25_documents": bm25_documents,
            "ocr_detections": ocr_detections,
            "asr_segments": asr_segments,
        }
    )
    for key, value in (
        ("clip_vectors", clip_vectors),
        ("bge_documents", bge_documents),
        ("bm25_documents", bm25_documents),
        ("ocr_detections", ocr_detections),
    ):
        _require_minimum(key, value, minimums, errors)
    if phase5.get("status") != "READY" or phase5.get("sqlite_integrity_check") != "ok":
        errors.append("Phase 5 manifest is not READY with SQLite integrity ok")
    if bge.get("status") != "READY":
        errors.append("BGE manifest is not READY")
    if bm25.get("status") != "READY" or bm25.get("database", {}).get("integrity_check") != "ok":
        errors.append("BM25 manifest is not READY with SQLite integrity ok")
    source_video_counts = bm25.get("corpus", {}).get("source_video_counts", {})
    ocr_videos = int(source_video_counts.get("ocr", 0))
    facts["ocr_videos"] = ocr_videos
    _require_minimum("ocr_videos", ocr_videos, minimums, errors)
    asr_videos = int(source_video_counts.get("asr", 0))
    facts["asr_videos"] = asr_videos
    if asr_videos < int(minimums.get("metadata_videos", 0)):
        warnings.append(f"ASR coverage remains partial: {asr_videos}/{minimums.get('metadata_videos', 0)} videos")
    if int(bge.get("corpus", {}).get("source_counts", {}).get("ocr", 0)) == 0:
        warnings.append("BGE uses the pre-OCR corpus; OCR retrieval uses Phase 5 FTS and BM25 v2")


def _require_local_model(name: str, raw_path: str, errors: list[str]) -> None:
    if not raw_path:
        errors.append(f"{name} is required for offline competition mode")
        return
    if not Path(raw_path).exists():
        errors.append(f"{name} does not exist: {raw_path}")


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {label} manifest: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"invalid {label} manifest: {exc}")
        return {}


def _require_minimum(
    key: str,
    actual: int,
    minimums: Mapping[str, Any],
    errors: list[str],
) -> None:
    expected = int(minimums.get(key, 0))
    if actual < expected:
        errors.append(f"{key} below competition minimum: {actual} < {expected}")
