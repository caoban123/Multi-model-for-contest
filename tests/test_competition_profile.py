from __future__ import annotations

import json
from pathlib import Path

from aic_retrieval.competition_profile import (
    build_retrieval_ui_command,
    preflight_competition_profile,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    paths = {
        "registry": "registry.json",
        "index_dir": "clip",
        "metadata_dir": "metadata",
        "object_store": "objects.sqlite",
        "object_aliases": "aliases.json",
        "phase5_store": "phase5.sqlite",
        "phase5_manifest": "phase5_manifest.json",
        "bge_index_dir": "bge",
        "bm25_index_dir": "bm25",
        "qa_store": "qa.sqlite",
        "trake_store": "trake.sqlite",
        "dense_frame_cache": "dense-cache",
    }
    for name in ("registry.json", "objects.sqlite", "aliases.json", "phase5.sqlite"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "L21_V001.json").write_text("{}", encoding="utf-8")
    _write_json(tmp_path / "clip" / "manifest.json", {"vector_count": 10})
    _write_json(
        tmp_path / "bge" / "manifest.json",
        {"status": "READY", "corpus": {"document_count": 5, "source_counts": {"ocr": 0}}},
    )
    _write_json(
        tmp_path / "bm25" / "manifest.json",
        {
            "status": "READY",
            "corpus": {"document_count": 7, "source_video_counts": {"ocr": 1, "asr": 0}},
            "database": {"integrity_check": "ok"},
        },
    )
    _write_json(
        tmp_path / "phase5_manifest.json",
        {"status": "READY", "sqlite_integrity_check": "ok", "ocr_detections": 12, "asr_segments": 0},
    )
    return {
        "schema_version": "aic-competition-profile-v1",
        "groups": ["L21"],
        "paths": paths,
        "minimums": {
            "clip_vectors": 10,
            "metadata_videos": 1,
            "ocr_detections": 12,
            "ocr_videos": 1,
            "bm25_documents": 7,
            "bge_documents": 5,
        },
        "runtime": {
            "require_keyframes": True,
            "enable_hybrid_retrieval": True,
            "clip_local_files_only": True,
            "bge_device": "cpu",
            "enable_dense_frame_localization": True,
        },
    }


def test_competition_preflight_accepts_complete_offline_profile(tmp_path: Path) -> None:
    profile = _fixture(tmp_path)
    clip_model = tmp_path / "clip-model"
    bge_model = tmp_path / "bge-model"
    clip_model.mkdir()
    bge_model.mkdir()
    result = preflight_competition_profile(
        profile,
        tmp_path,
        environment={
            "AIC_CLIP_MODEL_ID": str(clip_model),
            "AIC_BGE_MODEL_PATH": str(bge_model),
        },
    )

    assert result.ready
    assert result.facts["ocr_detections"] == 12
    assert any("ASR coverage remains partial" in warning for warning in result.warnings)
    assert any("pre-OCR corpus" in warning for warning in result.warnings)


def test_competition_preflight_blocks_missing_local_models(tmp_path: Path) -> None:
    result = preflight_competition_profile(_fixture(tmp_path), tmp_path, environment={})

    assert not result.ready
    assert any("AIC_CLIP_MODEL_ID" in error for error in result.errors)
    assert any("AIC_BGE_MODEL_PATH" in error for error in result.errors)


def test_competition_preflight_accepts_numpy_metadata_manifest_name(tmp_path: Path) -> None:
    profile = _fixture(tmp_path)
    (tmp_path / "clip" / "manifest.json").replace(tmp_path / "clip" / "metadata.json")
    clip_model = tmp_path / "clip-model"
    bge_model = tmp_path / "bge-model"
    clip_model.mkdir()
    bge_model.mkdir()

    result = preflight_competition_profile(
        profile,
        tmp_path,
        environment={
            "AIC_CLIP_MODEL_ID": str(clip_model),
            "AIC_BGE_MODEL_PATH": str(bge_model),
        },
    )

    assert result.ready


def test_competition_command_applies_full_profile(tmp_path: Path, monkeypatch) -> None:
    profile = _fixture(tmp_path)
    clip_model = tmp_path / "clip-model"
    bge_model = tmp_path / "bge-model"
    clip_model.mkdir()
    bge_model.mkdir()
    monkeypatch.setenv("AIC_BGE_MODEL_PATH", str(bge_model))
    result = preflight_competition_profile(
        profile,
        tmp_path,
        environment={
            "AIC_CLIP_MODEL_ID": str(clip_model),
            "AIC_BGE_MODEL_PATH": str(bge_model),
        },
    )

    command = build_retrieval_ui_command(
        profile,
        result,
        python_executable="python",
        repo_root=tmp_path,
        host="127.0.0.1",
        port=8765,
    )

    assert "--enable-hybrid-retrieval" in command
    assert "--require-keyframes" in command
    assert "--clip-local-files-only" in command
    assert "--enable-dense-frame-localization" in command
    assert command[command.index("--dense-frame-cache-dir") + 1].endswith("dense-cache")
    assert command[command.index("--phase5-store") + 1].endswith("phase5.sqlite")
    assert command[command.index("--bm25-index-dir") + 1].endswith("bm25")


def test_shipped_competition_profile_selects_ocr_enriched_bge_v2() -> None:
    profile_path = Path(__file__).resolve().parents[1] / "configs" / "competition_l21_l30.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["paths"]["bge_index_dir"].endswith("l21_l30_bge_v2")
    assert profile["minimums"]["bge_documents"] == 292_488
