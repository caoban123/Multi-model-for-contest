from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aic_retrieval.hybrid_audit import (
    audit_asr_jsonl,
    audit_benchmarks,
    audit_clip_index,
    audit_ocr_jsonl,
)


def test_clip_index_audit_detects_valid_index_and_missing_image(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    np.save(index_dir / "vectors.npy", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    refs = [
        {"video_id": "L21_V001", "group": "L21", "keyframe_id": 1, "frame_idx": 0, "pts_time": 0.0, "fps": 25.0, "keyframe_path": "data/keyframes/L21_V001/001.jpg"},
        {"video_id": "L21_V001", "group": "L21", "keyframe_id": 2, "frame_idx": 25, "pts_time": 1.0, "fps": 25.0, "keyframe_path": "data/keyframes/L21_V001/002.jpg"},
    ]
    (index_dir / "refs.json").write_text(json.dumps(refs), encoding="utf-8")
    (index_dir / "metadata.json").write_text(json.dumps({"vector_count": 2, "refs_count": 2}), encoding="utf-8")
    image = tmp_path / "data" / "keyframes" / "L21_V001" / "001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")

    result = audit_clip_index(index_dir, tmp_path, {"L21"})

    assert result["status"] == "READY"
    assert result["shape"] == [2, 2]
    assert result["keyframe_images"] == {"present": 1, "missing": 1, "coverage": 0.5}
    assert result["vector_stats"]["all_finite"] is True


def test_text_source_audit_reports_real_coverage(tmp_path: Path) -> None:
    ocr_path = tmp_path / "ocr.jsonl"
    ocr_path.write_text(
        json.dumps({"video_id": "L21_V001", "keyframe_id": 1, "detections": [{"text": "Xin chao"}]}) + "\n"
        + json.dumps({"video_id": "L21_V001", "keyframe_id": 2, "detections": []}) + "\n",
        encoding="utf-8",
    )
    asr_path = tmp_path / "asr.jsonl"
    asr_path.write_text(
        json.dumps({"video_id": "L21_V001", "status": "AVAILABLE", "segments": [{"text": "noi dung"}, {"text": ""}]}) + "\n",
        encoding="utf-8",
    )

    ocr = audit_ocr_jsonl(ocr_path, {("L21_V001", 1), ("L21_V001", 2)})
    asr = audit_asr_jsonl(asr_path, {"L21_V001", "L21_V002"})

    assert ocr["detections"] == 1
    assert ocr["mapping_coverage"] == 1.0
    assert ocr["usable_for_text_retrieval"] is True
    assert asr["segments"] == 2
    assert asr["nonempty_text_segments"] == 1
    assert asr["videos_known_to_clip_index"] == 1


def test_benchmark_audit_distinguishes_templates_from_judged_queries(tmp_path: Path) -> None:
    definitions = tmp_path / "benchmarks"
    artifacts = tmp_path / "artifacts"
    definitions.mkdir()
    artifacts.mkdir()
    (definitions / "hybrid_retrieval_queries_v1.json").write_text(
        json.dumps({"version": "v1", "queries": [{"query_id": "q1", "expected": []}, {"query_id": "q2", "expected": ["L21_V001"]}]}),
        encoding="utf-8",
    )

    result = audit_benchmarks(definitions, artifacts)

    assert result["query_count"] == 2
    assert result["judged_query_count"] == 1
    assert result["hybrid_retrieval_judged_query_count"] == 1
    assert result["hybrid_promotion_benchmark_ready"] is True
