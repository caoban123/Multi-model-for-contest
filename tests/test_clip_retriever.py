from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aic_retrieval.clip_retriever import ClipRetriever
from aic_retrieval.retrievers import RetrievalRequest
from aic_retrieval.search import FrameRef, save_numpy_index


class Encoder:
    def encode_text(self, text: str) -> np.ndarray:
        assert text
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_clip_adapter_searches_filters_and_preserves_frame_provenance(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    refs = [
        FrameRef("L21_V001", "L21", 1, 10, 0.5, 20.0, "one.jpg"),
        FrameRef("L21_V002", "L21", 2, 20, 1.0, 20.0, "two.jpg"),
    ]
    save_numpy_index(
        index_dir,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        refs,
        {"groups": ["L21"], "model_name": "fixture"},
    )
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"videos": []}), encoding="utf-8")
    retriever = ClipRetriever(tmp_path, index_dir, registry, Encoder(), allow_stale_index=True)
    hits = retriever.search(RetrievalRequest("q", "person", top_k=2))
    assert [hit.video_id for hit in hits] == ["L21_V001", "L21_V002"]
    assert hits[0].keyframe_id == 1
    assert hits[0].provenance["keyframe_path"] == "one.jpg"
    filtered = retriever.search(RetrievalRequest("q2", "person", top_k=2, filters={"video_ids": ("L21_V002",)}))
    assert [hit.video_id for hit in filtered] == ["L21_V002"]
    assert retriever.search(RetrievalRequest("q3", "person", top_k=2, filters={"source_types": ("asr",)})) == []


def test_clip_adapter_rejects_non_matching_group(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    save_numpy_index(
        index_dir,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [FrameRef("L21_V001", "L21", 1, 10, 0.5, 20.0, None)],
        {"groups": ["L21"]},
    )
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"videos": []}), encoding="utf-8")
    retriever = ClipRetriever(tmp_path, index_dir, registry, Encoder(), allow_stale_index=True)
    assert retriever.search(RetrievalRequest("q", "person", groups=("L22",), top_k=1)) == []
