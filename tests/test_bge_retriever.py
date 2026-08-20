from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from aic_retrieval.bge_retriever import BgeRetriever, build_bge_index
from aic_retrieval.retrievers import RetrievalRequest


class FakeEncoder:
    model_id = "test/bge"
    model_revision = "revision-1"
    dimension = 2
    max_seq_length = 32
    device = "cpu"

    def _vector(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0] if "xin" in text.casefold() else [0.0, 1.0], dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def token_length_stats(self, texts: Sequence[str]) -> dict[str, Any]:
        lengths = [len(text.split()) for text in texts]
        return {"document_count": len(texts), "min_tokens": min(lengths), "max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths), "model_max_seq_length": 32, "documents_over_limit": 0, "policy": "test"}


def _document(document_id: str, text: str, video_id: str, source_type: str, keyframe_id: int | None) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "source_type": source_type,
        "video_id": video_id,
        "text": text,
        "text_normalized": text.casefold(),
        "text_folded": text.casefold(),
        "content_checksum": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "keyframe_id": keyframe_id,
        "frame_idx": keyframe_id,
        "pts_time": float(keyframe_id) if keyframe_id is not None else None,
        "start_time": None,
        "end_time": None,
        "source_field": "test",
        "provenance": {},
    }


def _write_corpus(root: Path) -> tuple[Path, Path]:
    directory = root / "artifacts" / "corpora" / "l21_text"
    directory.mkdir(parents=True)
    documents = [
        _document("asr:1", "xin chào quý vị", "L21_V001", "asr", 1),
        *[_document(f"object:{index}", "person phone", f"L21_V{index + 2:03d}", "object", index + 1) for index in range(6)],
        _document("metadata:1", "bóng đá", "L21_V002", "metadata", None),
    ]
    corpus = directory / "documents.jsonl"
    with corpus.open("w", encoding="utf-8", newline="\n") as stream:
        for item in documents:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    checksum = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"group": "L21", "document_count": 8, "corpus_sha256": checksum, "input_fingerprint": "input", "source_counts": {"asr": 1, "metadata": 1, "ocr": 0, "object": 6}, "source_video_counts": {"asr": 1, "metadata": 1, "ocr": 0, "object": 6}}), encoding="utf-8")
    return corpus, manifest


def test_bge_index_build_reload_and_search(tmp_path: Path) -> None:
    corpus, corpus_manifest = _write_corpus(tmp_path)
    output = tmp_path / "artifacts" / "indexes" / "l21_bge"
    encoder = FakeEncoder()
    manifest = build_bge_index(root=tmp_path, corpus_path=corpus, corpus_manifest_path=corpus_manifest, output_dir=output, encoder=encoder)
    retriever = BgeRetriever(tmp_path, output, encoder)

    hits = retriever.search(RetrievalRequest("q1", "xin chào", ("L21",), 2))

    assert manifest["faiss"]["vector_count"] == 8
    assert manifest["model"]["dimension"] == 2
    assert hits[0].video_id == "L21_V001"
    assert hits[0].source_type == "asr"
    assert retriever.health().status == "DEGRADED"


def test_bge_retriever_filters_source_and_detects_corruption(tmp_path: Path) -> None:
    corpus, corpus_manifest = _write_corpus(tmp_path)
    output = tmp_path / "artifacts" / "indexes" / "l21_bge"
    encoder = FakeEncoder()
    build_bge_index(root=tmp_path, corpus_path=corpus, corpus_manifest_path=corpus_manifest, output_dir=output, encoder=encoder)
    retriever = BgeRetriever(tmp_path, output, encoder)

    hits = retriever.search(RetrievalRequest("q1", "xin chào", ("L21",), 1, {"source_types": ("metadata",)}))
    assert hits[0].source_type == "metadata"

    with (output / "vectors.faiss").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        BgeRetriever(tmp_path, output, encoder)
