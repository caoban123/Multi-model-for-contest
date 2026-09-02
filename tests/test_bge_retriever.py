from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from aic_retrieval.bge_retriever import BgeRetriever, build_bge_index
from aic_retrieval.retrievers import RetrievalRequest
from aic_retrieval.vector_store import VectorSearchResult


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


class InterruptingEncoder(FakeEncoder):
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return super().encode_documents(texts)


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


def test_bge_retriever_filters_source_video_and_detects_corruption(tmp_path: Path) -> None:
    corpus, corpus_manifest = _write_corpus(tmp_path)
    output = tmp_path / "artifacts" / "indexes" / "l21_bge"
    encoder = FakeEncoder()
    build_bge_index(root=tmp_path, corpus_path=corpus, corpus_manifest_path=corpus_manifest, output_dir=output, encoder=encoder)
    retriever = BgeRetriever(tmp_path, output, encoder)

    hits = retriever.search(RetrievalRequest("q1", "xin chào", ("L21",), 1, {"source_types": ("metadata",)}))
    assert hits[0].source_type == "metadata"

    video_hits = retriever.search(
        RetrievalRequest("q2", "xin chào", ("L21",), 5, {"video_ids": ("L21_V002",)})
    )
    assert video_hits
    assert {hit.video_id for hit in video_hits} == {"L21_V002"}

    with (output / "vectors.faiss").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        BgeRetriever(tmp_path, output, encoder)


def test_bge_multigroup_index_filters_requested_group(tmp_path: Path) -> None:
    corpus, corpus_manifest = _write_corpus(tmp_path)
    with corpus.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                _document("asr:l22", "xin chào", "L22_V001", "asr", 1),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    payload = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "group": None,
            "groups": ["L21", "L22"],
            "document_count": 9,
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        }
    )
    corpus_manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "artifacts" / "indexes" / "multi_bge"
    encoder = FakeEncoder()
    build_bge_index(
        root=tmp_path,
        corpus_path=corpus,
        corpus_manifest_path=corpus_manifest,
        output_dir=output,
        encoder=encoder,
    )

    hits = BgeRetriever(tmp_path, output, encoder).search(
        RetrievalRequest("q1", "xin chào", ("L22",), 5)
    )

    assert hits
    assert {hit.video_id.split("_", 1)[0] for hit in hits} == {"L22"}


def test_bge_filtered_search_expands_pool_until_enough_exact_hits() -> None:
    retriever = BgeRetriever.__new__(BgeRetriever)
    retriever.manifest = {"groups": ["L21"]}
    retriever.encoder = FakeEncoder()
    retriever.documents = [
        _document(
            f"ocr:{position}",
            "xin chào",
            "L21_TARGET" if position == 1500 else "L21_OTHER",
            "ocr",
            position + 1,
        )
        for position in range(3000)
    ]

    class RecordingStore:
        count = 3000
        calls: list[int] = []

        def search(self, _query, top_k):
            self.calls.append(top_k)
            return [VectorSearchResult(position, 1.0 - position / 10000.0) for position in range(top_k)]

    retriever.store = RecordingStore()
    hits = retriever.search(RetrievalRequest(
        "filtered",
        "xin chào",
        ("L21",),
        1,
        {"source_types": ("ocr",), "video_ids": ("L21_TARGET",)},
    ))

    assert retriever.store.calls == [1024, 2048]
    assert [hit.video_id for hit in hits] == ["L21_TARGET"]


def test_bge_build_resumes_from_last_completed_checkpoint(tmp_path: Path) -> None:
    corpus, corpus_manifest = _write_corpus(tmp_path)
    output = tmp_path / "index"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_bge_index(
            root=tmp_path,
            corpus_path=corpus,
            corpus_manifest_path=corpus_manifest,
            output_dir=output,
            encoder=InterruptingEncoder(fail_on_call=2),
            checkpoint_documents=3,
        )
    checkpoint = json.loads((output / "build_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["completed_documents"] == 3
    assert (output / "embeddings.npy.building").is_file()
    assert not (output / "vectors.faiss").exists()

    progress: list[tuple[int, int]] = []
    encoder = InterruptingEncoder()
    manifest = build_bge_index(
        root=tmp_path,
        corpus_path=corpus,
        corpus_manifest_path=corpus_manifest,
        output_dir=output,
        encoder=encoder,
        checkpoint_documents=3,
        progress=lambda processed, total: progress.append((processed, total)),
    )

    assert encoder.calls == 2
    assert progress == [(3, 8), (6, 8), (8, 8)]
    assert manifest["build"]["resumed_from_document"] == 3
    assert not (output / "build_checkpoint.json").exists()
    assert not (output / "embeddings.npy.building").exists()
