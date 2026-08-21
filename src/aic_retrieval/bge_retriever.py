from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import faiss
import numpy as np

from aic_retrieval.hybrid_audit import sha256_file
from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest, RetrieverHealth
from aic_retrieval.vector_store import FaissVectorStore


BGE_INDEX_SCHEMA_VERSION = "bge-faiss-index-v1"


class EmbeddingEncoder(Protocol):
    model_id: str
    model_revision: str | None
    dimension: int
    max_seq_length: int | None
    device: str

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def token_length_stats(self, texts: Sequence[str]) -> dict[str, Any]: ...


class BgeEncoder:
    def __init__(
        self,
        model_id: str,
        *,
        model_path: str | Path | None = None,
        cache_dir: Path | None = None,
        local_files_only: bool = True,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        source = str(model_path or model_id)
        self.model_id = model_id
        self.model_revision = _snapshot_revision(Path(source))
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            source,
            cache_folder=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
            device=device,
        )
        get_dimension = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)
        dimension = get_dimension()
        if dimension is None:
            raise ValueError("BGE model does not expose an embedding dimension")
        self.dimension = int(dimension)
        self.max_seq_length = int(self.model.max_seq_length) if self.model.max_seq_length else None
        self.device = str(self.model.device)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("BGE query must not be empty")
        return self._encode([text])[0]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("BGE inputs must contain non-empty text")
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(texts), self.dimension):
            raise ValueError(f"unexpected BGE embedding shape: {matrix.shape}")
        return matrix

    def token_length_stats(self, texts: Sequence[str]) -> dict[str, Any]:
        lengths: list[int] = []
        for start in range(0, len(texts), 256):
            encoded = self.model.tokenizer(
                list(texts[start : start + 256]),
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            lengths.extend(len(item) for item in encoded["input_ids"])
        limit = self.max_seq_length
        return {
            "document_count": len(lengths),
            "min_tokens": min(lengths) if lengths else None,
            "max_tokens": max(lengths) if lengths else None,
            "mean_tokens": round(sum(lengths) / len(lengths), 3) if lengths else None,
            "model_max_seq_length": limit,
            "documents_over_limit": sum(length > limit for length in lengths) if limit else None,
            "policy": "no_chunking; fail build when any document exceeds model_max_seq_length",
        }


def _snapshot_revision(path: Path) -> str | None:
    return path.name if path.parent.name == "snapshots" else None


def _load_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not str(payload.get("text", "")).strip():
                raise ValueError(f"invalid corpus document at line {line_number}")
            documents.append(payload)
    ids = [str(item.get("document_id", "")) for item in documents]
    if not documents or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("corpus must contain unique non-empty document IDs")
    return documents


def _load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    document_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"invalid corpus document at line {line_number}")
            document_id = str(payload.get("document_id", "")).strip()
            text = str(payload.get("text", "")).strip()
            if not document_id or not text:
                raise ValueError(f"invalid corpus document at line {line_number}")
            if document_id in document_ids:
                raise ValueError(f"duplicate corpus document ID: {document_id}")
            document_ids.add(document_id)
            texts.append(text)
    if not texts:
        raise ValueError("corpus must contain at least one document")
    return texts


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_bge_index(
    *,
    root: Path,
    corpus_path: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
    encoder: EmbeddingEncoder,
    checkpoint_documents: int = 512,
    progress: Callable[[int, int], None] | None = None,
    restart: bool = False,
) -> dict[str, Any]:
    if checkpoint_documents < 1:
        raise ValueError("checkpoint_documents must be positive")
    started = time.perf_counter()
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    corpus_checksum = sha256_file(corpus_path)
    if corpus_checksum != corpus_manifest.get("corpus_sha256"):
        raise ValueError("corpus checksum does not match its manifest")
    expected_count = int(corpus_manifest.get("document_count", -1))
    texts = _load_texts(corpus_path)
    if len(texts) != expected_count:
        raise ValueError("corpus document count does not match its manifest")

    token_stats = encoder.token_length_stats(texts)
    over_limit = token_stats.get("documents_over_limit")
    if over_limit not in (None, 0):
        raise ValueError(f"{over_limit} corpus documents exceed the model sequence limit")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "vectors.faiss"
    manifest_path = output_dir / "manifest.json"
    index_building_path = index_path.with_suffix(index_path.suffix + ".building")
    manifest_building_path = manifest_path.with_suffix(manifest_path.suffix + ".building")
    checkpoint_path = output_dir / "embeddings.npy.building"
    checkpoint_manifest_path = output_dir / "build_checkpoint.json"
    if restart:
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_manifest_path.unlink(missing_ok=True)
    index_building_path.unlink(missing_ok=True)
    manifest_building_path.unlink(missing_ok=True)

    checkpoint_identity = {
        "schema_version": "bge-build-checkpoint-v1",
        "corpus_sha256": corpus_checksum,
        "document_count": expected_count,
        "model_id": encoder.model_id,
        "model_revision": encoder.model_revision,
        "dimension": encoder.dimension,
    }
    resumed_from = 0
    encoding_elapsed_ms = 0.0
    if checkpoint_path.exists() != checkpoint_manifest_path.exists():
        raise ValueError("incomplete BGE checkpoint; rerun with restart=True")
    if checkpoint_path.exists():
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if any(checkpoint_manifest.get(key) != value for key, value in checkpoint_identity.items()):
            raise ValueError("BGE checkpoint does not match corpus/model; rerun with restart=True")
        resumed_from = int(checkpoint_manifest.get("completed_documents", 0))
        encoding_elapsed_ms = float(checkpoint_manifest.get("encoding_elapsed_ms", 0.0))
        if not 0 <= resumed_from <= expected_count:
            raise ValueError("BGE checkpoint has an invalid completed document count")
        vectors = np.load(checkpoint_path, mmap_mode="r+")
        if vectors.shape != (expected_count, encoder.dimension) or vectors.dtype != np.float32:
            raise ValueError("BGE checkpoint vector shape or dtype is invalid; rerun with restart=True")
    else:
        vectors = np.lib.format.open_memmap(
            checkpoint_path,
            mode="w+",
            dtype=np.float32,
            shape=(expected_count, encoder.dimension),
        )
        _write_json_atomic(
            checkpoint_manifest_path,
            {**checkpoint_identity, "completed_documents": 0, "encoding_elapsed_ms": 0.0},
        )

    if progress:
        progress(resumed_from, expected_count)
    for start in range(resumed_from, expected_count, checkpoint_documents):
        end = min(start + checkpoint_documents, expected_count)
        encode_started = time.perf_counter()
        batch = np.asarray(encoder.encode_documents(texts[start:end]), dtype=np.float32)
        encoding_elapsed_ms += (time.perf_counter() - encode_started) * 1000
        if batch.shape != (end - start, encoder.dimension):
            raise ValueError(f"embedding shape {batch.shape} does not match corpus/model")
        if not np.isfinite(batch).all() or np.any(np.linalg.norm(batch, axis=1) == 0):
            raise ValueError("BGE embeddings must be finite and non-zero")
        vectors[start:end] = batch
        vectors.flush()
        _write_json_atomic(
            checkpoint_manifest_path,
            {
                **checkpoint_identity,
                "completed_documents": end,
                "encoding_elapsed_ms": round(encoding_elapsed_ms, 3),
            },
        )
        if progress:
            progress(end, expected_count)

    store = FaissVectorStore.build_batched(vectors)
    if store.count != expected_count:
        raise ValueError("FAISS vector count does not match corpus")
    store.save(index_building_path)
    reloaded = FaissVectorStore.load(index_building_path)
    if reloaded.count != store.count or reloaded.dimension != store.dimension:
        raise ValueError("reloaded FAISS index does not match the built index")

    manifest = {
        "schema_version": BGE_INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "group": corpus_manifest.get("group"),
        "groups": corpus_manifest.get("groups") or ([corpus_manifest["group"]] if corpus_manifest.get("group") else []),
        "model": {
            "id": encoder.model_id,
            "revision": encoder.model_revision,
            "dimension": encoder.dimension,
            "max_seq_length": encoder.max_seq_length,
            "device": encoder.device,
            "normalization": "l2",
            "query_prompt": None,
        },
        "corpus": {
            "path": _portable_path(corpus_path, root),
            "manifest_path": _portable_path(corpus_manifest_path, root),
            "sha256": corpus_checksum,
            "input_fingerprint": corpus_manifest.get("input_fingerprint"),
            "document_count": expected_count,
            "source_counts": corpus_manifest.get("source_counts", {}),
            "source_video_counts": corpus_manifest.get("source_video_counts", {}),
        },
        "token_length_stats": token_stats,
        "faiss": {
            "type": "IndexFlatIP",
            "version": getattr(faiss, "__version__", "unknown"),
            "path": _portable_path(index_path, root),
            "sha256": sha256_file(index_building_path),
            "dimension": store.dimension,
            "vector_count": store.count,
            "size_bytes": index_building_path.stat().st_size,
        },
        "build": {
            "checkpoint_documents": checkpoint_documents,
            "resumed_from_document": resumed_from,
            "checkpoint_policy": "resume when corpus and model identity match; --restart discards partial embeddings",
        },
        "timing_ms": {
            "encoding": round(encoding_elapsed_ms, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "status": "READY",
    }
    manifest_building_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    del vectors
    os.replace(index_building_path, index_path)
    os.replace(manifest_building_path, manifest_path)
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_manifest_path.unlink(missing_ok=True)
    return manifest


@dataclass(frozen=True)
class BgeIndexPaths:
    index: Path
    manifest: Path
    corpus: Path


class BgeRetriever:
    name = "bge"

    def __init__(self, root: Path, index_dir: Path, encoder: EmbeddingEncoder) -> None:
        self.root = root.resolve()
        self.index_dir = index_dir.resolve()
        self.encoder = encoder
        self.manifest_path = self.index_dir / "manifest.json"
        self.index_path = self.index_dir / "vectors.faiss"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        raw_corpus = Path(str(self.manifest["corpus"]["path"]))
        self.corpus_path = raw_corpus if raw_corpus.is_absolute() else self.root / raw_corpus
        self.documents = _load_documents(self.corpus_path)
        self.store = FaissVectorStore.load(self.index_path)
        self._validate()

    def _validate(self) -> None:
        expected = self.manifest
        if expected.get("schema_version") != BGE_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported BGE index schema")
        if sha256_file(self.corpus_path) != expected["corpus"]["sha256"]:
            raise ValueError("BGE corpus is stale or corrupted")
        if sha256_file(self.index_path) != expected["faiss"]["sha256"]:
            raise ValueError("BGE FAISS index is corrupted")
        if self.store.count != len(self.documents) or self.store.count != int(expected["faiss"]["vector_count"]):
            raise ValueError("BGE index/document count mismatch")
        if self.store.dimension != self.encoder.dimension or self.store.dimension != int(expected["model"]["dimension"]):
            raise ValueError("BGE encoder/index dimension mismatch")
        if self.encoder.model_id != expected["model"]["id"]:
            raise ValueError("BGE encoder model ID does not match index manifest")

    def health(self) -> RetrieverHealth:
        source_counts = self.manifest["corpus"].get("source_counts", {})
        source_video_counts = self.manifest["corpus"].get("source_video_counts", {})
        group_video_count = int(source_video_counts.get("metadata", 0))
        warnings = tuple(
            warning
            for warning in (
                "OCR unavailable" if not source_counts.get("ocr") else None,
                "ASR coverage is partial" if int(source_video_counts.get("asr", 0)) < group_video_count else None,
            )
            if warning
        )
        return RetrieverHealth(
            self.name,
            "DEGRADED" if warnings else "READY",
            self.manifest.get("schema_version"),
            ("semantic_text", "multilingual"),
            warnings,
            {"documents": len(self.documents), "dimension": self.store.dimension},
        )

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        index_groups = {str(item) for item in self.manifest.get("groups", ()) if str(item)}
        if not index_groups and self.manifest.get("group"):
            index_groups = {str(self.manifest["group"])}
        active_groups = index_groups.intersection(request.groups)
        if not active_groups:
            return []
        source_filter = {str(item) for item in request.filters.get("source_types", ())}
        video_filter = {str(item) for item in request.filters.get("video_ids", ())}
        query_vector = self.encoder.encode_query(request.query_text)
        needs_group_filter = active_groups != index_groups
        needs_filter = bool(source_filter or video_filter or needs_group_filter)
        pool = self.store.count if needs_filter else min(self.store.count, request.top_k)
        raw = self.store.search(query_vector, pool)
        hits: list[RetrievalHit] = []
        for item in raw:
            document = self.documents[item.position]
            if needs_group_filter and str(document["video_id"]).split("_", 1)[0] not in active_groups:
                continue
            if source_filter and str(document["source_type"]) not in source_filter:
                continue
            if video_filter and str(document["video_id"]) not in video_filter:
                continue
            hits.append(
                RetrievalHit(
                    retriever=self.name,
                    rank=len(hits) + 1,
                    raw_score=item.score,
                    video_id=str(document["video_id"]),
                    source_type=str(document["source_type"]),
                    document_id=str(document["document_id"]),
                    keyframe_id=document.get("keyframe_id"),
                    frame_idx=document.get("frame_idx"),
                    pts_time=document.get("pts_time"),
                    matched_text=str(document["text"]),
                    provenance={
                        "content_checksum": document.get("content_checksum"),
                        "source_field": document.get("source_field"),
                        **dict(document.get("provenance", {})),
                    },
                )
            )
            if len(hits) >= request.top_k:
                break
        return hits
