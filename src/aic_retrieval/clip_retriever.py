from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest, RetrieverHealth
from aic_retrieval.search import FrameRef, load_numpy_index, search_numpy_index, validate_numpy_index


CLIP_RETRIEVER_VERSION = "clip-numpy-adapter-v1"


class ClipQueryEncoder(Protocol):
    def encode_text(self, text: str) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


class ClipRetriever:
    name = "clip"

    def __init__(
        self,
        root: Path,
        index_dir: Path,
        registry_path: Path,
        encoder: ClipQueryEncoder,
        *,
        groups: tuple[str, ...] = ("L21",),
        require_keyframes: bool = False,
        allow_stale_index: bool = False,
    ) -> None:
        self.root = root.resolve()
        self.index_dir = index_dir.resolve()
        self.registry_path = registry_path.resolve()
        self.encoder = encoder
        self.groups = tuple(groups)
        self.index, self.refs, self.metadata = load_numpy_index(self.index_dir)
        validate_numpy_index(
            self.index,
            self.refs,
            self.metadata,
            set(self.groups),
            require_keyframes,
            self.registry_path,
            self.root,
            allow_stale_index,
        )

    def health(self) -> RetrieverHealth:
        warnings = tuple(
            warning
            for warning in (
                "CLIP feature extraction provenance is incomplete"
                if "exact extraction provenance is not recorded" in str(self.metadata.get("feature_provenance", ""))
                else None,
                "CLIP model name is unknown in index metadata" if self.metadata.get("model_name") == "unknown" else None,
            )
            if warning
        )
        return RetrieverHealth(
            self.name,
            "DEGRADED" if warnings else "READY",
            CLIP_RETRIEVER_VERSION,
            ("visual_semantic", "frame_level"),
            warnings,
            {"vectors": len(self.refs), "dimension": int(self.index.shape[1]), "groups": self.groups},
        )

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        query_vector = self.encoder.encode_text(request.query_text)
        return self._search_vector(request, query_vector)

    def search_many(self, requests: Sequence[RetrievalRequest]) -> list[list[RetrievalHit]]:
        if not requests:
            return []
        texts = [request.query_text for request in requests]
        encode_many = getattr(self.encoder, "encode_texts", None)
        vectors = (
            encode_many(texts)
            if callable(encode_many)
            else np.stack([self.encoder.encode_text(text) for text in texts])
        )
        if len(vectors) != len(requests):
            raise ValueError("CLIP batch encoder returned an unexpected vector count")
        return [self._search_vector(request, vector) for request, vector in zip(requests, vectors)]

    def _search_vector(self, request: RetrievalRequest, query_vector: np.ndarray) -> list[RetrievalHit]:
        active_groups = set(request.groups) & set(self.groups)
        if not active_groups:
            return []
        source_filter = {str(item) for item in request.filters.get("source_types", ())}
        if source_filter and not ({"clip", "visual"} & source_filter):
            return []
        video_filter = {str(item) for item in request.filters.get("video_ids", ())}
        needs_filter = bool(video_filter or active_groups != set(self.groups))
        pool = len(self.refs) if needs_filter else min(len(self.refs), request.top_k)
        raw = search_numpy_index(self.index, self.refs, query_vector, pool)
        hits: list[RetrievalHit] = []
        for result in raw:
            if result.group not in active_groups:
                continue
            if video_filter and result.video_id not in video_filter:
                continue
            hits.append(
                RetrievalHit(
                    retriever=self.name,
                    rank=len(hits) + 1,
                    raw_score=result.score,
                    video_id=result.video_id,
                    source_type="clip",
                    document_id=f"clip:{result.video_id}:{result.keyframe_id}",
                    keyframe_id=result.keyframe_id,
                    frame_idx=result.frame_idx,
                    pts_time=result.pts_time,
                    matched_text=None,
                    provenance={
                        "group": result.group,
                        "fps": result.fps,
                        "keyframe_path": result.keyframe_path,
                    },
                )
            )
            if len(hits) >= request.top_k:
                break
        return hits
