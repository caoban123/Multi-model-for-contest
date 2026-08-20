from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aic_retrieval.hybrid_engine import HybridRetrievalEngine
from aic_retrieval.hybrid_query_planner import HybridQueryPlan
from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest


@dataclass(frozen=True)
class TrakeHybridChannels:
    hits: dict[str, tuple[RetrievalHit, ...]]
    failures: dict[str, str]
    skipped_without_frame: dict[str, int]


def retrieve_trake_hybrid_channels(
    engine: HybridRetrievalEngine,
    plan: HybridQueryPlan,
    *,
    query_id: str,
    groups: tuple[str, ...],
    top_k: int,
    exclude: tuple[str, ...] = ("clip",),
    strict: bool = False,
) -> TrakeHybridChannels:
    """Retrieve frame-addressable text evidence for one TRAKE event.

    TRAKE aligns moments, so video-level documents without a keyframe cannot
    enter the temporal candidate pool. They remain counted in diagnostics.
    """
    hits: dict[str, tuple[RetrievalHit, ...]] = {}
    failures: dict[str, str] = {}
    skipped: dict[str, int] = {}
    for name in plan.enabled_retrievers:
        if name in exclude:
            continue
        source_types = tuple(plan.source_filters.get(name, ()))
        request = RetrievalRequest(
            query_id=f"{query_id}:{name}",
            query_text=plan.query_for(name),
            groups=groups,
            top_k=top_k,
            filters={"source_types": source_types} if source_types else {},
        )
        try:
            raw_hits = engine.search_channel(name, request)
            frame_hits = tuple(hit for hit in raw_hits if hit.keyframe_id is not None)
            hits[name] = frame_hits
            skipped[name] = len(raw_hits) - len(frame_hits)
        except Exception as exc:
            if strict:
                raise
            failures[name] = f"{type(exc).__name__}: {exc}"
    return TrakeHybridChannels(hits, failures, skipped)
