from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from aic_retrieval.hybrid_query_planner import HybridQueryPlan
from aic_retrieval.retrievers import Retriever, RetrievalHit, RetrievalRequest
from aic_retrieval.rrf_fusion import RrfFusionConfig, fuse_video_hits


class HybridRetrievalEngine:
    def __init__(
        self,
        factories: Mapping[str, Callable[[], Retriever]],
        config: RrfFusionConfig = RrfFusionConfig(),
    ) -> None:
        self.factories = dict(factories)
        self.config = config
        self.instances: dict[str, Retriever] = {}

    def _retriever(self, name: str) -> Retriever:
        if name not in self.factories:
            raise KeyError(f"retriever is not configured: {name}")
        if name not in self.instances:
            self.instances[name] = self.factories[name]()
        return self.instances[name]

    def search_channel(self, name: str, request: RetrievalRequest) -> list[RetrievalHit]:
        """Run one configured channel without applying video-level fusion."""
        return self._retriever(name).search(request)

    def search(
        self,
        query_id: str,
        plan: HybridQueryPlan,
        *,
        groups: tuple[str, ...] = ("L21",),
        top_k: int = 12,
        strict: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        channels = {}
        failures: dict[str, str] = {}
        timings: dict[str, float] = {}
        health = {}
        pool = min(1000, max(top_k, self.config.candidate_pool))
        for name in plan.enabled_retrievers:
            channel_started = time.perf_counter()
            try:
                retriever = self._retriever(name)
                health[name] = retriever.health().to_dict()
                source_types = tuple(plan.source_filters.get(name, ()))
                channels[name] = retriever.search(
                    RetrievalRequest(
                        query_id,
                        plan.query_for(name),
                        groups=groups,
                        top_k=pool,
                        filters={"source_types": source_types} if source_types else {},
                    )
                )
            except Exception as exc:
                if strict:
                    raise
                failures[name] = f"{type(exc).__name__}: {exc}"
            timings[name] = round((time.perf_counter() - channel_started) * 1000, 3)
        if len(channels) == 1:
            name, hits = next(iter(channels.items()))
            fused = fuse_video_hits({name: hits}, self.config, top_k=top_k, failures=failures)
        else:
            fused = fuse_video_hits(channels, self.config, top_k=top_k, failures=failures)
        return {
            "schema_version": "hybrid-retrieval-response-v1",
            "query_id": query_id,
            "query_plan": plan.to_dict(),
            "profile": plan.profile,
            "fusion_method": "rrf" if len(channels) > 1 else "single_channel_rank",
            "health": health,
            "failures": failures,
            "channel_hit_counts": {name: len(hits) for name, hits in channels.items()},
            "latency_ms": {**timings, "total": round((time.perf_counter() - started) * 1000, 3)},
            "results": [hit.to_dict() for hit in fused],
        }
