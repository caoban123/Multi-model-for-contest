from __future__ import annotations

from dataclasses import dataclass

from aic_retrieval.hybrid_engine import HybridRetrievalEngine
from aic_retrieval.hybrid_query_planner import HybridQueryPlan
from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest, RetrieverHealth
from aic_retrieval.rrf_fusion import RrfFusionConfig


@dataclass
class FixtureRetriever:
    name: str
    video_id: str
    seen_query: str = ""
    seen_filters: dict | None = None

    def health(self) -> RetrieverHealth:
        return RetrieverHealth(self.name, "READY")

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        self.seen_query = request.query_text
        self.seen_filters = dict(request.filters)
        return [RetrievalHit(self.name, 1, 1.0, self.video_id, self.name, document_id=f"{self.name}:1")]


def test_engine_uses_per_channel_queries_and_lazy_factories() -> None:
    created = {}

    def factory(name: str, video_id: str):
        def make():
            created[name] = FixtureRetriever(name, video_id)
            return created[name]
        return make

    engine = HybridRetrievalEngine(
        {"clip": factory("clip", "V1"), "bm25": factory("bm25", "V2")},
        RrfFusionConfig(candidate_pool=10),
    )
    plan = HybridQueryPlan(
        "áo đỏ có chữ HTV", "a red shirt with HTV text", "áo đỏ có chữ HTV", "HTV",
        "mixed", ("clip", "bm25"), "clip_bm25_rrf", "rrf", ("mixed evidence",),
        source_filters={"bm25": ("metadata",)},
    )
    response = engine.search("q1", plan, top_k=5)
    assert created["clip"].seen_query == "a red shirt with HTV text"
    assert created["bm25"].seen_query == "HTV"
    assert created["bm25"].seen_filters == {"source_types": ("metadata",)}
    assert response["profile"] == "clip_bm25_rrf"
    assert response["channel_hit_counts"] == {"clip": 1, "bm25": 1}
    assert len(response["results"]) == 2


def test_engine_fail_open_records_missing_factory() -> None:
    plan = HybridQueryPlan(
        "query", "query", "query", "query", "semantic_text", ("bge",), "bge", "none", (),
    )
    response = HybridRetrievalEngine({}).search("q", plan)
    assert response["results"] == []
    assert "bge" in response["failures"]
