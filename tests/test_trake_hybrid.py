from __future__ import annotations

from aic_retrieval.hybrid_query_planner import HybridQueryPlan
from aic_retrieval.retrievers import RetrievalHit
from aic_retrieval.trake_hybrid import retrieve_trake_hybrid_channels


class FakeEngine:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.requests = []

    def search_channel(self, name, request):
        self.requests.append((name, request))
        if self.fail:
            raise RuntimeError("channel unavailable")
        return [
            RetrievalHit(name, 1, 2.0, "L21_V001", "asr", "d1", 3, 90, 3.0, "spoken text"),
            RetrievalHit(name, 2, 1.0, "L21_V002", "metadata", "d2", None, None, None, "title"),
        ]


def plan() -> HybridQueryPlan:
    return HybridQueryPlan(
        original_query="noi ve mua lon",
        visual_clip_query_en="a report about heavy rain",
        semantic_text_query="noi ve mua lon",
        lexical_text_query="mua lon",
        intent="mixed",
        enabled_retrievers=("clip", "bm25"),
        profile="clip_bm25_rrf",
        fusion_method="rrf",
        reasons=("test route",),
        source_filters={"bm25": ("asr",)},
        planner_source="local",
    )


def test_trake_bridge_keeps_only_frame_addressable_hits_and_filters_sources() -> None:
    engine = FakeEngine()
    result = retrieve_trake_hybrid_channels(engine, plan(), query_id="q:e1", groups=("L21",), top_k=20)

    assert [hit.document_id for hit in result.hits["bm25"]] == ["d1"]
    assert result.skipped_without_frame == {"bm25": 1}
    assert result.failures == {}
    assert engine.requests[0][1].query_text == "mua lon"
    assert engine.requests[0][1].filters == {"source_types": ("asr",)}


def test_trake_bridge_is_fail_open_unless_strict() -> None:
    result = retrieve_trake_hybrid_channels(FakeEngine(fail=True), plan(), query_id="q:e1", groups=("L21",), top_k=20)
    assert "RuntimeError" in result.failures["bm25"]

    try:
        retrieve_trake_hybrid_channels(FakeEngine(fail=True), plan(), query_id="q:e1", groups=("L21",), top_k=20, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict TRAKE hybrid retrieval must propagate channel failure")
