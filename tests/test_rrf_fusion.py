from __future__ import annotations

from dataclasses import dataclass

import pytest

from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest, RetrieverHealth
from aic_retrieval.rrf_fusion import HybridRrfRetriever, RrfFusionConfig, fuse_video_hits


def _hit(retriever: str, rank: int, video_id: str, document_id: str) -> RetrievalHit:
    return RetrievalHit(retriever, rank, 1.0 / rank, video_id, "fixture", document_id=document_id, matched_text=document_id)


def test_video_level_rrf_deduplicates_documents_and_preserves_trace() -> None:
    channels = {
        "bge": [_hit("bge", 1, "V1", "b1"), _hit("bge", 2, "V1", "b2"), _hit("bge", 3, "V2", "b3")],
        "bm25": [_hit("bm25", 1, "V2", "m1"), _hit("bm25", 2, "V1", "m2")],
    }
    hits = fuse_video_hits(channels, RrfFusionConfig(rrf_k=60), top_k=10)
    assert [hit.video_id for hit in hits] == ["V1", "V2"]
    assert hits[0].raw_score == pytest.approx(1 / 61 + 1 / 62)
    assert hits[0].provenance["retriever_ranks"] == {"bge": 1, "bm25": 2}
    assert hits[0].provenance["evidence"]["bge"]["document_id"] == "b1"


def test_rrf_ties_are_deterministic() -> None:
    channels = {"a": [_hit("a", 1, "V2", "a2"), _hit("a", 1, "V1", "a1")]}
    assert [hit.video_id for hit in fuse_video_hits(channels, top_k=10)] == ["V1", "V2"]


@dataclass
class FixtureRetriever:
    name: str
    hits: list[RetrievalHit] | None = None
    error: Exception | None = None

    def health(self) -> RetrieverHealth:
        return RetrieverHealth(self.name, "READY")

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        if self.error:
            raise self.error
        return list(self.hits or [])


def test_hybrid_retriever_degrades_when_one_channel_fails() -> None:
    good = FixtureRetriever("good", [_hit("good", 1, "V1", "d1")])
    bad = FixtureRetriever("bad", error=RuntimeError("offline"))
    retriever = HybridRrfRetriever((good, bad), RrfFusionConfig(candidate_pool=10))
    hits = retriever.search(RetrievalRequest("q", "query", top_k=5))
    assert [hit.video_id for hit in hits] == ["V1"]
    assert hits[0].provenance["failures"] == {"bad": "RuntimeError: offline"}
    assert retriever.last_failures == {"bad": "RuntimeError: offline"}

    strict = HybridRrfRetriever((good, bad), RrfFusionConfig(candidate_pool=10, strict=True))
    with pytest.raises(RuntimeError, match="offline"):
        strict.search(RetrievalRequest("q", "query", top_k=5))


def test_rrf_config_rejects_invalid_values_and_duplicate_names() -> None:
    with pytest.raises(ValueError):
        RrfFusionConfig(rrf_k=0)
    with pytest.raises(ValueError):
        RrfFusionConfig(weights={"bge": -1})
    same = FixtureRetriever("same")
    with pytest.raises(ValueError, match="unique"):
        HybridRrfRetriever((same, same))
