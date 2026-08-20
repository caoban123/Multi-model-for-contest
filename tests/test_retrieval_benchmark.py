from __future__ import annotations

from aic_retrieval.retrieval_benchmark import run_individual_benchmark
from aic_retrieval.retrievers import RetrievalHit, RetrieverHealth


class FixtureRetriever:
    name = "fixture"

    def health(self) -> RetrieverHealth:
        return RetrieverHealth(self.name, "READY")

    def search(self, request: object) -> list[RetrievalHit]:
        return [
            RetrievalHit(self.name, 1, 1.0, "V1", "metadata"),
            RetrievalHit(self.name, 2, 0.9, "V1", "asr"),
            RetrievalHit(self.name, 3, 0.8, "V2", "asr"),
        ]


def test_individual_benchmark_deduplicates_videos_and_scores_metrics() -> None:
    queries = [
        {"query_id": "q1", "query_text": "one", "expected_video_ids": ("V1",), "source_types": (), "category": "a", "language": "en"},
        {"query_id": "q2", "query_text": "two", "expected_video_ids": ("V2",), "source_types": (), "category": "b", "language": "en"},
    ]
    result = run_individual_benchmark(FixtureRetriever(), queries, groups=("L21",), document_top_k=3, video_top_k=2)

    assert result["metrics"]["mrr"] == 0.75
    assert result["metrics"]["recall_at_1"] == 0.5
    assert result["metrics"]["recall_at_5"] == 1.0
    assert [item["video_id"] for item in result["runs"][0]["video_results"]] == ["V1", "V2"]
