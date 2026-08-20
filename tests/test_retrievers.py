from __future__ import annotations

import pytest

from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest, RetrieverHealth


def test_retrieval_contracts_validate_and_serialize() -> None:
    request = RetrievalRequest("q1", "xin chao", ("L21",), 25, {"video_id": "L21_V001"}, "trace-1")
    hit = RetrievalHit("bm25", 1, 2.5, "L21_V001", "asr", "asr:1", 2, 90, 3.0, "xin chao", {"segment_id": 1})
    health = RetrieverHealth("bm25", "READY", "v1", ("lexical",))

    assert request.to_dict()["top_k"] == 25
    assert hit.to_dict()["keyframe_id"] == 2
    assert health.to_dict()["status"] == "READY"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RetrievalRequest("", "text"),
        lambda: RetrievalRequest("q", ""),
        lambda: RetrievalRequest("q", "text", top_k=0),
        lambda: RetrievalHit("bm25", 0, 1.0, "L21_V001", "asr"),
        lambda: RetrievalHit("bm25", 1, float("nan"), "L21_V001", "asr"),
        lambda: RetrieverHealth("bm25", "BROKEN"),
    ],
)
def test_retrieval_contracts_reject_invalid_values(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()
