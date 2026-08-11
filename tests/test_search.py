import numpy as np
import pytest

from aic_retrieval.search import normalize_query, search_numpy_index
from aic_retrieval.search import FrameRef


def test_numpy_search_returns_highest_cosine_match() -> None:
    index = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    index = index / np.linalg.norm(index, axis=1, keepdims=True)
    refs = [
        FrameRef("V1", "L21", 1, 10, 0.0, 30.0, None),
        FrameRef("V2", "L21", 1, 20, 0.0, 30.0, None),
        FrameRef("V3", "L21", 1, 30, 0.0, 30.0, None),
    ]

    results = search_numpy_index(index, refs, np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert [result.video_id for result in results] == ["V1", "V3"]
    assert results[0].score == pytest.approx(1.0)


def test_normalize_query_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="must not be all zeros"):
        normalize_query(np.zeros(3, dtype=np.float32), expected_dim=3)

