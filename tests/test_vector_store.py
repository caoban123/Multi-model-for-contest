from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic_retrieval.vector_store import FaissVectorStore


def test_faiss_vector_store_build_save_load_search(tmp_path: Path) -> None:
    store = FaissVectorStore.build(np.asarray([[2, 0], [0, 3], [1, 1]], dtype=np.float32))
    path = tmp_path / "vectors.faiss"
    store.save(path)
    loaded = FaissVectorStore.load(path)

    results = loaded.search(np.asarray([1, 0], dtype=np.float32), 3)

    assert loaded.dimension == 2
    assert loaded.count == 3
    assert [item.position for item in results] == [0, 2, 1]
    assert results[0].score == pytest.approx(1.0)


def test_batched_build_matches_regular_build() -> None:
    vectors = np.asarray([[2, 0], [0, 3], [1, 1], [-2, 1]], dtype=np.float32)
    regular = FaissVectorStore.build(vectors)
    batched = FaissVectorStore.build_batched(vectors, batch_size=2)

    regular_results = regular.search(np.asarray([1, 0], dtype=np.float32), 4)
    batched_results = batched.search(np.asarray([1, 0], dtype=np.float32), 4)

    assert [item.position for item in batched_results] == [item.position for item in regular_results]
    assert [item.score for item in batched_results] == pytest.approx([item.score for item in regular_results])


@pytest.mark.parametrize(
    "vectors",
    [
        np.asarray([], dtype=np.float32),
        np.asarray([[0, 0]], dtype=np.float32),
        np.asarray([[float("nan"), 1]], dtype=np.float32),
    ],
)
def test_faiss_vector_store_rejects_invalid_vectors(vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FaissVectorStore.build(vectors)
