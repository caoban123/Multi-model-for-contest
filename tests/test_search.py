import numpy as np
import pytest

import tools.vector_search as vector_search
from aic_retrieval.search import (
    diversify_results_by_video,
    find_asset,
    group_results_by_video,
    load_numpy_index,
    normalize_query,
    save_numpy_index,
    search_numpy_index,
)
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


def test_save_and_load_numpy_index_roundtrip(tmp_path) -> None:
    index = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    refs = [
        FrameRef("V1", "L21", 1, 10, 0.0, 30.0, "data/keyframes/V1/001.jpg"),
        FrameRef("V1", "L21", 2, 20, 1.0, 30.0, "data/keyframes/V1/002.jpg"),
    ]
    metadata = {"groups": ["L21"], "vectors": 2}

    save_numpy_index(tmp_path, index, refs, metadata)
    loaded_index, loaded_refs, loaded_metadata = load_numpy_index(tmp_path)

    np.testing.assert_allclose(loaded_index, index)
    assert loaded_refs == refs
    assert loaded_metadata == metadata


def test_find_asset_returns_matching_video() -> None:
    registry = {"videos": [{"video_id": "L21_V001"}, {"video_id": "L21_V002"}]}

    assert find_asset(registry, "L21_V002") == {"video_id": "L21_V002"}


def test_find_asset_rejects_missing_video() -> None:
    registry = {"videos": [{"video_id": "L21_V001"}]}

    with pytest.raises(ValueError, match="video_id not found"):
        find_asset(registry, "L21_V999")


def test_diversify_results_by_video_limits_frames_per_video() -> None:
    results = [
        _result(1, "V1", 1, 0.9),
        _result(2, "V1", 2, 0.8),
        _result(3, "V2", 1, 0.7),
    ]

    diversified = diversify_results_by_video(results, max_frames_per_video=1)

    assert [(item.rank, item.video_id, item.keyframe_id) for item in diversified] == [
        (1, "V1", 1),
        (2, "V2", 1),
    ]


def test_group_results_by_video_keeps_best_video_order() -> None:
    results = [
        _result(1, "V1", 1, 0.9),
        _result(2, "V1", 2, 0.8),
        _result(3, "V2", 1, 0.7),
    ]

    grouped = group_results_by_video(results, max_frames_per_video=2)

    assert [item.video_id for item in grouped] == ["V1", "V2"]
    assert grouped[0].best_score == pytest.approx(0.9)
    assert [frame.keyframe_id for frame in grouped[0].frames] == [1, 2]


def test_vector_search_accepts_text_query(tmp_path, monkeypatch, capsys) -> None:
    index_dir = tmp_path / "index"
    index = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    refs = [
        FrameRef("V1", "L21", 1, 10, 0.0, 30.0, None),
        FrameRef("V2", "L21", 1, 20, 0.0, 30.0, None),
    ]
    save_numpy_index(index_dir, index, refs, {"groups": ["L21"]})

    def fake_encode_clip_text(text, model_id, cache_dir, local_files_only):
        assert text == "a red car"
        assert model_id == "mock-model"
        assert str(cache_dir) == "D:\\AIC\\.cache\\huggingface"
        assert local_files_only is True
        return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(vector_search, "encode_clip_text", fake_encode_clip_text)
    monkeypatch.setattr(
        "sys.argv",
        [
            "vector_search.py",
            "--index-dir",
            str(index_dir),
            "--query-text",
            "a red car",
            "--clip-model-id",
            "mock-model",
            "--clip-cache-dir",
            "D:\\AIC\\.cache\\huggingface",
            "--clip-local-files-only",
            "--top-k",
            "1",
        ],
    )

    assert vector_search.main() == 0
    payload = capsys.readouterr().out
    assert '"query_source": "text:a red car"' in payload
    assert '"video_id": "V2"' in payload


def _result(rank: int, video_id: str, keyframe_id: int, score: float):
    from aic_retrieval.search import SearchResult

    return SearchResult(
        rank=rank,
        score=score,
        video_id=video_id,
        group="L21",
        keyframe_id=keyframe_id,
        frame_idx=keyframe_id * 10,
        pts_time=float(keyframe_id),
        fps=30.0,
        keyframe_path=None,
    )
