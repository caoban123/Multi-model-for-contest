import numpy as np
import pytest

import tools.vector_search as vector_search
from aic_retrieval.search import (
    IndexValidationError,
    aggregate_results_by_video,
    build_index_metadata,
    diversify_results_by_video,
    find_asset,
    group_results_by_video,
    load_numpy_index,
    normalize_query,
    save_numpy_index,
    search_numpy_index,
    validate_numpy_index,
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
    assert grouped[0].video_score == pytest.approx(0.9)
    assert grouped[0].frame_count == 2
    assert grouped[0].matched_frame_count == 2
    assert [frame.keyframe_id for frame in grouped[0].frames] == [1, 2]


def test_max_aggregation_groups_frames_and_selects_representative() -> None:
    results = [
        _result(1, "V1", 10, 0.71),
        _result(2, "V2", 10, 0.85),
        _result(3, "V1", 20, 0.91),
        _result(4, "V1", 30, 0.80),
        _result(5, "V2", 20, 0.84),
    ]

    videos = aggregate_results_by_video(results, max_frames_per_video=3)

    assert [video.video_id for video in videos] == ["V1", "V2"]
    assert videos[0].video_score == pytest.approx(0.91)
    assert videos[0].best_score == pytest.approx(0.91)
    assert videos[0].best_keyframe_id == 20
    assert videos[0].frame_count == 3
    assert [frame.keyframe_id for frame in videos[0].frames] == [20, 30, 10]


def test_video_aggregation_tie_breaks_by_original_candidate_rank() -> None:
    results = [
        _result(1, "V2", 20, 0.90),
        _result(2, "V1", 30, 0.90),
        _result(3, "V2", 10, 0.90),
    ]

    first = aggregate_results_by_video(results, max_frames_per_video=2)
    second = aggregate_results_by_video(results, max_frames_per_video=2)

    assert [video.video_id for video in first] == ["V2", "V1"]
    assert first == second
    assert first[0].best_keyframe_id == 20


def test_video_aggregation_returns_one_card_and_limits_attached_frames() -> None:
    results = [_result(rank, "V1", rank, 1.0 - rank / 100.0) for rank in range(1, 11)]

    videos = aggregate_results_by_video(results, max_frames_per_video=2)

    assert len(videos) == 1
    assert videos[0].frame_count == 10
    assert videos[0].matched_frame_count == 2
    assert len(videos[0].frames) == 2


def test_mean_top_n_is_available_but_max_remains_default() -> None:
    results = [
        _result(1, "V1", 1, 0.90),
        _result(2, "V1", 2, 0.70),
        _result(3, "V2", 1, 0.85),
        _result(4, "V2", 2, 0.84),
    ]

    default_videos = aggregate_results_by_video(results, max_frames_per_video=2)
    mean_videos = aggregate_results_by_video(
        results,
        max_frames_per_video=2,
        aggregation_method="mean_top_n",
        mean_top_n=2,
    )

    assert default_videos[0].video_id == "V1"
    assert default_videos[0].aggregation_method == "max"
    assert mean_videos[0].video_id == "V2"
    assert mean_videos[0].video_score == pytest.approx(0.845)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_frames_per_video": 0}, "max_frames_per_video"),
        ({"max_frames_per_video": 1, "mean_top_n": 0}, "mean_top_n"),
        ({"max_frames_per_video": 1, "aggregation_method": "weighted"}, "unsupported"),
    ],
)
def test_video_aggregation_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_results_by_video([_result(1, "V1", 1, 0.9)], **kwargs)


def test_numpy_search_tie_order_is_deterministic_by_index() -> None:
    index = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    refs = [FrameRef(f"V{index}", "L21", 1, index, 0.0, 30.0, None) for index in range(3)]

    results = search_numpy_index(index, refs, np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [result.video_id for result in results] == ["V0", "V1"]


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
            "--allow-stale-index",
        ],
    )

    assert vector_search.main() == 0
    payload = capsys.readouterr().out
    assert '"query_source": "text:a red car"' in payload
    assert '"video_id": "V2"' in payload


def test_index_validation_rejects_wrong_group(tmp_path) -> None:
    index = np.array([[1.0, 0.0]], dtype=np.float32)
    refs = [FrameRef("V1", "L21", 1, 0, 0.0, 30.0, None)]
    metadata = {
        "index_schema_version": "2.0",
        "groups": ["L21"],
        "require_keyframes": False,
        "registry_fingerprint": "unused",
        "feature_source_fingerprint": "unused",
        "mapping_source_fingerprint": "unused",
    }

    with pytest.raises(IndexValidationError, match="index groups"):
        validate_numpy_index(index, refs, metadata, {"L22"}, False, None, tmp_path, allow_stale_index=True)


def test_index_validation_rejects_keyframe_scope_mismatch(tmp_path) -> None:
    index = np.array([[1.0, 0.0]], dtype=np.float32)
    refs = [FrameRef("V1", "L21", 1, 0, 0.0, 30.0, None)]
    metadata = {
        "index_schema_version": "2.0",
        "groups": ["L21"],
        "require_keyframes": True,
        "registry_fingerprint": "unused",
        "feature_source_fingerprint": "unused",
        "mapping_source_fingerprint": "unused",
    }

    with pytest.raises(IndexValidationError, match="require_keyframes"):
        validate_numpy_index(index, refs, metadata, {"L21"}, False, None, tmp_path, allow_stale_index=True)


def test_index_validation_rejects_stale_registry(tmp_path) -> None:
    feature_dir = tmp_path / "data" / "clip-features-32"
    mapping_dir = tmp_path / "data" / "map-keyframes"
    feature_dir.mkdir(parents=True)
    mapping_dir.mkdir(parents=True)
    np.save(feature_dir / "L21_V001.npy", np.array([[1.0, 0.0]], dtype=np.float16))
    (mapping_dir / "L21_V001.csv").write_text("n,pts_time,fps,frame_idx\n1,0,30,0\n", encoding="utf-8")
    registry = {
        "version": "0.1",
        "videos": [
            {
                "video_id": "L21_V001",
                "group": "L21",
                "clip_feature_path": "data/clip-features-32/L21_V001.npy",
                "mapping_path": "data/map-keyframes/L21_V001.csv",
                "has_keyframe_images": False,
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(__import__("json").dumps(registry), encoding="utf-8")
    index = np.array([[1.0, 0.0]], dtype=np.float32)
    refs = [FrameRef("L21_V001", "L21", 1, 0, 0.0, 30.0, None)]
    metadata = build_index_metadata(registry_path, registry, tmp_path, {"L21"}, False, index, refs, 1.0)
    registry_path.write_text(__import__("json").dumps({**registry, "note": "changed"}), encoding="utf-8")

    with pytest.raises(IndexValidationError, match="STALE INDEX"):
        validate_numpy_index(index, refs, metadata, {"L21"}, False, registry_path, tmp_path)


def test_non_l21_result_keeps_mapping_without_image(tmp_path) -> None:
    feature_dir = tmp_path / "data" / "clip-features-32"
    mapping_dir = tmp_path / "data" / "map-keyframes"
    feature_dir.mkdir(parents=True)
    mapping_dir.mkdir(parents=True)
    np.save(feature_dir / "L22_V001.npy", np.array([[1.0, 0.0]], dtype=np.float16))
    (mapping_dir / "L22_V001.csv").write_text("n,pts_time,fps,frame_idx\n7,1.25,25,31\n", encoding="utf-8")
    registry = {"videos": [{"video_id": "L22_V001", "group": "L22", "clip_feature_path": "data/clip-features-32/L22_V001.npy", "mapping_path": "data/map-keyframes/L22_V001.csv", "keyframe_path": None, "has_keyframe_images": False}]}
    from aic_retrieval.search import build_numpy_index

    index, refs = build_numpy_index(registry, tmp_path, groups={"L22"})
    result = search_numpy_index(index, refs, np.array([1.0, 0.0], dtype=np.float32), top_k=1)[0]

    assert (result.video_id, result.keyframe_id, result.frame_idx, result.pts_time) == ("L22_V001", 7, 31, 1.25)
    assert result.keyframe_path is None


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
