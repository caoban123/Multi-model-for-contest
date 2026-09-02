from __future__ import annotations

import numpy as np
import pytest

from aic_retrieval.frame_localization import FrameLocalizationConfig, FrameLocalizer
from aic_retrieval.search import FrameRef


def _refs() -> list[FrameRef]:
    return [
        FrameRef("L21_V001", "L21", keyframe_id, keyframe_id * 25, float(keyframe_id), 25.0, f"one/{keyframe_id}.jpg")
        for keyframe_id in range(1, 9)
    ] + [
        FrameRef("L21_V002", "L21", 1, 25, 1.0, 25.0, "two/1.jpg"),
    ]


def test_localizer_rescores_only_inside_retrieved_video_and_preserves_video_rank() -> None:
    refs = _refs()
    index = np.zeros((len(refs), 2), dtype=np.float32)
    index[:, 1] = 1.0
    index[4] = np.asarray([1.0, 0.0], dtype=np.float32)
    candidates = [
        {"video_id": "L21_V001", "rank": 1, "score": 0.8, "frames": [{"keyframe_id": 3, "pts_time": 3.0}]},
        {"video_id": "L21_V002", "rank": 2, "score": 0.7, "frames": [{"keyframe_id": 1, "pts_time": 1.0}]},
    ]

    result = FrameLocalizer(index, refs).localize(
        candidates,
        np.asarray([1.0, 0.0], dtype=np.float32),
        config=FrameLocalizationConfig(radius_seconds=3.0, max_frames_per_video=3, min_gap_seconds=1.5),
        query_text="person turns around",
    )

    assert [item["rank"] for item in result] == [1, 2]
    assert result[0]["frames"][0]["keyframe_id"] == 5
    assert result[0]["best_frame_idx"] == 125
    assert result[0]["frame_localization"]["video_rank_preserved"] is True
    assert all(frame["video_id"] == "L21_V001" for frame in result[0]["frames"])


def test_localizer_searches_full_video_when_candidate_has_no_seed_frame() -> None:
    refs = _refs()
    index = np.zeros((len(refs), 2), dtype=np.float32)
    index[:, 1] = 1.0
    index[6] = np.asarray([1.0, 0.0], dtype=np.float32)

    result = FrameLocalizer(index, refs).localize(
        [{"video_id": "L21_V001", "rank": 1}],
        np.asarray([1.0, 0.0], dtype=np.float32),
        config=FrameLocalizationConfig(max_frames_per_video=2),
    )

    assert result[0]["frames"][0]["keyframe_id"] == 7
    assert result[0]["frame_localization"]["eligible_frames"] == 8


def test_localizer_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="radius_seconds"):
        FrameLocalizationConfig(radius_seconds=0)


def test_localizer_marks_video_without_index_refs_unavailable() -> None:
    refs = _refs()
    index = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (len(refs), 1))
    result = FrameLocalizer(index, refs).localize(
        [{"video_id": "L21_V999", "rank": 1}],
        np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert result[0]["frame_localization"]["status"] == "UNAVAILABLE"
