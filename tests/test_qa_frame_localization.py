from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from aic_retrieval.frame_localization import FrameLocalizer
from aic_retrieval.retrieval_ui import RetrievalUiService
from aic_retrieval.search import FrameRef


class _Encoder:
    def encode_text(self, text: str) -> np.ndarray:
        assert "answer focus" in text
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _Planner:
    def plan_with_trace(self, query: str, use_gemini: bool):
        assert "What is shown?" in query
        return SimpleNamespace(
            plan=SimpleNamespace(visual_clip_query_en="event with answer focus"),
            trace=SimpleNamespace(to_dict=lambda: {"status": "SUCCEEDED", "source": "fixture"}),
        )


def test_qa_stage_b_localizes_frames_without_changing_global_frame_results() -> None:
    refs = [
        FrameRef("L21_V001", "L21", 1, 25, 1.0, 25.0, "one/001.jpg"),
        FrameRef("L21_V001", "L21", 2, 50, 2.0, 25.0, "one/002.jpg"),
        FrameRef("L21_V001", "L21", 3, 75, 3.0, 25.0, "one/003.jpg"),
    ]
    index = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]], dtype=np.float32)
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.frame_localizer = FrameLocalizer(index, refs)
    service.hybrid_query_planner = _Planner()
    service.encoder = _Encoder()
    service._encoder = lambda: service.encoder
    raw_results = [{"video_id": "L21_V001", "keyframe_id": 1}]

    result = service._qa_localize_frames(
        {
            "query": "event",
            "video_results": [{"video_id": "L21_V001", "rank": 1, "frames": [{"keyframe_id": 1, "pts_time": 1.0}]}],
            "video_groups": [{"video_id": "L21_V001"}],
            "results": raw_results,
        },
        event_query="event",
        question="What is shown?",
        use_gemini_planner=False,
        radius_seconds=20,
        frame_limit=3,
    )

    assert result["frame_localization"]["status"] == "APPLIED"
    assert result["video_results"][0]["best_keyframe_id"] == 2
    assert result["video_groups"][0]["best_keyframe_id"] == 2
    assert result["results"] is raw_results


def test_qa_stage_b_falls_back_without_destroying_candidates() -> None:
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.hybrid_query_planner = _Planner()
    service._encoder = lambda: (_ for _ in ()).throw(RuntimeError("model unavailable"))
    response = {"query": "event", "video_results": [{"video_id": "L21_V001", "rank": 1}]}

    result = service._qa_localize_frames(
        response,
        event_query="event",
        question="What is shown?",
        use_gemini_planner=False,
        radius_seconds=40,
        frame_limit=12,
    )

    assert result["video_results"] == response["video_results"]
    assert result["frame_localization"]["status"] == "FALLBACK"


def test_qa_stage_b_promotes_decoded_frames_without_changing_video_rank() -> None:
    refs = [
        FrameRef("L21_V001", "L21", 1, 250, 10.0, 25.0, "one/001.jpg"),
        FrameRef("L21_V001", "L21", 2, 300, 12.0, 25.0, "one/002.jpg"),
    ]
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.frame_localizer = FrameLocalizer(np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32), refs)
    service.refs_by_video = {"L21_V001": refs}
    service.hybrid_query_planner = _Planner()
    service.encoder = _Encoder()
    service._encoder = lambda: service.encoder
    service.dense_frame_localizer = SimpleNamespace(localize=lambda **_kwargs: {
        "status": "APPLIED",
        "version": "dense-frame-localization-v2",
        "frames": [{
            "dense_rank": 1,
            "dense_score": 0.98,
            "frame_idx": 263,
            "pts_time": 10.52,
            "fps": 25.0,
            "keyframe_path": "dense/rank_01.jpg",
            "dense_frame": True,
            "mapping_status": "DECODED_FRAME_REQUIRES_VISUAL_CONFIRMATION",
        }],
    })

    result = service._qa_localize_frames(
        {
            "query": "event",
            "video_results": [{"video_id": "L21_V001", "rank": 1, "frames": [{"keyframe_id": 1, "pts_time": 10.0}]}],
        },
        event_query="event",
        question="What is shown?",
        use_gemini_planner=False,
        radius_seconds=20,
        frame_limit=3,
    )

    candidate = result["video_results"][0]
    assert candidate["rank"] == 1
    assert candidate["best_frame_idx"] == 263
    assert candidate["frames"][0]["dense_frame"] is True
    assert candidate["frames"][0]["source_keyframe_id"] == 1
    assert result["frame_localization"]["dense_video"]["videos_applied"] == 1
