from __future__ import annotations

from aic_retrieval.retrieval_ui import RetrievalUiService


class _Reranker:
    def rerank(self, query, frames):
        assert "Question" in query
        assert len(frames) == 3
        return {
            "status": "APPLIED",
            "version": "fixture",
            "raw_text": '{"candidates":[]}',
            "items": [{
                "candidate_id": "L21_V001:k3",
                "relevance": 0.95,
                "matched_events": ["answer visible"],
                "reason": "best evidence",
            }],
        }


def test_qa_gemini_reranker_can_change_frame_but_not_video_rank() -> None:
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.gemini_visual_reranker = _Reranker()
    rows = [
        {
            "video_id": "L21_V001",
            "rank": 1,
            "frames": [
                {"video_id": "L21_V001", "keyframe_id": 1, "frame_idx": 25, "pts_time": 1.0, "localization_rank": 1},
                {"video_id": "L21_V001", "keyframe_id": 3, "frame_idx": 75, "pts_time": 3.0, "localization_rank": 2},
            ],
        },
        {
            "video_id": "L21_V002",
            "rank": 2,
            "frames": [{"video_id": "L21_V002", "keyframe_id": 2, "frame_idx": 50, "pts_time": 2.0}],
        },
    ]
    result = service._qa_rerank_frames_with_gemini(
        {"video_results": rows, "video_groups": rows},
        query="event\nQuestion: what?",
    )

    assert [item["rank"] for item in result["video_results"]] == [1, 2]
    assert result["video_results"][0]["best_keyframe_id"] == 3
    assert result["video_results"][0]["frames"][0]["vlm_reason"] == "best evidence"
    assert result["gemini_frame_rerank"]["candidate_generation"] == "system_allowlist_only"


def test_qa_gemini_reranker_failure_preserves_baseline() -> None:
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.gemini_visual_reranker = type("Broken", (), {"rerank": lambda *args: (_ for _ in ()).throw(RuntimeError("offline"))})()
    response = {"video_results": [{"video_id": "L21_V001", "rank": 1, "frames": []}]}
    result = service._qa_rerank_frames_with_gemini(response, query="event")
    assert result["video_results"] == response["video_results"]
    assert result["gemini_frame_rerank"]["status"] == "FALLBACK"
