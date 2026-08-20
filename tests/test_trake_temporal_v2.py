from __future__ import annotations

from pathlib import Path

import pytest

from aic_retrieval.trake_alignment import build_candidate_windows, fast_temporal_feasibility
from aic_retrieval.trake_candidates import TrakeCandidate, VideoCandidate
from aic_retrieval.trake_config import load_trake_config
from aic_retrieval.trake_schema import TrakeEvent, TrakeRequest
from aic_retrieval.trake_candidates import EventCandidatePool
from aic_retrieval.trake_workflow import TrakeWorkflow, _gate_alignment_seeds


ROOT = Path(__file__).parents[1]


def event(order: int, **kwargs: object) -> TrakeEvent:
    return TrakeEvent(f"e{order}", order, f"event {order}", f"event {order}", **kwargs)


def candidate(event_id: str, video_id: str, keyframe: int, pts: float, score: float = 1.0) -> TrakeCandidate:
    return TrakeCandidate(event_id, f"{event_id}:{video_id}:{keyframe}", video_id, keyframe, keyframe * 25, pts, None, score, 1, {}, ("clip",), "fixture")


def video(video_id: str, **events: tuple[TrakeCandidate, ...]) -> VideoCandidate:
    return VideoCandidate(video_id, events, len(events), len(events), 0, 0, True)


def raw(event_id: str, video_id: str, frame: int, pts: float, score: float = 1.0) -> dict[str, object]:
    return {"event_id": event_id, "video_id": video_id, "keyframe_id": frame, "frame_idx": frame * 25, "pts_time": pts, "score": score}


def test_fast_feasibility_preserves_valid_path_and_rejects_wrong_order() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    valid_video = video(
        "L21_V001",
        e1=(candidate("e1", "L21_V001", 9, 9.0), candidate("e1", "L21_V001", 1, 1.0, 0.8)),
        e2=(candidate("e2", "L21_V001", 8, 8.0),),
    )
    invalid_video = video(
        "L21_V002",
        e1=(candidate("e1", "L21_V002", 8, 8.0),),
        e2=(candidate("e2", "L21_V002", 2, 2.0),),
    )

    valid = fast_temporal_feasibility(request, valid_video)
    invalid = fast_temporal_feasibility(request, invalid_video)

    assert valid.feasible and valid.witness_candidate_ids == ("e1:L21_V001:1", "e2:L21_V001:8")
    assert not invalid.feasible
    assert invalid.failure_stage == "TEMPORAL_FEASIBILITY_FAILURE"
    assert invalid.failure_reason == "NO_MONOTONIC_REQUIRED_PATH"


def test_feasibility_respects_same_frame_and_hard_gap_only() -> None:
    same_video = video("L21_V001", e1=(candidate("e1", "L21_V001", 1, 1.0),), e2=(candidate("e2", "L21_V001", 2, 1.0),))
    strict = TrakeRequest("q", "two", (event(1), event(2)))
    allowed = TrakeRequest("q2", "two", (event(1), event(2)), constraints={"allow_same_frame": True})
    soft_gap = TrakeRequest("q3", "two", (event(1), event(2, max_gap_seconds=1.0)), constraints={"gap_mode": "preferred"})
    hard_gap = TrakeRequest("q4", "two", (event(1), event(2, max_gap_seconds=1.0)), constraints={"gap_mode": "hard"})
    far_video = video("L21_V001", e1=(candidate("e1", "L21_V001", 1, 1.0),), e2=(candidate("e2", "L21_V001", 10, 10.0),))

    assert not fast_temporal_feasibility(strict, same_video).feasible
    assert fast_temporal_feasibility(allowed, same_video).feasible
    assert fast_temporal_feasibility(soft_gap, far_video).feasible
    assert not fast_temporal_feasibility(hard_gap, far_video).feasible


def test_candidate_windows_are_anchor_aware_padded_and_capped() -> None:
    request = TrakeRequest("q", "three", (event(1), event(2), event(3)))
    item = video(
        "L21_V001",
        e1=(candidate("e1", "L21_V001", 1, 10.0),),
        e2=(candidate("e2", "L21_V001", 2, 20.0), candidate("e2", "L21_V001", 3, 30.0, 0.8)),
        e3=(candidate("e3", "L21_V001", 4, 40.0),),
    )
    feasible = (fast_temporal_feasibility(request, item),)

    windows = build_candidate_windows(request, (item,), feasible, anchor_event_id="e2", padding_seconds=5.0, max_windows_per_video=1, max_total_windows=1)

    assert len(windows) == 1
    assert windows[0].anchor_event_id == "e2"
    assert windows[0].start_pts == 5.0 and windows[0].end_pts == 45.0
    assert set(windows[0].event_candidate_ids) == {"e1", "e2", "e3"}


def test_workflow_prunes_infeasible_video_before_alignment_and_persists_windows() -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")

    def batch(_request: TrakeRequest, _limit: int) -> dict[str, list[dict[str, object]]]:
        return {
            "e1": [raw("e1", "L21_V001", 1, 1.0), raw("e1", "L21_V002", 8, 8.0)],
            "e2": [raw("e2", "L21_V001", 2, 2.0), raw("e2", "L21_V002", 2, 2.0)],
        }

    workflow = TrakeWorkflow(lambda _event, _limit: [], runtime_config=config, request_retriever=batch)
    session_id = workflow.plan("q", "first then second")["state"]["session_id"]
    searched = workflow.search(session_id, event_pool_size=30, max_per_video=5, video_pool_size=10)
    aligned = workflow.align(session_id)

    diagnostics = searched["state"]["diagnostics"]["temporal_feasibility"]
    assert diagnostics["before_count"] == 2 and diagnostics["after_video_ids"] == ["L21_V001"]
    assert searched["state"]["candidate_windows"]
    assert all(window["video_id"] == "L21_V001" for window in searched["state"]["candidate_windows"])
    assert aligned["state"]["alignments"][0]["video_id"] == "L21_V001"
    assert "temporal_filter" in aligned["state"]["stage_timings_ms"]
    assert "window_build" in aligned["state"]["stage_timings_ms"]


def test_workflow_reports_controlled_no_feasible_video() -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    workflow = TrakeWorkflow(
        lambda _event, _limit: [],
        runtime_config=config,
        request_retriever=lambda _request, _limit: {
            "e1": [raw("e1", "L21_V001", 8, 8.0)],
            "e2": [raw("e2", "L21_V001", 2, 2.0)],
        },
    )
    session_id = workflow.plan("q", "first then second")["state"]["session_id"]
    workflow.search(session_id, event_pool_size=30, max_per_video=5, video_pool_size=10)

    with pytest.raises(ValueError, match="NO_FEASIBLE_VIDEO"):
        workflow.align(session_id)


def test_semantic_seed_gate_rejects_weak_frame_used_only_to_force_time_order() -> None:
    request = TrakeRequest("q", "one then two", (event(1), event(2)))
    strong_but_late = candidate("e1", "L21_V001", 10, 10.0, 1.0)
    weak_but_early = candidate("e1", "L21_V001", 1, 1.0, 0.60)
    second = candidate("e2", "L21_V001", 11, 10.0, 1.0)
    pools = (
        EventCandidatePool(request.events[0], (strong_but_late, weak_but_early), {}),
        EventCandidatePool(request.events[1], (second,), {}),
    )
    original = video("L21_V001", e1=(strong_but_late, weak_but_early), e2=(second,))

    gated, diagnostics = _gate_alignment_seeds(request, pools, (original,), 0.85)

    assert [item.candidate_id for item in gated[0].event_candidates["e1"]] == [strong_but_late.candidate_id]
    assert not fast_temporal_feasibility(request, gated[0]).feasible
    assert diagnostics["rejected_candidate_ids_by_video"]["L21_V001"]["e1"] == [weak_but_early.candidate_id]
