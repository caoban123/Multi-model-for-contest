from __future__ import annotations

import pytest

from aic_retrieval.trake_alignment import ChainEvent, TemporalAligner, TrakeChain, validate_chain
from aic_retrieval.trake_candidates import TrakeCandidate, VideoCandidate
from aic_retrieval.trake_schema import TrakeEvent, TrakeRequest
from aic_retrieval.trake_scoring import ScoreBreakdown


def event(order: int, required: bool = True, **kwargs: object) -> TrakeEvent:
    return TrakeEvent(f"e{order}", order, f"event {order}", f"event {order}", required=required, **kwargs)


def candidate(event_id: str, video: str, frame: int, score: float = 0.5, pts: float | None = None) -> TrakeCandidate:
    return TrakeCandidate(event_id, f"{event_id}:{video}:{frame}", video, frame, frame * 10, float(frame if pts is None else pts), None, score, 1, {}, ("clip",), "fixture")


def video(video_id: str, **events: tuple[TrakeCandidate, ...]) -> VideoCandidate:
    return VideoCandidate(video_id, events, len(events), len(events), 0, 0, True)


def test_same_video_and_strict_order_guard() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    score = ScoreBreakdown(0, 0, 0, 0, 0, 0, 0)
    cross = TrakeChain("c", "L21_V001", (ChainEvent("e1", True, candidate("e1", "L21_V001", 1)), ChainEvent("e2", True, candidate("e2", "L21_V002", 2))), score, ())
    assert "CROSS_VIDEO_CHAIN" in validate_chain(request, cross)[1]
    same_frame = TrakeChain("c2", "L21_V001", (ChainEvent("e1", True, candidate("e1", "L21_V001", 1)), ChainEvent("e2", True, candidate("e2", "L21_V001", 1))), score, ())
    assert "TEMPORAL_ORDER_INVALID" in validate_chain(request, same_frame)[1]


def test_dp_finds_chain_that_greedy_first_choice_misses() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    item = video(
        "L21_V001",
        e1=(candidate("e1", "L21_V001", 9, 1.0), candidate("e1", "L21_V001", 1, 0.8)),
        e2=(candidate("e2", "L21_V001", 8, 1.0),),
    )
    result = TemporalAligner().align_video(request, item)
    assert [entry.candidate.keyframe_id for entry in result.chains[0].events if entry.candidate] == [1, 8]


def test_wrong_order_only_has_no_chain() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    result = TemporalAligner().align_video(request, video("L21_V001", e1=(candidate("e1", "L21_V001", 8),), e2=(candidate("e2", "L21_V001", 2),)))
    assert result.chains == ()
    assert "TEMPORAL_ORDER_INVALID" in result.failures


def test_optional_event_can_be_skipped_with_penalty() -> None:
    request = TrakeRequest("q", "three", (event(1), event(2, required=False), event(3)))
    result = TemporalAligner().align_video(request, video("L21_V001", e1=(candidate("e1", "L21_V001", 1),), e2=(), e3=(candidate("e3", "L21_V001", 3),)))
    chain = result.chains[0]
    assert chain.events[1].candidate is None
    assert chain.score.optional_penalty > 0


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_aligns_two_to_five_ordered_events(count: int) -> None:
    events = tuple(event(index) for index in range(1, count + 1))
    request = TrakeRequest("q", "many", events)
    pools = {item.event_id: (candidate(item.event_id, "L21_V001", item.order),) for item in events}
    result = TemporalAligner().align_video(request, video("L21_V001", **pools))
    assert len(result.chains[0].events) == count


def test_required_missing_and_hard_gap_are_invalid() -> None:
    request = TrakeRequest("q", "missing", (event(1), event(2)))
    missing = TemporalAligner().align_video(request, video("L21_V001", e1=(candidate("e1", "L21_V001", 1),), e2=()))
    assert missing.chains == () and "REQUIRED_EVENT_MISSING:e2" in missing.failures
    gap_request = TrakeRequest("q", "gap", (event(1), event(2, max_gap_seconds=2)), constraints={"gap_mode": "hard"})
    gap = TemporalAligner().align_video(gap_request, video("L21_V001", e1=(candidate("e1", "L21_V001", 1),), e2=(candidate("e2", "L21_V001", 10),)))
    assert gap.chains == () and "HARD_GAP_CONSTRAINT_VIOLATED" in gap.failures


def test_preferred_gap_keeps_chain_with_warning_and_score_breakdown() -> None:
    request = TrakeRequest("q", "gap", (event(1), event(2, max_gap_seconds=2)), constraints={"gap_mode": "preferred"})
    chain = TemporalAligner().align_video(request, video("L21_V001", e1=(candidate("e1", "L21_V001", 1),), e2=(candidate("e2", "L21_V001", 10),))).chains[0]
    assert chain.score.gap_penalty > 0
    assert "PREFERRED_GAP_CONSTRAINT_VIOLATED" in chain.warnings
