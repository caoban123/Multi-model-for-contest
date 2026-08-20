from __future__ import annotations

import time

from aic_retrieval.trake_alignment import CandidateWindow, DanteInspiredAligner
from aic_retrieval.trake_candidates import TrakeCandidate, VideoCandidate
from aic_retrieval.trake_config import DanteSettings
from aic_retrieval.trake_refinement import DenseFrameCandidate, DenseWindowExpansion
from aic_retrieval.trake_schema import TrakeEvent, TrakeRequest


def event(order: int, *, required: bool = True, min_gap: float | None = None, max_gap: float | None = None) -> TrakeEvent:
    return TrakeEvent(
        f"e{order}",
        order,
        f"event {order}",
        f"event {order}",
        required=required,
        min_gap_seconds=min_gap,
        max_gap_seconds=max_gap,
    )


def candidate(event_id: str, pts: float, score: float = 0.8, *, suffix: str = "") -> TrakeCandidate:
    frame = round(pts * 25)
    return TrakeCandidate(
        event_id,
        f"{event_id}:{pts:.3f}:{suffix}",
        "L21_V001",
        frame,
        frame,
        pts,
        None,
        score,
        1,
        {"fps": 25.0},
        ("clip",),
        "fixture",
    )


def video(**values: tuple[TrakeCandidate, ...]) -> VideoCandidate:
    return VideoCandidate("L21_V001", values, len(values), len(values), 0, 0, True)


def selected_pts(result: object) -> list[float]:
    return [entry.candidate.pts_time for entry in result.chains[0].events if entry.candidate is not None]


def aligner(**settings: object) -> DanteInspiredAligner:
    return DanteInspiredAligner(dante_config=DanteSettings(**settings))


def test_dante_selects_correct_monotonic_path_and_rejects_reverse_only() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    valid = video(e1=(candidate("e1", 1.0),), e2=(candidate("e2", 2.0),))
    reverse = video(e1=(candidate("e1", 4.0),), e2=(candidate("e2", 2.0),))

    assert selected_pts(aligner().align_video(request, valid)) == [1.0, 2.0]
    rejected = aligner().align_video(request, reverse)
    assert not rejected.chains and "TEMPORAL_ORDER_INVALID" in rejected.failures


def test_dante_enforces_hard_min_and_max_gap() -> None:
    request = TrakeRequest(
        "q",
        "two",
        (event(1), event(2, min_gap=2.0, max_gap=4.0)),
        constraints={"gap_mode": "hard"},
    )
    too_short = video(e1=(candidate("e1", 1.0),), e2=(candidate("e2", 2.0),))
    too_long = video(e1=(candidate("e1", 1.0),), e2=(candidate("e2", 6.0),))
    valid = video(e1=(candidate("e1", 1.0),), e2=(candidate("e2", 4.0),))

    assert not aligner().align_video(request, too_short).chains
    assert not aligner().align_video(request, too_long).chains
    assert selected_pts(aligner().align_video(request, valid)) == [1.0, 4.0]


def test_dante_optional_event_has_explicit_configured_skip_transition() -> None:
    request = TrakeRequest("q", "three", (event(1), event(2, required=False), event(3)))
    item = video(e1=(candidate("e1", 1.0),), e2=(), e3=(candidate("e3", 3.0),))

    result = aligner(optional_skip_penalty=0.33).align_video(request, item)

    assert result.chains[0].events[1].candidate is None
    assert result.chains[0].score_components["optional_skip_penalty"] == 0.33


def test_dante_same_frame_is_configurable() -> None:
    item = video(e1=(candidate("e1", 1.0),), e2=(candidate("e2", 1.0),))
    strict = TrakeRequest("q", "two", (event(1), event(2)))
    allowed = TrakeRequest("q2", "two", (event(1), event(2)), constraints={"allow_same_frame": True})

    assert not aligner().align_video(strict, item).chains
    assert selected_pts(aligner().align_video(allowed, item)) == [1.0, 1.0]


def test_dante_keeps_multiple_paths_and_tie_breaks_by_candidate_id() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    item = video(
        e1=(candidate("e1", 1.0, suffix="b"), candidate("e1", 1.0, suffix="a")),
        e2=(candidate("e2", 2.0), candidate("e2", 3.0)),
    )

    result = aligner(transition_gap_penalty=0.0).align_video(request, item, top_k=4)

    assert len(result.chains) == 4
    assert result.chains[0].events[0].candidate.candidate_id.endswith(":a")
    assert [chain.score_components["dante_objective"] for chain in result.chains] == sorted(
        [chain.score_components["dante_objective"] for chain in result.chains], reverse=True
    )


def test_dante_dynamic_program_beats_greedy_first_event_choice() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    item = video(
        e1=(candidate("e1", 9.0, 1.0, suffix="greedy"), candidate("e1", 1.0, 0.7, suffix="dp")),
        e2=(candidate("e2", 8.0, 1.0),),
    )

    result = aligner(transition_gap_penalty=0.0).align_video(request, item)

    assert selected_pts(result) == [1.0, 8.0]


def test_dante_aligns_dense_window_and_preserves_seek_provenance() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    source = video(e1=(candidate("e1", 2.0),), e2=(candidate("e2", 8.0),))
    window = CandidateWindow("w1", "L21_V001", 0.0, 10.0, "e1", {"e1": "e1:2.000:", "e2": "e2:8.000:"}, 0.8)
    dense = DenseWindowExpansion(
        "AVAILABLE",
        "w1",
        "L21_V001",
        {
            "e1": (DenseFrameCandidate("e1", "e1:2.000:", 50, None, 2.49, 2.5, 25.0, 0.01, 2.5, 0.9, "fine", 1),),
            "e2": (DenseFrameCandidate("e2", "e2:8.000:", 200, 201, 8.03, 8.04, 25.0, 0.01, 8.04, 0.9, "fine", 1),),
        },
        2,
        {},
        1.0,
        "cache",
        {},
    )

    result = aligner().align_window(request, source, window, dense)

    first = result.chains[0].events[0].candidate
    assert selected_pts(result) == [2.5, 8.04]
    assert first.evidence["dense_refinement"] is True
    assert first.evidence["frame_index_is_estimate"] is True
    assert first.evidence["source_keyframe_id"] == 50


def test_dante_required_event_without_candidate_is_controlled() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    result = aligner().align_video(request, video(e1=(candidate("e1", 1.0),), e2=()))

    assert not result.chains
    assert "REQUIRED_EVENT_MISSING:e2" in result.failures


def test_dante_long_candidate_list_performance_sanity() -> None:
    request = TrakeRequest("q", "three", (event(1), event(2), event(3)))
    item = video(
        e1=tuple(candidate("e1", index / 10, 0.5) for index in range(1000)),
        e2=tuple(candidate("e2", 100 + index / 10, 0.5) for index in range(1000)),
        e3=tuple(candidate("e3", 200 + index / 10, 0.5) for index in range(1000)),
    )
    started = time.perf_counter()

    result = aligner().align_video(request, item, top_k=3, beam_size=20)

    assert result.chains
    assert time.perf_counter() - started < 3.0
