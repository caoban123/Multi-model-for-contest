from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from aic_retrieval.trake_alignment import CandidateWindow, ChainEvent, TrakeChain
from aic_retrieval.trake_candidates import TrakeCandidate
from aic_retrieval.trake_refinement import DenseWindowExpander, RefinementConfig, SampledFrame
from aic_retrieval.trake_schema import TrakeEvent, TrakeRequest
from aic_retrieval.trake_scoring import ScoreBreakdown


class FakeDecoder:
    def __init__(self) -> None:
        self.calls: list[list[float]] = []

    def sample(self, _path: Path, times: list[float]) -> list[SampledFrame]:
        self.calls.append(times)
        return [SampledFrame(value + 0.01, value, requested_pts=value, frame_index=round(value * 25)) for value in times]


def _candidate(event_id: str, keyframe_id: int, pts: float) -> TrakeCandidate:
    return TrakeCandidate(
        event_id,
        f"{event_id}:source",
        "L21_V001",
        keyframe_id,
        round(pts * 25),
        pts,
        f"keyframes/{keyframe_id}.jpg",
        0.8,
        1,
        {"fps": 25.0},
        ("clip",),
        "fixture",
    )


def _fixture(tmp_path: Path) -> tuple[DenseWindowExpander, CandidateWindow, TrakeRequest, dict[str, TrakeCandidate], FakeDecoder]:
    video = tmp_path / "data" / "video" / "L21_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture-video")
    decoder = FakeDecoder()
    expander = DenseWindowExpander(
        tmp_path,
        {"L21_V001": {"video_path": "data/video/L21_V001.mp4"}},
        tmp_path / "artifacts" / "cache" / "trake",
        lambda text, images: [-abs(float(image) - (4.0 if "one" in text else 7.0)) for image in images],
        decoder=decoder,
        config=RefinementConfig(coarse_window_seconds=5.0, coarse_fps=3.0, fine_window_seconds=1.0, fine_fps=12.0, max_frames_per_refinement=24),
        model_fingerprint="clip-test",
        config_fingerprint="config-test",
        index_fingerprint="index-test",
        candidates_per_event=5,
    )
    request = TrakeRequest(
        "q",
        "one then two",
        (
            TrakeEvent("e1", 1, "one", "one"),
            TrakeEvent("e2", 2, "two", "two"),
        ),
    )
    candidates = {"e1:source": _candidate("e1", 1, 3.5), "e2:source": _candidate("e2", 2, 7.5)}
    window = CandidateWindow(
        "L21_V001:window:1",
        "L21_V001",
        2.0,
        9.0,
        "e1",
        {"e1": "e1:source", "e2": "e2:source"},
        0.8,
    )
    return expander, window, request, candidates, decoder


def test_dense_expansion_is_two_stage_bounded_provenanced_and_cached(tmp_path: Path) -> None:
    expander, window, request, candidates, decoder = _fixture(tmp_path)

    coarse = expander.expand(window, request, candidates)

    assert coarse.status == "AVAILABLE"
    assert coarse.decoded_frame_count <= 24
    assert {item.stage for values in coarse.event_frames.values() for item in values} == {"coarse"}
    first = coarse.event_frames["e1"][0]
    assert first.requested_pts != first.decoded_pts
    assert first.seek_error_estimate == pytest.approx(0.01)
    assert first.frame_idx is not None
    assert coarse.provenance["video_fingerprint"]
    assert coarse.provenance["model_fingerprint"] == "clip-test"

    score = ScoreBreakdown(0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 1.8)
    chain = TrakeChain(
        "coarse",
        "L21_V001",
        (
            ChainEvent("e1", True, _candidate("e1", 1, 4.0)),
            ChainEvent("e2", True, _candidate("e2", 2, 7.0)),
        ),
        score,
        (),
    )
    fine = expander.expand(window, request, candidates, predicted_chain=chain)

    assert fine.status == "AVAILABLE"
    assert fine.decoded_frame_count <= 24
    assert {item.stage for values in fine.event_frames.values() for item in values} == {"fine"}
    assert min(item.requested_pts for item in fine.event_frames["e1"]) >= 3.0
    assert max(item.requested_pts for item in fine.event_frames["e1"]) <= 5.0
    calls_before_cache = len(decoder.calls)
    cached = expander.expand(window, request, candidates, predicted_chain=chain)
    assert cached.cache_key == fine.cache_key
    assert len(decoder.calls) == calls_before_cache


def test_dense_expansion_fails_safely_without_verified_fps(tmp_path: Path) -> None:
    expander, window, request, candidates, _decoder = _fixture(tmp_path)
    source = candidates["e1:source"]
    candidates["e1:source"] = TrakeCandidate(
        source.event_id,
        source.candidate_id,
        source.video_id,
        source.keyframe_id,
        source.frame_idx,
        source.pts_time,
        source.keyframe_path,
        source.local_score,
        source.local_rank,
        {},
        source.provenance,
        source.retrieval_method,
    )

    result = expander.expand(window, request, candidates)

    assert result.status == "REFINEMENT_UNAVAILABLE"
    assert result.decoded_frame_count == 0
    assert "unavailable or inconsistent" in result.warnings[0]


def test_dense_coarse_sampling_stays_local_to_each_source_candidate(tmp_path: Path) -> None:
    expander, window, request, candidates, decoder = _fixture(tmp_path)
    far = replace(window, start_pts=1.0, end_pts=1000.0)
    candidates["e2:source"] = _candidate("e2", 2, 900.0)

    result = expander.expand(far, request, candidates)

    assert result.status == "AVAILABLE"
    assert len(decoder.calls) == 2
    assert max(decoder.calls[0]) < 20.0
    assert min(decoder.calls[1]) > 800.0
    assert all(item.image_path is None for values in result.event_frames.values() for item in values)
