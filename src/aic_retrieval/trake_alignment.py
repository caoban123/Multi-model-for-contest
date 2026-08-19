"""Same-video temporal DP/beam alignment for Phase 8 TRAKE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aic_retrieval.trake_candidates import TrakeCandidate, VideoCandidate
from aic_retrieval.trake_schema import TrakeRequest
from aic_retrieval.trake_scoring import ScoreBreakdown, ScoreConfig, score_chain


ALIGNMENT_VERSION = "phase8-temporal-dp-v1"


@dataclass(frozen=True)
class ChainEvent:
    event_id: str
    required: bool
    candidate: TrakeCandidate | None
    selection: str = "algorithm"


@dataclass(frozen=True)
class TrakeChain:
    chain_id: str
    video_id: str
    events: tuple[ChainEvent, ...]
    score: ScoreBreakdown
    warnings: tuple[str, ...]
    valid: bool = True
    manual: bool = False
    version: str = ALIGNMENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    video_id: str
    chains: tuple[TrakeChain, ...]
    failures: tuple[str, ...] = ()


@dataclass
class _State:
    selected: tuple[TrakeCandidate | None, ...]
    gap_penalties: int
    warnings: tuple[str, ...]


def validate_chain(request: TrakeRequest, chain: TrakeChain, *, allow_same_frame: bool = False) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    selected = [entry.candidate for entry in chain.events if entry.candidate is not None]
    if any(item.video_id != chain.video_id for item in selected):
        failures.append("CROSS_VIDEO_CHAIN")
    times = [item.pts_time for item in selected]
    if allow_same_frame:
        if any(left > right for left, right in zip(times, times[1:])):
            failures.append("TEMPORAL_ORDER_INVALID")
    elif any(left >= right for left, right in zip(times, times[1:])):
        failures.append("TEMPORAL_ORDER_INVALID")
    selected_by_event = {entry.event_id: entry.candidate for entry in chain.events}
    for event in request.events:
        if event.required and selected_by_event.get(event.event_id) is None:
            failures.append(f"REQUIRED_EVENT_MISSING:{event.event_id}")
    return not failures, tuple(failures)


class TemporalAligner:
    def __init__(self, score_config: ScoreConfig | None = None) -> None:
        self.score_config = score_config or ScoreConfig()

    def align_video(
        self,
        request: TrakeRequest,
        video: VideoCandidate,
        *,
        top_k: int = 5,
        beam_size: int = 100,
    ) -> AlignmentResult:
        top_k = max(1, min(top_k, 5))
        beam_size = max(top_k, beam_size)
        allow_same_frame = bool(request.constraints.get("allow_same_frame", False))
        gap_mode = str(request.constraints.get("gap_mode", "preferred"))
        if gap_mode not in {"hard", "preferred"}:
            raise ValueError("gap_mode must be hard or preferred")
        states = [_State((), 0, ())]
        failures: set[str] = set(video.warnings)

        for event in request.events:
            candidates = sorted(
                video.event_candidates.get(event.event_id, ()),
                key=lambda item: (-item.local_score, item.pts_time, item.keyframe_id, item.candidate_id),
            )
            choices: list[TrakeCandidate | None] = list(candidates)
            if not event.required:
                choices.append(None)
            if event.required and not choices:
                return AlignmentResult(video.video_id, (), tuple(sorted({*failures, f"REQUIRED_EVENT_MISSING:{event.event_id}"})))
            next_states: list[_State] = []
            for state in states:
                previous = next((item for item in reversed(state.selected) if item is not None), None)
                for candidate in choices:
                    if candidate is not None and candidate.video_id != video.video_id:
                        failures.add("CROSS_VIDEO_CANDIDATE")
                        continue
                    warnings = list(state.warnings)
                    gap_penalties = state.gap_penalties
                    if previous is not None and candidate is not None:
                        if candidate.pts_time < previous.pts_time or (candidate.pts_time == previous.pts_time and not allow_same_frame):
                            failures.add("TEMPORAL_ORDER_INVALID")
                            continue
                        gap = candidate.pts_time - previous.pts_time
                        minimum = event.min_gap_seconds if event.min_gap_seconds is not None else request.constraints.get("min_gap_seconds")
                        maximum = event.max_gap_seconds if event.max_gap_seconds is not None else request.constraints.get("max_gap_seconds")
                        violated = (minimum is not None and gap < float(minimum)) or (maximum is not None and gap > float(maximum))
                        if violated and gap_mode == "hard":
                            failures.add("HARD_GAP_CONSTRAINT_VIOLATED")
                            continue
                        if violated:
                            gap_penalties += 1
                            warnings.append("PREFERRED_GAP_CONSTRAINT_VIOLATED")
                    if candidate is not None:
                        fps = candidate.evidence.get("fps") or candidate.evidence.get("raw_result", {}).get("fps")
                        if fps and abs((candidate.frame_idx / float(fps)) - candidate.pts_time) > float(request.constraints.get("frame_time_tolerance", 1.0)):
                            warnings.append("FRAME_TIME_MAPPING_MISMATCH")
                    next_states.append(_State((*state.selected, candidate), gap_penalties, tuple(dict.fromkeys(warnings))))
            if not next_states:
                return AlignmentResult(video.video_id, (), tuple(sorted(failures or {"NO_VALID_TEMPORAL_CHAIN"})))
            next_states.sort(key=lambda state: self._state_key(state, request))
            states = next_states[:beam_size]

        chains: list[TrakeChain] = []
        required = tuple(event.required for event in request.events)
        for index, state in enumerate(states, start=1):
            breakdown = score_chain(state.selected, required, state.gap_penalties, self.score_config)
            events = tuple(
                ChainEvent(event.event_id, event.required, candidate)
                for event, candidate in zip(request.events, state.selected)
            )
            chain = TrakeChain(f"{video.video_id}:chain:{index}", video.video_id, events, breakdown, state.warnings)
            valid, chain_failures = validate_chain(request, chain, allow_same_frame=allow_same_frame)
            if valid:
                chains.append(chain)
            else:
                failures.update(chain_failures)
        chains.sort(key=lambda chain: (-chain.score.final_score, self._chain_signature(chain)))
        return AlignmentResult(video.video_id, tuple(chains[:top_k]), tuple(sorted(failures)))

    def align_videos(
        self,
        request: TrakeRequest,
        videos: tuple[VideoCandidate, ...],
        *,
        top_k_per_video: int = 5,
        top_videos: int = 20,
    ) -> tuple[AlignmentResult, ...]:
        results = [self.align_video(request, video, top_k=top_k_per_video) for video in videos]
        results.sort(
            key=lambda result: (
                not bool(result.chains),
                -(result.chains[0].score.final_score if result.chains else float("-inf")),
                result.video_id,
            )
        )
        return tuple(results[:top_videos])

    def _state_key(self, state: _State, request: TrakeRequest) -> tuple[Any, ...]:
        required = tuple(event.required for event in request.events[: len(state.selected)])
        score = score_chain(state.selected, required, state.gap_penalties, self.score_config)
        signature = tuple(item.candidate_id if item else "~skip" for item in state.selected)
        return (-score.final_score, signature)

    @staticmethod
    def _chain_signature(chain: TrakeChain) -> tuple[str, ...]:
        return tuple(entry.candidate.candidate_id if entry.candidate else "~skip" for entry in chain.events)
