"""Same-video temporal DP/beam alignment for Phase 8 TRAKE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aic_retrieval.trake_candidates import TrakeCandidate, VideoCandidate
from aic_retrieval.trake_refinement import DenseWindowExpansion
from aic_retrieval.trake_schema import TrakeRequest
from aic_retrieval.trake_scoring import ScoreBreakdown, ScoreConfig, multimodal_event_score, score_chain


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
    score_components: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    video_id: str
    chains: tuple[TrakeChain, ...]
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalFeasibilityResult:
    video_id: str
    feasible: bool
    witness_candidate_ids: tuple[str, ...] = ()
    event_intervals: dict[str, tuple[float, float]] | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    version: str = "k-pointer-inspired-feasibility-v1"


@dataclass(frozen=True)
class CandidateWindow:
    window_id: str
    video_id: str
    start_pts: float
    end_pts: float
    anchor_event_id: str | None
    event_candidate_ids: dict[str, str]
    score: float
    source: str = "anchor-monotonic-window-v1"
    warnings: tuple[str, ...] = ()


def fast_temporal_feasibility(request: TrakeRequest, video: VideoCandidate) -> TemporalFeasibilityResult:
    """Cheap monotonic feasibility check over required event candidate timestamps."""
    required_events = [event for event in request.events if event.required]
    intervals = {
        event.event_id: (
            min((candidate.pts_time for candidate in video.event_candidates.get(event.event_id, ())), default=0.0),
            max((candidate.pts_time for candidate in video.event_candidates.get(event.event_id, ())), default=0.0),
        )
        for event in required_events
        if video.event_candidates.get(event.event_id)
    }
    for event in required_events:
        if not video.event_candidates.get(event.event_id):
            return TemporalFeasibilityResult(
                video.video_id,
                False,
                event_intervals=intervals,
                failure_stage="EVENT_RECALL_FAILURE",
                failure_reason=f"REQUIRED_EVENT_MISSING:{event.event_id}",
            )

    allow_same = bool(request.constraints.get("allow_same_frame", False))
    hard_gap = str(request.constraints.get("gap_mode", "preferred")) == "hard"
    states: list[tuple[TrakeCandidate, ...]] = [()]
    for event in required_events:
        next_states: dict[str, tuple[TrakeCandidate, ...]] = {}
        candidates = sorted(video.event_candidates[event.event_id], key=lambda item: (item.pts_time, -item.local_score, item.candidate_id))
        for candidate in candidates:
            best_path: tuple[TrakeCandidate, ...] | None = None
            for path in states:
                previous = path[-1] if path else None
                if previous is not None:
                    ordered = candidate.pts_time >= previous.pts_time if allow_same else candidate.pts_time > previous.pts_time
                    if not ordered:
                        continue
                    if hard_gap and not _gap_compatible(request, event, previous.pts_time, candidate.pts_time):
                        continue
                proposal = (*path, candidate)
                if best_path is None or _path_key(proposal) < _path_key(best_path):
                    best_path = proposal
            if best_path is not None:
                next_states[candidate.candidate_id] = best_path
        states = list(next_states.values())
        if not states:
            return TemporalFeasibilityResult(
                video.video_id,
                False,
                event_intervals=intervals,
                failure_stage="TEMPORAL_FEASIBILITY_FAILURE",
                failure_reason="NO_MONOTONIC_REQUIRED_PATH",
            )
    witness = min(states, key=_path_key) if states else ()
    return TemporalFeasibilityResult(
        video.video_id,
        True,
        tuple(candidate.candidate_id for candidate in witness),
        intervals,
    )


def build_candidate_windows(
    request: TrakeRequest,
    videos: tuple[VideoCandidate, ...],
    feasibility: tuple[TemporalFeasibilityResult, ...],
    *,
    anchor_event_id: str | None,
    padding_seconds: float,
    max_windows_per_video: int,
    max_total_windows: int,
) -> tuple[CandidateWindow, ...]:
    """Build bounded anchor-aware windows without decoding video frames."""
    feasible_ids = {item.video_id for item in feasibility if item.feasible}
    windows: list[CandidateWindow] = []
    required_events = [event for event in request.events if event.required]
    for video in videos:
        if video.video_id not in feasible_ids:
            continue
        anchor_candidates = list(video.event_candidates.get(anchor_event_id or "", ()))
        if not anchor_candidates:
            witness = next((item for item in feasibility if item.video_id == video.video_id), None)
            candidate_by_id = {candidate.candidate_id: candidate for values in video.event_candidates.values() for candidate in values}
            anchor_candidates = [candidate_by_id[witness.witness_candidate_ids[0]]] if witness and witness.witness_candidate_ids else []
        candidates_for_windows = sorted(anchor_candidates, key=lambda item: (-item.local_score, item.pts_time, item.candidate_id))[:max_windows_per_video]
        per_video: list[CandidateWindow] = []
        for anchor in candidates_for_windows:
            selected = _path_around_anchor(request, video, required_events, anchor_event_id, anchor)
            if selected is None:
                continue
            times = [candidate.pts_time for candidate in selected.values()]
            signature = "|".join(selected[event.event_id].candidate_id for event in required_events)
            window = CandidateWindow(
                window_id=f"{video.video_id}:window:{len(per_video) + 1}",
                video_id=video.video_id,
                start_pts=max(0.0, min(times) - padding_seconds),
                end_pts=max(times) + padding_seconds,
                anchor_event_id=anchor_event_id,
                event_candidate_ids={event_id: candidate.candidate_id for event_id, candidate in selected.items()},
                score=sum(candidate.local_score for candidate in selected.values()) / max(1, len(selected)),
                warnings=() if signature else ("WINDOW_EMPTY",),
            )
            if not any(existing.event_candidate_ids == window.event_candidate_ids for existing in per_video):
                per_video.append(window)
        windows.extend(sorted(per_video, key=lambda item: (-item.score, item.start_pts, item.window_id))[:max_windows_per_video])
    windows.sort(key=lambda item: (-item.score, item.video_id, item.start_pts, item.window_id))
    return tuple(windows[:max_total_windows])


def _path_around_anchor(
    request: TrakeRequest,
    video: VideoCandidate,
    events: list[Any],
    anchor_event_id: str | None,
    anchor: TrakeCandidate,
) -> dict[str, TrakeCandidate] | None:
    anchor_index = next((index for index, event in enumerate(events) if event.event_id == anchor_event_id), 0)
    selected: dict[str, TrakeCandidate] = {events[anchor_index].event_id: anchor}
    next_candidate = anchor
    next_event_schema = events[anchor_index]
    for event in reversed(events[:anchor_index]):
        compatible = [candidate for candidate in video.event_candidates.get(event.event_id, ()) if _ordered(candidate.pts_time, next_candidate.pts_time, request) and _gap_compatible(request, next_event_schema, candidate.pts_time, next_candidate.pts_time)]
        if not compatible:
            return None
        chosen = min(compatible, key=lambda item: (-item.local_score, -item.pts_time, item.candidate_id))
        selected[event.event_id] = chosen
        next_candidate = chosen
        next_event_schema = event
    previous = anchor
    for event in events[anchor_index + 1:]:
        compatible = [candidate for candidate in video.event_candidates.get(event.event_id, ()) if _ordered(previous.pts_time, candidate.pts_time, request) and _gap_compatible(request, event, previous.pts_time, candidate.pts_time)]
        if not compatible:
            return None
        chosen = min(compatible, key=lambda item: (-item.local_score, item.pts_time, item.candidate_id))
        selected[event.event_id] = chosen
        previous = chosen
    return selected


def _ordered(left: float, right: float, request: TrakeRequest) -> bool:
    return left <= right if bool(request.constraints.get("allow_same_frame", False)) else left < right


def _gap_compatible(request: TrakeRequest, event: Any, left: float, right: float) -> bool:
    if str(request.constraints.get("gap_mode", "preferred")) != "hard":
        return True
    gap = right - left
    minimum = event.min_gap_seconds if event.min_gap_seconds is not None else request.constraints.get("min_gap_seconds")
    maximum = event.max_gap_seconds if event.max_gap_seconds is not None else request.constraints.get("max_gap_seconds")
    return not ((minimum is not None and gap < float(minimum)) or (maximum is not None and gap > float(maximum)))


def _path_key(path: tuple[TrakeCandidate, ...]) -> tuple[Any, ...]:
    return (-sum(candidate.local_score for candidate in path), tuple(candidate.candidate_id for candidate in path))


@dataclass
class _State:
    selected: tuple[TrakeCandidate | None, ...]
    gap_penalties: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _DanteState:
    selected: tuple[TrakeCandidate | None, ...]
    objective: float
    semantic_sum: float
    transition_penalty: float
    optional_skip_penalty: float
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
    alignment_version = ALIGNMENT_VERSION

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
            chain = TrakeChain(
                f"{video.video_id}:chain:{index}",
                video.video_id,
                events,
                breakdown,
                state.warnings,
                version=self.alignment_version,
            )
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
        beam_size: int = 100,
    ) -> tuple[AlignmentResult, ...]:
        results = [self.align_video(request, video, top_k=top_k_per_video, beam_size=beam_size) for video in videos]
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


class TrakeV2FoundationAligner(TemporalAligner):
    """V2 selection boundary while later coarse-to-fine stages are added.

    P8.V2.0 deliberately delegates the verified legacy temporal search.  It is
    a distinct runtime type/version so A/B selection is real and inspectable;
    later V2 milestones extend this class without replacing the legacy path.
    """

    alignment_version = "trake-v2-foundation"


class DanteInspiredAligner(TemporalAligner):
    """Event-frame dynamic program inspired by DANTE, not an exact reproduction.

    State expansion is bounded by ``beam_size`` and therefore scales as
    O(events * candidates * beam), rather than forming an unbounded T x T
    transition matrix. Every state retains an inspectable backpointer path.
    """

    alignment_version = "dante-inspired-coarse-fine-v1"

    def __init__(
        self,
        score_config: ScoreConfig | None = None,
        dante_config: Any | None = None,
        modality_availability: dict[str, Any] | None = None,
        reranking_config: Any | None = None,
    ) -> None:
        super().__init__(score_config)
        self.dante_config = dante_config
        self.modality_availability = modality_availability or {}
        self.reranking_config = reranking_config

    def align_video(
        self,
        request: TrakeRequest,
        video: VideoCandidate,
        *,
        top_k: int = 5,
        beam_size: int = 100,
    ) -> AlignmentResult:
        top_k = max(1, min(top_k, 20))
        beam_size = max(top_k, beam_size)
        semantic_weight = float(getattr(self.dante_config, "semantic_weight", 1.0))
        transition_weight = float(getattr(self.dante_config, "transition_gap_penalty", 0.25))
        skip_weight = float(getattr(self.dante_config, "optional_skip_penalty", 0.15))
        allow_same = bool(request.constraints.get("allow_same_frame", False))
        gap_mode = str(request.constraints.get("gap_mode", "preferred"))
        if gap_mode not in {"hard", "preferred"}:
            raise ValueError("gap_mode must be hard or preferred")
        states = [_DanteState((), 0.0, 0.0, 0.0, 0.0, 0, ())]
        failures: set[str] = set(video.warnings)

        for event in request.events:
            candidates = sorted(
                video.event_candidates.get(event.event_id, ()),
                key=lambda item: (item.pts_time, -item.local_score, item.candidate_id),
            )
            if event.required and not candidates:
                return AlignmentResult(video.video_id, (), tuple(sorted({*failures, f"REQUIRED_EVENT_MISSING:{event.event_id}"})))
            next_states: dict[tuple[str, ...], _DanteState] = {}
            for state in states:
                if not event.required:
                    skipped = _DanteState(
                        (*state.selected, None),
                        state.objective - skip_weight,
                        state.semantic_sum,
                        state.transition_penalty,
                        state.optional_skip_penalty + skip_weight,
                        state.gap_penalties,
                        state.warnings,
                    )
                    _keep_dante_state(next_states, skipped)
                previous = next((item for item in reversed(state.selected) if item is not None), None)
                for candidate in candidates:
                    if candidate.video_id != video.video_id:
                        failures.add("CROSS_VIDEO_CANDIDATE")
                        continue
                    penalty = 0.0
                    gap_penalties = state.gap_penalties
                    warnings = list(state.warnings)
                    if previous is not None:
                        ordered = candidate.pts_time >= previous.pts_time if allow_same else candidate.pts_time > previous.pts_time
                        if not ordered:
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
                            penalty += transition_weight
                            gap_penalties += 1
                            warnings.append("PREFERRED_GAP_CONSTRAINT_VIOLATED")
                        # Smooth continuity cost; explicit hard gaps above remain authoritative.
                        penalty += transition_weight * (gap / (gap + 60.0))
                    event_score = multimodal_event_score(
                        event,
                        candidate,
                        self.modality_availability,
                        getattr(self.reranking_config, "multimodal_weights", None),
                    )
                    semantic = semantic_weight * event_score.score
                    proposal = _DanteState(
                        (*state.selected, candidate),
                        state.objective + semantic - penalty,
                        state.semantic_sum + semantic,
                        state.transition_penalty + penalty,
                        state.optional_skip_penalty,
                        gap_penalties,
                        tuple(dict.fromkeys(warnings)),
                    )
                    _keep_dante_state(next_states, proposal)
            if not next_states:
                return AlignmentResult(video.video_id, (), tuple(sorted(failures or {"NO_VALID_TEMPORAL_CHAIN"})))
            states = sorted(next_states.values(), key=_dante_state_key)[:beam_size]

        chains: list[TrakeChain] = []
        required = tuple(event.required for event in request.events)
        for index, state in enumerate(sorted(states, key=_dante_state_key)[:top_k], start=1):
            breakdown = score_chain(state.selected, required, state.gap_penalties, self.score_config)
            multimodal_scores = [
                multimodal_event_score(
                    event,
                    candidate,
                    self.modality_availability,
                    getattr(self.reranking_config, "multimodal_weights", None),
                ).to_dict()
                for event, candidate in zip(request.events, state.selected)
                if candidate is not None
            ]
            chain = TrakeChain(
                f"{video.video_id}:chain:{index}",
                video.video_id,
                tuple(ChainEvent(event.event_id, event.required, candidate) for event, candidate in zip(request.events, state.selected)),
                breakdown,
                state.warnings,
                version=self.alignment_version,
                score_components={
                    "dante_objective": state.objective,
                    "event_semantic_score": state.semantic_sum,
                    "transition_penalty": state.transition_penalty,
                    "optional_skip_penalty": state.optional_skip_penalty,
                    "event_multimodal_scores": multimodal_scores,
                    "multimodal_scoring_method": "event-aware-reciprocal-rank-v1",
                },
            )
            valid, chain_failures = validate_chain(request, chain, allow_same_frame=allow_same)
            if valid:
                chains.append(chain)
            else:
                failures.update(chain_failures)
        return AlignmentResult(video.video_id, tuple(chains), tuple(sorted(failures)))

    def align_videos(
        self,
        request: TrakeRequest,
        videos: tuple[VideoCandidate, ...],
        *,
        top_k_per_video: int = 5,
        top_videos: int = 20,
        beam_size: int = 100,
    ) -> tuple[AlignmentResult, ...]:
        results = [self.align_video(request, video, top_k=top_k_per_video, beam_size=beam_size) for video in videos]
        results.sort(key=_dante_result_key)
        return tuple(results[:top_videos])

    def align_window(
        self,
        request: TrakeRequest,
        video: VideoCandidate,
        window: CandidateWindow,
        expansion: DenseWindowExpansion | None,
        *,
        top_k: int = 5,
        beam_size: int = 100,
    ) -> AlignmentResult:
        """Align one bounded window, using dense candidates when available."""
        source_by_id = {
            candidate.candidate_id: candidate
            for values in video.event_candidates.values()
            for candidate in values
        }
        event_candidates: dict[str, tuple[TrakeCandidate, ...]] = {}
        for event in request.events:
            source_values = tuple(
                candidate
                for candidate in video.event_candidates.get(event.event_id, ())
                if window.start_pts <= candidate.pts_time <= window.end_pts
            )
            dense_values: tuple[TrakeCandidate, ...] = ()
            if expansion is not None and expansion.status == "AVAILABLE":
                dense_values = tuple(
                    _candidate_from_dense(item, source_by_id)
                    for item in expansion.event_frames.get(event.event_id, ())
                    if item.source_candidate_id in source_by_id
                )
            event_candidates[event.event_id] = dense_values or source_values
        bounded_video = VideoCandidate(
            video.video_id,
            event_candidates,
            sum(bool(event_candidates.get(event.event_id)) for event in request.events if event.required),
            sum(event.required for event in request.events),
            sum(bool(event_candidates.get(event.event_id)) for event in request.events if not event.required),
            sum(not event.required for event in request.events),
            all(bool(event_candidates.get(event.event_id)) for event in request.events if event.required),
            video.warnings,
        )
        result = self.align_video(request, bounded_video, top_k=top_k, beam_size=beam_size)
        chains = tuple(
            TrakeChain(
                f"{window.window_id}:chain:{index}",
                chain.video_id,
                chain.events,
                chain.score,
                chain.warnings,
                chain.valid,
                chain.manual,
                chain.version,
                {**(chain.score_components or {}), "window_id": window.window_id, "window_score": window.score},
            )
            for index, chain in enumerate(result.chains, start=1)
        )
        return AlignmentResult(result.video_id, chains, result.failures)


def _keep_dante_state(target: dict[tuple[str, ...], _DanteState], state: _DanteState) -> None:
    signature = tuple(item.candidate_id if item is not None else "~skip" for item in state.selected)
    current = target.get(signature)
    if current is None or _dante_state_key(state) < _dante_state_key(current):
        target[signature] = state


def _dante_state_key(state: _DanteState) -> tuple[Any, ...]:
    signature = tuple(item.candidate_id if item is not None else "~skip" for item in state.selected)
    return (-state.objective, signature)


def _dante_result_key(result: AlignmentResult) -> tuple[Any, ...]:
    objective = float(result.chains[0].score_components.get("dante_objective", float("-inf"))) if result.chains and result.chains[0].score_components else float("-inf")
    return (not bool(result.chains), -objective, result.video_id)


def _candidate_from_dense(item: Any, source_by_id: dict[str, TrakeCandidate]) -> TrakeCandidate:
    source = source_by_id[item.source_candidate_id]
    frame_index_is_estimate = item.frame_idx is None
    frame_idx = int(item.frame_idx) if item.frame_idx is not None else round(float(item.decoded_pts) * float(item.fps))
    normalized = max(0.0, min(1.0, (float(item.semantic_score) + 1.0) / 2.0))
    return TrakeCandidate(
        item.event_id,
        f"{item.event_id}:{source.video_id}:dense:{item.stage}:{item.decoded_pts:.6f}",
        source.video_id,
        source.keyframe_id,
        frame_idx,
        float(item.decoded_pts),
        item.image_path,
        normalized,
        item.rank,
        {
            **source.evidence,
            "dense_refinement": True,
            "dense_stage": item.stage,
            "source_candidate_id": source.candidate_id,
            "source_keyframe_id": source.keyframe_id,
            "requested_pts": item.requested_pts,
            "decoded_pts": item.decoded_pts,
            "fps": item.fps,
            "frame_index_if_known": item.frame_idx,
            "frame_index_is_estimate": frame_index_is_estimate,
            "seek_error_estimate": item.seek_error_estimate,
            "dense_frame_asset_path": item.image_path,
            "visual_embedding": list(item.visual_signature),
        },
        tuple(dict.fromkeys((*source.provenance, "dense_clip"))),
        "dense_window_clip",
        source.warnings,
        {**source.raw_scores, "dense_clip": float(item.semantic_score)},
        {**source.raw_ranks, "dense_clip": item.rank},
    )
