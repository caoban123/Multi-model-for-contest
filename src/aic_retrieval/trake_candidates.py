"""Typed event candidate pools and same-video grouping for TRAKE."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Iterable

from aic_retrieval.trake_schema import Availability, TrakeEvent, TrakeRequest


CANDIDATE_CONTRACT_VERSION = "phase8-candidate-v1"


@dataclass(frozen=True)
class TrakeCandidate:
    event_id: str
    candidate_id: str
    video_id: str
    keyframe_id: int
    frame_idx: int
    pts_time: float
    keyframe_path: str | None
    local_score: float
    local_rank: int
    evidence: dict[str, Any]
    provenance: tuple[str, ...]
    retrieval_method: str
    warnings: tuple[str, ...] = ()
    raw_scores: dict[str, float | None] = field(default_factory=dict)
    raw_ranks: dict[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_id", "candidate_id", "video_id", "retrieval_method"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.keyframe_id < 0 or self.frame_idx < 0 or self.pts_time < 0:
            raise ValueError("candidate frame/time fields must be non-negative")
        if self.local_rank < 1 or not 0.0 <= self.local_score <= 1.0:
            raise ValueError("candidate rank/normalized score is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventCandidatePool:
    event: TrakeEvent
    candidates: tuple[TrakeCandidate, ...]
    modality_availability: dict[str, Availability]
    warnings: tuple[str, ...] = ()
    topk_used: int | None = None
    distinctiveness: float | None = None
    distinctiveness_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoCandidate:
    video_id: str
    event_candidates: dict[str, tuple[TrakeCandidate, ...]]
    required_coverage: int
    required_total: int
    optional_coverage: int
    optional_total: int
    complete: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rank_normalized_score(rank: int, pool_size: int) -> float:
    """Map rank to [0, 1] without combining heterogeneous raw scores."""
    if rank < 1 or pool_size < 1:
        raise ValueError("rank and pool_size must be positive")
    if pool_size == 1:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((rank - 1) / (pool_size - 1))))


def with_distinctiveness(pool: EventCandidatePool, settings: Any) -> EventCandidatePool:
    """Attach an inspectable, configurable event distinctiveness score."""
    candidates = pool.candidates
    margin = 0.0
    if candidates:
        first = float(candidates[0].raw_scores.get("score") or candidates[0].local_score)
        second = float(candidates[1].raw_scores.get("score") or candidates[1].local_score) if len(candidates) > 1 else 0.0
        margin = max(0.0, min(1.0, (first - second) / max(abs(first), 1e-9)))
    counts = Counter(candidate.video_id for candidate in candidates)
    concentration = max(counts.values(), default=0) / max(1, len(candidates))
    event = pool.event
    structured_count = (
        len(event.object_constraints)
        + len(event.attribute_constraints)
        + len(event.ocr_terms)
        + len(event.asr_terms)
        + max(0, len(event.modalities) - 1)
    )
    constraint = min(1.0, structured_count / 4.0)
    lexical = min(1.0, len((event.visual_query or event.clip_query).split()) / 10.0)
    weights = {
        "margin": float(settings.margin_weight),
        "video_concentration": float(settings.video_concentration_weight),
        "constraint": float(settings.constraint_weight),
        "lexical": float(settings.lexical_weight),
    }
    components = {
        "margin": margin,
        "video_concentration": concentration,
        "constraint": constraint,
        "lexical": lexical,
    }
    total_weight = sum(weights.values())
    score = sum(components[name] * weights[name] for name in components) / total_weight
    return replace(pool, distinctiveness=round(score, 6), distinctiveness_components=components)


def select_anchor_event(
    request: TrakeRequest,
    pools: Iterable[EventCandidatePool],
) -> tuple[str | None, dict[str, Any]]:
    """Select one evidence-driven required anchor; never a fixed event index."""
    by_event = {pool.event.event_id: pool for pool in pools}
    eligible = [event for event in request.events if event.required and event.event_id in by_event]
    if not eligible:
        return None, {"reason": "NO_REQUIRED_EVENT_POOL", "scores": {}}
    scores = {event.event_id: float(by_event[event.event_id].distinctiveness or 0.0) for event in eligible}
    anchor = min(
        eligible,
        key=lambda event: (
            -scores[event.event_id],
            -max((candidate.local_score for candidate in by_event[event.event_id].candidates), default=0.0),
            event.order,
            event.event_id,
        ),
    )
    return anchor.event_id, {"reason": "MAX_DISTINCTIVENESS_REQUIRED_EVENT", "scores": scores}


class TrakeRetrievalAdapter:
    """Adapt existing retrieval results into deterministic per-event pools.

    ``retriever`` receives ``(event, pool_size)`` and returns existing
    retrieval dictionaries. The adapter never assumes raw CLIP/BM25/object
    scores share a scale; ``local_score`` is derived only from local rank.
    """

    def __init__(self, retriever: Callable[[TrakeEvent, int], Iterable[dict[str, Any]]]) -> None:
        self.retriever = retriever

    def retrieve_event(
        self,
        event: TrakeEvent,
        *,
        pool_size: int = 60,
        max_per_video: int = 8,
        availability: dict[str, Availability] | None = None,
        raw_results: Iterable[dict[str, Any]] | None = None,
    ) -> EventCandidatePool:
        if not 1 <= pool_size <= 1000:
            raise ValueError("event pool_size must be between 1 and 1000")
        if not 1 <= max_per_video <= 64:
            raise ValueError("max_per_video must be between 1 and 64")
        raw = list(self.retriever(event, pool_size) if raw_results is None else raw_results)[:pool_size]
        counts: dict[str, int] = {}
        candidates: list[TrakeCandidate] = []
        warnings: list[str] = []
        statuses = {modality: Availability.UNKNOWN for modality in event.modalities}
        statuses.update(availability or {})
        for modality in event.modalities:
            if modality == "clip" and modality not in (availability or {}):
                statuses[modality] = Availability.AVAILABLE
            if statuses[modality] is Availability.UNAVAILABLE:
                warnings.append(f"{modality.upper()}_UNAVAILABLE")

        for raw_rank, item in enumerate(raw, start=1):
            video_id = str(item.get("video_id") or "").strip()
            keyframe_id = item.get("keyframe_id")
            frame_idx = item.get("frame_idx")
            pts_time = item.get("pts_time")
            if not video_id or keyframe_id is None or frame_idx is None or pts_time is None:
                warnings.append("INCOMPLETE_RETRIEVAL_RECORD")
                continue
            if counts.get(video_id, 0) >= max_per_video:
                continue
            counts[video_id] = counts.get(video_id, 0) + 1
            local_rank = len(candidates) + 1
            raw_scores = {
                key: item.get(key)
                for key in ("score", "clip_score", "object_score", "attribute_score", "ocr_score", "asr_score", "metadata_score")
                if key in item
            }
            raw_scores.update({str(key): value for key, value in dict(item.get("raw_modality_scores") or {}).items()})
            raw_ranks = {str(key): int(value) if value is not None else None for key, value in dict(item.get("raw_modality_ranks") or {}).items()}
            provenance = tuple(item.get("provenance") or ("clip",))
            evidence = dict(item.get("evidence") or {})
            candidate_warnings = tuple(item.get("warnings") or ())
            candidates.append(
                TrakeCandidate(
                    event_id=event.event_id,
                    candidate_id=f"{event.event_id}:{video_id}:{int(keyframe_id)}",
                    video_id=video_id,
                    keyframe_id=int(keyframe_id),
                    frame_idx=int(frame_idx),
                    pts_time=float(pts_time),
                    keyframe_path=item.get("keyframe_path"),
                    local_score=rank_normalized_score(raw_rank, max(1, len(raw))),
                    local_rank=local_rank,
                    evidence=evidence or {"raw_result": item},
                    provenance=provenance,
                    retrieval_method=str(item.get("retrieval_method") or "existing_rank_adapter"),
                    warnings=candidate_warnings,
                    raw_scores=raw_scores,
                    raw_ranks=raw_ranks,
                )
            )
        return EventCandidatePool(event, tuple(candidates), statuses, tuple(dict.fromkeys(warnings)), topk_used=pool_size)

    def retrieve_request(
        self,
        request: TrakeRequest,
        *,
        pool_size: int = 60,
        max_per_video: int = 8,
        availability: dict[str, Availability] | None = None,
        pool_sizes: dict[str, int] | None = None,
        raw_by_event: dict[str, Iterable[dict[str, Any]]] | None = None,
    ) -> tuple[EventCandidatePool, ...]:
        return tuple(
            self.retrieve_event(
                event,
                pool_size=(pool_sizes or {}).get(event.event_id, pool_size),
                max_per_video=max_per_video,
                availability=availability,
                raw_results=(raw_by_event or {}).get(event.event_id),
            )
            for event in request.events
        )


def group_candidates_by_video(
    request: TrakeRequest,
    pools: Iterable[EventCandidatePool],
    *,
    video_pool_size: int = 20,
    anchor_event_id: str | None = None,
) -> tuple[VideoCandidate, ...]:
    if not 1 <= video_pool_size <= 100:
        raise ValueError("video_pool_size must be between 1 and 100")
    by_event = {pool.event.event_id: pool for pool in pools}
    video_ids = sorted({item.video_id for pool in by_event.values() for item in pool.candidates})
    required = [event for event in request.events if event.required]
    optional = [event for event in request.events if not event.required]
    grouped: list[VideoCandidate] = []
    for video_id in video_ids:
        event_candidates = {
            event.event_id: tuple(item for item in by_event.get(event.event_id, EventCandidatePool(event, (), {})).candidates if item.video_id == video_id)
            for event in request.events
        }
        required_coverage = sum(bool(event_candidates[event.event_id]) for event in required)
        optional_coverage = sum(bool(event_candidates[event.event_id]) for event in optional)
        complete = required_coverage == len(required)
        warnings = () if complete else ("REQUIRED_EVENT_MISSING",)
        grouped.append(
            VideoCandidate(
                video_id=video_id,
                event_candidates=event_candidates,
                required_coverage=required_coverage,
                required_total=len(required),
                optional_coverage=optional_coverage,
                optional_total=len(optional),
                complete=complete,
                warnings=warnings,
            )
        )
    grouped.sort(
        key=lambda item: (
            not item.complete,
            -item.required_coverage,
            -item.optional_coverage,
            -max((candidate.local_score for candidate in item.event_candidates.get(anchor_event_id or "", ())), default=0.0),
            -sum(candidate.local_score for values in item.event_candidates.values() for candidate in values[:1]),
            item.video_id,
        )
    )
    return tuple(grouped[:video_pool_size])
