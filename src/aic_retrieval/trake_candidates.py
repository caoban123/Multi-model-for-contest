"""Typed event candidate pools and same-video grouping for TRAKE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    ) -> EventCandidatePool:
        if not 30 <= pool_size <= 100:
            raise ValueError("event pool_size must be between 30 and 100")
        if not 5 <= max_per_video <= 10:
            raise ValueError("max_per_video must be between 5 and 10")
        raw = list(self.retriever(event, pool_size))[:pool_size]
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
                )
            )
        return EventCandidatePool(event, tuple(candidates), statuses, tuple(dict.fromkeys(warnings)))

    def retrieve_request(
        self,
        request: TrakeRequest,
        *,
        pool_size: int = 60,
        max_per_video: int = 8,
        availability: dict[str, Availability] | None = None,
    ) -> tuple[EventCandidatePool, ...]:
        return tuple(
            self.retrieve_event(event, pool_size=pool_size, max_per_video=max_per_video, availability=availability)
            for event in request.events
        )


def group_candidates_by_video(
    request: TrakeRequest,
    pools: Iterable[EventCandidatePool],
    *,
    video_pool_size: int = 20,
) -> tuple[VideoCandidate, ...]:
    if not 10 <= video_pool_size <= 30:
        raise ValueError("video_pool_size must be between 10 and 30")
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
            -sum(candidate.local_score for values in item.event_candidates.values() for candidate in values[:1]),
            item.video_id,
        )
    )
    return tuple(grouped[:video_pool_size])
