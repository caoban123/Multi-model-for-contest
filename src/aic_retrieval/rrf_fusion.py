from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from aic_retrieval.retrievers import Retriever, RetrievalHit, RetrievalRequest, RetrieverHealth


RRF_FUSION_VERSION = "generic-video-rrf-v1"


@dataclass(frozen=True)
class RrfFusionConfig:
    rrf_k: int = 60
    candidate_pool: int = 200
    weights: Mapping[str, float] = field(default_factory=dict)
    strict: bool = False

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not 1 <= self.candidate_pool <= 1000:
            raise ValueError("candidate_pool must be between 1 and 1000")
        if any(float(weight) < 0 for weight in self.weights.values()):
            raise ValueError("RRF weights must be non-negative")


def _video_hits(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    ordered = sorted(hits, key=lambda hit: (hit.rank, hit.video_id, hit.document_id or ""))
    seen: set[str] = set()
    unique: list[RetrievalHit] = []
    for hit in ordered:
        if hit.video_id in seen:
            continue
        seen.add(hit.video_id)
        unique.append(hit)
    return unique


def fuse_video_hits(
    channels: Mapping[str, Sequence[RetrievalHit]],
    config: RrfFusionConfig = RrfFusionConfig(),
    *,
    top_k: int = 100,
    failures: Mapping[str, str] | None = None,
) -> list[RetrievalHit]:
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k must be between 1 and 1000")
    accumulators: dict[str, dict] = {}
    for retriever_name in sorted(channels):
        weight = float(config.weights.get(retriever_name, 1.0))
        if weight == 0:
            continue
        for video_rank, hit in enumerate(_video_hits(channels[retriever_name]), 1):
            contribution = weight / (config.rrf_k + video_rank)
            item = accumulators.setdefault(
                hit.video_id,
                {
                    "score": 0.0,
                    "best_rank": 10**9,
                    "ranks": {},
                    "raw_scores": {},
                    "contributions": {},
                    "evidence": {},
                    "representative": None,
                    "representative_key": None,
                },
            )
            item["score"] += contribution
            item["best_rank"] = min(item["best_rank"], video_rank)
            item["ranks"][retriever_name] = video_rank
            item["raw_scores"][retriever_name] = hit.raw_score
            item["contributions"][retriever_name] = contribution
            item["evidence"][retriever_name] = hit.to_dict()
            representative_key = (-contribution, video_rank, retriever_name)
            if item["representative_key"] is None or representative_key < item["representative_key"]:
                item["representative_key"] = representative_key
                item["representative"] = hit

    ordered = sorted(accumulators.items(), key=lambda pair: (-pair[1]["score"], pair[1]["best_rank"], pair[0]))
    output: list[RetrievalHit] = []
    for rank, (video_id, item) in enumerate(ordered[:top_k], 1):
        representative: RetrievalHit = item["representative"]
        output.append(
            RetrievalHit(
                retriever="rrf",
                rank=rank,
                raw_score=float(item["score"]),
                video_id=video_id,
                source_type="hybrid",
                document_id=representative.document_id,
                keyframe_id=representative.keyframe_id,
                frame_idx=representative.frame_idx,
                pts_time=representative.pts_time,
                matched_text=representative.matched_text,
                provenance={
                    "fusion_version": RRF_FUSION_VERSION,
                    "rrf_k": config.rrf_k,
                    "retriever_ranks": item["ranks"],
                    "retriever_raw_scores": item["raw_scores"],
                    "contributions": item["contributions"],
                    "evidence": item["evidence"],
                    "failures": dict(failures or {}),
                },
            )
        )
    return output


class HybridRrfRetriever:
    name = "rrf"

    def __init__(self, retrievers: Sequence[Retriever], config: RrfFusionConfig = RrfFusionConfig()) -> None:
        if not retrievers:
            raise ValueError("hybrid RRF requires at least one retriever")
        names = [retriever.name for retriever in retrievers]
        if len(names) != len(set(names)):
            raise ValueError("hybrid RRF retriever names must be unique")
        self.retrievers = tuple(retrievers)
        self.config = config
        self.last_failures: dict[str, str] = {}

    def health(self) -> RetrieverHealth:
        statuses: dict[str, str] = {}
        warnings: list[str] = []
        for retriever in self.retrievers:
            try:
                health = retriever.health()
                statuses[retriever.name] = health.status
                warnings.extend(f"{retriever.name}: {warning}" for warning in health.warnings)
            except Exception as exc:
                statuses[retriever.name] = "UNAVAILABLE"
                warnings.append(f"{retriever.name}: {type(exc).__name__}: {exc}")
        available = [status for status in statuses.values() if status != "UNAVAILABLE"]
        status = "UNAVAILABLE" if not available else ("READY" if all(value == "READY" for value in statuses.values()) else "DEGRADED")
        return RetrieverHealth(
            self.name,
            status,
            RRF_FUSION_VERSION,
            tuple(f"fusion:{name}" for name in statuses),
            tuple(warnings),
            {"retrievers": statuses, "rrf_k": self.config.rrf_k, "candidate_pool": self.config.candidate_pool},
        )

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        pool = min(1000, max(request.top_k, self.config.candidate_pool))
        child_request = RetrievalRequest(
            query_id=request.query_id,
            query_text=request.query_text,
            groups=request.groups,
            top_k=pool,
            filters=request.filters,
            trace_id=request.trace_id,
        )
        channels: dict[str, Sequence[RetrievalHit]] = {}
        failures: dict[str, str] = {}
        for retriever in self.retrievers:
            try:
                channels[retriever.name] = retriever.search(child_request)
            except Exception as exc:
                if self.config.strict:
                    raise
                failures[retriever.name] = f"{type(exc).__name__}: {exc}"
        self.last_failures = failures
        return fuse_video_hits(channels, self.config, top_k=request.top_k, failures=failures)
