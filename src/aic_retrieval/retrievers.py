from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


RETRIEVER_CONTRACT_VERSION = "retriever-contract-v1"
HEALTH_STATUSES = {"READY", "DEGRADED", "UNAVAILABLE"}


@dataclass(frozen=True)
class RetrievalRequest:
    query_id: str
    query_text: str
    groups: tuple[str, ...] = ("L21",)
    top_k: int = 100
    filters: Mapping[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be empty")
        if not self.query_text.strip():
            raise ValueError("query_text must not be empty")
        if not self.groups or any(not group.strip() for group in self.groups):
            raise ValueError("groups must contain non-empty values")
        if not 1 <= self.top_k <= 1000:
            raise ValueError("top_k must be between 1 and 1000")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    retriever: str
    rank: int
    raw_score: float
    video_id: str
    source_type: str
    document_id: str | None = None
    keyframe_id: int | None = None
    frame_idx: int | None = None
    pts_time: float | None = None
    matched_text: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.retriever.strip() or not self.video_id.strip() or not self.source_type.strip():
            raise ValueError("retriever, video_id and source_type must not be empty")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if not math.isfinite(self.raw_score):
            raise ValueError("raw_score must be finite")
        for name, value in (("keyframe_id", self.keyframe_id), ("frame_idx", self.frame_idx)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.pts_time is not None and (not math.isfinite(self.pts_time) or self.pts_time < 0):
            raise ValueError("pts_time must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrieverHealth:
    name: str
    status: str
    index_version: str | None = None
    capabilities: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("retriever health name must not be empty")
        if self.status not in HEALTH_STATUSES:
            raise ValueError(f"unsupported retriever health status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def health(self) -> RetrieverHealth: ...

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]: ...
