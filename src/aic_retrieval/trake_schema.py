"""Versioned domain contracts for Phase 8 TRAKE.

The contracts are deliberately independent from retrieval and HTTP code so
sequence validity can be checked again at every persistence/export boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


TRAKE_SCHEMA_VERSION = "phase8-trake-v1"
SUPPORTED_GROUPS = {"L21"}
SUPPORTED_MODALITIES = {"clip", "object", "attribute", "ocr", "asr", "metadata"}


class EventSource(str, Enum):
    MANUAL = "manual"
    RULE = "rule"
    API = "api"


class Availability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


def _text(name: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True)
class TrakeEvent:
    event_id: str
    order: int
    text: str
    clip_query: str
    modalities: tuple[str, ...] = ("clip",)
    required: bool = True
    object_constraints: tuple[dict[str, Any], ...] = ()
    attribute_constraints: tuple[dict[str, Any], ...] = ()
    ocr_constraints: tuple[dict[str, Any], ...] = ()
    asr_constraints: tuple[dict[str, Any], ...] = ()
    min_gap_seconds: float | None = None
    max_gap_seconds: float | None = None
    confidence: float = 1.0
    source: EventSource = EventSource.RULE

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text("event_id", self.event_id))
        object.__setattr__(self, "text", _text("text", self.text))
        object.__setattr__(self, "clip_query", _text("clip_query", self.clip_query))
        object.__setattr__(self, "modalities", tuple(dict.fromkeys(self.modalities)))
        if self.order < 1:
            raise ValueError("event order must start at 1")
        unsupported = set(self.modalities) - SUPPORTED_MODALITIES
        if unsupported:
            raise ValueError(f"unsupported modalities: {sorted(unsupported)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        for name, value in (("min_gap_seconds", self.min_gap_seconds), ("max_gap_seconds", self.max_gap_seconds)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_gap_seconds is not None and self.max_gap_seconds is not None and self.min_gap_seconds > self.max_gap_seconds:
            raise ValueError("min_gap_seconds must not exceed max_gap_seconds")


@dataclass(frozen=True)
class TrakeRequest:
    query_id: str
    original_query: str
    events: tuple[TrakeEvent, ...]
    constraints: dict[str, Any] = field(default_factory=dict)
    group: str = "L21"
    source_versions: dict[str, str] | None = None
    schema_version: str = TRAKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _text("query_id", self.query_id))
        object.__setattr__(self, "original_query", _text("original_query", self.original_query))
        object.__setattr__(self, "events", tuple(self.events))
        if self.schema_version != TRAKE_SCHEMA_VERSION:
            raise ValueError(f"unsupported TRAKE schema version: {self.schema_version}")
        if self.group not in SUPPORTED_GROUPS:
            raise ValueError(f"unsupported group: {self.group}")
        if not 1 <= len(self.events) <= 5:
            raise ValueError("TRAKE requires between 1 and 5 events")
        ids = [event.event_id for event in self.events]
        orders = [event.order for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event_id")
        if len(orders) != len(set(orders)):
            raise ValueError("duplicate event order")
        if sorted(orders) != list(range(1, len(self.events) + 1)):
            raise ValueError("event order must be contiguous from 1")
        if orders != sorted(orders):
            raise ValueError("events must be stored in temporal order")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for event in payload["events"]:
            event["source"] = event["source"].value if isinstance(event["source"], EventSource) else event["source"]
        return payload


def event_from_dict(payload: dict[str, Any]) -> TrakeEvent:
    values = dict(payload)
    for name in ("modalities", "object_constraints", "attribute_constraints", "ocr_constraints", "asr_constraints"):
        if name in values:
            values[name] = tuple(values[name])
    if "source" in values:
        values["source"] = EventSource(values["source"])
    return TrakeEvent(**values)


def request_from_dict(payload: dict[str, Any]) -> TrakeRequest:
    values = dict(payload)
    values["events"] = tuple(event_from_dict(item) for item in values.get("events", ()))
    return TrakeRequest(**values)
