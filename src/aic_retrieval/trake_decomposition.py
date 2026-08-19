"""Deterministic local event decomposition for TRAKE queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from aic_retrieval.trake_schema import EventSource, TrakeEvent, TrakeRequest


DECOMPOSER_VERSION = "phase8-rule-decomposer-v1"
REVIEW_WARNING = "DECOMPOSITION_REVIEW_REQUIRED"

_FORWARD_CONNECTOR = re.compile(
    r"\s*(?:→|\b(?:sau\s+đó|rồi|tiếp\s+theo|then|after\s+that|followed\s+by)\b)\s*",
    flags=re.IGNORECASE,
)
_BEFORE_CONNECTOR = re.compile(r"\s+\b(?:trước\s+khi|before)\b\s+", flags=re.IGNORECASE)
_TEMPORAL_HINT = re.compile(r"→|\b(?:sau|trước|rồi|tiếp|then|after|before|followed)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class DecompositionResult:
    request: TrakeRequest
    warnings: tuple[str, ...] = ()
    version: str = DECOMPOSER_VERSION


def _clean(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip(" \t\r\n,;.").split())


def _event(text: str, order: int, source: EventSource = EventSource.RULE) -> TrakeEvent:
    return TrakeEvent(event_id=f"e{order}", order=order, text=text, clip_query=text, source=source)


class RuleBasedTrakeDecomposer:
    """Split only explicit temporal connectors and flag ambiguous hints."""

    def decompose(self, query_id: str, query: str, *, group: str = "L21") -> DecompositionResult:
        original = _clean(query)
        if not original:
            raise ValueError("original_query must not be empty")
        warnings: list[str] = []

        forward_parts = [_clean(part) for part in _FORWARD_CONNECTOR.split(original)]
        forward_parts = [part for part in forward_parts if part]
        if len(forward_parts) > 1:
            parts = forward_parts
        else:
            before_parts = [_clean(part) for part in _BEFORE_CONNECTOR.split(original)]
            before_parts = [part for part in before_parts if part]
            parts = before_parts if len(before_parts) > 1 else [original]
            if len(parts) == 1 and _TEMPORAL_HINT.search(original):
                warnings.append(REVIEW_WARNING)

        if len(parts) > 5:
            raise ValueError("TRAKE requires at most 5 events")
        request = TrakeRequest(
            query_id=query_id,
            original_query=original,
            events=tuple(_event(text, index) for index, text in enumerate(parts, start=1)),
            constraints={},
            group=group,
            source_versions={"decomposer": DECOMPOSER_VERSION},
        )
        return DecompositionResult(request=request, warnings=tuple(warnings))

    def manual_override(
        self,
        query_id: str,
        original_query: str,
        event_texts: list[str] | tuple[str, ...],
        *,
        group: str = "L21",
        constraints: dict[str, object] | None = None,
    ) -> DecompositionResult:
        cleaned = [_clean(value) for value in event_texts]
        if any(not value for value in cleaned):
            raise ValueError("manual event text must not be empty")
        request = TrakeRequest(
            query_id=query_id,
            original_query=original_query,
            events=tuple(_event(text, index, EventSource.MANUAL) for index, text in enumerate(cleaned, start=1)),
            constraints=dict(constraints or {}),
            group=group,
            source_versions={"decomposer": DECOMPOSER_VERSION},
        )
        return DecompositionResult(request=request)
