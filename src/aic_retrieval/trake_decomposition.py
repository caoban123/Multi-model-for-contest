"""Deterministic local event decomposition for TRAKE queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from aic_retrieval.trake_schema import EventSource, TRAKE_V2_SCHEMA_VERSION, TrakeEvent, TrakeRequest


DECOMPOSER_VERSION = "phase8-rule-decomposer-v1"
PLANNER_V2_VERSION = "trake-rule-planner-v2"
REVIEW_WARNING = "DECOMPOSITION_REVIEW_REQUIRED"
PLANNER_LOW_CONFIDENCE = "PLANNER_LOW_CONFIDENCE"

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


_CONTEXT_RULES = (
    (("football", "soccer", "stadium", "bong da", "san van dong"), "football match / stadium"),
    (("kitchen", "cook", "cooking", "nau an", "nha bep"), "cooking / kitchen"),
    (("classroom", "teacher", "student", "lop hoc", "giao vien"), "classroom / education"),
    (("road", "street", "traffic", "duong pho", "giao thong"), "road / traffic"),
    (("concert", "stage", "audience", "khán giả", "sân khấu"), "live event / stage"),
    (("store", "shop", "cua hang", "sieu thi"), "retail / store"),
)
_OCR_MARKERS = ("text", "word", "sign", "screen", "subtitle", "logo", "chu", "bien", "man hinh", "phu de")
_ASR_MARKERS = ("say", "says", "said", "speak", "speech", "shout", "hear", "noi", "phat bieu", "het")
_COLOR_WORDS = ("red", "blue", "green", "yellow", "black", "white", "pink", "orange", "purple", "do", "xanh", "vang", "den", "trang", "hong")
_OBJECT_WORDS = ("ball", "phone", "trophy", "cup", "car", "bicycle", "door", "table", "chair", "dog", "cat", "book", "bong", "dien thoai", "xe", "cua", "ban", "ghe", "cho", "meo")
_SUBJECT_WORDS = ("man", "woman", "person", "player", "goalkeeper", "child", "crowd", "nguoi dan ong", "nguoi phu nu", "cau thu", "thu mon", "tre em", "khan gia")
_ACTION_WORDS = ("enter", "open", "hold", "carry", "run", "walk", "kick", "save", "celebrate", "sit", "stand", "buoc vao", "mo", "cam", "chay", "di bo", "sut", "can", "an mung", "ngoi", "dung")
_LOCATION_WORDS = ("stadium", "kitchen", "classroom", "street", "store", "stage", "room", "san van dong", "nha bep", "lop hoc", "duong", "cua hang", "san khau", "phong")
_SYNONYMS = (
    (r"\bman\b", "person"),
    (r"\bwoman\b", "person"),
    (r"\bholding\b", "carrying"),
    (r"\bholds\b", "carries"),
    (r"\benters\b", "walks into"),
    (r"\bcelebrating\b", "cheering"),
)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    return " ".join("".join(char for char in normalized if unicodedata.category(char) != "Mn").split())


def _first_phrase(folded: str, phrases: tuple[str, ...]) -> str | None:
    return next((phrase for phrase in phrases if re.search(rf"\b{re.escape(phrase)}\b", folded)), None)


def _quoted_terms(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.strip() for match in re.findall(r'["“”]([^"“”]+)["“”]', value) if match.strip()))


def _query_variants(visual_query: str) -> tuple[str, ...]:
    variants: list[str] = []
    for pattern, replacement in _SYNONYMS:
        candidate = re.sub(pattern, replacement, visual_query, flags=re.IGNORECASE)
        if candidate != visual_query and candidate not in variants:
            variants.append(candidate)
    compact = re.sub(r"\b(?:a|an|the)\b", " ", visual_query, flags=re.IGNORECASE)
    compact = " ".join(compact.split())
    if compact and compact != visual_query and compact not in variants:
        variants.append(compact)
    return tuple(variants[:3])


class TrakePlannerV2:
    """Deterministic structured planner layered on the verified rule parser."""

    version = PLANNER_V2_VERSION

    def __init__(self, sequence_default_max_gap_seconds: float | None = None, parser: RuleBasedTrakeDecomposer | None = None) -> None:
        self.sequence_default_max_gap_seconds = sequence_default_max_gap_seconds
        self.parser = parser or RuleBasedTrakeDecomposer()

    def decompose(self, query_id: str, query: str, *, group: str = "L21") -> DecompositionResult:
        parsed = self.parser.decompose(query_id, query, group=group)
        confidence = 0.65 if parsed.warnings else 0.9
        return self._enrich(parsed, confidence=confidence)

    def manual_override(
        self,
        query_id: str,
        original_query: str,
        event_texts: list[str] | tuple[str, ...],
        *,
        group: str = "L21",
        constraints: dict[str, object] | None = None,
    ) -> DecompositionResult:
        parsed = self.parser.manual_override(
            query_id,
            original_query,
            event_texts,
            group=group,
            constraints=constraints,
        )
        return self._enrich(parsed, confidence=1.0)

    def _enrich(self, parsed: DecompositionResult, *, confidence: float) -> DecompositionResult:
        context = self._global_context(parsed.request.original_query)
        events = tuple(self._enrich_event(event, confidence=confidence) for event in parsed.request.events)
        constraints = dict(parsed.request.constraints)
        if len(events) > 1 and self.sequence_default_max_gap_seconds is not None:
            gap = self._sequence_gap_seconds(parsed.request.original_query)
            events = tuple(
                replace(
                    event,
                    max_gap_seconds=event.max_gap_seconds if event.max_gap_seconds is not None else gap,
                    expected_gap_class=event.expected_gap_class or self._gap_class(gap),
                ) if index else event
                for index, event in enumerate(events)
            )
            constraints.setdefault("gap_mode", "hard")
            constraints.setdefault("sequence_default_max_gap_seconds", gap)
        warnings = list(parsed.warnings)
        if confidence < 0.7:
            warnings.append(PLANNER_LOW_CONFIDENCE)
        request = TrakeRequest(
            query_id=parsed.request.query_id,
            original_query=parsed.request.original_query,
            events=events,
            constraints=constraints,
            group=parsed.request.group,
            source_versions={
                **(parsed.request.source_versions or {}),
                "planner": self.version,
            },
            schema_version=TRAKE_V2_SCHEMA_VERSION,
            global_context=context,
            planner_version=self.version,
        )
        return DecompositionResult(request=request, warnings=tuple(dict.fromkeys(warnings)), version=self.version)

    def _sequence_gap_seconds(self, query: str) -> float:
        folded = _fold(query)
        default = float(self.sequence_default_max_gap_seconds or 60.0)
        if any(marker in folded for marker in ("immediately", "right after", "ngay sau")):
            return min(default, 15.0)
        if any(marker in folded for marker in ("later", "after a while", "mot luc sau")):
            return max(default, 300.0)
        return default

    @staticmethod
    def _gap_class(gap: float) -> str:
        return "short" if gap <= 60 else "medium" if gap <= 300 else "long"

    @staticmethod
    def _global_context(query: str) -> str | None:
        folded = _fold(query)
        for markers, context in _CONTEXT_RULES:
            if any(marker in folded for marker in markers):
                return context
        return None

    @staticmethod
    def _enrich_event(event: TrakeEvent, *, confidence: float) -> TrakeEvent:
        raw = event.text
        folded = _fold(raw)
        quoted = _quoted_terms(raw)
        modalities = ["clip"]
        if any(marker in folded for marker in _OCR_MARKERS):
            modalities.append("ocr")
        if any(marker in folded for marker in _ASR_MARKERS):
            modalities.append("asr")
        color = _first_phrase(folded, _COLOR_WORDS)
        if color:
            modalities.append("attribute")
        object_label = _first_phrase(folded, _OBJECT_WORDS)
        if object_label:
            modalities.append("object")
        visual_query = raw
        expected_gap_class = "short" if any(marker in folded for marker in ("immediately", "right after", "ngay sau")) else None
        if expected_gap_class is None and any(marker in folded for marker in ("later", "after a while", "mot luc sau")):
            expected_gap_class = "long"
        return TrakeEvent(
            event_id=event.event_id,
            order=event.order,
            text=event.text,
            clip_query=visual_query,
            modalities=tuple(modalities),
            required=event.required,
            object_constraints=({"label": object_label},) if object_label else (),
            attribute_constraints=({"color": color, "filter_mode": "soft"},) if color else (),
            ocr_constraints=({"terms": quoted},) if "ocr" in modalities and quoted else (),
            asr_constraints=({"terms": quoted},) if "asr" in modalities and quoted else (),
            min_gap_seconds=event.min_gap_seconds,
            max_gap_seconds=event.max_gap_seconds,
            confidence=confidence,
            source=event.source,
            raw_text=raw,
            visual_query=visual_query,
            query_variants=_query_variants(visual_query),
            subject=_first_phrase(folded, _SUBJECT_WORDS),
            action=_first_phrase(folded, _ACTION_WORDS),
            location=_first_phrase(folded, _LOCATION_WORDS),
            ocr_terms=quoted if "ocr" in modalities else (),
            asr_terms=quoted if "asr" in modalities else (),
            expected_gap_class=expected_gap_class,
            distinctiveness=None,
            planner_confidence=confidence,
        )
