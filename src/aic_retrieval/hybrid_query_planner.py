from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib import request

from aic_retrieval.query_planner import RuleBasedQueryPlanner, TEXT_MARKERS, contains_phrase, normalize, quoted_phrases
from aic_retrieval.translation import TranslationConfig, http_error_detail, parse_gemini_translation


HYBRID_PLAN_VERSION = "hybrid-query-plan-v2"
AGENT_TRACE_VERSION = "agent-planning-trace-v1"
MAX_RAW_TRACE_CHARS = 16_384
ALLOWED_RETRIEVERS = ("clip", "bge", "bm25")
ALLOWED_INTENTS = {"visual", "semantic_text", "lexical_text", "mixed"}
CONTENT_MARKERS = (
    "nội dung", "chủ đề", "bản tin", "về", "đề cập", "nói", "phát biểu",
    "content", "topic", "report", "about", "mentions", "says", "speech",
)


@dataclass(frozen=True)
class HybridQueryPlan:
    original_query: str
    visual_clip_query_en: str
    semantic_text_query: str
    lexical_text_query: str
    intent: str
    enabled_retrievers: tuple[str, ...]
    profile: str
    fusion_method: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    planner_source: str = "local_rules"
    source_filters: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    structured_filter_suggestions: Mapping[str, Any] = field(default_factory=dict)
    version: str = HYBRID_PLAN_VERSION

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError("original_query must not be empty")
        if self.intent not in ALLOWED_INTENTS:
            raise ValueError(f"unsupported hybrid intent: {self.intent}")
        if not self.enabled_retrievers or any(name not in ALLOWED_RETRIEVERS for name in self.enabled_retrievers):
            raise ValueError("enabled_retrievers contains an unsupported or empty route")
        if len(self.enabled_retrievers) != len(set(self.enabled_retrievers)):
            raise ValueError("enabled_retrievers must be unique")
        required_queries = {
            "clip": self.visual_clip_query_en,
            "bge": self.semantic_text_query,
            "bm25": self.lexical_text_query,
        }
        if any(not required_queries[name].strip() for name in self.enabled_retrievers):
            raise ValueError("every enabled retriever requires a non-empty routed query")

    def query_for(self, retriever: str) -> str:
        return {
            "clip": self.visual_clip_query_en,
            "bge": self.semantic_text_query,
            "bm25": self.lexical_text_query,
        }[retriever]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridPlanningTrace:
    requested: bool
    provider: str
    model: str
    status: str
    raw_text: str | None
    parsed_output: Mapping[str, Any] | None
    validated_plan: Mapping[str, Any]
    normalization_notes: tuple[str, ...]
    warnings: tuple[str, ...]
    fallback: str | None
    latency_ms: float
    raw_text_truncated: bool = False
    version: str = AGENT_TRACE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridPlanningResult:
    plan: HybridQueryPlan
    trace: HybridPlanningTrace


def _profile(routes: tuple[str, ...]) -> str:
    return "_".join(routes) + ("_rrf" if len(routes) > 1 else "")


class LocalHybridQueryPlanner:
    def __init__(self) -> None:
        self.base = RuleBasedQueryPlanner()

    def plan(self, query: str, visual_clip_query_en: str | None = None) -> HybridQueryPlan:
        base = self.base.plan(query)
        text = normalize(query)
        modalities = set(base.recommended_modalities)
        quoted = quoted_phrases(query)
        explicit_ocr = any(contains_phrase(text, marker) for marker in TEXT_MARKERS)
        has_number = bool(re.search(r"\d", query))
        lexical = bool({"ocr", "asr", "metadata"} & modalities or quoted or has_number)
        semantic = bool("asr" in modalities or any(marker in text for marker in CONTENT_MARKERS))
        metadata_only = "metadata" in modalities and not ({"objects", "attributes"} & modalities) and not semantic

        routes: list[str] = []
        reasons: list[str] = []
        if not metadata_only:
            routes.append("clip")
            reasons.append("visual semantics remain the baseline route")
        if semantic:
            routes.append("bge")
            reasons.append("query asks about spoken or semantic content")
        if lexical:
            routes.append("bm25")
            reasons.append("query contains exact text, metadata, speech markers, quotes, or numbers")
        if not routes:
            routes.append("clip")
        ordered_routes = tuple(name for name in ALLOWED_RETRIEVERS if name in routes)
        intent = (
            "mixed" if len(ordered_routes) > 1 else
            "semantic_text" if ordered_routes == ("bge",) else
            "lexical_text" if ordered_routes == ("bm25",) else
            "visual"
        )
        warnings: list[str] = []
        if "asr" in modalities:
            warnings.append("ASR coverage is partial for L21")
        if "ocr" in modalities and (explicit_ocr or "asr" not in modalities):
            warnings.append("OCR evidence is unavailable in the current L21 corpus")
        text_sources: list[str] = []
        if "asr" in modalities:
            text_sources.append("asr")
        if "ocr" in modalities and (explicit_ocr or "asr" not in modalities):
            text_sources.append("ocr")
        if "metadata" in modalities:
            text_sources.append("metadata")
        source_filters = {
            name: tuple(dict.fromkeys(text_sources))
            for name in ("bge", "bm25")
            if name in ordered_routes and text_sources
        }
        color = next((candidate for candidate in ("red", "green", "blue", "white", "black", "yellow", "đỏ", "xanh", "trắng", "đen", "vàng") if contains_phrase(text, candidate)), None)
        structured_suggestions = {
            "enable_objects": bool(base.objects),
            "object_label": base.objects[0] if base.objects else None,
            "enable_attributes": bool(color),
            "attribute_color": color,
            "enable_ocr": "ocr" in modalities,
            "enable_asr": "asr" in modalities,
            "enable_metadata": "metadata" in modalities,
            "metadata_author": None,
            "metadata_date": None,
            "metadata_title": None,
        }
        return HybridQueryPlan(
            original_query=query.strip(),
            visual_clip_query_en=(visual_clip_query_en or base.visual_query).strip(),
            semantic_text_query=query.strip(),
            lexical_text_query=" ".join((*quoted, query.strip())) if quoted else query.strip(),
            intent=intent,
            enabled_retrievers=ordered_routes,
            profile=_profile(ordered_routes),
            fusion_method="rrf" if len(ordered_routes) > 1 else "none",
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            source_filters=source_filters,
            structured_filter_suggestions=structured_suggestions,
        )


def gemini_hybrid_plan_payload(query: str) -> dict[str, Any]:
    prompt = (
        "Analyze this video-retrieval query. Return one compact JSON object only, no markdown, with keys: "
        '"clip_query" (concise concrete English visual prompt), "semantic_query" (semantic content query, keep Vietnamese if Vietnamese), '
        '"lexical_query" (short exact Vietnamese/entity/number terms), "intent" (visual|semantic_text|lexical_text|mixed), '
        '"routes" (array containing only clip, bge, bm25), "reasons" (array of at most 3 short strings), and '
        '"structured_filters" with optional keys object_label, attribute_color, enable_ocr, enable_asr, '
        'metadata_author, metadata_date, metadata_title. Use null/false when a constraint is not explicit. '
        "Use clip for visible scene/action/object/color, bge for semantic transcript/content, bm25 for exact OCR/ASR/metadata/entity/date terms. "
        "Structured filters are suggestions only, so include a value only when the user explicitly states it. "
        "Do not answer the query and do not invent evidence.\n\n"
        f"Query: {query.strip()}"
    )
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }


def parse_gemini_hybrid_plan(payload: dict[str, Any], original_query: str) -> HybridQueryPlan:
    raw_text = parse_gemini_translation(payload)
    return parse_gemini_hybrid_plan_text(raw_text, original_query)


def parse_gemini_hybrid_plan_text(raw_text: str, original_query: str) -> HybridQueryPlan:
    raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini hybrid plan is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("Gemini hybrid plan must be a JSON object")
    routes_raw = raw.get("routes")
    if not isinstance(routes_raw, list) or not routes_raw:
        raise ValueError("Gemini hybrid plan routes must be a non-empty array")
    routes = tuple(name for name in ALLOWED_RETRIEVERS if name in routes_raw)
    if len(routes) != len(set(str(item) for item in routes_raw)) or len(routes) != len(routes_raw):
        raise ValueError("Gemini hybrid plan contains invalid routes")
    reasons_raw = raw.get("reasons", [])
    reasons = tuple(str(item).strip() for item in reasons_raw[:3] if str(item).strip()) if isinstance(reasons_raw, list) else ()
    intent = str(raw.get("intent", "mixed"))
    return HybridQueryPlan(
        original_query=original_query.strip(),
        visual_clip_query_en=str(raw.get("clip_query", "")).strip(),
        semantic_text_query=str(raw.get("semantic_query", "")).strip(),
        lexical_text_query=str(raw.get("lexical_query", "")).strip(),
        intent=intent,
        enabled_retrievers=routes,
        profile=_profile(routes),
        fusion_method="rrf" if len(routes) > 1 else "none",
        reasons=reasons,
        planner_source="gemini",
        structured_filter_suggestions=_structured_filter_suggestions(raw.get("structured_filters")),
    )


def _structured_filter_suggestions(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}

    def short_text(key: str) -> str | None:
        candidate = raw.get(key)
        if candidate is None:
            return None
        text = " ".join(str(candidate).strip().split())
        return text[:160] or None

    object_label = short_text("object_label")
    attribute_color = short_text("attribute_color")
    metadata_author = short_text("metadata_author")
    metadata_date = short_text("metadata_date")
    metadata_title = short_text("metadata_title")
    return {
        "enable_objects": bool(object_label),
        "object_label": object_label,
        "enable_attributes": bool(attribute_color),
        "attribute_color": attribute_color,
        "enable_ocr": raw.get("enable_ocr") is True,
        "enable_asr": raw.get("enable_asr") is True,
        "enable_metadata": any((metadata_author, metadata_date, metadata_title)),
        "metadata_author": metadata_author,
        "metadata_date": metadata_date,
        "metadata_title": metadata_title,
    }


def _safe_parsed_output(raw_text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    allowed = ("clip_query", "semantic_query", "lexical_query", "intent", "routes", "reasons", "structured_filters")
    return {key: value[key] for key in allowed if key in value}


def _trace_text(raw_text: str) -> tuple[str, bool]:
    if len(raw_text) <= MAX_RAW_TRACE_CHARS:
        return raw_text, False
    return raw_text[:MAX_RAW_TRACE_CHARS], True


class HybridQueryPlanner:
    def __init__(self, config: TranslationConfig | None = None) -> None:
        self.config = config or TranslationConfig()
        self.local = LocalHybridQueryPlanner()

    @property
    def gemini_configured(self) -> bool:
        return bool(self.config.provider == "gemini" and self.config.api_key and self.config.api_url and self.config.model)

    def plan(self, query: str, use_gemini: bool = True) -> HybridQueryPlan:
        return self.plan_with_trace(query, use_gemini=use_gemini).plan

    def plan_with_trace(self, query: str, use_gemini: bool = True) -> HybridPlanningResult:
        started = time.perf_counter()
        local = self.local.plan(query)
        if not use_gemini or not self.gemini_configured:
            warning = "Gemini planner unavailable; deterministic local routing used" if use_gemini else ""
            plan = replace(local, warnings=tuple((*local.warnings, warning)) if warning else local.warnings)
            status = "UNAVAILABLE" if use_gemini else "SKIPPED"
            return HybridPlanningResult(plan, self._trace(
                plan,
                requested=use_gemini,
                status=status,
                warnings=plan.warnings,
                fallback="local_rules" if use_gemini else None,
                latency_ms=(time.perf_counter() - started) * 1000,
            ))
        try:
            plan, raw_text = self._plan_with_gemini(query)
            safe_text, truncated = _trace_text(raw_text)
            notes = []
            if plan.source_filters:
                notes.append("Local coverage rules added source filters for available text evidence")
            if plan.warnings:
                notes.append("Local data coverage warnings were appended to the Gemini plan")
            if any(plan.structured_filter_suggestions.values()):
                notes.append("Structured filters are validated suggestions and are not auto-applied to this Agent run")
            return HybridPlanningResult(plan, self._trace(
                plan,
                requested=True,
                status="SUCCEEDED",
                raw_text=safe_text,
                raw_text_truncated=truncated,
                parsed_output=_safe_parsed_output(raw_text),
                normalization_notes=tuple(notes),
                warnings=plan.warnings,
                latency_ms=(time.perf_counter() - started) * 1000,
            ))
        except Exception as exc:
            plan = replace(
                local,
                warnings=tuple((*local.warnings, f"Gemini planner fallback: {type(exc).__name__}: {exc}")),
            )
            return HybridPlanningResult(plan, self._trace(
                plan,
                requested=True,
                status="FALLBACK",
                warnings=plan.warnings,
                fallback=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            ))

    def _trace(
        self,
        plan: HybridQueryPlan,
        *,
        requested: bool,
        status: str,
        raw_text: str | None = None,
        parsed_output: Mapping[str, Any] | None = None,
        normalization_notes: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
        fallback: str | None = None,
        latency_ms: float,
        raw_text_truncated: bool = False,
    ) -> HybridPlanningTrace:
        return HybridPlanningTrace(
            requested=requested,
            provider=self.config.provider,
            model=self.config.model,
            status=status,
            raw_text=raw_text,
            parsed_output=parsed_output,
            validated_plan=plan.to_dict(),
            normalization_notes=normalization_notes,
            warnings=warnings,
            fallback=fallback,
            latency_ms=round(latency_ms, 3),
            raw_text_truncated=raw_text_truncated,
        )

    def _plan_with_gemini(self, query: str) -> tuple[HybridQueryPlan, str]:
        model = self.config.model if self.config.model.startswith("models/") else f"models/{self.config.model}"
        url = f"{self.config.api_url.rstrip('/')}/{model}:generateContent"
        req = request.Request(
            url,
            data=json.dumps(gemini_hybrid_plan_payload(query)).encode("utf-8"),
            headers={"x-goog-api-key": self.config.api_key or "", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Gemini planner request failed: {http_error_detail(exc)}") from exc
        raw_text = parse_gemini_translation(payload)
        plan = parse_gemini_hybrid_plan_text(raw_text, query)
        local = self.local.plan(query, visual_clip_query_en=plan.visual_clip_query_en)
        base_plan = self.local.base.plan(query)
        warnings = list(local.warnings)
        if "asr" in base_plan.recommended_modalities and "ASR coverage is partial for L21" not in warnings:
            warnings.append("ASR coverage is partial for L21")
        return replace(plan, warnings=tuple(warnings), source_filters=local.source_filters), raw_text
