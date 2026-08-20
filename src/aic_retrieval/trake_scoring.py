"""Transparent scoring primitives for TRAKE temporal chains."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from aic_retrieval.trake_candidates import TrakeCandidate
from aic_retrieval.trake_schema import TrakeEvent, TrakeRequest


SCORING_VERSION = "phase8-chain-score-v1"
MULTIMODAL_SCORING_VERSION = "event-aware-reciprocal-rank-v1"
CONTEXT_SCORING_VERSION = "verified-evidence-token-overlap-v1"
LINKAGE_SCORING_VERSION = "inspectable-adjacent-linkage-v1"
FINAL_RERANK_VERSION = "trake-final-rerank-v1"


@dataclass(frozen=True)
class ScoreConfig:
    required_coverage_weight: float = 2.0
    optional_coverage_weight: float = 0.5
    evidence_weight: float = 0.03
    preferred_gap_penalty: float = 0.25
    optional_skip_penalty: float = 0.15
    duplicate_penalty: float = 1.0


@dataclass(frozen=True)
class ScoreBreakdown:
    coverage_score: float
    event_score_sum: float
    evidence_score: float
    gap_penalty: float
    optional_penalty: float
    duplicate_penalty: float
    final_score: float
    version: str = SCORING_VERSION


@dataclass(frozen=True)
class MultimodalEventScore:
    score: float
    normalized_by_modality: dict[str, float]
    raw_scores: dict[str, float | None]
    raw_ranks: dict[str, int | None]
    weights_used: dict[str, float]
    availability: dict[str, str]
    support_score: float
    method: str = MULTIMODAL_SCORING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextScore:
    status: str
    score: float | None
    components: dict[str, dict[str, Any]]
    context: str | None
    method: str = CONTEXT_SCORING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinkageScore:
    status: str
    score: float | None
    adjacent_pairs: tuple[dict[str, Any], ...]
    method: str = LINKAGE_SCORING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_chain(
    selected: tuple[TrakeCandidate | None, ...],
    required: tuple[bool, ...],
    gap_penalty_count: int,
    config: ScoreConfig,
) -> ScoreBreakdown:
    coverage_score = sum(
        config.required_coverage_weight if is_required else config.optional_coverage_weight
        for item, is_required in zip(selected, required)
        if item is not None
    )
    event_score_sum = sum(item.local_score for item in selected if item is not None)
    evidence_score = sum(len(item.provenance) for item in selected if item is not None) * config.evidence_weight
    optional_penalty = sum(
        config.optional_skip_penalty
        for item, is_required in zip(selected, required)
        if item is None and not is_required
    )
    ids = [(item.video_id, item.keyframe_id) for item in selected if item is not None]
    duplicate_penalty = max(0, len(ids) - len(set(ids))) * config.duplicate_penalty
    gap_penalty = gap_penalty_count * config.preferred_gap_penalty
    final_score = coverage_score + event_score_sum + evidence_score - gap_penalty - optional_penalty - duplicate_penalty
    return ScoreBreakdown(
        coverage_score=coverage_score,
        event_score_sum=event_score_sum,
        evidence_score=evidence_score,
        gap_penalty=gap_penalty,
        optional_penalty=optional_penalty,
        duplicate_penalty=duplicate_penalty,
        final_score=final_score,
    )


def multimodal_event_score(
    event: TrakeEvent,
    candidate: TrakeCandidate,
    availability: dict[str, Any] | None = None,
    modality_weights: dict[str, float] | None = None,
) -> MultimodalEventScore:
    """Normalize heterogeneous channels by rank before event-aware weighting."""
    availability = availability or {}
    modality_weights = modality_weights or {}
    normalized: dict[str, float] = {}
    raw_scores: dict[str, float | None] = {}
    raw_ranks: dict[str, int | None] = {}
    weights: dict[str, float] = {}
    statuses: dict[str, str] = {}
    routed = tuple(dict.fromkeys(event.modalities))
    for modality in routed:
        rank = _rank_for_modality(candidate.raw_ranks, modality)
        raw = _score_for_modality(candidate.raw_scores, modality)
        provenance_match = any(_canonical_modality(value) == modality for value in candidate.provenance)
        status = _availability_value(availability.get(modality))
        if rank is not None or raw is not None or provenance_match:
            status = "AVAILABLE"
        statuses[modality] = status
        raw_scores[modality] = raw
        raw_ranks[modality] = rank
        if status == "UNAVAILABLE" or (status == "UNKNOWN" and not provenance_match and rank is None and raw is None):
            continue
        if rank is not None:
            value = 1.0 / max(1, rank)
        elif modality == "clip" and provenance_match:
            value = candidate.local_score
        else:
            value = 0.0
        normalized[modality] = value
        weights[modality] = float(modality_weights.get(modality, _default_modality_weight(modality)))
    if not normalized:
        normalized = {"adapter_rank": candidate.local_score}
        weights = {"adapter_rank": 1.0}
        statuses["adapter_rank"] = "AVAILABLE"
    denominator = sum(weights.values()) or 1.0
    score = sum(normalized[name] * weights[name] for name in normalized) / denominator
    positive = sum(value > 0 for value in normalized.values())
    support = positive / max(1, len(normalized))
    return MultimodalEventScore(
        round(max(0.0, min(1.0, score)), 6),
        normalized,
        raw_scores,
        raw_ranks,
        weights,
        statuses,
        round(support, 6),
    )


def global_context_score(
    request: TrakeRequest,
    chain: Any,
    availability: dict[str, Any] | None = None,
) -> ContextScore:
    context_tokens = _tokens(request.global_context or "")
    if not context_tokens:
        return ContextScore("UNKNOWN", None, {}, request.global_context)
    components: dict[str, dict[str, Any]] = {}
    for modality in ("metadata", "asr", "ocr", "object"):
        if _availability_value((availability or {}).get(modality)) == "UNAVAILABLE":
            components[modality] = {"status": "UNAVAILABLE", "score": None, "matched_tokens": []}
            continue
        values: list[str] = []
        for entry in chain.events:
            if entry.candidate is not None:
                values.extend(_evidence_strings(entry.candidate.evidence.get(modality)))
        evidence_tokens = _tokens(" ".join(values))
        if not evidence_tokens:
            components[modality] = {"status": "UNKNOWN", "score": None, "matched_tokens": []}
            continue
        matched = sorted(context_tokens & evidence_tokens)
        components[modality] = {
            "status": "AVAILABLE",
            "score": len(matched) / len(context_tokens),
            "matched_tokens": matched,
        }
    available_scores = [item["score"] for item in components.values() if item["status"] == "AVAILABLE"]
    if not available_scores:
        return ContextScore("UNKNOWN", None, components, request.global_context)
    return ContextScore("AVAILABLE", round(sum(available_scores) / len(available_scores), 6), components, request.global_context)


def adjacent_linkage_score(
    chain: Any,
    availability: dict[str, Any] | None = None,
    *,
    temporal_scale_seconds: float = 60.0,
) -> LinkageScore:
    selected = [entry for entry in chain.events if entry.candidate is not None]
    pairs: list[dict[str, Any]] = []
    for left, right in zip(selected, selected[1:]):
        left_candidate = left.candidate
        right_candidate = right.candidate
        gap = right_candidate.pts_time - left_candidate.pts_time
        components: dict[str, dict[str, Any]] = {
            "temporal_proximity": {
                "status": "AVAILABLE",
                "score": math.exp(-max(0.0, gap) / temporal_scale_seconds),
                "gap_seconds": gap,
            },
            "visual_continuity": _pair_evidence_component(left_candidate, right_candidate, "visual_embedding"),
            "scene_continuity": _pair_scalar_component(left_candidate, right_candidate, "scene_id"),
            "object_entity_continuity": _pair_token_component(left_candidate, right_candidate, ("object", "metadata")),
            "asr_continuity": (
                {"status": "UNAVAILABLE", "score": None}
                if _availability_value((availability or {}).get("asr")) == "UNAVAILABLE"
                else _pair_token_component(left_candidate, right_candidate, ("asr",))
            ),
        }
        scores = [value["score"] for value in components.values() if value.get("status") == "AVAILABLE"]
        pairs.append({
            "left_event_id": left.event_id,
            "right_event_id": right.event_id,
            "score": sum(scores) / len(scores),
            "components": components,
        })
    if not pairs:
        return LinkageScore("UNKNOWN", None, ())
    return LinkageScore("AVAILABLE", round(sum(pair["score"] for pair in pairs) / len(pairs), 6), tuple(pairs))


def linkage_gate_reason(chain: Any, settings: Any, availability: dict[str, Any] | None = None) -> str | None:
    """Return a conservative continuity rejection reason for dense TRAKE chains.

    Dense candidates carry a small visual scene descriptor from the decoded
    frame.  We only hard-gate when a chain actually used dense refinement;
    legacy/index-only paths remain reviewable rather than being mislabeled as
    visual continuity evidence.
    """
    if not bool(getattr(settings, "linkage_hard_gate", False)):
        return None
    selected = [entry.candidate for entry in chain.events if entry.candidate is not None]
    if not any(candidate.evidence.get("dense_refinement") for candidate in selected):
        return None
    linkage = adjacent_linkage_score(chain, availability, temporal_scale_seconds=settings.temporal_link_scale_seconds)
    threshold = float(getattr(settings, "min_visual_continuity", 0.0))
    for pair in linkage.adjacent_pairs:
        visual = pair["components"]["visual_continuity"]
        if visual.get("status") != "AVAILABLE":
            return "CONTINUITY_EVIDENCE_MISSING"
        if float(visual.get("score") or 0.0) < threshold:
            return "VISUAL_CONTINUITY_BELOW_THRESHOLD"
    return None


def rerank_alignments(
    request: TrakeRequest,
    alignments: tuple[Any, ...],
    settings: Any,
    availability: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Late rerank with explicit components; missing evidence is omitted, not negative."""
    reranked_results: list[Any] = []
    for result in alignments:
        reranked_chains: list[Any] = []
        for chain in result.chains:
            event_scores = [
                multimodal_event_score(event, entry.candidate, availability, settings.multimodal_weights)
                for event, entry in zip(request.events, chain.events)
                if entry.candidate is not None
            ]
            context = global_context_score(request, chain, availability) if settings.context_enabled else ContextScore("DISABLED", None, {}, request.global_context)
            linkage = adjacent_linkage_score(chain, availability, temporal_scale_seconds=settings.temporal_link_scale_seconds) if settings.linkage_enabled else LinkageScore("DISABLED", None, ())
            required_total = sum(event.required for event in request.events)
            required_selected = sum(event.required and entry.candidate is not None for event, entry in zip(request.events, chain.events))
            if required_selected != required_total:
                continue
            gate_reason = linkage_gate_reason(chain, settings, availability)
            if gate_reason is not None:
                continue
            semantic = sum(item.score for item in event_scores) / max(1, len(event_scores))
            temporal = 1.0 / (1.0 + float((chain.score_components or {}).get("transition_penalty", chain.score.gap_penalty)))
            coverage = sum(entry.candidate is not None for entry in chain.events) / len(request.events)
            multimodal = sum(item.support_score for item in event_scores) / max(1, len(event_scores))
            values: dict[str, float | None] = {
                "semantic": semantic,
                "temporal": temporal,
                "linkage": linkage.score,
                "context": context.score,
                "multimodal": multimodal,
                "coverage": coverage,
            }
            used = {name: float(settings.weights.get(name, 0.0)) for name, value in values.items() if value is not None and float(settings.weights.get(name, 0.0)) > 0}
            denominator = sum(used.values()) or 1.0
            final = sum(float(values[name]) * weight for name, weight in used.items()) / denominator
            components = {
                **(chain.score_components or {}),
                "event_semantic_score": semantic,
                "temporal_score": temporal,
                "context_score": context.to_dict(),
                "linkage_score": linkage.to_dict(),
                "linkage_gate": {"enabled": bool(getattr(settings, "linkage_hard_gate", False)), "status": "PASSED"},
                # Dense CLIP refinement and the visual-continuity gate establish a
                # coherent *candidate* timeline. They do not prove a human
                # action (for example, that someone actually sat down). That
                # requires an action-capable verifier or the existing human
                # review step, so never expose this score as an automatic
                # semantic confirmation.
                "semantic_verification": {
                    "status": "REQUIRES_MANUAL_REVIEW",
                    "reason": "CLIP_AND_SCENE_CONTINUITY_DO_NOT_VERIFY_ACTION_EXECUTION",
                },
                "coverage_score": coverage,
                "multimodal_evidence_score": multimodal,
                "event_multimodal_scores": [item.to_dict() for item in event_scores],
                "final_rerank_score": round(final, 6),
                "final_rerank_weights_used": used,
                "final_rerank_version": FINAL_RERANK_VERSION,
            }
            reranked_chains.append(replace(chain, score_components=components))
        reranked_chains.sort(key=lambda chain: (-float((chain.score_components or {}).get("final_rerank_score", 0.0)), chain.chain_id))
        reranked_results.append(replace(result, chains=tuple(reranked_chains)))
    reranked_results.sort(
        key=lambda result: (
            not bool(result.chains),
            -float((result.chains[0].score_components or {}).get("final_rerank_score", 0.0)) if result.chains else 0.0,
            result.video_id,
        )
    )
    return tuple(reranked_results)


def _rank_for_modality(values: dict[str, int | None], modality: str) -> int | None:
    ranks = [int(value) for key, value in values.items() if value is not None and _canonical_modality(key) == modality]
    return min(ranks, default=None)


def _score_for_modality(values: dict[str, float | None], modality: str) -> float | None:
    scores = [float(value) for key, value in values.items() if value is not None and _canonical_modality(key) == modality]
    return max(scores, default=None)


def _canonical_modality(value: str) -> str:
    folded = value.lower()
    if folded.startswith("dense_clip") or folded.startswith("clip"):
        return "clip"
    return folded.split("_", 1)[0]


def _availability_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "UNKNOWN").upper()


def _default_modality_weight(modality: str) -> float:
    return {"ocr": 3.0, "asr": 3.0, "object": 2.0, "attribute": 2.0}.get(modality, 1.0)


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE) if len(token) >= 2}


def _evidence_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _evidence_strings(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _evidence_strings(child)]
    return []


def _pair_scalar_component(left: TrakeCandidate, right: TrakeCandidate, key: str) -> dict[str, Any]:
    left_value = left.evidence.get(key)
    right_value = right.evidence.get(key)
    if left_value is None or right_value is None:
        return {"status": "UNKNOWN", "score": None}
    return {"status": "AVAILABLE", "score": 1.0 if left_value == right_value else 0.0}


def _pair_evidence_component(left: TrakeCandidate, right: TrakeCandidate, key: str) -> dict[str, Any]:
    left_value = left.evidence.get(key)
    right_value = right.evidence.get(key)
    if not isinstance(left_value, (list, tuple)) or not isinstance(right_value, (list, tuple)) or len(left_value) != len(right_value):
        return {"status": "UNKNOWN", "score": None}
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left_value))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right_value))
    if not left_norm or not right_norm:
        return {"status": "UNKNOWN", "score": None}
    cosine = sum(float(a) * float(b) for a, b in zip(left_value, right_value)) / (left_norm * right_norm)
    return {"status": "AVAILABLE", "score": max(0.0, min(1.0, (cosine + 1.0) / 2.0))}


def _pair_token_component(left: TrakeCandidate, right: TrakeCandidate, modalities: tuple[str, ...]) -> dict[str, Any]:
    left_tokens = _tokens(" ".join(value for modality in modalities for value in _evidence_strings(left.evidence.get(modality))))
    right_tokens = _tokens(" ".join(value for modality in modalities for value in _evidence_strings(right.evidence.get(modality))))
    if not left_tokens or not right_tokens:
        return {"status": "UNKNOWN", "score": None}
    union = left_tokens | right_tokens
    return {"status": "AVAILABLE", "score": len(left_tokens & right_tokens) / len(union), "shared_tokens": sorted(left_tokens & right_tokens)}
