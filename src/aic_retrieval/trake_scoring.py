"""Transparent scoring primitives for TRAKE temporal chains."""

from __future__ import annotations

from dataclasses import dataclass

from aic_retrieval.trake_candidates import TrakeCandidate


SCORING_VERSION = "phase8-chain-score-v1"


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
