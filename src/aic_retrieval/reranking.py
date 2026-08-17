from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from aic_retrieval.query_planner import QueryPlan, normalize

RERANKER_VERSION = "phase6-lightweight-reranker-v1"


@dataclass(frozen=True)
class RerankerConfig:
    top_n: int = 20
    original_score_weight: float = 1.0
    modality_match_weight: float = 0.08
    lexical_evidence_weight: float = 0.12
    planner_alignment_weight: float = 0.06

    def __post_init__(self) -> None:
        if self.top_n <= 0 or self.top_n > 100:
            raise ValueError("reranker top_n must be between 1 and 100")
        if min(self.original_score_weight, self.modality_match_weight, self.lexical_evidence_weight, self.planner_alignment_weight) < 0:
            raise ValueError("reranker weights must not be negative")


def load_reranker_config(path: Path) -> RerankerConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != RERANKER_VERSION:
        raise ValueError(f"unsupported reranker config version: {payload.get('version')}")
    return RerankerConfig(**payload["config"])


def with_top_n(config: RerankerConfig, top_n: int) -> RerankerConfig:
    return replace(config, top_n=top_n)


def rerank_video_results(results: list[dict[str, Any]], plan: QueryPlan, config: RerankerConfig = RerankerConfig()) -> list[dict[str, Any]]:
    """Rerank only the first Top-N candidates and preserve the tail order."""
    head = results[: config.top_n]
    tail = results[config.top_n :]
    if not head:
        return []
    max_score = max(abs(float(item.get("fusion_score", item.get("video_score", item.get("score", 0.0))))) for item in head) or 1.0
    reranked = []
    requested = set(plan.recommended_modalities)
    query_tokens = set(normalize(" ".join(plan.variants)).split())
    for original_rank, item in enumerate(head, start=1):
        base = float(item.get("fusion_score", item.get("video_score", item.get("score", 0.0))))
        ranks = item.get("modality_ranks", {})
        matched = {name for name, rank in ranks.items() if rank is not None}
        modality_matches = len(requested & matched)
        evidence_text = evidence_strings(item)
        evidence_tokens = set(normalize(" ".join(evidence_text)).split())
        lexical_overlap = len(query_tokens & evidence_tokens) / max(1, len(query_tokens))
        planner_alignment = len(requested & matched) / max(1, len(requested))
        contributions = {
            "normalized_original": config.original_score_weight * (base / max_score),
            "modality_match": config.modality_match_weight * modality_matches,
            "lexical_evidence": config.lexical_evidence_weight * lexical_overlap,
            "planner_alignment": config.planner_alignment_weight * planner_alignment,
        }
        modality_contributions = {
            modality: config.modality_match_weight if modality in matched else 0.0
            for modality in sorted(requested)
        }
        rerank_score = sum(contributions.values())
        reranked.append({
            **item,
            "pre_rerank_rank": item.get("rank", original_rank),
            "pre_rerank_score": base,
            "rerank_score": rerank_score,
            "rerank_explanation": {
                "matched_modalities": sorted(matched),
                "requested_modalities": sorted(requested),
                "lexical_overlap": lexical_overlap,
                "contributions": contributions,
                "modality_contributions": modality_contributions,
            },
        })
    reranked.sort(key=lambda item: (-item["rerank_score"], item["pre_rerank_rank"], item["video_id"]))
    output = [{**item, "rank": rank, "video_score":item["rerank_score"], "post_rerank_score": item["rerank_score"]} for rank, item in enumerate(reranked, start=1)]
    output.extend({**item, "rank": rank} for rank, item in enumerate(tail, start=len(output) + 1))
    return output


def evidence_strings(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for frame in item.get("frames", []):
        evidence = frame.get("evidence", {})
        for modality in ("ocr", "asr"):
            for match in evidence.get(modality, []):
                values.append(str(match.get("matched_text") or match.get("text_raw") or ""))
        metadata = evidence.get("metadata") or {}
        for match in metadata.get("evidence", []):
            values.append(str(match.get("value", "")))
    return values


def reranker_metadata(config: RerankerConfig) -> dict[str, Any]:
    return {"version": RERANKER_VERSION, "config": asdict(config), "scope": "top_n_only", "network_required": False}
