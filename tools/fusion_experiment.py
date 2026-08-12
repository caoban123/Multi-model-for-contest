from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.fusion import rerank_results_with_objects
from aic_retrieval.judgements import load_judgement_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline object-fusion rerank experiment on benchmark candidates.")
    parser.add_argument("--benchmark", default="artifacts/benchmarks/l21_text_benchmark.json")
    parser.add_argument("--judgements", default="artifacts/benchmarks/l21_text_benchmark_judged.csv")
    parser.add_argument("--object-root", default="data/objects")
    parser.add_argument("--object-weight", type=float, default=0.05)
    parser.add_argument("--interaction-weight", type=float, default=0.0)
    parser.add_argument("--result-set", choices=["results", "raw_results"], default="results")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="artifacts/benchmarks/l21_fusion_experiment.json")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    judgement_lookup = build_judgement_lookup(load_judgement_rows(Path(args.judgements)))

    query_reports = []
    baseline_totals = empty_metrics()
    fusion_totals = empty_metrics()
    for query in benchmark.get("queries", []):
        query_id = str(query.get("id", ""))
        query_text = str(query.get("text", ""))
        baseline_results = query.get(args.result_set, query.get("results", []))[: args.top_k]
        fused_results = rerank_results_with_objects(
            query_text,
            baseline_results,
            object_root=Path(args.object_root),
            object_weight=args.object_weight,
            interaction_weight=args.interaction_weight,
        )[: args.top_k]

        baseline_ranked = attach_judgements(baseline_results, judgement_lookup, query_id)
        fused_ranked = attach_judgements(fused_results, judgement_lookup, query_id)
        baseline_metrics = metrics_for_ranked(baseline_ranked)
        fused_metrics = metrics_for_ranked(fused_ranked)
        add_metrics(baseline_totals, baseline_metrics)
        add_metrics(fusion_totals, fused_metrics)

        query_reports.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "baseline": {
                    "metrics": baseline_metrics,
                    "results": baseline_ranked,
                },
                "fusion": {
                    "metrics": fused_metrics,
                    "results": fused_ranked,
                },
                "delta": delta_metrics(fused_metrics, baseline_metrics),
            }
        )

    query_count = len(query_reports)
    payload = {
        "benchmark": args.benchmark,
        "judgements": args.judgements,
        "object_root": args.object_root,
        "object_weight": args.object_weight,
        "interaction_weight": args.interaction_weight,
        "result_set": args.result_set,
        "top_k": args.top_k,
        "query_count": query_count,
        "baseline": average_metrics(baseline_totals, query_count),
        "fusion": average_metrics(fusion_totals, query_count),
        "delta": delta_metrics(average_metrics(fusion_totals, query_count), average_metrics(baseline_totals, query_count)),
        "queries": query_reports,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "object_weight": args.object_weight,
                "interaction_weight": args.interaction_weight,
                "result_set": args.result_set,
                "baseline": payload["baseline"],
                "fusion": payload["fusion"],
                "delta": payload["delta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_judgement_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], str]:
    lookup = {}
    for row in rows:
        lookup[
            (
                row.get("query_id", ""),
                row.get("video_id", ""),
                int(row.get("keyframe_id") or 0),
            )
        ] = row.get("manual_judgement", "").strip().lower()
    return lookup


def attach_judgements(
    results: list[dict[str, Any]],
    judgement_lookup: dict[tuple[str, str, int], str],
    query_id: str,
) -> list[dict[str, Any]]:
    ranked = []
    for rank, result in enumerate(results, start=1):
        video_id = str(result.get("video_id", ""))
        keyframe_id = int(result.get("keyframe_id") or 0)
        item = dict(result)
        item["rank"] = rank
        item["manual_judgement"] = judgement_lookup.get((query_id, video_id, keyframe_id), "")
        ranked.append(item)
    return ranked


def metrics_for_ranked(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "good_at_k": 0.0,
            "good_or_partial_at_k": 0.0,
            "bad_at_k": 0.0,
            "mean_relevance_at_k": 0.0,
            "mrr_good": 0.0,
            "ndcg_at_k": 0.0,
            "top1_good": 0.0,
            "top1_bad": 0.0,
            "top1_relevance": 0.0,
        }
    judgements = [str(result.get("manual_judgement", "")).lower() for result in results]
    k = len(judgements)
    relevance = [relevance_score(item) for item in judgements]
    return {
        "good_at_k": sum(1 for item in judgements if item == "good") / k,
        "good_or_partial_at_k": sum(1 for item in judgements if item in {"good", "partial"}) / k,
        "bad_at_k": sum(1 for item in judgements if item == "bad") / k,
        "mean_relevance_at_k": sum(relevance) / k,
        "mrr_good": reciprocal_rank(judgements, "good"),
        "ndcg_at_k": ndcg(relevance),
        "top1_good": 1.0 if judgements[0] == "good" else 0.0,
        "top1_bad": 1.0 if judgements[0] == "bad" else 0.0,
        "top1_relevance": relevance[0],
    }


def empty_metrics() -> dict[str, float]:
    return defaultdict(float)


def add_metrics(total: dict[str, float], item: dict[str, float]) -> None:
    for key, value in item.items():
        total[key] += value


def average_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        return {}
    return {key: value / count for key, value in total.items()}


def delta_metrics(new: dict[str, float], old: dict[str, float]) -> dict[str, float]:
    return {key: new.get(key, 0.0) - old.get(key, 0.0) for key in sorted(set(new) | set(old))}


def relevance_score(judgement: str) -> float:
    if judgement == "good":
        return 1.0
    if judgement == "partial":
        return 0.5
    return 0.0


def reciprocal_rank(judgements: list[str], target: str) -> float:
    for index, judgement in enumerate(judgements, start=1):
        if judgement == target:
            return 1.0 / index
    return 0.0


def ndcg(relevance: list[float]) -> float:
    if not relevance:
        return 0.0
    dcg = sum(value / log2_rank(index) for index, value in enumerate(relevance, start=1))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(value / log2_rank(index) for index, value in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def log2_rank(rank: int) -> float:
    import math

    return math.log2(rank + 1)


if __name__ == "__main__":
    raise SystemExit(main())
