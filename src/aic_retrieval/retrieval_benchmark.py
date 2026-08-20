from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from aic_retrieval.hybrid_audit import sha256_file
from aic_retrieval.retrievers import RetrievalRequest, Retriever


BENCHMARK_RESULT_SCHEMA = "individual-retriever-benchmark-v1"


def load_benchmark_queries(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("benchmark must contain a queries list")
    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, raw in enumerate(payload["queries"], 1):
        if not isinstance(raw, dict):
            raise ValueError(f"query #{number} must be an object")
        query_id = str(raw.get("query_id", raw.get("id", ""))).strip()
        query_text = str(raw.get("query_text", raw.get("query", raw.get("text", "")))).strip()
        if not query_id or not query_text or query_id in seen:
            raise ValueError(f"invalid or duplicate benchmark query at position {number}")
        seen.add(query_id)
        expected = tuple(str(item) for item in raw.get("expected_video_ids", ()) if str(item))
        source_types = tuple(str(item) for item in raw.get("source_types", ()) if str(item))
        queries.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "expected_video_ids": expected,
                "source_types": source_types,
                "category": str(raw.get("category", "unknown")),
                "language": str(raw.get("language", "unknown")),
            }
        )
    return payload, queries


def _video_results(hits: Iterable[Any]) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.video_id in seen:
            continue
        seen.add(hit.video_id)
        videos.append(
            {
                "video_id": hit.video_id,
                "document_rank": hit.rank,
                "score": hit.raw_score,
                "source_type": hit.source_type,
                "document_id": hit.document_id,
                "keyframe_id": hit.keyframe_id,
                "frame_idx": hit.frame_idx,
                "pts_time": hit.pts_time,
                "matched_text": hit.matched_text,
            }
        )
    return videos


def _first_expected_rank(videos: list[dict[str, Any]], expected: set[str]) -> int | None:
    return next((rank for rank, item in enumerate(videos, 1) if item["video_id"] in expected), None)


def _ndcg(videos: list[dict[str, Any]], expected: set[str], k: int) -> float | None:
    if not expected:
        return None
    gains = [1.0 if item["video_id"] in expected else 0.0 for item in videos[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def run_individual_benchmark(
    retriever: Retriever,
    queries: list[dict[str, Any]],
    *,
    groups: tuple[str, ...] = ("L21",),
    document_top_k: int = 100,
    video_top_k: int = 10,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in queries:
        filters = {"source_types": query["source_types"]} if query.get("source_types") else {}
        request = RetrievalRequest(query["query_id"], query["query_text"], groups, document_top_k, filters)
        started = time.perf_counter()
        hits = retriever.search(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        videos = _video_results(hits)[:video_top_k]
        expected = set(query.get("expected_video_ids", ()))
        first_rank = _first_expected_rank(videos, expected) if expected else None
        runs.append(
            {
                **query,
                "latency_ms": round(elapsed_ms, 3),
                "document_hit_count": len(hits),
                "video_results": videos,
                "first_expected_rank": first_rank,
                "reciprocal_rank": 1.0 / first_rank if first_rank else (0.0 if expected else None),
                "recall_at_1": float(first_rank is not None and first_rank <= 1) if expected else None,
                "recall_at_5": float(first_rank is not None and first_rank <= 5) if expected else None,
                "recall_at_10": float(first_rank is not None and first_rank <= 10) if expected else None,
                "ndcg_at_10": _ndcg(videos, expected, 10),
            }
        )
    judged = [run for run in runs if run["expected_video_ids"]]
    metrics = {
        "judged_query_count": len(judged),
        "mrr": statistics.mean(run["reciprocal_rank"] for run in judged) if judged else None,
        "recall_at_1": statistics.mean(run["recall_at_1"] for run in judged) if judged else None,
        "recall_at_5": statistics.mean(run["recall_at_5"] for run in judged) if judged else None,
        "recall_at_10": statistics.mean(run["recall_at_10"] for run in judged) if judged else None,
        "ndcg_at_10": statistics.mean(run["ndcg_at_10"] for run in judged) if judged else None,
    }
    return {
        "schema_version": BENCHMARK_RESULT_SCHEMA,
        "retriever": retriever.name,
        "query_count": len(runs),
        "document_top_k": document_top_k,
        "video_top_k": video_top_k,
        "health": retriever.health().to_dict(),
        "metrics": metrics,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "runs": runs,
    }


def write_benchmark_result(path: Path, payload: dict[str, Any], query_path: Path) -> None:
    payload["query_set_path"] = str(query_path)
    payload["query_set_sha256"] = sha256_file(query_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
