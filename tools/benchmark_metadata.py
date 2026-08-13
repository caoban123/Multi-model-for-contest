from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.metadata_search import load_metadata_documents, search_metadata
from aic_retrieval.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark metadata-only video retrieval against a versioned expected-ID query set.")
    parser.add_argument("--queries", type=Path, default=ROOT / "benchmarks" / "metadata_queries.json")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "tests" / "fixtures" / "metadata")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-match", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "metadata_benchmark.json")
    args = parser.parse_args()

    query_set = load_queries(args.queries)
    groups = set(args.groups) if args.groups else None
    docs = load_metadata_documents(args.media_dir, groups=groups)
    query_reports = []
    latencies = []
    for query in query_set["queries"]:
        start = time.perf_counter()
        results = search_metadata(docs, query["query"], top_k=args.top_k, min_match=args.min_match)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        result_ids = [result.video_id for result in results]
        expected = set(query["expected_video_ids"])
        first_rank = next((rank for rank, video_id in enumerate(result_ids, start=1) if video_id in expected), None)
        query_reports.append(
            {
                **query,
                "latency_ms": latency_ms,
                "first_relevant_rank": first_rank,
                "recall_at_1": 1.0 if first_rank and first_rank <= 1 else 0.0,
                "recall_at_5": 1.0 if first_rank and first_rank <= 5 else 0.0,
                "recall_at_10": 1.0 if first_rank and first_rank <= 10 else 0.0,
                "mrr": 1.0 / first_rank if first_rank else 0.0,
                "false_positives_before_first_relevant": result_ids[: first_rank - 1] if first_rank else result_ids,
                "result_video_ids": result_ids,
            }
        )

    payload = {
        "benchmark_version": query_set["benchmark_version"],
        "dataset_scope": query_set.get("dataset_scope", "unknown"),
        "query_set_fingerprint": sha256_file(args.queries),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_mode": "metadata_only_experimental",
        "media_dir": str(args.media_dir),
        "media_fingerprint": metadata_fingerprint(args.media_dir),
        "groups": sorted(groups) if groups else None,
        "top_k": args.top_k,
        "min_match": args.min_match,
        "document_count": len(docs),
        "query_count": len(query_reports),
        "metrics": aggregate_metrics(query_reports),
        "query_type_breakdown": breakdown_by_type(query_reports),
        "latency_ms": latency_summary(latencies),
        "queries": query_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": payload["metrics"], "latency_ms": payload["latency_ms"]}, ensure_ascii=False, indent=2))
    return 0


def load_queries(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("metadata query set must be an object with a queries list")
    required = {"id", "query", "query_type", "expected_video_ids"}
    seen = set()
    for item in payload["queries"]:
        missing = required - set(item)
        if missing:
            raise ValueError(f"metadata query is missing fields: {', '.join(sorted(missing))}")
        if item["id"] in seen:
            raise ValueError(f"duplicate metadata query id: {item['id']}")
        seen.add(item["id"])
    return payload


def aggregate_metrics(reports: list[dict[str, Any]]) -> dict[str, float]:
    if not reports:
        return {"video_recall_at_1": 0.0, "video_recall_at_5": 0.0, "video_recall_at_10": 0.0, "mrr": 0.0}
    count = len(reports)
    return {
        "video_recall_at_1": sum(item["recall_at_1"] for item in reports) / count,
        "video_recall_at_5": sum(item["recall_at_5"] for item in reports) / count,
        "video_recall_at_10": sum(item["recall_at_10"] for item in reports) / count,
        "mrr": sum(item["mrr"] for item in reports) / count,
    }


def breakdown_by_type(reports: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[report["query_type"]].append(report)
    return {query_type: aggregate_metrics(items) for query_type, items in sorted(grouped.items())}


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
    }


def percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def metadata_fingerprint(media_dir: Path) -> str | None:
    if not media_dir.exists():
        return None
    from aic_retrieval.provenance import fingerprint_paths

    return fingerprint_paths(media_dir.glob("*.json"), media_dir)


if __name__ == "__main__":
    raise SystemExit(main())
