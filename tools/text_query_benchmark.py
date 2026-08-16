from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.search import (
    aggregate_results_by_video,
    build_numpy_index,
    diversify_results_by_video,
    load_numpy_index,
    load_registry,
    search_numpy_index,
    validate_numpy_index,
)
from aic_retrieval.provenance import sha256_file
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a batch text-query benchmark over the NumPy CLIP index.")
    parser.add_argument("--queries", default="benchmarks/text_queries_l21.json")
    parser.add_argument("--registry", default="artifacts/registry/data_registry.json")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--allow-stale-index", action="store_true", help="Allow an index whose provenance cannot be verified; debug only.")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--require-keyframes", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=25)
    parser.add_argument("--max-frames-per-video", type=int, default=1)
    parser.add_argument("--matched-frames-per-video", type=int, default=5)
    parser.add_argument("--aggregation-method", choices=["max", "mean_top_n"], default="max")
    parser.add_argument("--mean-top-n", type=int, default=3)
    parser.add_argument("--clip-model-id", default=os.environ.get("AIC_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID))
    parser.add_argument("--clip-cache-dir", default=os.environ.get("AIC_CLIP_CACHE_DIR"))
    parser.add_argument("--clip-local-files-only", action="store_true")
    parser.add_argument("--output", default="artifacts/benchmarks/l21_text_benchmark.json")
    parser.add_argument("--csv-output", default="artifacts/benchmarks/l21_text_benchmark.csv")
    parser.add_argument(
        "--csv-result-set",
        choices=["results", "raw_results"],
        default="results",
        help="Which result list to export to CSV for manual judgement.",
    )
    args = parser.parse_args()

    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    queries = load_queries(Path(args.queries))

    start = time.perf_counter()
    registry = None
    index_source = "rebuilt"
    index_metadata = {}
    if args.index_dir:
        index, refs, index_metadata = load_numpy_index(Path(args.index_dir))
        validate_numpy_index(
            index,
            refs,
            index_metadata,
            requested_groups=groups,
            require_keyframes=args.require_keyframes,
            registry_path=Path(args.registry),
            repo_root=ROOT,
            allow_stale_index=args.allow_stale_index,
        )
        index_source = args.index_dir
    else:
        registry = load_registry(Path(args.registry))
        index, refs = build_numpy_index(registry, repo_root=ROOT, groups=groups, require_keyframes=args.require_keyframes)
    index_ready_ms = (time.perf_counter() - start) * 1000

    encoder_start = time.perf_counter()
    encoder = ClipTextEncoder(
        model_id=args.clip_model_id,
        cache_dir=Path(args.clip_cache_dir) if args.clip_cache_dir else None,
        local_files_only=args.clip_local_files_only,
    )
    query_vectors = encoder.encode_texts([query["text"] for query in queries])
    encode_ms = (time.perf_counter() - encoder_start) * 1000

    candidate_pool = max(args.candidate_pool, args.top_k)
    for name, value in (
        ("top-k", args.top_k),
        ("candidate-pool", args.candidate_pool),
        ("max-frames-per-video", args.max_frames_per_video),
        ("matched-frames-per-video", args.matched_frames_per_video),
        ("mean-top-n", args.mean_top_n),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    query_results = []
    rows = []
    search_total_ms = 0.0
    retrieval_total_ms = 0.0
    aggregation_total_ms = 0.0
    for query, query_vector in zip(queries, query_vectors):
        search_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        raw_results = search_numpy_index(index, refs, query_vector, top_k=candidate_pool)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        aggregation_start = time.perf_counter()
        results = diversify_results_by_video(raw_results, args.max_frames_per_video)[: args.top_k]
        video_results = aggregate_results_by_video(
            raw_results,
            max_frames_per_video=args.matched_frames_per_video,
            aggregation_method=args.aggregation_method,
            mean_top_n=args.mean_top_n,
        )[: args.top_k]
        aggregation_ms = (time.perf_counter() - aggregation_start) * 1000
        search_ms = (time.perf_counter() - search_start) * 1000
        search_total_ms += search_ms
        retrieval_total_ms += retrieval_ms
        aggregation_total_ms += aggregation_ms

        query_payload = {
            "id": query["id"],
            "text": query["text"],
            "notes": query.get("notes", ""),
            "query_type": query.get("query_type", "unknown"),
            "language": query.get("language", "unknown"),
            "search_ms": search_ms,
            "retrieval_ms": retrieval_ms,
            "aggregation_ms": aggregation_ms,
            "raw_results": [asdict(result) for result in raw_results],
            "results": [asdict(result) for result in results],
            "video_results": [asdict(result) for result in video_results],
            "video_groups": [asdict(result) for result in video_results],
        }
        query_results.append(query_payload)
        csv_results = raw_results if args.csv_result_set == "raw_results" else results
        for result in csv_results:
            rows.append(
                {
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "rank": result.rank,
                    "score": result.score,
                    "video_id": result.video_id,
                    "keyframe_id": result.keyframe_id,
                    "frame_idx": result.frame_idx,
                    "pts_time": result.pts_time,
                    "keyframe_path": result.keyframe_path or "",
                    "manual_judgement": "",
                    "manual_notes": "",
                }
            )

    elapsed_ms = (time.perf_counter() - start) * 1000
    payload = {
        "benchmark_version": "3.0",
        "benchmark_kind": "MANUAL_RELEVANCE_BENCHMARK — not an official AIC score",
        "query_set_fingerprint": sha256_file(Path(args.queries)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "queries_path": str(Path(args.queries)),
        "query_count": len(queries),
        "groups": sorted(groups),
        "top_k": args.top_k,
        "candidate_pool": candidate_pool,
        "csv_result_set": args.csv_result_set,
        "max_frames_per_video": args.max_frames_per_video,
        "matched_frames_per_video": args.matched_frames_per_video,
        "aggregation_method": args.aggregation_method,
        "mean_top_n": args.mean_top_n,
        "index_source": index_source,
        "index_metadata": index_metadata,
        "index_vectors": int(index.shape[0]),
        "index_dim": int(index.shape[1]),
        "clip_model_id": args.clip_model_id,
        "clip_cache_dir": args.clip_cache_dir,
        "clip_local_files_only": args.clip_local_files_only,
        "index_ready_ms": index_ready_ms,
        "encode_ms": encode_ms,
        "search_total_ms": search_total_ms,
        "retrieval_total_ms": retrieval_total_ms,
        "aggregation_total_ms": aggregation_total_ms,
        "search_latency_ms": latency_summary([item["search_ms"] for item in query_results]),
        "retrieval_latency_ms": latency_summary([item["retrieval_ms"] for item in query_results]),
        "aggregation_latency_ms": latency_summary([item["aggregation_ms"] for item in query_results]),
        "elapsed_ms": elapsed_ms,
        "queries": query_results,
    }

    write_json(Path(args.output), payload)
    write_csv(Path(args.csv_output), rows)
    summary = {
        "query_count": len(queries),
        "output": args.output,
        "csv_output": args.csv_output,
        "index_vectors": int(index.shape[0]),
        "index_dim": int(index.shape[1]),
        "encode_ms": encode_ms,
        "search_total_ms": search_total_ms,
        "retrieval_total_ms": retrieval_total_ms,
        "aggregation_total_ms": aggregation_total_ms,
        "elapsed_ms": elapsed_ms,
        "top1": [
            {
                "query_id": item["id"],
                "video_id": item["results"][0]["video_id"] if item["results"] else None,
                "keyframe_id": item["results"][0]["keyframe_id"] if item["results"] else None,
                "score": item["results"][0]["score"] if item["results"] else None,
            }
            for item in query_results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_queries(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("queries file must contain a JSON list")

    queries: list[dict[str, str]] = []
    seen_ids = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"query #{index} must be an object")
        query_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not query_id:
            raise ValueError(f"query #{index} is missing id")
        if not text:
            raise ValueError(f"query {query_id} is missing text")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query id: {query_id}")
        seen_ids.add(query_id)
        queries.append(
            {
                "id": query_id,
                "text": text,
                "notes": str(item.get("notes", "")).strip(),
                "query_type": str(item.get("query_type", "unknown")).strip() or "unknown",
                "language": str(item.get("language", "unknown")).strip() or "unknown",
            }
        )
    return queries


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_text",
        "rank",
        "score",
        "video_id",
        "keyframe_id",
        "frame_idx",
        "pts_time",
        "keyframe_path",
        "manual_judgement",
        "manual_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
    }


def percentile(values: list[float], fraction: float) -> float:
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]


if __name__ == "__main__":
    raise SystemExit(main())
