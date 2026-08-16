from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.provenance import fingerprint_paths, sha256_file
from aic_retrieval.search import aggregate_results_by_video, load_numpy_index, search_numpy_index, validate_numpy_index
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder
from tools.text_query_benchmark import latency_summary


QUERY_TYPES = {
    "general_semantic", "object_specific", "object_count", "object_position", "interaction",
    "title", "channel", "date", "metadata_context", "attribute_color",
}


def load_phase4_queries(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("Phase 4 query file must contain an object with a queries list")
    version = str(payload.get("query_set_version", "")).strip()
    if not version:
        raise ValueError("query_set_version is required")
    seen: set[str] = set()
    queries: list[dict[str, Any]] = []
    for number, raw in enumerate(payload["queries"], start=1):
        required = {"query_id", "query_text_vi", "query_text_en", "query_type", "split", "expected_video_ids", "structured_constraints"}
        missing = sorted(required - set(raw)) if isinstance(raw, dict) else sorted(required)
        if missing:
            raise ValueError(f"query #{number} missing fields: {', '.join(missing)}")
        query_id = str(raw["query_id"]).strip()
        if not query_id or query_id in seen:
            raise ValueError(f"duplicate or empty query_id: {query_id}")
        if raw["query_type"] not in QUERY_TYPES:
            raise ValueError(f"unsupported query_type: {raw['query_type']}")
        if raw["split"] not in {"development", "holdout"}:
            raise ValueError(f"unsupported split: {raw['split']}")
        if not isinstance(raw["expected_video_ids"], list) or not isinstance(raw["structured_constraints"], dict):
            raise ValueError(f"invalid labels/constraints for {query_id}")
        if not str(raw["query_text_en"]).strip() or not str(raw["query_text_vi"]).strip():
            raise ValueError(f"query text is required for {query_id}")
        seen.add(query_id)
        queries.append(dict(raw))
    present = {item["query_type"] for item in queries}
    if present != QUERY_TYPES:
        raise ValueError(f"query types mismatch; missing={sorted(QUERY_TYPES - present)}, extra={sorted(present - QUERY_TYPES)}")
    return version, queries


def git_context(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD")}


def stable_signature(results: list[Any]) -> list[tuple[str, int, int, float]]:
    return [(item.video_id, item.keyframe_id, item.frame_idx, round(float(item.score), 8)) for item in results]


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the deterministic Phase 4 CLIP-only baseline.")
    parser.add_argument("--queries", default="benchmarks/phase4_queries_v1.json")
    parser.add_argument("--registry", default="artifacts/registry/data_registry.json")
    parser.add_argument("--index-dir", default="artifacts/indexes/l21_numpy")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--candidate-pool", type=int, default=40)
    parser.add_argument("--aggregation-method", choices=["max", "mean_top_n"], default="max")
    parser.add_argument("--mean-top-n", type=int, default=3)
    parser.add_argument("--matched-frames-per-video", type=int, default=5)
    parser.add_argument("--determinism-runs", type=int, default=2)
    parser.add_argument("--allow-stale-index", action="store_true")
    parser.add_argument("--clip-model-id", default=os.environ.get("AIC_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID))
    parser.add_argument("--clip-cache-dir", default=os.environ.get("AIC_CLIP_CACHE_DIR"))
    parser.add_argument("--clip-local-files-only", action="store_true")
    parser.add_argument("--output", default="artifacts/benchmarks/phase4/clip_baseline_v1.json")
    parser.add_argument("--csv-output", default="artifacts/benchmarks/phase4/clip_baseline_v1.csv")
    args = parser.parse_args()
    for name in ("top_k", "candidate_pool", "mean_top_n", "matched_frames_per_video", "determinism_runs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    query_path, index_dir, registry_path = Path(args.queries), Path(args.index_dir), Path(args.registry)
    version, queries = load_phase4_queries(query_path)
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    index, refs, index_metadata = load_numpy_index(index_dir)
    validate_numpy_index(index, refs, index_metadata, groups, False, registry_path, ROOT, args.allow_stale_index)
    index_files = [index_dir / name for name in ("vectors.npy", "refs.json", "metadata.json")]
    encoder_started = time.perf_counter()
    encoder = ClipTextEncoder(args.clip_model_id, Path(args.clip_cache_dir) if args.clip_cache_dir else None, args.clip_local_files_only)
    model_load_ms = (time.perf_counter() - encoder_started) * 1000
    vectors = []
    encoding_latencies: list[float] = []
    for query in queries:
        encode_started = time.perf_counter()
        vectors.append(encoder.encode_text(query["query_text_en"]))
        encoding_latencies.append((time.perf_counter() - encode_started) * 1000)

    rows: list[dict[str, Any]] = []
    query_payloads: list[dict[str, Any]] = []
    retrieval_latencies: list[float] = []
    aggregation_latencies: list[float] = []
    total_latencies: list[float] = []
    deterministic = True
    pool = max(args.candidate_pool, args.top_k)
    for query, vector in zip(queries, vectors):
        signatures = []
        first_raw = first_videos = None
        run_times = []
        for _ in range(args.determinism_runs):
            total_started = time.perf_counter()
            retrieval_started = time.perf_counter()
            raw = search_numpy_index(index, refs, vector, top_k=pool)
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            aggregation_started = time.perf_counter()
            videos = aggregate_results_by_video(raw, args.matched_frames_per_video, args.aggregation_method, args.mean_top_n)[: args.top_k]
            aggregation_ms = (time.perf_counter() - aggregation_started) * 1000
            total_ms = (time.perf_counter() - total_started) * 1000
            signatures.append(stable_signature(raw))
            run_times.append({"retrieval_ms": retrieval_ms, "aggregation_ms": aggregation_ms, "total_ms": total_ms})
            if first_raw is None:
                first_raw, first_videos = raw, videos
                retrieval_latencies.append(retrieval_ms)
                aggregation_latencies.append(aggregation_ms)
                total_latencies.append(encoding_latencies[len(query_payloads)] + total_ms)
        query_deterministic = all(signature == signatures[0] for signature in signatures[1:])
        deterministic = deterministic and query_deterministic
        result_payload = [asdict(item) for item in first_videos or []]
        query_payloads.append({**query, "clip_query": query["query_text_en"], "deterministic": query_deterministic, "runs": run_times, "video_results": result_payload})
        for item in result_payload:
            rows.append({
                "query_id": query["query_id"], "split": query["split"], "query_type": query["query_type"],
                "query_text_vi": query["query_text_vi"], "query_text_en": query["query_text_en"],
                "rank": item["rank"], "video_id": item["video_id"], "best_keyframe_id": item["best_keyframe_id"],
                "score": item["video_score"], "manual_judgement": "", "manual_notes": "",
            })

    git = git_context(ROOT)
    payload = {
        "benchmark_version": "phase4-clip-baseline-1.0", "query_set_version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(), **git, "mode": "clip_only", "hybrid_enabled": False,
        "query_set_path": str(query_path), "query_set_fingerprint": sha256_file(query_path),
        "query_count": len(queries), "split_counts": {split: sum(q["split"] == split for q in queries) for split in ("development", "holdout")},
        "index_manifest": index_metadata, "index_fingerprint": fingerprint_paths(index_files, ROOT),
        "model": args.clip_model_id, "feature_provenance": index_metadata.get("feature_provenance", "unknown"),
        "config": {"groups": sorted(groups), "candidate_pool": pool, "top_k": args.top_k, "aggregation_method": args.aggregation_method,
                   "mean_top_n": args.mean_top_n, "matched_frames_per_video": args.matched_frames_per_video, "determinism_runs": args.determinism_runs},
        "machine": {"platform": platform.platform(), "python": platform.python_version(), "device": getattr(encoder, "device", "unknown")},
        "index": {"vectors": int(index.shape[0]), "dimension": int(index.shape[1])},
        "determinism": {"passed": deterministic, "runs": args.determinism_runs},
        "latency_ms": {"model_load": model_load_ms, "query_encoding": latency_summary(encoding_latencies),
                       "retrieval": latency_summary(retrieval_latencies), "aggregation": latency_summary(aggregation_latencies), "total": latency_summary(total_latencies)},
        "quality_metrics": {"status": "unavailable", "reason": "No verified relevance ground truth; use exported manual_judgement columns."},
        "queries": query_payloads,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "split", "query_type", "query_text_vi", "query_text_en", "rank", "video_id", "best_keyframe_id", "score", "manual_judgement", "manual_notes"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output": str(output), "csv_output": str(csv_path), "query_count": len(queries), "deterministic": deterministic, "latency_ms": payload["latency_ms"]}, ensure_ascii=False, indent=2))
    return 0 if deterministic else 2


if __name__ == "__main__":
    raise SystemExit(main())
