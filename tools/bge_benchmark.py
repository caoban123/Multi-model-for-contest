from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.bge_retriever import BgeEncoder, BgeRetriever
from aic_retrieval.retrieval_benchmark import load_benchmark_queries, run_individual_benchmark, write_benchmark_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible BGE-only benchmark over the L21 text corpus.")
    parser.add_argument("--queries", type=Path, default=ROOT / "benchmarks" / "hybrid_retrieval_smoke_v1.json")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bge")
    parser.add_argument("--model-id", default=os.environ.get("AIC_BGE_MODEL_ID", "BAAI/bge-m3"))
    parser.add_argument("--model-path", default=os.environ.get("AIC_BGE_MODEL_PATH"))
    parser.add_argument("--device", default=os.environ.get("AIC_BGE_DEVICE"))
    parser.add_argument("--groups", default="L21", help="Comma-separated groups included in benchmark requests.")
    parser.add_argument("--document-top-k", type=int, default=100)
    parser.add_argument("--video-top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "hybrid" / "bge_smoke_v1.json")
    args = parser.parse_args()
    groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    if not groups:
        parser.error("--groups must contain at least one group")

    load_started = time.perf_counter()
    encoder = BgeEncoder(args.model_id, model_path=args.model_path, local_files_only=True, device=args.device)
    retriever = BgeRetriever(ROOT, args.index_dir, encoder)
    model_load_ms = (time.perf_counter() - load_started) * 1000
    definition, queries = load_benchmark_queries(args.queries)
    result = run_individual_benchmark(
        retriever,
        queries,
        groups=groups,
        document_top_k=args.document_top_k,
        video_top_k=args.video_top_k,
    )
    result["groups"] = list(groups)
    result["benchmark_definition"] = {
        "version": definition.get("version"),
        "status": definition.get("status"),
        "description": definition.get("description"),
    }
    result["model_load_ms"] = round(model_load_ms, 3)
    result["index_manifest"] = retriever.manifest
    write_benchmark_result(args.output, result, args.queries)
    print(json.dumps({"output": str(args.output), "model_load_ms": result["model_load_ms"], "metrics": result["metrics"], "latency_ms": result["latency_ms"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
