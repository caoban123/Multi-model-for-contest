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
from aic_retrieval.bm25_retriever import Bm25Retriever
from aic_retrieval.retrieval_benchmark import load_benchmark_queries, run_individual_benchmark, write_benchmark_result
from aic_retrieval.rrf_fusion import HybridRrfRetriever, RrfFusionConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark opt-in BGE+BM25 video-level RRF.")
    parser.add_argument("--queries", type=Path, default=ROOT / "benchmarks" / "hybrid_retrieval_smoke_v1.json")
    parser.add_argument("--bge-index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bge")
    parser.add_argument("--bm25-index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bm25")
    parser.add_argument("--model-id", default=os.environ.get("AIC_BGE_MODEL_ID", "BAAI/bge-m3"))
    parser.add_argument("--model-path", default=os.environ.get("AIC_BGE_MODEL_PATH"))
    parser.add_argument("--device", default=os.environ.get("AIC_BGE_DEVICE"))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-pool", type=int, default=200)
    parser.add_argument("--document-top-k", type=int, default=100)
    parser.add_argument("--video-top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "hybrid" / "bge_bm25_rrf_smoke_v1.json")
    args = parser.parse_args()

    definition, queries = load_benchmark_queries(args.queries)
    load_started = time.perf_counter()
    encoder = BgeEncoder(args.model_id, model_path=args.model_path, local_files_only=True, device=args.device)
    bge = BgeRetriever(ROOT, args.bge_index_dir, encoder)
    bm25 = Bm25Retriever(ROOT, args.bm25_index_dir)
    retriever = HybridRrfRetriever(
        (bge, bm25),
        RrfFusionConfig(rrf_k=args.rrf_k, candidate_pool=args.candidate_pool),
    )
    load_ms = (time.perf_counter() - load_started) * 1000
    result = run_individual_benchmark(retriever, queries, document_top_k=args.document_top_k, video_top_k=args.video_top_k)
    result["benchmark_definition"] = {
        "version": definition.get("version"),
        "status": definition.get("status"),
        "description": definition.get("description"),
    }
    result["load_ms"] = round(load_ms, 3)
    result["fusion_config"] = {
        "version": "generic-video-rrf-v1",
        "rrf_k": args.rrf_k,
        "candidate_pool": args.candidate_pool,
        "weights": {"bge": 1.0, "bm25": 1.0},
    }
    write_benchmark_result(args.output, result, args.queries)
    print(json.dumps({"output": str(args.output), "load_ms": result["load_ms"], "metrics": result["metrics"], "latency_ms": result["latency_ms"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
