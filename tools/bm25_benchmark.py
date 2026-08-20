from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.bm25_retriever import Bm25Retriever
from aic_retrieval.retrieval_benchmark import load_benchmark_queries, run_individual_benchmark, write_benchmark_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible BM25-only benchmark over the L21 text corpus.")
    parser.add_argument("--queries", type=Path, default=ROOT / "benchmarks" / "hybrid_retrieval_smoke_v1.json")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bm25")
    parser.add_argument("--document-top-k", type=int, default=100)
    parser.add_argument("--video-top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "hybrid" / "bm25_smoke_v1.json")
    args = parser.parse_args()
    definition, queries = load_benchmark_queries(args.queries)
    retriever = Bm25Retriever(ROOT, args.index_dir)
    result = run_individual_benchmark(retriever, queries, document_top_k=args.document_top_k, video_top_k=args.video_top_k)
    result["benchmark_definition"] = {
        "version": definition.get("version"),
        "status": definition.get("status"),
        "description": definition.get("description"),
    }
    result["index_manifest"] = retriever.manifest
    write_benchmark_result(args.output, result, args.queries)
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"], "latency_ms": result["latency_ms"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
