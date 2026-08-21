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

from aic_retrieval.clip_retriever import ClipRetriever
from aic_retrieval.retrieval_benchmark import load_benchmark_queries, run_individual_benchmark, write_benchmark_result
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CLIP through the generic retriever contract.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry" / "data_registry.json")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_numpy")
    parser.add_argument("--model-id", default=os.environ.get("AIC_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ["AIC_CLIP_CACHE_DIR"]) if os.environ.get("AIC_CLIP_CACHE_DIR") else None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--groups", default="L21", help="Comma-separated groups included in benchmark requests.")
    parser.add_argument("--require-keyframes", action="store_true")
    parser.add_argument("--document-top-k", type=int, default=200)
    parser.add_argument("--video-top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    if not groups:
        parser.error("--groups must contain at least one group")
    definition, queries = load_benchmark_queries(args.queries)
    load_started = time.perf_counter()
    encoder = ClipTextEncoder(args.model_id, args.cache_dir, args.local_files_only)
    retriever = ClipRetriever(
        ROOT,
        args.index_dir,
        args.registry,
        encoder,
        groups=groups,
        require_keyframes=args.require_keyframes,
    )
    load_ms = (time.perf_counter() - load_started) * 1000
    result = run_individual_benchmark(
        retriever,
        queries,
        groups=groups,
        document_top_k=args.document_top_k,
        video_top_k=args.video_top_k,
    )
    result["groups"] = list(groups)
    result["require_keyframes"] = args.require_keyframes
    result["benchmark_definition"] = {key: definition.get(key) for key in ("version", "status", "description")}
    result["load_ms"] = round(load_ms, 3)
    result["index_metadata"] = retriever.metadata
    write_benchmark_result(args.output, result, args.queries)
    print(json.dumps({"output": str(args.output), "load_ms": result["load_ms"], "metrics": result["metrics"], "latency_ms": result["latency_ms"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
