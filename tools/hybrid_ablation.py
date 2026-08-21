from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.bge_retriever import BgeEncoder, BgeRetriever
from aic_retrieval.bm25_retriever import Bm25Retriever
from aic_retrieval.clip_retriever import ClipRetriever
from aic_retrieval.hybrid_audit import sha256_file
from aic_retrieval.retrieval_benchmark import load_benchmark_queries, run_individual_benchmark
from aic_retrieval.rrf_fusion import HybridRrfRetriever, RrfFusionConfig
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run independent and equal-weight RRF ablations for CLIP, BGE and BM25.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry" / "data_registry.json")
    parser.add_argument("--clip-index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_numpy")
    parser.add_argument("--bge-index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bge")
    parser.add_argument("--bm25-index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bm25")
    parser.add_argument("--clip-model-id", default=os.environ.get("AIC_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID))
    parser.add_argument("--clip-cache-dir", type=Path, default=Path(os.environ["AIC_CLIP_CACHE_DIR"]) if os.environ.get("AIC_CLIP_CACHE_DIR") else None)
    parser.add_argument("--bge-model-id", default=os.environ.get("AIC_BGE_MODEL_ID", "BAAI/bge-m3"))
    parser.add_argument("--bge-model-path", default=os.environ.get("AIC_BGE_MODEL_PATH"))
    parser.add_argument("--device", default=os.environ.get("AIC_BGE_DEVICE"))
    parser.add_argument("--groups", default="L21", help="Comma-separated groups included in benchmark requests.")
    parser.add_argument("--require-keyframes", action="store_true")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-pool", type=int, default=200)
    parser.add_argument("--document-top-k", type=int, default=200)
    parser.add_argument("--video-top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    if not groups:
        parser.error("--groups must contain at least one group")

    load_started = time.perf_counter()
    clip_encoder = ClipTextEncoder(args.clip_model_id, args.clip_cache_dir, True)
    clip = ClipRetriever(
        ROOT,
        args.clip_index_dir,
        args.registry,
        clip_encoder,
        groups=groups,
        require_keyframes=args.require_keyframes,
    )
    bge_encoder = BgeEncoder(args.bge_model_id, model_path=args.bge_model_path, local_files_only=True, device=args.device)
    bge = BgeRetriever(ROOT, args.bge_index_dir, bge_encoder)
    bm25 = Bm25Retriever(ROOT, args.bm25_index_dir)
    load_ms = (time.perf_counter() - load_started) * 1000
    definition, queries = load_benchmark_queries(args.queries)

    retrievers = {"clip": clip, "bge": bge, "bm25": bm25}
    profiles = {
        "clip": ("clip",),
        "bge": ("bge",),
        "bm25": ("bm25",),
        "clip_bge_rrf": ("clip", "bge"),
        "clip_bm25_rrf": ("clip", "bm25"),
        "bge_bm25_rrf": ("bge", "bm25"),
        "clip_bge_bm25_rrf": ("clip", "bge", "bm25"),
    }
    runs = {}
    for profile_name, names in profiles.items():
        if len(names) == 1:
            retriever = retrievers[names[0]]
        else:
            retriever = HybridRrfRetriever(
                tuple(retrievers[name] for name in names),
                RrfFusionConfig(rrf_k=args.rrf_k, candidate_pool=args.candidate_pool),
            )
        result = run_individual_benchmark(
            retriever,
            queries,
            groups=groups,
            document_top_k=args.document_top_k,
            video_top_k=args.video_top_k,
        )
        runs[profile_name] = result

    baseline_mrr = float(runs["clip"]["metrics"]["mrr"])
    summary = {
        name: {
            "retrievers": list(profiles[name]),
            "metrics": result["metrics"],
            "latency_ms": result["latency_ms"],
            "delta_mrr_vs_clip": float(result["metrics"]["mrr"]) - baseline_mrr,
        }
        for name, result in runs.items()
    }
    payload = {
        "schema_version": "hybrid-ablation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {key: definition.get(key) for key in ("version", "status", "description")},
        "query_path": str(args.queries),
        "query_sha256": sha256_file(args.queries),
        "query_count": len(queries),
        "load_ms": round(load_ms, 3),
        "config": {
            "fusion": "equal_weight_video_rrf",
            "rrf_k": args.rrf_k,
            "candidate_pool": args.candidate_pool,
            "document_top_k": args.document_top_k,
            "video_top_k": args.video_top_k,
            "groups": list(groups),
            "require_keyframes": args.require_keyframes,
        },
        "summary": summary,
        "runs": runs,
        "promotion_decision": {
            "eligible": False,
            "reason": "This benchmark contains partial CLIP-pooled judgments and cannot independently support default promotion.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "load_ms": payload["load_ms"], "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
