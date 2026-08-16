from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.search import (
    aggregate_results_by_video,
    build_numpy_index,
    diversify_results_by_video,
    load_numpy_index,
    load_query_vector_from_asset,
    load_query_vector,
    load_registry,
    search_numpy_index,
    validate_numpy_index,
    write_results,
)
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, encode_clip_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a NumPy cosine search over CLIP image vectors.")
    parser.add_argument("--registry", default="artifacts/registry/data_registry.json")
    parser.add_argument("--query-vector", default=None, help="Path to a .npy query vector with dim 512.")
    parser.add_argument("--query-text", default=None, help="Natural-language text query encoded with CLIP.")
    parser.add_argument("--query-from-video-id", default=None, help="Debug: use a keyframe vector from this video.")
    parser.add_argument("--query-keyframe-id", type=int, default=None, help="Debug: keyframe id for --query-from-video-id.")
    parser.add_argument("--clip-model-id", default=os.environ.get("AIC_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID))
    parser.add_argument(
        "--clip-cache-dir",
        default=os.environ.get("AIC_CLIP_CACHE_DIR"),
        help="Optional HuggingFace cache dir for CLIP model files.",
    )
    parser.add_argument(
        "--allow-stale-index",
        action="store_true",
        help="Explicitly allow a persistent index whose registry/source fingerprint cannot be verified. Intended only for debugging.",
    )
    parser.add_argument("--clip-local-files-only", action="store_true", help="Use only local CLIP files; never download.")
    parser.add_argument("--groups", default="L21", help="Comma-separated groups to search, e.g. L21,L22.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=None,
        help="Raw candidate pool size before grouping/diversity. Defaults to top-k.",
    )
    parser.add_argument("--max-frames-per-video", type=int, default=1)
    parser.add_argument("--matched-frames-per-video", type=int, default=5)
    parser.add_argument("--aggregation-method", choices=["max", "mean_top_n"], default="max")
    parser.add_argument("--mean-top-n", type=int, default=3)
    parser.add_argument("--group-by-video", action="store_true")
    parser.add_argument("--require-keyframes", action="store_true")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Optional persistent NumPy index directory containing vectors.npy and refs.json.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON result output path.")
    args = parser.parse_args()

    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
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
        index, refs = build_numpy_index(
            registry,
            repo_root=ROOT,
            groups=groups,
            require_keyframes=args.require_keyframes,
        )
    index_ready_ms = (time.perf_counter() - start) * 1000

    search_start = time.perf_counter()
    query_source = ""
    if args.query_vector:
        query = load_query_vector(Path(args.query_vector))
        query_source = args.query_vector
    elif args.query_text:
        query = encode_clip_text(
            args.query_text,
            model_id=args.clip_model_id,
            cache_dir=Path(args.clip_cache_dir) if args.clip_cache_dir else None,
            local_files_only=args.clip_local_files_only,
        )
        query_source = f"text:{args.query_text}"
    elif args.query_from_video_id and args.query_keyframe_id is not None:
        if registry is None:
            registry = load_registry(Path(args.registry))
        query = load_query_vector_from_asset(
            registry,
            repo_root=ROOT,
            video_id=args.query_from_video_id,
            keyframe_id=args.query_keyframe_id,
        )
        query_source = f"{args.query_from_video_id}:{args.query_keyframe_id}"
    else:
        raise ValueError("provide --query-vector, --query-text, or both --query-from-video-id and --query-keyframe-id")

    candidate_pool = args.candidate_pool or args.top_k
    for name, value in (
        ("top-k", args.top_k),
        ("candidate-pool", candidate_pool),
        ("max-frames-per-video", args.max_frames_per_video),
        ("matched-frames-per-video", args.matched_frames_per_video),
        ("mean-top-n", args.mean_top_n),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if candidate_pool < args.top_k:
        candidate_pool = args.top_k
    retrieval_start = time.perf_counter()
    raw_results = search_numpy_index(index, refs, query, top_k=candidate_pool)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    aggregation_start = time.perf_counter()
    diversified_results = diversify_results_by_video(raw_results, args.max_frames_per_video)[: args.top_k]
    video_results = aggregate_results_by_video(
        raw_results,
        max_frames_per_video=args.matched_frames_per_video,
        aggregation_method=args.aggregation_method,
        mean_top_n=args.mean_top_n,
    )[: args.top_k]
    aggregation_ms = (time.perf_counter() - aggregation_start) * 1000
    search_ms = (time.perf_counter() - search_start) * 1000
    elapsed_ms = (time.perf_counter() - start) * 1000

    payload = {
        "groups": sorted(groups),
        "top_k": args.top_k,
        "candidate_pool": candidate_pool,
        "max_frames_per_video": args.max_frames_per_video,
        "matched_frames_per_video": args.matched_frames_per_video,
        "aggregation_method": args.aggregation_method,
        "mean_top_n": args.mean_top_n,
        "group_by_video": args.group_by_video,
        "index_source": index_source,
        "index_metadata": index_metadata,
        "index_vectors": int(index.shape[0]),
        "index_dim": int(index.shape[1]),
        "query_source": query_source,
        "index_ready_ms": index_ready_ms,
        "search_ms": search_ms,
        "retrieval_ms": retrieval_ms,
        "aggregation_ms": aggregation_ms,
        "elapsed_ms": elapsed_ms,
        "raw_results": [asdict(result) for result in raw_results],
        "results": [asdict(result) for result in diversified_results],
        "video_results": [asdict(result) for result in video_results],
        "video_groups": [asdict(result) for result in video_results] if args.group_by_video else [],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        write_results(Path(args.output), diversified_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
