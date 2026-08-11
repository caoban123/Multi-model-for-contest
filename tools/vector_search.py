from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.search import (
    build_numpy_index,
    load_numpy_index,
    load_query_vector,
    load_registry,
    search_numpy_index,
    write_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a NumPy cosine search over CLIP image vectors.")
    parser.add_argument("--registry", default="artifacts/registry/data_registry.json")
    parser.add_argument("--query-vector", required=True, help="Path to a .npy query vector with dim 512.")
    parser.add_argument("--groups", default="L21", help="Comma-separated groups to search, e.g. L21,L22.")
    parser.add_argument("--top-k", type=int, default=10)
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
    index_source = "rebuilt"
    index_metadata = {}
    if args.index_dir:
        index, refs, index_metadata = load_numpy_index(Path(args.index_dir))
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
    query = load_query_vector(Path(args.query_vector))
    results = search_numpy_index(index, refs, query, top_k=args.top_k)
    search_ms = (time.perf_counter() - search_start) * 1000
    elapsed_ms = (time.perf_counter() - start) * 1000

    payload = {
        "groups": sorted(groups),
        "top_k": args.top_k,
        "index_source": index_source,
        "index_metadata": index_metadata,
        "index_vectors": int(index.shape[0]),
        "index_dim": int(index.shape[1]),
        "index_ready_ms": index_ready_ms,
        "search_ms": search_ms,
        "elapsed_ms": elapsed_ms,
        "results": [asdict(result) for result in results],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        write_results(Path(args.output), results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
