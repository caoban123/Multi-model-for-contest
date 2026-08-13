from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.search import build_index_metadata, build_numpy_index, load_registry, save_numpy_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a persistent local NumPy vector index.")
    parser.add_argument("--registry", default="artifacts/registry/data_registry.json")
    parser.add_argument("--groups", default="L21", help="Comma-separated groups to index.")
    parser.add_argument("--require-keyframes", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/indexes/l21_numpy")
    args = parser.parse_args()

    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    start = time.perf_counter()
    registry = load_registry(Path(args.registry))
    index, refs = build_numpy_index(
        registry,
        repo_root=ROOT,
        groups=groups,
        require_keyframes=args.require_keyframes,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    metadata = build_index_metadata(
        registry_path=Path(args.registry),
        registry=registry,
        repo_root=ROOT,
        groups=groups,
        require_keyframes=args.require_keyframes,
        index=index,
        refs=refs,
        build_elapsed_ms=elapsed_ms,
    )
    save_numpy_index(Path(args.output_dir), index, refs, metadata)

    print(json.dumps({"output_dir": args.output_dir, **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
