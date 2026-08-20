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
from aic_retrieval.retrievers import RetrievalRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the opt-in L21 BM25 index.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bm25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--video", action="append", dest="videos")
    args = parser.parse_args()
    filters = {}
    if args.sources:
        filters["source_types"] = tuple(args.sources)
    if args.videos:
        filters["video_ids"] = tuple(args.videos)
    retriever = Bm25Retriever(ROOT, args.index_dir)
    hits = retriever.search(RetrievalRequest("cli", args.query, top_k=args.top_k, filters=filters))
    print(json.dumps({"health": retriever.health().to_dict(), "hits": [hit.to_dict() for hit in hits]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
