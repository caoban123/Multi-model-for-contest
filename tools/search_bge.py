from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.bge_retriever import BgeEncoder, BgeRetriever
from aic_retrieval.retrievers import RetrievalRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the opt-in L21 BGE index.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bge")
    parser.add_argument("--model-id", default=os.environ.get("AIC_BGE_MODEL_ID", "BAAI/bge-m3"))
    parser.add_argument("--model-path", default=os.environ.get("AIC_BGE_MODEL_PATH"))
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("AIC_BGE_CACHE_DIR"))
    parser.add_argument("--device", default=os.environ.get("AIC_BGE_DEVICE"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source-types", default="")
    args = parser.parse_args()
    encoder = BgeEncoder(args.model_id, model_path=args.model_path, cache_dir=args.cache_dir, local_files_only=True, device=args.device)
    retriever = BgeRetriever(ROOT, args.index_dir, encoder)
    source_types = tuple(item.strip() for item in args.source_types.split(",") if item.strip())
    request = RetrievalRequest("cli", args.query, ("L21",), args.top_k, {"source_types": source_types} if source_types else {})
    print(json.dumps({"health": retriever.health().to_dict(), "results": [item.to_dict() for item in retriever.search(request)]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
