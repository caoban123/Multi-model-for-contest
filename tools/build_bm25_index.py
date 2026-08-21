from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.bm25_retriever import build_bm25_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SQLite FTS5/BM25 index over the canonical L21 text corpus.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts" / "corpora" / "l21_text" / "documents.jsonl")
    parser.add_argument("--corpus-manifest", type=Path, default=ROOT / "artifacts" / "corpora" / "l21_text" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bm25")
    parser.add_argument("--progress", choices=("tqdm", "none"), default="tqdm")
    args = parser.parse_args()
    progress_bar = None
    progress = None
    if args.progress == "tqdm":
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("tqdm is unavailable; continuing without a progress bar.", file=sys.stderr, flush=True)
        else:
            def report_progress(processed: int, total: int) -> None:
                nonlocal progress_bar
                if progress_bar is None:
                    progress_bar = tqdm(total=total, desc="BM25 documents", unit="doc", dynamic_ncols=True, smoothing=0.1)
                progress_bar.update(max(0, processed - progress_bar.n))
                if processed == total:
                    progress_bar.set_description("BM25: finalizing", refresh=False)
                progress_bar.refresh()

            progress = report_progress
    try:
        manifest = build_bm25_index(
            root=ROOT,
            corpus_path=args.corpus,
            corpus_manifest_path=args.corpus_manifest,
            output_dir=args.output_dir,
            progress=progress,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
