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

from aic_retrieval.bge_retriever import BgeEncoder, build_bge_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an opt-in BGE FAISS index from the canonical L21 text corpus.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts" / "corpora" / "l21_text" / "documents.jsonl")
    parser.add_argument("--corpus-manifest", type=Path, default=ROOT / "artifacts" / "corpora" / "l21_text" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_bge")
    parser.add_argument("--model-id", default=os.environ.get("AIC_BGE_MODEL_ID", "BAAI/bge-m3"))
    parser.add_argument("--model-path", default=os.environ.get("AIC_BGE_MODEL_PATH"))
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("AIC_BGE_CACHE_DIR"))
    parser.add_argument("--device", default=os.environ.get("AIC_BGE_DEVICE"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-documents", type=int, default=512)
    parser.add_argument("--progress", choices=("tqdm", "none"), default="tqdm")
    parser.add_argument("--restart", action="store_true", help="Discard a partial embedding checkpoint and start over.")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    encoder = BgeEncoder(
        args.model_id,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
        device=args.device,
        batch_size=args.batch_size,
    )
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
                    progress_bar = tqdm(total=total, desc="BGE embeddings", unit="doc", dynamic_ncols=True, smoothing=0.1)
                progress_bar.update(max(0, processed - progress_bar.n))
                if processed == total:
                    progress_bar.set_description("BGE: finalizing FAISS", refresh=False)
                progress_bar.refresh()

            progress = report_progress
    try:
        manifest = build_bge_index(
            root=ROOT,
            corpus_path=args.corpus,
            corpus_manifest_path=args.corpus_manifest,
            output_dir=args.output_dir,
            encoder=encoder,
            checkpoint_documents=args.checkpoint_documents,
            progress=progress,
            restart=args.restart,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    print(json.dumps({"output": str(args.output_dir), "model": manifest["model"], "faiss": manifest["faiss"], "timing_ms": manifest["timing_ms"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
