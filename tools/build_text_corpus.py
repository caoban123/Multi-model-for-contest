from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.text_corpus import CorpusBuildConfig, build_text_corpus, write_text_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic text corpus for BGE and BM25 retrievers.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--group", default="L21")
    parser.add_argument("--object-min-confidence", type=float, default=0.3)
    parser.add_argument("--object-max-labels-per-frame", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "corpora" / "l21_text")
    args = parser.parse_args()
    config = CorpusBuildConfig(args.group, args.object_min_confidence, args.object_max_labels_per_frame)
    documents, manifest = build_text_corpus(args.root, config)
    result = write_text_corpus(documents, manifest, args.output_dir)
    print(json.dumps({**result, "source_counts": manifest["source_counts"], "warnings": manifest["warnings"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
