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
    parser.add_argument("--group", help="Single group compatibility option, for example L21.")
    parser.add_argument("--groups", help="Comma-separated groups. Overrides --group when provided.")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--phase5-store", type=Path)
    parser.add_argument("--object-store", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--object-min-confidence", type=float, default=0.3)
    parser.add_argument("--object-max-labels-per-frame", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "corpora" / "l21_text")
    args = parser.parse_args()
    groups = tuple(item.strip() for item in (args.groups or args.group or "L21").split(",") if item.strip())
    config = CorpusBuildConfig(
        group=groups[0],
        groups=groups,
        object_min_confidence=args.object_min_confidence,
        object_max_labels_per_frame=args.object_max_labels_per_frame,
    )
    documents, manifest = build_text_corpus(
        args.root,
        config,
        registry_path=args.registry,
        phase5_store_path=args.phase5_store,
        object_store_path=args.object_store,
        audit_path=args.audit,
    )
    result = write_text_corpus(documents, manifest, args.output_dir)
    print(json.dumps({**result, "source_counts": manifest["source_counts"], "warnings": manifest["warnings"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
