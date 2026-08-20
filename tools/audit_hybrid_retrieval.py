from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.hybrid_audit import build_hybrid_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit L21 inputs for modular CLIP/BGE/BM25 retrieval.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--group", default="L21")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "audits" / "l21_hybrid_retrieval_audit.json")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero code when blockers are present.")
    args = parser.parse_args()

    payload = build_hybrid_audit(args.root, group=args.group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output": str(args.output),
        "status": payload["status"],
        "videos": payload["registry"].get("videos", 0),
        "clip_vectors": payload["clip_index"].get("vector_count", 0),
        "ocr_detections": payload["text_sources"]["ocr"].get("detections", 0),
        "asr_segments": payload["text_sources"]["asr"].get("segments", 0),
        "blockers": payload["blockers"],
        "warnings": payload["warnings"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.strict and payload["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
