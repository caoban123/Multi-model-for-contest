from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.metadata import audit_media_info, audit_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit media-info metadata encoding and field coverage.")
    parser.add_argument("--media-dir", default="data/media-info")
    parser.add_argument("--output", default="artifacts/metadata/media_info_audit.json")
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args()

    audit = audit_media_info(Path(args.media_dir), sample_limit=args.sample_limit)
    payload = audit_to_dict(audit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
