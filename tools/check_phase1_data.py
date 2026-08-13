from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.phase1_data import load_phase1_manifest, preflight_phase1_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Phase-1 data readiness without downloading or changing data.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "phase1_data_manifest.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry" / "data_registry.json")
    parser.add_argument(
        "--groups",
        default="L21",
        help="Comma-separated Phase-1 scope to validate. Defaults to L21, the local KIS baseline.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    payload = preflight_phase1_data(args.data_root, load_phase1_manifest(args.manifest), args.registry, groups=groups)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if payload["overall_status"] in {"OK", "WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
