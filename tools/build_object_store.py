from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.object_store import build_object_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 4 normalized SQLite object store.")
    parser.add_argument("--refs", default="artifacts/indexes/l21_numpy/refs.json")
    parser.add_argument("--object-root", default="data/objects")
    parser.add_argument("--aliases", default="config/object_aliases_v1.json")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--output", default="artifacts/structured/l21_objects.sqlite")
    parser.add_argument("--manifest", default="artifacts/structured/l21_objects_manifest.json")
    args = parser.parse_args()
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    manifest = build_object_store(Path(args.refs), Path(args.object_root), Path(args.aliases), Path(args.output), groups)
    manifest["store_path"] = args.output
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
