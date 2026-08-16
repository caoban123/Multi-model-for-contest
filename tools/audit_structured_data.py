from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.structured_audit import audit_metadata, audit_objects, load_frame_identities


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit object and metadata inputs for Phase 4.")
    parser.add_argument("--refs", default="artifacts/indexes/l21_numpy/refs.json")
    parser.add_argument("--object-root", default="data/objects")
    parser.add_argument("--metadata-dir", default="data/media-info")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--output", default="artifacts/audits/phase4_structured_data_audit.json")
    args = parser.parse_args()
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    frames = load_frame_identities(Path(args.refs), groups)
    payload = {
        "audit_version": "phase4-structured-audit-1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": {"groups": sorted(groups), "frame_registry": args.refs, "object_root": args.object_root, "metadata_dir": args.metadata_dir},
        "object_audit": audit_objects(Path(args.object_root), frames),
        "metadata_audit_scope": audit_metadata(Path(args.metadata_dir), groups),
        "metadata_audit_all_groups": audit_metadata(Path(args.metadata_dir), None),
        "fusion_enabled": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "objects": {key: payload["object_audit"][key] for key in ("videos_total", "videos_with_object_data", "frames_total", "frames_with_object_data", "invalid_json_count", "length_mismatch_count", "invalid_bbox_count")}, "metadata": payload["metadata_audit_all_groups"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
