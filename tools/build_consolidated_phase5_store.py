from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_retrieval.phase5_consolidated import ConsolidatedImportStats, build_consolidated_phase5_store


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map consolidated L21-L30 OCR to official refs and build an atomic Phase 5 SQLite store."
    )
    parser.add_argument("--consolidated-root", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--groups", default="L21,L22,L23,L24,L25,L26,L27,L28,L29,L30")
    parser.add_argument("--base-store", type=Path, help="Existing Phase 5 store whose ASR rows are preserved.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-mapping-distance", type=float, default=5.0)
    parser.add_argument("--max-invalid-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    last_reported = 0

    def progress(stats: ConsolidatedImportStats) -> None:
        nonlocal last_reported
        if stats.source_frames - last_reported < 5_000:
            return
        last_reported = stats.source_frames
        print(
            f"frames={stats.source_frames:,} mapped={stats.mapped_frames:,} "
            f"unmapped={stats.unmapped_frames:,} detections={stats.imported_detections:,}",
            file=sys.stderr,
            flush=True,
        )

    result = build_consolidated_phase5_store(
        ocr_root=args.consolidated_root,
        refs_path=args.refs,
        groups=groups,
        output=args.output,
        base_store=args.base_store,
        manifest_output=args.manifest,
        max_mapping_distance=args.max_mapping_distance,
        max_invalid_rows=args.max_invalid_rows,
        overwrite=args.overwrite,
        progress=progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
