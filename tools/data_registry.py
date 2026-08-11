from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.registry import scan_data_root, validate_assets, write_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the local AIC data registry.")
    parser.add_argument("--data-root", default="data", help="Local data directory.")
    parser.add_argument(
        "--output",
        default="artifacts/registry/data_registry.json",
        help="Registry JSON output path.",
    )
    parser.add_argument(
        "--validation-output",
        default="artifacts/registry/validation_report.json",
        help="Validation JSON output path.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    assets = scan_data_root(data_root)
    validation = validate_assets(assets)

    write_registry(Path(args.output), data_root, assets, validation)
    validation_path = Path(args.validation_output)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"registry={args.output}")
    print(f"validation={args.validation_output}")
    print(f"total_videos={validation.total_videos}")
    print(f"errors={validation.error_count}")
    print(f"warnings={validation.warning_count}")
    print(f"group_counts={validation.group_counts}")
    print(f"keyframe_group_counts={validation.keyframe_group_counts}")

    return 1 if validation.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

