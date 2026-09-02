from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.competition_profile import (
    build_retrieval_ui_command,
    load_competition_profile,
    preflight_competition_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight and run the AIC L21-L30 competition UI profile.")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs" / "competition_l21_l30.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--allow-online-models",
        action="store_true",
        help="Allow missing local model paths. Not recommended during the contest.",
    )
    args = parser.parse_args()

    profile = load_competition_profile(args.profile)
    preflight = preflight_competition_profile(
        profile,
        ROOT,
        allow_online_models=args.allow_online_models,
    )
    print(json.dumps(preflight.to_dict(), ensure_ascii=False, indent=2))
    if not preflight.ready:
        print("Competition UI preflight is BLOCKED. Fix the errors above before starting.", file=sys.stderr)
        return 2
    if args.check_only:
        return 0

    command = build_retrieval_ui_command(
        profile,
        preflight,
        python_executable=sys.executable,
        repo_root=ROOT,
        host=args.host,
        port=args.port,
    )
    print(f"Starting competition UI at http://{args.host}:{args.port}/submission")
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
