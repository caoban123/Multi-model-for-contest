from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aic_retrieval.trake_workflow import TrakeWorkflow


def run_fixture(path: Path) -> dict[str, object]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    by_event = fixture["candidates"]
    workflow = TrakeWorkflow(lambda event, _pool: by_event[event.event_id])
    planned = workflow.plan("fixture", fixture["query"], manual_events=fixture["events"])
    session_id = planned["state"]["session_id"]
    workflow.search(session_id, event_pool_size=30, max_per_video=5, video_pool_size=10)
    aligned = workflow.align(session_id)
    chains = aligned["state"]["alignments"][0]["chains"]
    actual = [item["candidate"]["keyframe_id"] for item in chains[0]["events"] if item["candidate"]]
    return {
        "fixture_only": True,
        "pipeline_pass": actual == fixture["expected_keyframe_ids"],
        "expected_keyframe_ids": fixture["expected_keyframe_ids"],
        "actual_keyframe_ids": actual,
        "quality_claim": None,
        "stage_timings_ms": aligned["state"]["stage_timings_ms"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=ROOT / "benchmarks" / "trake_fixture_v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_fixture(args.fixture)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["pipeline_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
