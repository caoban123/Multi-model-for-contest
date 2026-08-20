from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any


PROFILE_VERSION = "phase8-profile-v1"
STAGES = ("planner", "embedding", "retrieval", "fusion", "temporal_filter", "window_build", "dense_decode", "alignment", "rerank", "total")
ALIASES = {"plan": "planner", "search": "retrieval", "align": "alignment"}


def profile_store(path: Path, *, algorithm: str | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "store": str(path),
        "algorithm_filter": algorithm,
        "session_count": 0,
        "warm_session_count": 0,
        "stage_latency_ms": {stage: _summary([]) for stage in STAGES},
        "warm_stage_latency_ms": {stage: _summary([]) for stage in STAGES},
        "quality_claim": None,
        "notes": [],
    }
    if not path.is_file():
        output["availability"] = "unavailable"
        output["notes"].append("TRAKE store is missing.")
        return output
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT state_json FROM sessions ORDER BY created_at,session_id").fetchall()
    states = [json.loads(row[0]) for row in rows]
    if algorithm is not None:
        states = [state for state in states if state.get("algorithm") == algorithm]
    samples = [_normalized_timings(state.get("stage_timings_ms", {})) for state in states]
    warm = samples[1:] if len(samples) > 1 else []
    output["availability"] = "available" if samples else "unavailable"
    output["session_count"] = len(samples)
    output["warm_session_count"] = len(warm)
    for stage in STAGES:
        output["stage_latency_ms"][stage] = _summary([sample[stage] for sample in samples if sample.get(stage) is not None])
        output["warm_stage_latency_ms"][stage] = _summary([sample[stage] for sample in warm if sample.get(stage) is not None])
    output["notes"].append("Warm statistics omit the first persisted session; latency evidence does not imply retrieval quality.")
    return output


def render_markdown(profile: dict[str, Any]) -> str:
    lines = ["# Phase 8 TRAKE latency profile", "", f"- Sessions: {profile['session_count']}", f"- Warm sessions: {profile['warm_session_count']}", f"- Algorithm filter: `{profile.get('algorithm_filter')}`", "", "| Stage | p50 ms | p95 ms | n | Warm p50 ms | Warm p95 ms | Warm n |", "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        all_values = profile["stage_latency_ms"][stage]
        warm = profile["warm_stage_latency_ms"][stage]
        lines.append(f"| {stage} | {_display(all_values['p50'])} | {_display(all_values['p95'])} | {all_values['sample_count']} | {_display(warm['p50'])} | {_display(warm['p95'])} | {warm['sample_count']} |")
    lines.extend(["", "Quality claim: `null` — this profile measures latency only."])
    return "\n".join(lines) + "\n"


def _normalized_timings(raw: dict[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {stage: None for stage in STAGES}
    for key, value in raw.items():
        target = ALIASES.get(key, key)
        if target in values and value is not None:
            values[target] = float(value)
    if values["total"] is None:
        recorded = [value for stage, value in values.items() if stage != "total" and value is not None]
        values["total"] = sum(recorded) if recorded else None
    return values


def _summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered) if ordered else None,
        "p95": ordered[max(0, int((len(ordered) * 0.95) + 0.999999) - 1)] if ordered else None,
        "sample_count": len(ordered),
    }


def _display(value: float | None) -> str:
    return "null" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile persisted TRAKE stage latencies read-only.")
    parser.add_argument("--trake-store", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("legacy", "trake_v2"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    profile = profile_store(args.trake_store, algorithm=args.algorithm)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(profile), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
