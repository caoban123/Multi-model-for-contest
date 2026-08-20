from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


BENCHMARK_VERSION = "phase8-benchmark-v2"
STAGE_KEYS = (
    "planner",
    "embedding",
    "retrieval",
    "fusion",
    "temporal_filter",
    "window_build",
    "dense_decode",
    "alignment",
    "rerank",
    "total",
)
METRIC_KEYS = (
    "event_recall_at_20",
    "event_recall_at_50",
    "event_recall_at_100",
    "video_recall_at_5",
    "video_recall_at_10",
    "video_recall_at_20",
    "chain_success_rate",
    "correct_video_top1",
    "correct_video_top5",
    "required_event_coverage",
    "temporal_order_accuracy",
    "mean_absolute_timestamp_error_seconds",
    "median_absolute_timestamp_error_seconds",
    "event_within_1s",
    "event_within_3s",
    "event_within_5s",
    "manual_correction_rate",
    "latency_ms_median",
    "latency_ms_p95",
    # Backward-compatible aliases. Timestamp metrics above are authoritative.
    "video_top1_accuracy",
    "video_topk_recall",
    "mean_frame_distance",
    "frame_tolerance_accuracy",
)

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "event_recall_at_20": {"definition": "Required labelled events with a same-video candidate inside the labelled PTS interval among the first 20 event candidates.", "unit": "ratio", "denominator": "required events with timestamp labels", "null_behavior": "null when no required timestamp-labelled events are evaluable"},
    "event_recall_at_50": {"definition": "Required labelled events with a same-video candidate inside the labelled PTS interval among the first 50 event candidates.", "unit": "ratio", "denominator": "required events with timestamp labels", "null_behavior": "null when no required timestamp-labelled events are evaluable"},
    "event_recall_at_100": {"definition": "Required labelled events with a same-video candidate inside the labelled PTS interval among the first 100 event candidates.", "unit": "ratio", "denominator": "required events with timestamp labels", "null_behavior": "null when no required timestamp-labelled events are evaluable"},
    "video_recall_at_5": {"definition": "Expected video appears in the first 5 grouped candidate videos before alignment.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "null when no labelled queries are evaluable"},
    "video_recall_at_10": {"definition": "Expected video appears in the first 10 grouped candidate videos before alignment.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "null when no labelled queries are evaluable"},
    "video_recall_at_20": {"definition": "Expected video appears in the first 20 grouped candidate videos before alignment.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "null when no labelled queries are evaluable"},
    "chain_success_rate": {"definition": "The final chain uses the expected video and every required event is selected inside its labelled PTS interval.", "unit": "ratio", "denominator": "queries whose required events all have timestamp labels", "null_behavior": "null when no fully timestamp-labelled required chains are evaluable"},
    "correct_video_top1": {"definition": "Top-ranked final chain uses the expected video.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "null when no labelled queries are evaluable"},
    "correct_video_top5": {"definition": "Expected video occurs in the first 5 ranked chains.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "null when no labelled queries are evaluable"},
    "required_event_coverage": {"definition": "Selected required events divided by total required events; optional events are excluded.", "unit": "ratio", "denominator": "required events in labelled queries", "null_behavior": "null when a query has no required events"},
    "temporal_order_accuracy": {"definition": "A final chain exists and its selected events have valid monotonic PTS order; no-chain is false.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "null when no labelled queries are evaluable"},
    "mean_absolute_timestamp_error_seconds": {"definition": "Mean distance from predicted PTS to the labelled interval; distance is zero inside the interval and otherwise to the nearest boundary.", "unit": "seconds", "denominator": "selected required event predictions on the expected video with timestamp labels", "null_behavior": "null when no comparable timestamp prediction exists"},
    "median_absolute_timestamp_error_seconds": {"definition": "Median distance from predicted PTS to the labelled interval.", "unit": "seconds", "denominator": "selected required event predictions on the expected video with timestamp labels", "null_behavior": "null when no comparable timestamp prediction exists"},
    "event_within_1s": {"definition": "Required timestamp-labelled events predicted on the expected video within 1 second of the labelled interval; missing/wrong-video predictions fail.", "unit": "ratio", "denominator": "required events with timestamp labels", "null_behavior": "null when no required timestamp-labelled events are evaluable"},
    "event_within_3s": {"definition": "Required timestamp-labelled events predicted on the expected video within 3 seconds of the labelled interval; missing/wrong-video predictions fail.", "unit": "ratio", "denominator": "required events with timestamp labels", "null_behavior": "null when no required timestamp-labelled events are evaluable"},
    "event_within_5s": {"definition": "Required timestamp-labelled events predicted on the expected video within 5 seconds of the labelled interval; missing/wrong-video predictions fail.", "unit": "ratio", "denominator": "required events with timestamp labels", "null_behavior": "null when no required timestamp-labelled events are evaluable"},
    "manual_correction_rate": {"definition": "Queries whose persisted final state contains a manual chain.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "null when no labelled queries are evaluable"},
    "latency_ms_median": {"definition": "Median recorded total pipeline latency.", "unit": "milliseconds", "denominator": "evaluated queries with recorded stage timing", "null_behavior": "null when no timing is recorded"},
    "latency_ms_p95": {"definition": "Nearest-rank p95 recorded total pipeline latency.", "unit": "milliseconds", "denominator": "evaluated queries with recorded stage timing", "null_behavior": "null when no timing is recorded"},
    "video_top1_accuracy": {"definition": "Deprecated alias of correct_video_top1.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "same as correct_video_top1"},
    "video_topk_recall": {"definition": "Deprecated alias of correct_video_top5.", "unit": "ratio", "denominator": "labelled queries", "null_behavior": "same as correct_video_top5"},
    "mean_frame_distance": {"definition": "Deprecated keyframe-ID debug metric; never the primary temporal metric.", "unit": "keyframe id", "denominator": "legacy expected_event_frames paired with selected events", "null_behavior": "null when legacy keyframe labels are absent"},
    "frame_tolerance_accuracy": {"definition": "Deprecated keyframe-ID debug tolerance metric.", "unit": "ratio", "denominator": "legacy expected_event_frames paired with selected events", "null_behavior": "null when legacy keyframe labels are absent"},
}


def evaluate(
    queries_path: Path,
    store_path: Path,
    *,
    top_k: int = 5,
    frame_tolerance: int = 3,
    config_fingerprint: str | None = None,
) -> dict[str, Any]:
    spec = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = list(spec.get("queries", []))
    labelled = [item for item in queries if _expected_video(item)]
    output: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "query_version": spec.get("version"),
        "split": spec.get("split"),
        "query_count": len(queries),
        "labelled_query_count": len(labelled),
        "availability": "unavailable" if not labelled else "available",
        "quality_claim": None,
        "config_fingerprint_filter": config_fingerprint,
        "metrics": {key: None for key in METRIC_KEYS},
        "metric_definitions": METRIC_DEFINITIONS,
        "stage_latency_ms": {stage: {"p50": None, "p95": None, "sample_count": 0} for stage in STAGE_KEYS},
        "failure_diagnostics": [],
        "ablations": {},
        "notes": [],
    }
    if not labelled:
        output["notes"].append("Metrics are null/unavailable because expected labels are absent; null is not zero.")
        return output
    if not store_path.is_file():
        output["availability"] = "unavailable"
        output["notes"].append("TRAKE SQLite store is missing.")
        return output

    sessions = _load_sessions(store_path)
    if config_fingerprint is not None:
        sessions = [state for state in sessions if state.get("config_fingerprint") == config_fingerprint]
    rows = [_evaluate_query(query, sessions, frame_tolerance=frame_tolerance) for query in labelled]
    output["evaluated_query_count"] = len(rows)
    output["matching_session_count"] = sum(not row["missing_session"] for row in rows)
    output["missing_session_count"] = sum(row["missing_session"] for row in rows)
    output["failure_diagnostics"] = [row["diagnostic"] for row in rows]

    metric_sources = {
        "event_recall_at_20": "event_recall_20",
        "event_recall_at_50": "event_recall_50",
        "event_recall_at_100": "event_recall_100",
        "video_recall_at_5": "video_recall_5",
        "video_recall_at_10": "video_recall_10",
        "video_recall_at_20": "video_recall_20",
        "chain_success_rate": "chain_success",
        "correct_video_top1": "correct_video_top1",
        "correct_video_top5": "correct_video_top5",
        "required_event_coverage": "required_coverage",
        "temporal_order_accuracy": "temporal_order",
        "event_within_1s": "within_1s",
        "event_within_3s": "within_3s",
        "event_within_5s": "within_5s",
        "manual_correction_rate": "manual",
        "mean_frame_distance": "legacy_frame_distance",
        "frame_tolerance_accuracy": "legacy_frame_tolerance",
    }
    for metric, source in metric_sources.items():
        output["metrics"][metric] = _mean(_values(rows, source))

    timestamp_errors = [value for row in rows for value in row["timestamp_errors"]]
    output["metrics"]["mean_absolute_timestamp_error_seconds"] = _mean(timestamp_errors)
    output["metrics"]["median_absolute_timestamp_error_seconds"] = statistics.median(timestamp_errors) if timestamp_errors else None

    total_latencies = _values(rows, "total_latency")
    output["metrics"]["latency_ms_median"] = statistics.median(total_latencies) if total_latencies else None
    output["metrics"]["latency_ms_p95"] = _percentile(total_latencies, 0.95) if total_latencies else None
    output["metrics"]["video_top1_accuracy"] = output["metrics"]["correct_video_top1"]
    output["metrics"]["video_topk_recall"] = _mean(_values(rows, f"correct_video_top{top_k}")) if top_k != 5 else output["metrics"]["correct_video_top5"]

    for stage in STAGE_KEYS:
        values = [row["latencies"][stage] for row in rows if row["latencies"].get(stage) is not None]
        output["stage_latency_ms"][stage] = {
            "p50": statistics.median(values) if values else None,
            "p95": _percentile(values, 0.95) if values else None,
            "sample_count": len(values),
        }
    if output["missing_session_count"]:
        output["notes"].append("Labelled queries without persisted sessions are counted as query/event failures, not silently omitted.")
    output["notes"].append("Timestamp metrics use PTS seconds. Keyframe-ID metrics are deprecated debug-only aliases.")
    return output


def _evaluate_query(query: dict[str, Any], sessions: list[dict[str, Any]], *, frame_tolerance: int) -> dict[str, Any]:
    query_id = str(query.get("query_id") or "")
    expected_video = _expected_video(query)
    matching = [state for state in sessions if state.get("request", {}).get("query_id") == query_id]
    labels = _event_labels(query)
    required_labels = [label for label in labels if label["required"]]
    timestamp_labels = [label for label in required_labels if label["interval"] is not None]
    missing_session = not matching
    state = matching[-1] if matching else {}

    ranked = [chain for result in state.get("alignments", []) for chain in result.get("chains", [])]
    ranked.sort(key=lambda chain: (-float(chain.get("score", {}).get("final_score", 0.0)), str(chain.get("chain_id", ""))))
    manual = state.get("manual_chain")
    best = manual or (ranked[0] if ranked else None)
    ranked_for_video = ([manual] if manual else []) + [chain for chain in ranked if not manual or chain.get("chain_id") != manual.get("chain_id")]
    selected_by_event = {
        str(entry.get("event_id")): entry.get("candidate")
        for entry in (best or {}).get("events", [])
        if entry.get("candidate") is not None
    }
    selected_required = sum(label["event_id"] in selected_by_event for label in required_labels)
    required_coverage = selected_required / len(required_labels) if required_labels else None

    event_pools = {str(pool.get("event", {}).get("event_id")): list(pool.get("candidates", [])) for pool in state.get("pools", [])}
    event_hits: dict[int, list[bool]] = {20: [], 50: [], 100: []}
    timestamp_errors: list[float] = []
    tolerance_hits: dict[float, list[bool]] = {1.0: [], 3.0: [], 5.0: []}
    for label in timestamp_labels:
        pool = event_pools.get(label["event_id"], [])
        for cutoff in event_hits:
            event_hits[cutoff].append(any(_candidate_matches(candidate, expected_video, label["interval"]) for candidate in pool[:cutoff]))
        selected = selected_by_event.get(label["event_id"])
        comparable = bool(selected and str(selected.get("video_id")) == expected_video)
        error = _timestamp_distance(float(selected["pts_time"]), label["interval"]) if comparable else None
        if error is not None:
            timestamp_errors.append(error)
        for tolerance in tolerance_hits:
            tolerance_hits[tolerance].append(error is not None and error <= tolerance)

    video_ids = [str(item.get("video_id")) for item in state.get("videos", [])]
    correct_video_top1 = bool(best and str(best.get("video_id")) == expected_video)
    requested_top_k_values = {1, 5}
    correct_video_at = {cutoff: any(str(chain.get("video_id")) == expected_video for chain in ranked_for_video[:cutoff]) for cutoff in requested_top_k_values}
    selected_times = [float(entry["candidate"]["pts_time"]) for entry in (best or {}).get("events", []) if entry.get("candidate")]
    allow_same = bool(state.get("request", {}).get("constraints", {}).get("allow_same_frame", False))
    temporal_order = bool(best) and all(left <= right if allow_same else left < right for left, right in zip(selected_times, selected_times[1:]))

    fully_labelled = bool(required_labels) and len(timestamp_labels) == len(required_labels)
    chain_success = None
    if fully_labelled:
        chain_success = correct_video_top1 and all(
            label["event_id"] in selected_by_event
            and _candidate_matches(selected_by_event[label["event_id"]], expected_video, label["interval"])
            for label in required_labels
        )

    legacy_expected = list(query.get("expected_event_frames") or [])
    legacy_selected = [entry.get("candidate") for entry in (best or {}).get("events", []) if entry.get("candidate")]
    legacy_distances = [abs(int(actual["keyframe_id"]) - int(expected)) for actual, expected in zip(legacy_selected, legacy_expected)]
    latencies = _stage_latencies(state.get("stage_timings_ms", {}))
    diagnostic = _diagnostic(
        query_id,
        missing_session=missing_session,
        event_recall_100=_mean(event_hits[100]),
        video_recall_20=expected_video in video_ids[:20],
        has_chain=bool(best),
        correct_video=correct_video_top1,
        required_coverage=required_coverage,
        timestamp_errors=timestamp_errors,
    )
    return {
        "missing_session": missing_session,
        "event_recall_20": _mean(event_hits[20]),
        "event_recall_50": _mean(event_hits[50]),
        "event_recall_100": _mean(event_hits[100]),
        "video_recall_5": expected_video in video_ids[:5],
        "video_recall_10": expected_video in video_ids[:10],
        "video_recall_20": expected_video in video_ids[:20],
        "correct_video_top1": correct_video_top1,
        "correct_video_top5": correct_video_at[5],
        "correct_video_top1_dynamic": correct_video_at[1],
        "required_coverage": required_coverage,
        "temporal_order": temporal_order,
        "chain_success": chain_success,
        "within_1s": _mean(tolerance_hits[1.0]),
        "within_3s": _mean(tolerance_hits[3.0]),
        "within_5s": _mean(tolerance_hits[5.0]),
        "timestamp_errors": timestamp_errors,
        "manual": bool(manual),
        "legacy_frame_distance": _mean(legacy_distances),
        "legacy_frame_tolerance": _mean([distance <= frame_tolerance for distance in legacy_distances]),
        "latencies": latencies,
        "total_latency": latencies.get("total"),
        "diagnostic": diagnostic,
        **{f"correct_video_top{cutoff}": value for cutoff, value in correct_video_at.items()},
    }


def _event_labels(query: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for index, event in enumerate(query.get("events") or [], start=1):
        event_id = str(event.get("event_id") or f"e{index}")
        start = event.get("start_pts")
        end = event.get("end_pts")
        representative = event.get("representative_pts")
        interval: tuple[float, float] | None = None
        if start is not None or end is not None:
            left = float(start if start is not None else end)
            right = float(end if end is not None else start)
            interval = (min(left, right), max(left, right))
        elif representative is not None:
            value = float(representative)
            interval = (value, value)
        labels.append({"event_id": event_id, "required": bool(event.get("required", True)), "interval": interval})
    return labels


def _candidate_matches(candidate: dict[str, Any], expected_video: str, interval: tuple[float, float] | None) -> bool:
    return bool(interval is not None and str(candidate.get("video_id")) == expected_video and candidate.get("pts_time") is not None and _timestamp_distance(float(candidate["pts_time"]), interval) == 0.0)


def _timestamp_distance(value: float, interval: tuple[float, float]) -> float:
    start, end = interval
    if start <= value <= end:
        return 0.0
    return start - value if value < start else value - end


def _stage_latencies(raw: dict[str, Any]) -> dict[str, float | None]:
    aliases = {"plan": "planner", "search": "retrieval", "align": "alignment"}
    values: dict[str, float | None] = {stage: None for stage in STAGE_KEYS}
    for key, value in raw.items():
        target = aliases.get(key, key)
        if target in values and value is not None:
            values[target] = float(value)
    recorded = [value for key, value in values.items() if key != "total" and value is not None]
    values["total"] = float(raw["total"]) if raw.get("total") is not None else (sum(recorded) if recorded else None)
    return values


def _diagnostic(
    query_id: str,
    *,
    missing_session: bool,
    event_recall_100: float | None,
    video_recall_20: bool,
    has_chain: bool,
    correct_video: bool,
    required_coverage: float | None,
    timestamp_errors: list[float],
) -> dict[str, Any]:
    stage = reason = None
    if missing_session:
        reason = "SESSION_NOT_PERSISTED"
    elif event_recall_100 is not None and event_recall_100 < 1.0:
        stage, reason = "EVENT_RECALL_FAILURE", "A_REQUIRED_EVENT_DID_NOT_SURVIVE_TOP_100"
    elif not video_recall_20:
        stage, reason = "VIDEO_RECALL_FAILURE", "EXPECTED_VIDEO_DID_NOT_SURVIVE_TOP_20"
    elif not has_chain:
        stage, reason = "TEMPORAL_FEASIBILITY_FAILURE", "NO_SELECTED_CHAIN"
    elif not correct_video:
        stage, reason = "ALIGNMENT_FAILURE", "TOP_CHAIN_USES_WRONG_VIDEO"
    elif required_coverage is not None and required_coverage < 1.0:
        stage, reason = "ALIGNMENT_FAILURE", "REQUIRED_EVENT_MISSING_FROM_CHAIN"
    elif timestamp_errors and max(timestamp_errors) > 5.0:
        stage, reason = "FRAME_GRANULARITY_FAILURE", "EVENT_TIMESTAMP_ERROR_EXCEEDS_5_SECONDS"
    return {"query_id": query_id, "failure_stage": stage, "failure_reason": reason}


def _expected_video(query: dict[str, Any]) -> str:
    return str(query.get("expected_video_id") or query.get("video_id") or "").strip()


def _load_sessions(path: Path) -> list[dict[str, Any]]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT state_json FROM sessions ORDER BY updated_at").fetchall()
    return [json.loads(row[0]) for row in rows]


def _values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _mean(values: Iterable[float | bool]) -> float | None:
    items = list(values)
    return statistics.mean(items) if items else None


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--trake-store", type=Path, default=ROOT / "artifacts/trake/phase8_trake.sqlite3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate(args.queries, args.trake_store)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
