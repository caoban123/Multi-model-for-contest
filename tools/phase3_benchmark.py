from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.search import SearchResult, aggregate_results_by_video


LABEL_VALUE = {"good": 1.0, "partial": 0.5, "bad": 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare raw frame ranking with Phase-3 max-score video ranking."
    )
    parser.add_argument("--input", required=True, help="Text benchmark JSON containing raw_results.")
    parser.add_argument("--judgements-csv", default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--candidate-pool", type=int, default=None)
    parser.add_argument("--matched-frames-per-video", type=int, default=5)
    parser.add_argument("--aggregation-method", choices=["max", "mean_top_n"], default="max")
    parser.add_argument("--mean-top-n", type=int, default=3)
    parser.add_argument("--output", default="artifacts/benchmarks/phase3_comparison.json")
    args = parser.parse_args()

    for name, value in (
        ("top-k", args.top_k),
        ("matched-frames-per-video", args.matched_frames_per_video),
        ("mean-top-n", args.mean_top_n),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.candidate_pool is not None and args.candidate_pool <= 0:
        raise ValueError("candidate-pool must be positive")

    source_path = Path(args.input)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    queries = source.get("queries")
    if not isinstance(queries, list):
        raise ValueError("input benchmark must contain a queries list")
    judgements = load_judgements(Path(args.judgements_csv)) if args.judgements_csv else {}

    query_reports: list[dict[str, Any]] = []
    for query in queries:
        raw_payload = query.get("raw_results")
        if not isinstance(raw_payload, list):
            raise ValueError(f"query {query.get('id', '<unknown>')} has no raw_results list")
        available_pool = len(raw_payload)
        requested_pool = args.candidate_pool or int(source.get("candidate_pool") or available_pool)
        candidate_pool = min(requested_pool, available_pool)
        raw_results = [search_result(item) for item in raw_payload[:candidate_pool]]
        baseline = raw_results[: args.top_k]

        aggregation_started = time.perf_counter()
        video_results = aggregate_results_by_video(
            raw_results,
            max_frames_per_video=args.matched_frames_per_video,
            aggregation_method=args.aggregation_method,
            mean_top_n=args.mean_top_n,
        )[: args.top_k]
        aggregation_ms = (time.perf_counter() - aggregation_started) * 1000
        retrieval_ms = float(query.get("retrieval_ms", query.get("search_ms", 0.0)))
        query_id = str(query.get("id", ""))
        frame_quality = frame_quality_metrics(query_id, baseline, judgements)
        video_quality = video_quality_metrics(query_id, video_results, judgements)
        top_video_rank = next(
            (video.rank for video in video_results if baseline and video.video_id == baseline[0].video_id),
            None,
        )

        query_reports.append(
            {
                "id": query_id,
                "text": query.get("text", ""),
                "query_type": query.get("query_type", "unknown"),
                "language": query.get("language", "unknown"),
                "candidate_pool_size": candidate_pool,
                "frame_ranking": diversity_metrics(baseline, args.top_k),
                "video_ranking": diversity_metrics(video_results, args.top_k),
                "frame_quality": frame_quality,
                "video_quality": video_quality,
                "top_frame_video_rank_after_aggregation": top_video_rank,
                "top_frame_video_preserved_at_rank_1": top_video_rank == 1,
                "retrieval_ms": retrieval_ms,
                "aggregation_ms": aggregation_ms,
                "total_retrieval_and_aggregation_ms": retrieval_ms + aggregation_ms,
                "video_results": [asdict(video) for video in video_results],
            }
        )

    payload = {
        "benchmark_version": "1.0",
        "benchmark_kind": "PHASE_3_FRAME_VS_VIDEO_COMPARISON — not an official AIC score",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(source_path),
        "source_benchmark_version": source.get("benchmark_version"),
        "judgements_csv": args.judgements_csv,
        "quality_status": "AVAILABLE" if judgements else "UNAVAILABLE_NO_MANUAL_JUDGEMENTS",
        "top_k": args.top_k,
        "requested_candidate_pool": args.candidate_pool,
        "aggregation_method": args.aggregation_method,
        "mean_top_n": args.mean_top_n,
        "matched_frames_per_video": args.matched_frames_per_video,
        "query_count": len(query_reports),
        "overall": comparison_summary(query_reports),
        "by_query_type": breakdown(query_reports, "query_type"),
        "by_language": breakdown(query_reports, "language"),
        "queries": query_reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["overall"]}, ensure_ascii=False, indent=2))
    return 0


def search_result(item: dict[str, Any]) -> SearchResult:
    return SearchResult(
        rank=int(item["rank"]),
        score=float(item["score"]),
        video_id=str(item["video_id"]),
        group=str(item.get("group") or str(item["video_id"]).split("_")[0]),
        keyframe_id=int(item["keyframe_id"]),
        frame_idx=int(item["frame_idx"]),
        pts_time=float(item["pts_time"]),
        fps=float(item.get("fps", 0.0)),
        keyframe_path=item.get("keyframe_path") or None,
    )


def diversity_metrics(results: list[Any], requested_k: int) -> dict[str, Any]:
    counts = Counter(item.video_id for item in results)
    unique_count = len(counts)
    return {
        "result_count": len(results),
        "unique_video_count": unique_count,
        "unique_video_at_k": unique_count / requested_k,
        "duplicate_result_count": len(results) - unique_count,
        "max_results_from_same_video": max(counts.values(), default=0),
    }


def load_judgements(path: Path) -> dict[tuple[str, str, int], str]:
    judgements: dict[tuple[str, str, int], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("manual_judgement", "")).strip().lower()
            if label not in LABEL_VALUE:
                continue
            try:
                keyframe_id = int(row["keyframe_id"])
            except (KeyError, TypeError, ValueError):
                continue
            judgements[(str(row.get("query_id", "")), str(row.get("video_id", "")), keyframe_id)] = label
    return judgements


def frame_quality_metrics(
    query_id: str,
    frames: list[SearchResult],
    judgements: dict[tuple[str, str, int], str],
) -> dict[str, Any]:
    labels = [
        judgements.get((query_id, frame.video_id, frame.keyframe_id))
        for frame in frames
    ]
    return quality_metrics(labels)


def video_quality_metrics(query_id: str, videos: list[Any], judgements: dict[tuple[str, str, int], str]) -> dict[str, Any]:
    labels: list[str | None] = []
    for video in videos:
        frame_labels = [
            judgements.get((query_id, frame.video_id, frame.keyframe_id))
            for frame in video.frames
        ]
        known = [label for label in frame_labels if label in LABEL_VALUE]
        labels.append(max(known, key=LABEL_VALUE.get) if known else None)
    return quality_metrics(labels)


def quality_metrics(labels: list[str | None]) -> dict[str, Any]:
    known = [(rank, label) for rank, label in enumerate(labels, start=1) if label in LABEL_VALUE]
    counts = Counter(label for _, label in known)
    return {
        "judged_result_count": len(known),
        "good": counts["good"],
        "partial": counts["partial"],
        "bad": counts["bad"],
        "mean_relevance": (
            statistics.mean(LABEL_VALUE[label] for _, label in known) if known else None
        ),
        "best_relevant_rank": next(
            (rank for rank, label in known if LABEL_VALUE[label] > 0),
            None,
        ),
    }


def comparison_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frame_ranking": aggregate_diversity(reports, "frame_ranking"),
        "video_ranking": aggregate_diversity(reports, "video_ranking"),
        "frame_quality": aggregate_quality(reports, "frame_quality"),
        "video_quality": aggregate_quality(reports, "video_quality"),
        "top_frame_video_preservation_rate": mean(
            [1.0 if report["top_frame_video_preserved_at_rank_1"] else 0.0 for report in reports]
        ),
        "retrieval_latency_ms": latency_summary([report["retrieval_ms"] for report in reports]),
        "aggregation_latency_ms": latency_summary([report["aggregation_ms"] for report in reports]),
        "total_latency_ms": latency_summary(
            [report["total_retrieval_and_aggregation_ms"] for report in reports]
        ),
    }


def aggregate_diversity(reports: list[dict[str, Any]], key: str) -> dict[str, float]:
    metrics = [report[key] for report in reports]
    return {
        "mean_unique_video_at_k": mean([item["unique_video_at_k"] for item in metrics]),
        "mean_duplicate_result_count": mean([item["duplicate_result_count"] for item in metrics]),
        "mean_max_results_from_same_video": mean(
            [item["max_results_from_same_video"] for item in metrics]
        ),
    }


def aggregate_quality(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [report[key] for report in reports]
    judged = sum(item["judged_result_count"] for item in metrics)
    weighted_values = [
        item["mean_relevance"]
        for item in metrics
        for _ in range(item["judged_result_count"])
        if item["mean_relevance"] is not None
    ]
    ranks = [item["best_relevant_rank"] for item in metrics if item["best_relevant_rank"] is not None]
    return {
        "judged_result_count": judged,
        "good": sum(item["good"] for item in metrics),
        "partial": sum(item["partial"] for item in metrics),
        "bad": sum(item["bad"] for item in metrics),
        "mean_relevance": mean(weighted_values) if weighted_values else None,
        "mean_best_relevant_rank": mean(ranks) if ranks else None,
    }


def breakdown(reports: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[str(report.get(field, "unknown"))].append(report)
    return {value: comparison_summary(items) for value, items in sorted(grouped.items())}


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": mean(values),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]


if __name__ == "__main__":
    raise SystemExit(main())
