from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_retrieval.qa_normalization import AnswerNormalizer
from aic_retrieval.qa_schema import ReviewDecision
from aic_retrieval.qa_store import QaStore


def evaluate(queries: list[dict[str, Any]], observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalizer = AnswerNormalizer()
    runs = []
    judged: list[dict[str, Any]] = []
    confirmed = []
    for query in queries:
        observation = observations.get(query["query_id"])
        expected_videos = set(query.get("expected_video_ids") or [])
        expected_answers = set(query.get("expected_answers") or [])
        expected_range = query.get("expected_frame_range")
        has_labels = bool(expected_videos and expected_answers)
        run = {
            "query_id": query["query_id"], "question_type": query.get("question_type"),
            "has_manual_labels": has_labels, "observation": observation,
            "video_correct": None, "frame_correct": None, "answer_exact": None,
            "answer_normalized_exact": None, "evidence_accuracy": None,
        }
        if observation and observation.get("confirmed"):
            confirmed.append(observation)
        if has_labels:
            observed_video = observation.get("video_id") if observation else None
            observed_frame = observation.get("frame_id") if observation else None
            observed_answer = observation.get("answer") if observation else None
            run["video_correct"] = observed_video in expected_videos
            if expected_range and observed_frame is not None and len(expected_range) == 2:
                run["frame_correct"] = bool(expected_range[0] <= observed_frame <= expected_range[1])
            elif expected_range:
                run["frame_correct"] = False
            run["answer_exact"] = observed_answer in expected_answers
            expected_normalized = {normalizer.normalize(answer).normalized_answer for answer in expected_answers}
            run["answer_normalized_exact"] = normalizer.normalize(observed_answer).normalized_answer in expected_normalized if observed_answer else False
            accepted = set(query.get("accepted_evidence") or [])
            if accepted:
                run["evidence_accuracy"] = bool(observation and set(observation.get("evidence_ids", ())) & accepted)
            judged.append(run)
        runs.append(run)

    def mean(key: str) -> float | None:
        values = [float(item[key]) for item in judged if item.get(key) is not None]
        return round(statistics.mean(values), 6) if values else None

    reviewed = [item for item in observations.values() if item.get("review_decision")]
    metrics = {
        "availability": "available" if judged else "unavailable",
        "labelled_query_count": len(judged),
        "observed_query_count": len(observations),
        "video_accuracy": mean("video_correct"),
        "frame_accuracy": mean("frame_correct"),
        "answer_exact_match": mean("answer_exact"),
        "normalized_answer_exact_match": mean("answer_normalized_exact"),
        "manual_semantic_match": None,
        "evidence_accuracy": mean("evidence_accuracy"),
        "evidence_coverage": round(sum(bool(item.get("evidence_ids")) for item in confirmed) / len(confirmed), 6) if confirmed else None,
        "unsupported_answer_rate": round(sum(not item.get("evidence_ids") for item in confirmed) / len(confirmed), 6) if confirmed else None,
        "review_edit_rate": round(sum(item.get("review_decision") == ReviewDecision.EDITED.value for item in reviewed) / len(reviewed), 6) if reviewed else None,
        "review_reject_rate": round(sum(item.get("review_decision") == ReviewDecision.REJECTED.value for item in reviewed) / len(reviewed), 6) if reviewed else None,
        "latency_ms": {"retrieval": None, "evidence_build": None, "draft_answer": None, "normalization": None, "save": None, "export_record": None},
    }
    return {"metrics": metrics, "runs": runs}


def observations_from_store(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    store = QaStore(path)
    observations: dict[str, dict[str, Any]] = {}
    for summary in store.list_sessions():
        state = store.load_state(summary["session_id"])
        review = state.reviews[-1] if state.reviews else None
        if review is None:
            continue
        by_id = {item.evidence_id: item for item in state.pack.evidence_refs}
        frame_ref = next((by_id[item] for item in review.selected_evidence_refs if item in by_id and by_id[item].keyframe_id is not None), None)
        observations.setdefault(state.session.request.query_id, {
            "session_id": state.session.session_id,
            "review_id": review.review_id,
            "review_decision": review.decision.value,
            "confirmed": review.decision in {ReviewDecision.CONFIRMED, ReviewDecision.EDITED},
            "video_id": frame_ref.video_id if frame_ref else None,
            "frame_id": frame_ref.keyframe_id if frame_ref else None,
            "answer": review.final_answer,
            "evidence_ids": list(review.selected_evidence_refs),
        })
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate auditable Phase-7 Q&A records without fabricating missing labels.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "holdout"], required=True)
    parser.add_argument("--qa-store", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = [item for item in source["queries"] if item["split"] == args.split]
    result = evaluate(queries, observations_from_store(args.qa_store))
    payload = {"version": source["version"], "split": args.split, "query_count": len(queries), "qa_store": str(args.qa_store) if args.qa_store else None, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
