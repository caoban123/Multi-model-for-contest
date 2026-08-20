from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VALIDATION_VERSION = "phase8-annotation-validator-v1"
CATEGORIES = {
    "simple_visual", "2_event", "3_event", "4_event", "5_event",
    "generic_to_distinctive", "distinctive_to_generic", "OCR", "ASR",
    "object", "attribute_color", "short_gap", "medium_gap", "long_gap",
    "same_scene", "cross_scene", "ambiguous", "OOD_rare_entity",
}


def validate_annotations(path: Path, config_path: Path | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[dict[str, str]] = []
    split = str(payload.get("split") or "")
    status = str(payload.get("annotation_status") or "")
    queries = list(payload.get("queries") or ())
    if split not in {"development", "holdout"}:
        errors.append({"path": "split", "message": "split must be development or holdout"})
    if status not in {"UNLABELLED_TEMPLATE", "UNLABELLED_FROZEN_TEMPLATE", "IN_PROGRESS", "LABELLED"}:
        errors.append({"path": "annotation_status", "message": "unsupported annotation status"})
    if split == "holdout" and status == "LABELLED" and config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not bool(config.get("benchmark", {}).get("development_config_frozen")):
            errors.append({"path": "benchmark.development_config_frozen", "message": "holdout labels/evaluation require a frozen development config"})
    labelled_count = 0
    for query_index, query in enumerate(queries):
        prefix = f"queries[{query_index}]"
        expected_video = query.get("expected_video_id")
        if expected_video is None:
            if status == "LABELLED":
                errors.append({"path": f"{prefix}.expected_video_id", "message": "labelled query requires an expected video"})
            continue
        labelled_count += 1
        if not re.fullmatch(r"L21_V\d{3}", str(expected_video)):
            errors.append({"path": f"{prefix}.expected_video_id", "message": "expected video must match L21_V###"})
        if not str(query.get("query") or "").strip():
            errors.append({"path": f"{prefix}.query", "message": "labelled query text is empty"})
        categories = list(query.get("categories") or ())
        unknown = sorted(set(categories) - CATEGORIES)
        if not categories or unknown:
            errors.append({"path": f"{prefix}.categories", "message": f"categories are missing or unsupported: {unknown}"})
        events = list(query.get("events") or ())
        if not 1 <= len(events) <= 5:
            errors.append({"path": f"{prefix}.events", "message": "TRAKE requires 1–5 annotated events"})
        seen_ids: set[str] = set()
        previous_start: float | None = None
        for event_index, event in enumerate(events):
            event_prefix = f"{prefix}.events[{event_index}]"
            event_id = str(event.get("event_id") or "")
            if not event_id or event_id in seen_ids:
                errors.append({"path": f"{event_prefix}.event_id", "message": "event IDs must be non-empty and unique"})
            seen_ids.add(event_id)
            interval = _interval(event)
            if bool(event.get("required", True)) and interval is None:
                errors.append({"path": event_prefix, "message": "required event needs a PTS interval or representative PTS"})
                continue
            if interval is not None:
                start, end = interval
                if start < 0 or end < start:
                    errors.append({"path": event_prefix, "message": "PTS interval must be non-negative and ordered"})
                if previous_start is not None and start < previous_start:
                    errors.append({"path": event_prefix, "message": "event intervals must follow query order"})
                previous_start = start
        if not str(query.get("annotator") or "").strip() or not query.get("annotated_at"):
            errors.append({"path": prefix, "message": "labelled query requires annotator and annotated_at"})
    return {
        "validation_version": VALIDATION_VERSION,
        "path": str(path),
        "split": split,
        "annotation_status": status,
        "query_count": len(queries),
        "labelled_query_count": labelled_count,
        "valid": not errors,
        "errors": errors,
        "quality_claim": None,
    }


def _interval(event: dict[str, Any]) -> tuple[float, float] | None:
    start = event.get("start_pts")
    end = event.get("end_pts")
    representative = event.get("representative_pts")
    if start is None and end is None and representative is None:
        return None
    if start is None and end is None:
        value = float(representative)
        return value, value
    left = float(start if start is not None else end)
    right = float(end if end is not None else start)
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manually authored TRAKE annotations.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--trake-config", type=Path)
    args = parser.parse_args()
    result = validate_annotations(args.annotations, args.trake_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
