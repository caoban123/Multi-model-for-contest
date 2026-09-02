from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VALIDATION_VERSION = "phase10-qa-annotation-validator-v1"
QUESTION_TYPES = {"OBJECT", "VISUAL", "COUNT", "OCR", "ASR", "METADATA", "TEMPORAL"}
MODALITIES = {"keyframe", "clip", "object", "attribute", "metadata", "ocr", "asr", "temporal"}
STATUSES = {"UNLABELLED_TEMPLATE", "UNLABELLED_FROZEN_TEMPLATE", "IN_PROGRESS", "LABELLED"}


def validate_annotations(path: Path, config_path: Path | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[dict[str, str]] = []
    split = str(payload.get("split") or "")
    status = str(payload.get("annotation_status") or "")
    queries = list(payload.get("queries") or ())
    if split not in {"development", "holdout"}:
        errors.append({"path": "split", "message": "split must be development or holdout"})
    if status not in STATUSES:
        errors.append({"path": "annotation_status", "message": "unsupported annotation status"})
    if split == "holdout" and status == "LABELLED" and config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not bool(config.get("benchmark", {}).get("development_config_frozen")):
            errors.append({"path": "benchmark.development_config_frozen", "message": "holdout labels/evaluation require a frozen development config"})

    labelled_count = 0
    seen_query_ids: set[str] = set()
    for index, query in enumerate(queries):
        prefix = f"queries[{index}]"
        query_id = str(query.get("query_id") or "").strip()
        answerable = not bool(query.get("unanswerable_from_available_evidence", False))
        has_labels = bool(query.get("expected_video_ids") or query.get("expected_answers") or query.get("accepted_pts_ranges") or query.get("accepted_frame_ranges"))
        if status == "LABELLED" or has_labels:
            labelled_count += 1
            if not query_id:
                errors.append({"path": f"{prefix}.query_id", "message": "labelled query requires query_id"})
            elif query_id in seen_query_ids:
                errors.append({"path": f"{prefix}.query_id", "message": "query_id must be unique"})
            seen_query_ids.add(query_id)
            if not str(query.get("event_query_vi") or "").strip() or not str(query.get("question_vi") or "").strip():
                errors.append({"path": prefix, "message": "labelled query requires event_query_vi and question_vi"})
            question_type = str(query.get("question_type") or "")
            if question_type not in QUESTION_TYPES:
                errors.append({"path": f"{prefix}.question_type", "message": "unsupported question_type"})
            videos = list(query.get("expected_video_ids") or ())
            if answerable and not videos:
                errors.append({"path": f"{prefix}.expected_video_ids", "message": "answerable query requires an expected video"})
            for video_id in videos:
                if not re.fullmatch(r"L(?:2[1-9]|30)_V\d{3}", str(video_id)):
                    errors.append({"path": f"{prefix}.expected_video_ids", "message": "video ID must match L21-L30_V###"})
            answers = [str(value).strip() for value in query.get("expected_answers") or () if str(value).strip()]
            if answerable and not answers:
                errors.append({"path": f"{prefix}.expected_answers", "message": "answerable query requires at least one accepted answer"})
            if any(len(answer) > 100 for answer in answers):
                errors.append({"path": f"{prefix}.expected_answers", "message": "accepted answers must fit the 100-character submission limit"})
            modalities = set(query.get("required_modalities") or ())
            if answerable and "keyframe" not in modalities:
                errors.append({"path": f"{prefix}.required_modalities", "message": "answerable query requires keyframe evidence for submission mapping"})
            unknown_modalities = sorted(modalities - MODALITIES)
            if unknown_modalities:
                errors.append({"path": f"{prefix}.required_modalities", "message": f"unsupported modalities: {unknown_modalities}"})
            _validate_ranges(query.get("accepted_pts_ranges") or (), f"{prefix}.accepted_pts_ranges", errors, float)
            _validate_ranges(query.get("accepted_frame_ranges") or (), f"{prefix}.accepted_frame_ranges", errors, int)
            if answerable and not query.get("accepted_pts_ranges") and not query.get("accepted_frame_ranges"):
                errors.append({"path": prefix, "message": "answerable query requires an accepted PTS or official frame range"})
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
        "ready_for_quality_benchmark": status == "LABELLED" and labelled_count == len(queries) and not errors,
        "errors": errors,
        "quality_claim": None,
    }


def _validate_ranges(values: Any, path: str, errors: list[dict[str, str]], caster: type[float] | type[int]) -> None:
    for index, value in enumerate(values):
        try:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError
            start, end = caster(value[0]), caster(value[1])
            if start < 0 or end < start:
                raise ValueError
        except (TypeError, ValueError):
            errors.append({"path": f"{path}[{index}]", "message": "range must be [non-negative start, ordered end]"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manually authored Phase-10 Q&A annotations.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = validate_annotations(args.annotations, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
