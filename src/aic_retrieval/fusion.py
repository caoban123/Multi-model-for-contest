from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OBJECT_RULES = {
    "phone": {
        "terms": ["phone", "mobile"],
        "positive_entities": {
            "Mobile phone": 1.0,
            "Telephone": 0.7,
            "Human hand": 0.35,
            "Human arm": 0.2,
        },
        "ignored_entities": {"Person", "Clothing", "Man", "Woman", "Human face"},
    },
    "vehicle": {
        "terms": ["car", "motorbike", "motorcycle", "road", "street", "vehicle"],
        "positive_entities": {
            "Car": 1.0,
            "Motorcycle": 1.0,
            "Vehicle": 0.7,
            "Land vehicle": 0.7,
            "Wheel": 0.45,
            "Traffic sign": 0.25,
            "Truck": 0.55,
            "Bus": 0.55,
        },
        "ignored_entities": {"Person", "Clothing", "Man", "Woman", "Human face"},
    },
    "table_food": {
        "terms": ["table", "food", "plate"],
        "positive_entities": {
            "Table": 1.0,
            "Food": 1.0,
            "Plate": 1.0,
            "Dishware": 0.75,
            "Bowl": 0.65,
            "Drink": 0.35,
        },
        "ignored_entities": {"Person", "Clothing", "Man", "Woman", "Human face"},
    },
    "screen": {
        "terms": ["computer", "television", "screen", "monitor"],
        "positive_entities": {
            "Computer monitor": 1.0,
            "Television": 1.0,
            "Laptop": 0.75,
            "Computer keyboard": 0.5,
            "Mobile phone": 0.25,
        },
        "ignored_entities": {"Person", "Clothing", "Man", "Woman", "Human face"},
    },
    "kitchen": {
        "terms": ["kitchen"],
        "positive_entities": {
            "Kitchen appliance": 1.0,
            "Refrigerator": 0.9,
            "Oven": 0.9,
            "Microwave oven": 0.9,
            "Sink": 0.8,
            "Countertop": 0.7,
            "Kitchen & dining room table": 0.55,
            "Table": 0.35,
        },
        "ignored_entities": {"Person", "Clothing", "Man", "Woman", "Human face"},
    },
}


@dataclass(frozen=True)
class FusedResult:
    rank: int
    baseline_rank: int
    video_id: str
    keyframe_id: int
    clip_score: float
    object_score: float
    fused_score: float
    matched_rules: list[str]
    matched_entities: list[dict[str, Any]]
    keyframe_path: str


def rerank_results_with_objects(
    query_text: str,
    results: list[dict[str, Any]],
    object_root: Path,
    object_weight: float = 0.05,
    max_object_score: float = 1.0,
) -> list[dict[str, Any]]:
    fused: list[FusedResult] = []
    for baseline_rank, result in enumerate(results, start=1):
        object_score, matched_rules, matched_entities = score_objects_for_query(
            query_text,
            object_root,
            str(result.get("video_id", "")),
            int(result.get("keyframe_id") or 0),
            max_object_score=max_object_score,
        )
        clip_score = float(result.get("score", 0.0))
        fused.append(
            FusedResult(
                rank=0,
                baseline_rank=baseline_rank,
                video_id=str(result.get("video_id", "")),
                keyframe_id=int(result.get("keyframe_id") or 0),
                clip_score=clip_score,
                object_score=object_score,
                fused_score=clip_score + (object_weight * object_score),
                matched_rules=matched_rules,
                matched_entities=matched_entities,
                keyframe_path=str(result.get("keyframe_path") or ""),
            )
        )

    ranked = sorted(fused, key=lambda item: (-item.fused_score, item.baseline_rank))
    return [asdict(_with_rank(item, rank)) for rank, item in enumerate(ranked, start=1)]


def score_objects_for_query(
    query_text: str,
    object_root: Path,
    video_id: str,
    keyframe_id: int,
    max_object_score: float = 1.0,
) -> tuple[float, list[str], list[dict[str, Any]]]:
    active_rules = matching_rules(query_text)
    if not active_rules:
        return 0.0, [], []

    detections = load_object_detections(object_root, video_id, keyframe_id)
    score = 0.0
    matched_entities = []
    for entity, detection_score in detections:
        for rule_name in active_rules:
            rule = DEFAULT_OBJECT_RULES[rule_name]
            if entity in rule["ignored_entities"]:
                continue
            weight = rule["positive_entities"].get(entity)
            if weight is None:
                continue
            contribution = float(detection_score) * float(weight)
            score += contribution
            matched_entities.append(
                {
                    "rule": rule_name,
                    "entity": entity,
                    "detection_score": float(detection_score),
                    "weight": float(weight),
                    "contribution": contribution,
                }
            )
    return min(score, max_object_score), active_rules, matched_entities


def matching_rules(query_text: str) -> list[str]:
    normalized = query_text.lower()
    matches = []
    for rule_name, rule in DEFAULT_OBJECT_RULES.items():
        if any(term in normalized for term in rule["terms"]):
            matches.append(rule_name)
    return matches


def load_object_detections(object_root: Path, video_id: str, keyframe_id: int) -> list[tuple[str, float]]:
    path = object_root / video_id / f"{keyframe_id:03d}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = payload.get("detection_class_entities", [])
    scores = payload.get("detection_scores", [])
    return [(str(entity), float(score)) for entity, score in zip(entities, scores)]


def _with_rank(result: FusedResult, rank: int) -> FusedResult:
    return FusedResult(
        rank=rank,
        baseline_rank=result.baseline_rank,
        video_id=result.video_id,
        keyframe_id=result.keyframe_id,
        clip_score=result.clip_score,
        object_score=result.object_score,
        fused_score=result.fused_score,
        matched_rules=result.matched_rules,
        matched_entities=result.matched_entities,
        keyframe_path=result.keyframe_path,
    )
