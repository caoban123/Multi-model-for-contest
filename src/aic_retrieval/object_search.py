from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from aic_retrieval.object_store import AliasDictionary

LEFT_CENTER_BOUNDARY = 0.33
CENTER_RIGHT_BOUNDARY = 0.66


@dataclass(frozen=True)
class Detection:
    label_raw: str
    label_normalized: str
    confidence: float
    bbox_y1: float
    bbox_x1: float
    bbox_y2: float
    bbox_x2: float
    center_x: float
    center_y: float
    area: float


@dataclass(frozen=True)
class ObjectPredicate:
    labels: tuple[str, ...]
    count_operator: str = ">="
    count: int = 1
    horizontal: str = "any"
    vertical: str = "any"


@dataclass(frozen=True)
class ObjectSearchConfig:
    min_confidence: float = 0.3
    nms_iou_threshold: float = 0.5


def iou(first: Detection, second: Detection) -> float:
    y1, x1 = max(first.bbox_y1, second.bbox_y1), max(first.bbox_x1, second.bbox_x1)
    y2, x2 = min(first.bbox_y2, second.bbox_y2), min(first.bbox_x2, second.bbox_x2)
    intersection = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections: Iterable[Detection], threshold: float) -> list[Detection]:
    if not 0 <= threshold <= 1:
        raise ValueError("nms_iou_threshold must be between 0 and 1")
    kept: list[Detection] = []
    ordered = sorted(detections, key=lambda item: (-item.confidence, item.label_normalized, item.bbox_y1, item.bbox_x1, item.bbox_y2, item.bbox_x2))
    for detection in ordered:
        if all(detection.label_normalized != existing.label_normalized or iou(detection, existing) <= threshold for existing in kept):
            kept.append(detection)
    return kept


def horizontal_region(center_x: float) -> str:
    if center_x < LEFT_CENTER_BOUNDARY:
        return "left"
    if center_x < CENTER_RIGHT_BOUNDARY:
        return "center"
    return "right"


def vertical_region(center_y: float) -> str:
    if center_y < LEFT_CENTER_BOUNDARY:
        return "top"
    if center_y < CENTER_RIGHT_BOUNDARY:
        return "middle"
    return "bottom"


def position_matches(detection: Detection, horizontal: str = "any", vertical: str = "any") -> bool:
    if horizontal not in {"any", "left", "center", "right"}:
        raise ValueError(f"unsupported horizontal region: {horizontal}")
    if vertical not in {"any", "top", "middle", "bottom"}:
        raise ValueError(f"unsupported vertical region: {vertical}")
    if detection.center_x is None or detection.center_y is None:
        return False
    return (horizontal == "any" or horizontal_region(detection.center_x) == horizontal) and (vertical == "any" or vertical_region(detection.center_y) == vertical)


def count_matches(actual: int, operator: str, expected: int) -> bool:
    if expected < 0:
        raise ValueError("count must not be negative")
    functions = {"=": lambda: actual == expected, ">=": lambda: actual >= expected, "<=": lambda: actual <= expected}
    if operator not in functions:
        raise ValueError(f"unsupported count operator: {operator}")
    return functions[operator]()


def canonical_predicate(predicate: ObjectPredicate, aliases: AliasDictionary) -> ObjectPredicate:
    labels = tuple(sorted({aliases.normalize(label) for label in predicate.labels if label.strip()}))
    if not labels:
        raise ValueError("at least one object label is required")
    return ObjectPredicate(labels, predicate.count_operator, predicate.count, predicate.horizontal, predicate.vertical)


def row_to_detection(row: sqlite3.Row) -> Detection:
    return Detection(*(row[name] for name in ("label_raw", "label_normalized", "confidence", "bbox_y1", "bbox_x1", "bbox_y2", "bbox_x2", "center_x", "center_y", "area")))


class ObjectSearchService:
    def __init__(self, store_path: Path, aliases: AliasDictionary) -> None:
        self.store_path = store_path
        self.aliases = aliases

    def search(self, predicate: ObjectPredicate, config: ObjectSearchConfig = ObjectSearchConfig()) -> dict[str, Any]:
        if not 0 <= config.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        query = canonical_predicate(predicate, self.aliases)
        connection = sqlite3.connect(self.store_path)
        connection.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in query.labels)
        rows = connection.execute(
            f"SELECT video_id,keyframe_id,label_raw,label_normalized,confidence,bbox_y1,bbox_x1,bbox_y2,bbox_x2,center_x,center_y,area FROM detections WHERE label_normalized IN ({placeholders}) AND confidence >= ? ORDER BY video_id,keyframe_id,confidence DESC,label_raw",
            (*query.labels, config.min_confidence),
        ).fetchall()
        unknown_rows = connection.execute("SELECT video_id,keyframe_id FROM frames WHERE object_status='UNKNOWN' ORDER BY video_id,keyframe_id").fetchall()
        available_rows = connection.execute("SELECT video_id,keyframe_id FROM frames WHERE object_status='AVAILABLE' ORDER BY video_id,keyframe_id").fetchall()
        connection.close()
        grouped: dict[tuple[str, int], list[Detection]] = {}
        for row in rows:
            grouped.setdefault((row["video_id"], row["keyframe_id"]), []).append(row_to_detection(row))
        results = []
        for available in available_rows:
            video_id, keyframe_id = available["video_id"], available["keyframe_id"]
            detections = grouped.get((video_id, keyframe_id), [])
            filtered = [item for item in nms(detections, config.nms_iou_threshold) if position_matches(item, query.horizontal, query.vertical)]
            if not count_matches(len(filtered), query.count_operator, query.count):
                continue
            score = max((item.confidence for item in filtered), default=0.0)
            results.append({
                "video_id": video_id, "keyframe_id": keyframe_id,
                "matched_labels": sorted({item.label_normalized for item in filtered}),
                "confidence": score, "bounding_boxes": [[item.bbox_y1, item.bbox_x1, item.bbox_y2, item.bbox_x2] for item in filtered],
                "object_score": score, "data_status": "MATCH", "matched_count": len(filtered),
                "detections": [asdict(item) for item in filtered],
            })
        results.sort(key=lambda item: (-item["object_score"], item["video_id"], item["keyframe_id"]))
        for rank, item in enumerate(results, start=1):
            item["object_rank"] = rank
        return {
            "predicate": asdict(query), "config": asdict(config), "results": results,
            "unknown_frames": [{"video_id": row["video_id"], "keyframe_id": row["keyframe_id"], "data_status": "UNKNOWN"} for row in unknown_rows],
            "unknown_count": len(unknown_rows), "no_match_semantics": "AVAILABLE frames omitted from results are NO_MATCH",
        }

    def video_data_status(self, video_id: str) -> dict[str, int | str]:
        connection = sqlite3.connect(self.store_path)
        available, unknown = connection.execute(
            "SELECT SUM(object_status='AVAILABLE'), SUM(object_status='UNKNOWN') FROM frames WHERE video_id=?",
            (video_id,),
        ).fetchone()
        connection.close()
        available, unknown = int(available or 0), int(unknown or 0)
        status = "AVAILABLE" if available and not unknown else ("PARTIAL_UNKNOWN" if available else "UNKNOWN")
        return {"status": status, "available_frames": available, "unknown_frames": unknown}

    def phone_near_hand(self, config: ObjectSearchConfig = ObjectSearchConfig(), max_center_distance: float = 0.35) -> dict[str, Any]:
        if max_center_distance <= 0:
            raise ValueError("max_center_distance must be positive")
        connection = sqlite3.connect(self.store_path)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT video_id,keyframe_id,label_raw,label_normalized,confidence,bbox_y1,bbox_x1,bbox_y2,bbox_x2,center_x,center_y,area FROM detections WHERE label_normalized IN ('phone','hand') AND confidence >= ? ORDER BY video_id,keyframe_id,confidence DESC",
            (config.min_confidence,),
        ).fetchall()
        connection.close()
        grouped: dict[tuple[str, int], list[Detection]] = {}
        for row in rows:
            grouped.setdefault((row["video_id"], row["keyframe_id"]), []).append(row_to_detection(row))
        results = []
        for (video_id, keyframe_id), items in grouped.items():
            kept = nms(items, config.nms_iou_threshold)
            phones, hands = [item for item in kept if item.label_normalized == "phone"], [item for item in kept if item.label_normalized == "hand"]
            pairs = []
            for phone in phones:
                for hand in hands:
                    distance = math.hypot(phone.center_x - hand.center_x, phone.center_y - hand.center_y)
                    if distance <= max_center_distance:
                        proximity = max(0.0, 1.0 - distance / max_center_distance)
                        pairs.append({"phone_confidence": phone.confidence, "hand_confidence": hand.confidence, "center_distance": distance, "proximity_score": proximity,
                                      "phone_bbox": [phone.bbox_y1, phone.bbox_x1, phone.bbox_y2, phone.bbox_x2], "hand_bbox": [hand.bbox_y1, hand.bbox_x1, hand.bbox_y2, hand.bbox_x2]})
            if pairs:
                best = max(pairs, key=lambda item: (item["proximity_score"], item["phone_confidence"], item["hand_confidence"]))
                results.append({"video_id": video_id, "keyframe_id": keyframe_id, "object_score": best["proximity_score"] * math.sqrt(best["phone_confidence"] * best["hand_confidence"]),
                                "data_status": "MATCH", "interaction": "phone_near_hand", "claim": "heuristic_proximity_not_holding_fact", "evidence": best})
        results.sort(key=lambda item: (-item["object_score"], item["video_id"], item["keyframe_id"]))
        for rank, item in enumerate(results, start=1): item["object_rank"] = rank
        return {"interaction": "phone_near_hand", "experimental": True, "max_center_distance": max_center_distance, "config": asdict(config), "results": results}
