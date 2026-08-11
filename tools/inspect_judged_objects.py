from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect object detections for judged benchmark rows.")
    parser.add_argument("--input", default="artifacts/benchmarks/l21_text_benchmark_judged.csv")
    parser.add_argument("--object-root", default="data/objects")
    parser.add_argument("--judgements", default="bad,partial")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--output", default="artifacts/benchmarks/l21_judged_object_inspection.json")
    args = parser.parse_args()

    wanted = {item.strip().lower() for item in args.judgements.split(",") if item.strip()}
    rows = load_csv(Path(args.input))
    inspected = []
    aggregate = Counter()
    for row in rows:
        judgement = row.get("manual_judgement", "").strip().lower()
        if judgement not in wanted:
            continue
        entities = load_top_entities(
            Path(args.object_root),
            row.get("video_id", ""),
            row.get("keyframe_id", ""),
            args.top_n,
        )
        aggregate.update(entity["entity"] for entity in entities)
        inspected.append(
            {
                "query_id": row.get("query_id", ""),
                "query_text": row.get("query_text", ""),
                "rank": int(row.get("rank") or 0),
                "judgement": judgement,
                "video_id": row.get("video_id", ""),
                "keyframe_id": int(row.get("keyframe_id") or 0),
                "keyframe_path": row.get("keyframe_path", ""),
                "objects": entities,
            }
        )

    payload = {
        "input": args.input,
        "judgements": sorted(wanted),
        "rows": len(inspected),
        "top_entities": [{"entity": name, "count": count} for name, count in aggregate.most_common(30)],
        "results": inspected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(inspected), "top_entities": payload["top_entities"][:10]}, ensure_ascii=False, indent=2))
    return 0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_top_entities(object_root: Path, video_id: str, keyframe_id: str, top_n: int) -> list[dict[str, Any]]:
    path = object_root / video_id / f"{int(keyframe_id):03d}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = payload.get("detection_class_entities", [])
    scores = payload.get("detection_scores", [])
    items = []
    for entity, score in zip(entities, scores):
        items.append({"entity": str(entity), "score": float(score)})
    return items[:top_n]


if __name__ == "__main__":
    raise SystemExit(main())
