from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a partial video-level benchmark from manually reviewed UI CSV exports.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "kq" / "1")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "hybrid_reviewed_clip_pool_v1.json")
    args = parser.parse_args()

    grouped: dict[str, dict] = defaultdict(lambda: {"positive_videos": set(), "negative_videos": set(), "files": [], "rows": 0})
    source_hash = hashlib.sha256()
    for path in sorted(args.input_dir.glob("*.csv")):
        payload = path.read_bytes()
        source_hash.update(path.name.encode("utf-8"))
        source_hash.update(hashlib.sha256(payload).digest())
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        query = str(rows[0].get("clip_query", "")).strip()
        if not query:
            continue
        item = grouped[query]
        item["files"].append(path.name)
        item["rows"] += len(rows)
        for row in rows:
            video_id = str(row.get("video_id", "")).strip()
            judgement = str(row.get("manual_judgement", "")).strip().lower()
            if not video_id:
                continue
            if judgement == "good":
                item["positive_videos"].add(video_id)
            elif judgement == "bad":
                item["negative_videos"].add(video_id)

    queries = []
    for number, query in enumerate(sorted(grouped), 1):
        item = grouped[query]
        positives = sorted(item["positive_videos"])
        if not positives:
            continue
        queries.append(
            {
                "query_id": f"reviewed-{number:03d}",
                "query_text": query,
                "expected_video_ids": positives,
                "source_types": [],
                "category": "reviewed_visual_pool",
                "language": "en",
                "review": {
                    "csv_files": item["files"],
                    "reviewed_rows": item["rows"],
                    "positive_video_count": len(positives),
                    "negative_video_ids": sorted(item["negative_videos"] - item["positive_videos"]),
                },
            }
        )
    output = {
        "version": "hybrid-reviewed-clip-pool-v1",
        "status": "PARTIAL_JUDGMENTS_NOT_PROMOTION_GROUND_TRUTH",
        "description": "Video positives derived from manual review of CLIP-pooled UI exports. Unseen videos are unjudged, not negative.",
        "source_fingerprint": source_hash.hexdigest(),
        "query_count": len(queries),
        "queries": queries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "query_count": len(queries), "source_fingerprint": output["source_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
