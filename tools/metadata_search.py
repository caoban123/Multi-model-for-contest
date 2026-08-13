from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.metadata_search import load_metadata_documents, results_to_dict, search_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search video-level media-info metadata.")
    parser.add_argument("--media-dir", type=Path, default=Path("data/media-info"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional groups such as L21 L22.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-match", type=int, default=1, help="Minimum number of query terms that must match.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = set(args.groups) if args.groups else None
    docs = load_metadata_documents(args.media_dir, groups=groups)
    results = search_metadata(docs, args.query, top_k=args.top_k, min_match=args.min_match)
    payload = {
        "query": args.query,
        "media_dir": str(args.media_dir),
        "groups": sorted(groups) if groups else None,
        "total_documents": len(docs),
        "top_k": args.top_k,
        "min_match": args.min_match,
        "results": results_to_dict(results),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv_output:
        write_csv(args.csv_output, payload["results"])

    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "video_id",
        "score",
        "matched_terms",
        "title",
        "author",
        "publish_date",
        "keywords",
        "watch_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "video_id": row["video_id"],
                    "score": row["score"],
                    "matched_terms": " ".join(row["matched_terms"]),
                    "title": row["title"],
                    "author": row["author"],
                    "publish_date": row["publish_date"],
                    "keywords": " | ".join(row["keywords"]),
                    "watch_url": row["watch_url"],
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
