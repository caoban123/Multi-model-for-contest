from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_reviewed_benchmark_builder_uses_clip_query_and_positive_videos(tmp_path: Path) -> None:
    input_dir = tmp_path / "reviews"
    input_dir.mkdir()
    path = input_dir / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_text", "clip_query", "video_id", "manual_judgement"])
        writer.writeheader()
        writer.writerow({"query_text": "stale", "clip_query": "a red shirt", "video_id": "V1", "manual_judgement": "good"})
        writer.writerow({"query_text": "stale", "clip_query": "a red shirt", "video_id": "V2", "manual_judgement": "bad"})
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [sys.executable, "tools/build_reviewed_hybrid_benchmark.py", "--input-dir", str(input_dir), "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PARTIAL_JUDGMENTS_NOT_PROMOTION_GROUND_TRUTH"
    assert payload["queries"][0]["query_text"] == "a red shirt"
    assert payload["queries"][0]["expected_video_ids"] == ["V1"]
    assert payload["queries"][0]["review"]["negative_video_ids"] == ["V2"]
