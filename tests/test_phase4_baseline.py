import json
from pathlib import Path

import pytest

from tools.phase4_baseline import QUERY_TYPES, load_phase4_queries, stable_signature


def test_repository_phase4_query_set_is_versioned_balanced_and_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    version, queries = load_phase4_queries(root / "benchmarks" / "phase4_queries_v1.json")
    assert version == "phase4-v1.0"
    assert len(queries) == 20
    assert {item["query_type"] for item in queries} == QUERY_TYPES
    assert sum(item["split"] == "development" for item in queries) == 10
    assert sum(item["split"] == "holdout" for item in queries) == 10
    assert all(item["expected_video_ids"] == [] for item in queries)


def test_phase4_query_loader_rejects_missing_type_coverage(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"query_set_version": "test", "queries": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="query types mismatch"):
        load_phase4_queries(path)


def test_stable_signature_ignores_timing_and_rounds_float_noise() -> None:
    class Result:
        video_id = "L21_V001"
        keyframe_id = 1
        frame_idx = 2
        score = 0.123456789

    assert stable_signature([Result()]) == [("L21_V001", 1, 2, 0.12345679)]
