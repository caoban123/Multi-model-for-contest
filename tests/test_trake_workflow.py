from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_retrieval.trake_workflow import TrakeWorkflow
from tools.phase8_fixture_benchmark import run_fixture


def candidates(event_id: str) -> list[dict[str, object]]:
    frame = 1 if event_id == "e1" else 2
    return [{"video_id": "L21_V001", "keyframe_id": frame, "frame_idx": frame * 10, "pts_time": float(frame), "score": 1.0}]


def test_workflow_plan_search_align_and_stage_timings() -> None:
    workflow = TrakeWorkflow(lambda event, _pool: candidates(event.event_id))
    planned = workflow.plan("q1", "first then second")
    session_id = planned["state"]["session_id"]
    searched = workflow.search(session_id, event_pool_size=30, max_per_video=5, video_pool_size=10)
    aligned = workflow.align(session_id)
    assert searched["state"]["videos"][0]["complete"]
    assert aligned["state"]["alignments"][0]["chains"][0]["valid"]
    assert set(aligned["state"]["stage_timings_ms"]) == {"plan", "search", "align"}


def test_expected_domain_failures_are_explicit() -> None:
    workflow = TrakeWorkflow(lambda _event, _pool: [])
    with pytest.raises(KeyError, match="unknown TRAKE session"):
        workflow.search("missing", event_pool_size=30)
    planned = workflow.plan("q", "one")
    with pytest.raises(ValueError, match="search must run"):
        workflow.align(planned["state"]["session_id"])


def test_synthetic_fixture_benchmark_proves_dp_pipeline(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "benchmarks" / "trake_fixture_v1.json"
    payload = run_fixture(fixture)
    assert payload["pipeline_pass"] is True
    assert payload["quality_claim"] is None
