from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools.phase8_benchmark import METRIC_KEYS, evaluate


def write_benchmark_session(path: Path, state: dict[str, object]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sessions(state_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        connection.execute("INSERT INTO sessions VALUES(?,?)", (json.dumps(state), "2026-08-19T00:00:00Z"))


def labelled_query(*, optional: bool = False) -> dict[str, object]:
    events: list[dict[str, object]] = [
        {"event_id": "e1", "required": True, "start_pts": 9.0, "end_pts": 11.0, "representative_pts": 10.0},
    ]
    if optional:
        events.append({"event_id": "e2", "required": False, "start_pts": 19.0, "end_pts": 21.0, "representative_pts": 20.0})
    return {"query_id": "q", "query": "event", "expected_video_id": "L21_V001", "events": events}


def state_with_chain(chain: dict[str, object] | None, *, candidates: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "request": {"query_id": "q", "constraints": {}},
        "pools": [{"event": {"event_id": "e1"}, "candidates": candidates or []}],
        "videos": [{"video_id": "L21_V001"}],
        "alignments": [{"chains": [chain] if chain else []}],
        "stage_timings_ms": {"planner": 1.0, "retrieval": 2.0, "alignment": 3.0},
        "manual_chain": None,
    }


def test_unlabelled_benchmark_reports_null_not_zero(tmp_path: Path) -> None:
    queries=tmp_path/'queries.json'; queries.write_text(json.dumps({'version':'v','split':'development','queries':[{'query_id':'q','query':'','events':[],'expected_video_id':None,'expected_event_frames':[]}]}),encoding='utf-8')
    result=evaluate(queries,tmp_path/'missing.sqlite3')
    assert result['availability']=='unavailable' and result['labelled_query_count']==0
    assert all(result['metrics'][key] is None for key in METRIC_KEYS)
    assert all(value is None for value in result['ablations'].values())


def test_phase8_config_keeps_vlm_off_and_holdout_frozen() -> None:
    root=Path(__file__).parents[1]; config=json.loads((root/'configs/phase8_trake_v1.json').read_text(encoding='utf-8'))
    assert config['p8_7_vlm_extension']['enabled'] is False
    assert config['p8_7_vlm_extension']['network_calls_allowed'] is False
    assert config['benchmark']['development_config_frozen'] is True


def test_no_chain_is_order_failure_and_required_coverage_zero(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"version": "v2", "split": "development", "queries": [labelled_query()]}), encoding="utf-8")
    store = tmp_path / "trake.sqlite3"
    write_benchmark_session(store, state_with_chain(None))

    result = evaluate(queries, store)

    assert result["metrics"]["temporal_order_accuracy"] == 0
    assert result["metrics"]["required_event_coverage"] == 0
    assert result["metrics"]["chain_success_rate"] == 0
    assert result["failure_diagnostics"][0]["failure_reason"] == "A_REQUIRED_EVENT_DID_NOT_SURVIVE_TOP_100"


def test_required_coverage_excludes_optional_and_timestamp_uses_interval(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"version": "v2", "split": "development", "queries": [labelled_query(optional=True)]}), encoding="utf-8")
    store = tmp_path / "trake.sqlite3"
    candidate = {"event_id": "e1", "video_id": "L21_V001", "keyframe_id": 999, "pts_time": 10.5}
    chain = {"chain_id": "c", "video_id": "L21_V001", "score": {"final_score": 1.0}, "events": [{"event_id": "e1", "candidate": candidate}]}
    write_benchmark_session(store, state_with_chain(chain, candidates=[candidate]))

    result = evaluate(queries, store)

    assert result["metrics"]["required_event_coverage"] == 1
    assert result["metrics"]["mean_absolute_timestamp_error_seconds"] == 0
    assert result["metrics"]["event_within_1s"] == 1
    assert result["metrics"]["chain_success_rate"] == 1
    assert result["metrics"]["mean_frame_distance"] is None


def test_event_recall_cutoffs_and_nearest_interval_boundary(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"version": "v2", "split": "development", "queries": [labelled_query()]}), encoding="utf-8")
    store = tmp_path / "trake.sqlite3"
    distractors = [{"event_id": "e1", "video_id": "L21_V002", "keyframe_id": index, "pts_time": float(index)} for index in range(20)]
    correct = {"event_id": "e1", "video_id": "L21_V001", "keyframe_id": 21, "pts_time": 10.0}
    selected = {"event_id": "e1", "video_id": "L21_V001", "keyframe_id": 22, "pts_time": 13.5}
    chain = {"chain_id": "c", "video_id": "L21_V001", "score": {"final_score": 1.0}, "events": [{"event_id": "e1", "candidate": selected}]}
    write_benchmark_session(store, state_with_chain(chain, candidates=[*distractors, correct]))

    result = evaluate(queries, store)

    assert result["metrics"]["event_recall_at_20"] == 0
    assert result["metrics"]["event_recall_at_50"] == 1
    assert result["metrics"]["mean_absolute_timestamp_error_seconds"] == 2.5
    assert result["metrics"]["event_within_1s"] == 0
    assert result["metrics"]["event_within_3s"] == 1
