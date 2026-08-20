from __future__ import annotations

from pathlib import Path

import pytest

from aic_retrieval.trake_config import load_trake_config
from aic_retrieval.trake_schema import TrakeRequest
from aic_retrieval.trake_store import TrakeStore
from aic_retrieval.trake_workflow import TrakeDomainError, TrakeWorkflow


ROOT = Path(__file__).parents[1]


def raw(event_id: str, keyframe: int, pts: float) -> dict[str, object]:
    return {
        "event_id": event_id,
        "video_id": "L21_V001",
        "keyframe_id": keyframe,
        "frame_idx": keyframe * 25,
        "pts_time": pts,
        "score": 1.0,
        "evidence": {"fps": 25.0},
        "raw_modality_ranks": {"clip_variant_0": 1},
    }


def test_successful_query_exposes_every_debug_stage_and_score_breakdown() -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")

    def batch(request: TrakeRequest, _limit: int) -> dict[str, list[dict[str, object]]]:
        return {event.event_id: [raw(event.event_id, event.order, float(event.order))] for event in request.events}

    workflow = TrakeWorkflow(lambda _event, _limit: [], runtime_config=config, request_retriever=batch)
    session = workflow.plan("q", "first then second")["state"]["session_id"]
    workflow.search(session, event_pool_size=30)
    payload = workflow.align(session)["state"]
    diagnostics = payload["diagnostics"]

    assert set(("planner", "retrieval", "temporal_feasibility", "candidate_windows", "dense_refinement", "alignment", "final_reranking", "latency_ms")) <= set(diagnostics)
    event_debug = diagnostics["retrieval"]["events"][0]
    assert set(("topk_used", "selected_modalities", "top_score", "score_margin", "distinctiveness", "warnings")) <= set(event_debug)
    assert diagnostics["alignment"]["paths"][0]["score_breakdown"]["final_rerank_version"] == "trake-final-rerank-v1"
    assert payload["stage_timings_ms"]["total"] >= 0


def test_no_candidate_failure_is_typed_persisted_and_debuggable(tmp_path: Path) -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    store = TrakeStore(tmp_path / "trake.sqlite3")
    workflow = TrakeWorkflow(
        lambda _event, _limit: [],
        store=store,
        runtime_config=config,
        request_retriever=lambda request, _limit: {event.event_id: [] for event in request.events},
    )
    session = workflow.plan("q", "first then second")["state"]["session_id"]
    workflow.search(session, event_pool_size=30)

    with pytest.raises(TrakeDomainError, match="NO_CANDIDATES") as caught:
        workflow.align(session)

    assert caught.value.code == "NO_CANDIDATES"
    reopened = TrakeWorkflow(lambda _event, _limit: [], store=TrakeStore(store.path), runtime_config=config)
    failure = reopened.get(session)["state"]["diagnostics"]["failure"]
    assert failure == {"code": "NO_CANDIDATES", "failure_stage": "EVENT_RECALL_FAILURE", "message": "no candidate video covers any planned event"}
