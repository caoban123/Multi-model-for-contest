from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aic_retrieval.trake_alignment import DanteInspiredAligner, TemporalAligner
from aic_retrieval.trake_config import RetrievalSettings, TrakeRuntimeConfig, VlmSettings, load_trake_config
from aic_retrieval.trake_workflow import TrakeWorkflow


ROOT = Path(__file__).parents[1]


def candidate_rows(_event: object, _pool_size: int) -> list[dict[str, object]]:
    return [{"video_id": "L21_V001", "keyframe_id": 1, "frame_idx": 25, "pts_time": 1.0, "score": 1.0}]


def test_v2_config_is_typed_fingerprinted_and_vlm_off() -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    assert config.algorithm == "trake_v2"
    assert config.retrieval.topk_max == 200
    assert len(config.fingerprint) == 64
    assert config.vlm.enabled is False
    assert config.vlm.network_calls_allowed is False


def test_algorithm_selection_changes_aligner_and_persists_provenance() -> None:
    legacy = TrakeWorkflow(candidate_rows, runtime_config=TrakeRuntimeConfig.legacy())
    v2_config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    v2 = TrakeWorkflow(candidate_rows, runtime_config=v2_config)

    assert type(legacy.aligner) is TemporalAligner
    assert type(v2.aligner) is DanteInspiredAligner
    planned = v2.plan("q", "event")
    assert planned["workflow_version"] == "trake-workflow-v2"
    assert planned["state"]["algorithm"] == "trake_v2"
    assert planned["state"]["config_fingerprint"] == v2_config.fingerprint
    assert planned["state"]["provenance"]["clip_model_fingerprint"] is None


def test_workflow_preserves_planner_sequence_constraints() -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    planned = TrakeWorkflow(candidate_rows, runtime_config=config).plan("q", "first → second")

    assert planned["state"]["request"]["constraints"]["gap_mode"] == "hard"
    assert planned["state"]["request"]["events"][1]["max_gap_seconds"] == 60.0


def test_topk_max_changes_runtime_acceptance() -> None:
    base = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    limited_retrieval = RetrievalSettings(**{**base.to_dict()["retrieval"], "topk_default": 50, "topk_max": 50})
    limited = replace(base, retrieval=limited_retrieval)
    limited_workflow = TrakeWorkflow(candidate_rows, runtime_config=limited)
    limited_session = limited_workflow.plan("limited", "event")["state"]["session_id"]
    with pytest.raises(ValueError, match="topk_max"):
        limited_workflow.search(limited_session, event_pool_size=100)

    full_workflow = TrakeWorkflow(candidate_rows, runtime_config=base)
    full_session = full_workflow.plan("full", "event")["state"]["session_id"]
    assert full_workflow.search(full_session, event_pool_size=100)["state"]["pools"]


def test_vlm_disabled_prevents_injected_verifier_call() -> None:
    calls: list[dict[str, object]] = []
    workflow = TrakeWorkflow(
        candidate_rows,
        runtime_config=load_trake_config(ROOT / "configs" / "phase8_trake_v2.json"),
        vlm_verifier=lambda payload: calls.append(payload) or {"accepted": True},
    )

    result = workflow.maybe_verify_with_vlm({"chain_id": "c"})

    assert result == {"status": "DISABLED", "called": False}
    assert calls == []


def test_vlm_requires_explicit_approval_and_only_receives_low_confidence_valid_top_n() -> None:
    base = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    with pytest.raises(ValueError, match="decision_log"):
        VlmSettings(enabled=True, network_calls_allowed=True, provider="fixture", model="fixture")
    enabled = replace(
        base,
        features=replace(base.features, conditional_vlm=True),
        vlm=VlmSettings(
            enabled=True,
            network_calls_allowed=True,
            decision_log_approved=True,
            provider="fixture",
            model="fixture",
            top_n=1,
        ),
    )
    calls: list[dict[str, object]] = []
    workflow = TrakeWorkflow(candidate_rows, runtime_config=enabled, vlm_verifier=lambda payload: calls.append(payload) or {"verdict": "ambiguous"})
    valid_chain = {
        "chain_id": "c1",
        "video_id": "L21_V001",
        "valid": True,
        "score_components": {"final_rerank_score": 0.50, "linkage_score": {"score": 0.2}},
        "events": [
            {"event_id": "e1", "required": True, "candidate": {"video_id": "L21_V001", "pts_time": 1.0, "keyframe_path": "a.jpg"}},
            {"event_id": "e2", "required": True, "candidate": {"video_id": "L21_V001", "pts_time": 2.0, "keyframe_path": "b.jpg"}},
        ],
    }
    second = {**valid_chain, "chain_id": "c2", "score_components": {"final_rerank_score": 0.49}}

    result = workflow.maybe_verify_with_vlm({"query": "q", "event_descriptions": ["one", "two"], "chains": [valid_chain, second], "secret": "must-not-leak"})

    assert result["called"] is True and result["advisory_only"] is True
    assert result["trigger_reasons"] == ["SMALL_TOP1_TOP2_MARGIN", "LOW_LINKAGE"]
    assert len(calls) == 1 and len(calls[0]["chains"]) == 1
    assert "secret" not in calls[0]


def test_vlm_never_receives_server_invalid_chain() -> None:
    base = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    enabled = replace(base, features=replace(base.features, conditional_vlm=True), vlm=VlmSettings(True, "fixture", "fixture", True, True))
    calls: list[dict[str, object]] = []
    workflow = TrakeWorkflow(candidate_rows, runtime_config=enabled, vlm_verifier=lambda payload: calls.append(payload) or {})
    invalid = {
        "chain_id": "bad",
        "valid": False,
        "score_components": {"linkage_score": {"score": 0.1}},
        "events": [],
    }

    result = workflow.maybe_verify_with_vlm({"chains": [invalid]})

    assert result["status"] == "SKIPPED_INVALID" and result["called"] is False
    assert calls == []


def test_opt_in_gemini_config_extends_frozen_v2_baseline() -> None:
    baseline = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    enabled = load_trake_config(ROOT / "configs" / "phase10_trake_gemini_opt_in.json")
    assert enabled.algorithm == baseline.algorithm
    assert enabled.temporal == baseline.temporal
    assert enabled.reranking == baseline.reranking
    assert enabled.vlm.enabled is True
    assert enabled.vlm.decision_log_approved is True
    assert enabled.features.conditional_vlm is True
