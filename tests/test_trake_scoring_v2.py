from __future__ import annotations

from aic_retrieval.trake_alignment import AlignmentResult, ChainEvent, TrakeChain
from aic_retrieval.trake_candidates import TrakeCandidate
from aic_retrieval.trake_config import RerankingSettings
from aic_retrieval.trake_schema import Availability, TrakeEvent, TrakeRequest
from aic_retrieval.trake_scoring import (
    ScoreBreakdown,
    adjacent_linkage_score,
    global_context_score,
    multimodal_event_score,
    rerank_alignments,
)


def candidate(
    event_id: str,
    pts: float,
    *,
    evidence: dict[str, object] | None = None,
    provenance: tuple[str, ...] = ("clip",),
    raw_scores: dict[str, float | None] | None = None,
    raw_ranks: dict[str, int | None] | None = None,
) -> TrakeCandidate:
    frame = round(pts * 25)
    return TrakeCandidate(
        event_id,
        f"{event_id}:{pts}",
        "L21_V001",
        frame,
        frame,
        pts,
        None,
        0.8,
        1,
        evidence or {"fps": 25.0},
        provenance,
        "fixture",
        raw_scores=raw_scores or {},
        raw_ranks=raw_ranks or {},
    )


def chain(chain_id: str, first: TrakeCandidate, second: TrakeCandidate | None) -> TrakeChain:
    score = ScoreBreakdown(4.0 if second else 2.0, 1.6 if second else 0.8, 0.0, 0.0, 0.0, 0.0, 5.6 if second else 2.8)
    return TrakeChain(
        chain_id,
        "L21_V001",
        (ChainEvent("e1", True, first), ChainEvent("e2", True, second)),
        score,
        (),
        score_components={"transition_penalty": 0.0},
    )


def test_multimodal_score_is_event_routed_rank_normalized_and_raw_preserving() -> None:
    item = candidate(
        "e1",
        1.0,
        provenance=("clip", "ocr"),
        raw_scores={"clip_variant_0": 0.91, "ocr": 12.7},
        raw_ranks={"clip_variant_0": 2, "ocr": 1},
    )
    visual = TrakeEvent("e1", 1, "runner", "runner", modalities=("clip",))
    ocr = TrakeEvent("e1", 1, "sign EXIT", "sign EXIT", modalities=("clip", "ocr"), ocr_terms=("EXIT",))
    weights = {"clip": 1.0, "ocr": 3.0}

    visual_score = multimodal_event_score(visual, item, {"clip": Availability.AVAILABLE}, weights)
    ocr_score = multimodal_event_score(ocr, item, {"clip": Availability.AVAILABLE, "ocr": Availability.AVAILABLE}, weights)

    assert visual_score.score == 0.5
    assert ocr_score.score == 0.875
    assert ocr_score.normalized_by_modality == {"clip": 0.5, "ocr": 1.0}
    assert ocr_score.raw_scores == {"clip": 0.91, "ocr": 12.7}
    assert ocr_score.method == "event-aware-reciprocal-rank-v1"


def test_missing_unknown_modality_is_omitted_not_negative() -> None:
    event = TrakeEvent("e1", 1, "spoken words", "spoken words", modalities=("clip", "asr"))
    item = candidate("e1", 1.0, raw_ranks={"clip_variant_0": 2})

    score = multimodal_event_score(event, item, {"clip": Availability.AVAILABLE, "asr": Availability.UNKNOWN})

    assert score.score == 0.5
    assert score.availability["asr"] == "UNKNOWN"
    assert "asr" not in score.normalized_by_modality


def test_context_uses_verified_late_evidence_and_unknown_is_not_zero() -> None:
    request = TrakeRequest(
        "q",
        "football sequence",
        (TrakeEvent("e1", 1, "kick", "kick"), TrakeEvent("e2", 2, "celebrate", "celebrate")),
        global_context="football match",
    )
    matching = chain(
        "matching",
        candidate("e1", 1.0, evidence={"metadata": [{"title": "football match highlights"}]}),
        candidate("e2", 2.0, evidence={"metadata": [{"description": "football match celebration"}]}),
    )
    absent = chain("absent", candidate("e1", 1.0), candidate("e2", 2.0))

    match_score = global_context_score(request, matching, {"metadata": Availability.AVAILABLE})
    unknown_score = global_context_score(request, absent, {"metadata": Availability.UNKNOWN})

    assert match_score.status == "AVAILABLE" and match_score.score == 1.0
    assert match_score.components["metadata"]["matched_tokens"] == ["football", "match"]
    assert unknown_score.status == "UNKNOWN" and unknown_score.score is None


def test_linkage_breakdown_keeps_available_and_unknown_components_separate() -> None:
    left = candidate(
        "e1",
        1.0,
        evidence={"visual_embedding": [1.0, 0.0], "scene_id": "s1", "object": [{"label": "ball"}]},
    )
    right = candidate(
        "e2",
        11.0,
        evidence={"visual_embedding": [1.0, 0.0], "scene_id": "s1", "object": [{"label": "ball"}]},
    )

    result = adjacent_linkage_score(chain("linked", left, right), {"asr": Availability.UNAVAILABLE}, temporal_scale_seconds=10.0)
    components = result.adjacent_pairs[0]["components"]

    assert result.status == "AVAILABLE"
    assert components["temporal_proximity"]["score"] > 0
    assert components["visual_continuity"]["score"] == 1.0
    assert components["scene_continuity"]["score"] == 1.0
    assert components["object_entity_continuity"]["score"] == 1.0
    assert components["asr_continuity"] == {"status": "UNAVAILABLE", "score": None}


def test_final_rerank_is_explicit_reorders_late_and_drops_incomplete_required_chain() -> None:
    request = TrakeRequest(
        "q",
        "football",
        (TrakeEvent("e1", 1, "one", "one"), TrakeEvent("e2", 2, "two", "two")),
        global_context="football match",
    )
    wrong = chain(
        "a-wrong",
        candidate("e1", 1.0, evidence={"metadata": [{"title": "concert crowd"}]}),
        candidate("e2", 2.0, evidence={"metadata": [{"title": "concert stage"}]}),
    )
    correct = chain(
        "z-correct",
        candidate("e1", 1.0, evidence={"metadata": [{"title": "football match"}]}),
        candidate("e2", 2.0, evidence={"metadata": [{"title": "football match"}]}),
    )
    incomplete = chain("incomplete", candidate("e1", 1.0), None)
    settings = RerankingSettings(
        context_enabled=True,
        linkage_enabled=False,
        weights={"context": 1.0},
    )

    results = rerank_alignments(
        request,
        (AlignmentResult("L21_V001", (wrong, incomplete, correct)),),
        settings,
        {"metadata": Availability.AVAILABLE},
    )

    assert [item.chain_id for item in results[0].chains] == ["z-correct", "a-wrong"]
    breakdown = results[0].chains[0].score_components
    assert breakdown["final_rerank_score"] == 1.0
    assert breakdown["context_score"]["status"] == "AVAILABLE"
    assert set(("event_semantic_score", "temporal_score", "linkage_score", "context_score", "multimodal_evidence_score", "coverage_score")) <= set(breakdown)


def test_dense_chain_with_discontinuous_visual_evidence_is_hard_rejected() -> None:
    request = TrakeRequest("q", "one then two", (TrakeEvent("e1", 1, "one", "one"), TrakeEvent("e2", 2, "two", "two")))
    discontinuous = chain(
        "bad-continuity",
        candidate("e1", 1.0, evidence={"dense_refinement": True, "visual_embedding": [1.0, 0.0]}),
        candidate("e2", 2.0, evidence={"dense_refinement": True, "visual_embedding": [0.0, 1.0]}),
    )
    settings = RerankingSettings(linkage_hard_gate=True, min_visual_continuity=0.8)

    results = rerank_alignments(request, (AlignmentResult("L21_V001", (discontinuous,)),), settings)

    assert results[0].chains == ()


def test_dense_clip_chain_requires_manual_semantic_review() -> None:
    request = TrakeRequest("q", "one then two", (TrakeEvent("e1", 1, "one", "one"), TrakeEvent("e2", 2, "two", "two")))
    contiguous = chain(
        "scene-consistent",
        candidate("e1", 1.0, evidence={"dense_refinement": True, "visual_embedding": [1.0, 0.0]}),
        candidate("e2", 2.0, evidence={"dense_refinement": True, "visual_embedding": [0.9, 0.1]}),
    )
    settings = RerankingSettings(linkage_hard_gate=True, min_visual_continuity=0.8)

    results = rerank_alignments(request, (AlignmentResult("L21_V001", (contiguous,)),), settings)

    verification = results[0].chains[0].score_components["semantic_verification"]
    assert verification["status"] == "REQUIRES_MANUAL_REVIEW"
