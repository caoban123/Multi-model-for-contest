import json

import pytest

from aic_retrieval.fusion import (
    matching_rules,
    rerank_results_with_objects,
    score_interactions_for_query,
    score_objects_for_query,
)
import tools.fusion_experiment as fusion_experiment


def test_matching_rules_detects_phone_query() -> None:
    assert matching_rules("a hand holding a phone") == ["phone"]


def test_score_objects_for_query_uses_non_generic_entities(tmp_path) -> None:
    object_dir = tmp_path / "V1"
    object_dir.mkdir()
    (object_dir / "001.json").write_text(
        json.dumps(
            {
                "detection_class_entities": ["Person", "Mobile phone", "Human hand"],
                "detection_scores": ["0.99", "0.8", "0.5"],
            }
        ),
        encoding="utf-8",
    )

    score, rules, entities = score_objects_for_query("phone", tmp_path, "V1", 1)

    assert rules == ["phone"]
    assert score == pytest.approx(0.975)
    assert [item["entity"] for item in entities] == ["Mobile phone", "Human hand"]


def test_rerank_results_with_objects_can_promote_matching_candidate(tmp_path) -> None:
    for video_id, score in [("V1", 0.1), ("V2", 0.9)]:
        object_dir = tmp_path / video_id
        object_dir.mkdir()
        (object_dir / "001.json").write_text(
            json.dumps(
                {
                    "detection_class_entities": ["Mobile phone"],
                    "detection_scores": [str(score)],
                    "detection_boxes": [[0.1, 0.1, 0.2, 0.2]],
                }
            ),
            encoding="utf-8",
        )

    results = [
        {"video_id": "V1", "keyframe_id": 1, "score": 0.50, "keyframe_path": "a.jpg"},
        {"video_id": "V2", "keyframe_id": 1, "score": 0.49, "keyframe_path": "b.jpg"},
    ]

    reranked = rerank_results_with_objects("phone", results, tmp_path, object_weight=0.05)

    assert reranked[0]["video_id"] == "V2"
    assert reranked[0]["baseline_rank"] == 2


def test_score_interactions_for_query_scores_phone_near_hand(tmp_path) -> None:
    object_dir = tmp_path / "V1"
    object_dir.mkdir()
    (object_dir / "001.json").write_text(
        json.dumps(
            {
                "detection_class_entities": ["Mobile phone", "Human hand"],
                "detection_scores": ["0.9", "0.8"],
                "detection_boxes": [[0.1, 0.1, 0.2, 0.2], [0.12, 0.12, 0.22, 0.22]],
            }
        ),
        encoding="utf-8",
    )

    assert score_interactions_for_query("a hand holding a phone", tmp_path, "V1", 1) > 0.5


def test_fusion_experiment_metrics_for_ranked() -> None:
    metrics = fusion_experiment.metrics_for_ranked(
        [
            {"manual_judgement": "bad"},
            {"manual_judgement": "good"},
            {"manual_judgement": "partial"},
        ]
    )

    assert metrics["good_at_k"] == pytest.approx(1 / 3)
    assert metrics["good_or_partial_at_k"] == pytest.approx(2 / 3)
    assert metrics["mean_relevance_at_k"] == pytest.approx(0.5)
    assert metrics["mrr_good"] == pytest.approx(0.5)
    assert metrics["ndcg_at_k"] > 0.5
    assert metrics["top1_bad"] == 1.0
    assert metrics["top1_relevance"] == 0.0
