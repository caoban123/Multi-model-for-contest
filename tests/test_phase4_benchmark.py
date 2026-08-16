from tools.phase4_benchmark import ABLATIONS, THRESHOLDS, constraints_from_query, summarize_runs, unavailable_quality_metrics


def test_ablation_matrix_and_metrics_contract() -> None:
    assert list(ABLATIONS) == ["A_clip_only","B_object_only","C_metadata_only","D_clip_object","E_clip_metadata","F_clip_object_metadata"]
    assert THRESHOLDS == (0.1,0.2,0.3,0.5)
    metrics=unavailable_quality_metrics()
    assert metrics["status"] == "unavailable"
    assert all(metrics[key] is None for key in ("Recall@1","Recall@5","Recall@20","Recall@50","Recall@100","MRR","NDCG","Top-1_good","Top-1_bad"))


def test_constraints_from_versioned_query_supports_object_and_metadata() -> None:
    raw={"structured_constraints":{"objects":[{"label":"person","min_count":2,"position":"left","mode":"hard"}],"metadata":{"title":"News","date_from":"2024-08-01","keywords":["weather"],"mode":"soft"}}}
    objects,metadata=constraints_from_query(raw,0.5)
    assert objects[0].labels == ("person",) and objects[0].count == 2 and objects[0].horizontal == "left" and objects[0].filter_mode == "hard"
    assert objects[0].min_confidence == 0.5
    assert metadata.title_phrase == "News" and metadata.date_from == "2024-08-01" and metadata.keywords == ("weather",)


def test_summary_keeps_missing_quality_unavailable() -> None:
    run={"candidate_count":10,"candidate_coverage":.1,"latency_ms":{"candidate_generation":1.,"fusion":2.,"total":3.},"missing_modality_rate":{"objects":.2}}
    summary=summarize_runs([run,run])
    assert summary["query_count"] == 2 and summary["candidate_count_mean"] == 10
    assert summary["object_missing_rate_mean"] == .2
    assert summary["quality_metrics"]["MRR"] is None
