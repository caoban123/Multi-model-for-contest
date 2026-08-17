import pytest

from aic_retrieval.query_planner import RuleBasedQueryPlanner


def test_local_planner_extracts_modalities_objects_temporal_and_variants():
    plan=RuleBasedQueryPlanner().plan('Người cầm điện thoại, sau đó nói "xin chào"')
    assert set(plan.recommended_modalities) >= {"clip","objects","asr"}
    assert plan.objects == ("phone","person")
    assert "sau đó" in plan.temporal_hints
    assert "xin chào" in plan.variants
    assert len(plan.events)==2


def test_planner_detects_ocr_metadata_and_color_without_network():
    plan=RuleBasedQueryPlanner().plan("biển hiệu chữ đỏ của kênh Tuổi Trẻ")
    assert set(plan.recommended_modalities) >= {"clip","ocr","attributes","metadata"}
    assert plan.version.startswith("phase6-rule-planner")


def test_planner_rejects_empty_query_and_caps_variants():
    with pytest.raises(ValueError,match="must not be empty"): RuleBasedQueryPlanner().plan("  ")
    assert len(RuleBasedQueryPlanner().plan('nói "a" rồi "b"',max_variants=1).variants)==1


def test_user_can_disable_individual_variants_with_original_fallback():
    planner=RuleBasedQueryPlanner(); plan=planner.plan('nói "xin chào"')
    assert planner.select_variants(plan,("xin chào",)).variants == ("xin chào",)
    assert planner.select_variants(plan,()).variants == (plan.original_query,)
