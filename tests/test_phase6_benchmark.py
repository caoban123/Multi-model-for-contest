from tools.phase6_benchmark import reciprocal_rank


def test_phase6_reciprocal_rank_uses_video_level_judgement():
    results=[{"video_id":"V1"},{"video_id":"V2"}]
    assert reciprocal_rank(results,{"V2"})==0.5
    assert reciprocal_rank(results,{"V9"})==0.0
