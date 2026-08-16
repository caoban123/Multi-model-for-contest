from dataclasses import asdict

from aic_retrieval.hybrid_ranking import RrfConfig, contribution, rank_video_candidates
from aic_retrieval.search import FrameRef, SearchResult, aggregate_results_by_video
from aic_retrieval.structured_query import ObjectConstraint, StructuredQuery


def candidate(video, keyframe, clip_rank=None, clip_score=None, object_rank=None, object_score=None, metadata_rank=None):
    return {"video_id":video,"keyframe_id":keyframe,"group":"L21","frame_idx":keyframe*10,"pts_time":float(keyframe),"fps":30.0,"keyframe_path":None,
            "clip_rank":clip_rank,"clip_score":clip_score,"object_rank":object_rank,"object_score":object_score,"metadata_rank":metadata_rank,"metadata_score":None,"provenance":[],"evidence":{}}


def test_rrf_formula_and_deterministic_tie_break() -> None:
    config = RrfConfig(rrf_k=60, weight_clip=1, weight_object=.5, weight_metadata=.3)
    assert contribution(1, 1, 60) == 1/61
    payload = {"candidates":[candidate("L21_V002",1,clip_rank=1,clip_score=.9),candidate("L21_V001",1,clip_rank=1,clip_score=.9)]}
    query = StructuredQuery("x", clip_candidate_pool=2)
    first=rank_video_candidates(payload,query,config); second=rank_video_candidates(payload,query,config)
    assert first == second
    assert [item["video_id"] for item in first["video_results"]] == ["L21_V001","L21_V002"]


def test_clip_only_reproduces_phase3_video_order_and_representative() -> None:
    raw = [
        SearchResult(1,.9,"V2","L21",1,10,1,30,None), SearchResult(2,.8,"V1","L21",2,20,2,30,None),
        SearchResult(3,.7,"V2","L21",3,30,3,30,None), SearchResult(4,.6,"V3","L21",1,10,1,30,None),
    ]
    phase3=aggregate_results_by_video(raw,5,"max",3)
    payload={"candidates":[candidate(item.video_id,item.keyframe_id,item.rank,item.score) for item in raw]}
    ranked=rank_video_candidates(payload,StructuredQuery("x",clip_candidate_pool=4),RrfConfig(),top_k=3)
    assert [item.video_id for item in phase3] == [item["video_id"] for item in ranked["video_results"]]
    assert [item.best_keyframe_id for item in phase3] == [item["best_keyframe_id"] for item in ranked["video_results"]]
    assert all(item["representative_rule"] == "highest_clip_score_visual_only" for item in ranked["video_results"])
    assert set(ranked) >= {"raw_candidates","raw_results","results","video_results","video_groups"}
    assert [item["keyframe_id"] for item in ranked["results"]] == [item.best_keyframe_id for item in phase3]


def test_hard_object_representative_is_highest_clip_satisfying_frame() -> None:
    payload={"candidates":[candidate("V1",1,1,.9,None,None),candidate("V1",2,2,.8,1,.95),candidate("V1",3,3,.7,2,.8)]}
    query=StructuredQuery("x",enable_objects=True,object_constraints=(ObjectConstraint(("phone",),filter_mode="hard"),),clip_candidate_pool=3)
    result=rank_video_candidates(payload,query)["video_results"][0]
    assert result["best_keyframe_id"] == 2
    assert result["representative_rule"] == "highest_clip_among_hard_object_matches"


def test_hybrid_video_ranking_uses_modality_ranks_not_raw_score_addition() -> None:
    payload={"candidates":[candidate("V1",1,clip_rank=1,clip_score=.01),candidate("V2",1,clip_rank=2,clip_score=.99,object_rank=1,object_score=.01,metadata_rank=1)]}
    query=StructuredQuery("x",enable_objects=True,enable_metadata=True,object_constraints=(ObjectConstraint(("phone",)),),metadata_mode="soft",clip_candidate_pool=2)
    result=rank_video_candidates(payload,query,RrfConfig())["video_results"]
    assert result[0]["video_id"] == "V2"
    assert result[0]["fusion_score"] == contribution(2,1,60)+contribution(1,.5,60)+contribution(1,.3,60)
