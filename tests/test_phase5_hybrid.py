import numpy as np
from aic_retrieval.hybrid_candidates import StructuredCandidateGenerator
from aic_retrieval.hybrid_ranking import RrfConfig,rank_video_candidates
from aic_retrieval.phase5_schema import OcrDetection,OcrFrame
from aic_retrieval.phase5_store import Phase5SearchService,create_store,ingest_ocr
from aic_retrieval.search import FrameRef
from aic_retrieval.structured_query import StructuredQuery

def test_ocr_channel_generates_evidence_and_rrf_rank(tmp_path):
    refs=[FrameRef("L21_V001","L21",1,1,1,1,None),FrameRef("L21_V002","L21",1,1,1,1,None)]
    path=tmp_path/"store.sqlite"; db=create_store(path); box=((0,0),(1,0),(1,1),(0,1)); ingest_ocr(db,[OcrFrame("L21_V002",1,1,1,1,(OcrDetection("Samsung",.9,box),))]); db.close()
    service=Phase5SearchService(path,refs); generator=StructuredCandidateGenerator(np.eye(2,dtype=np.float32),refs,None,[],None,service)
    query=StructuredQuery("Samsung",enable_clip=False,enable_ocr=True,clip_mode="disabled",ocr_mode="soft")
    candidates=generator.generate(query,None); ranked=rank_video_candidates(candidates,query,RrfConfig(),top_k=2)
    assert candidates["channel_counts"] == {"clip_frames":0,"object_frames":0,"attribute_frames":0,"metadata_videos":0,"ocr_frames":1}
    assert ranked["video_results"][0]["video_id"]=="L21_V002"
    assert ranked["video_results"][0]["modality_ranks"]["ocr"]==1
    assert ranked["results"][0]["evidence"]["ocr"][0]["matched_text"]=="Samsung"
