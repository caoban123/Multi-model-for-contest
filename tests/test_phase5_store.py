from pathlib import Path
from types import SimpleNamespace
from aic_retrieval.phase5_schema import AsrSegment,AsrTranscript,OcrDetection,OcrFrame
from aic_retrieval.phase5_store import Phase5SearchService,create_store,ingest_asr,ingest_ocr

BOX=((0,0),(1,0),(1,1),(0,1))

def build(path:Path):
    refs=[SimpleNamespace(video_id="L21_V001",keyframe_id=1,frame_idx=25,pts_time=1)]
    db=create_store(path)
    ingest_ocr(db,[OcrFrame("L21_V001",1,25,1,25,(OcrDetection("THỜI SỰ 19H",.9,BOX),),run_id="ocr-test")])
    ingest_asr(db,[AsrTranscript("L21_V001","AVAILABLE",(AsrSegment(0,.5,1.5,"Xin chào quý vị",.8),),duration=2,run_id="asr-test")],refs)
    db.close(); return Phase5SearchService(path,refs)

def test_fts_search_is_accent_tolerant_and_maps_asr(tmp_path):
    service=build(tmp_path/"phase5.sqlite")
    ocr=service.search_ocr("thoi su",10,.5); assert (ocr[0]["video_id"],ocr[0]["keyframe_id"],ocr[0]["matched_text"])==("L21_V001",1,"THỜI SỰ 19H")
    asr=service.search_asr("xin chao"); assert (asr[0]["mapping_kind"],asr[0]["keyframe_id"])==("inside_segment",1)

def test_evidence_for_timeline_contains_bbox_and_segment(tmp_path):
    service=build(tmp_path/"phase5.sqlite"); evidence=service.evidence_for_frame("L21_V001",1)
    assert evidence["ocr"][0]["bbox"]==[[0,0],[1,0],[1,1],[0,1]]
    assert evidence["asr"][0]["start_time"]==.5

def test_ocr_minimum_confidence_filters_result(tmp_path):
    service=build(tmp_path/"phase5.sqlite"); assert service.search_ocr("thời sự",10,.95)==[]
