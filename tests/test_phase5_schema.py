import pytest
from types import SimpleNamespace
from aic_retrieval.phase5_schema import AsrSegment,AsrTranscript,OcrDetection,OcrFrame,map_segment_to_frame,normalize_text

BOX=((0.1,0.1),(0.9,0.1),(0.9,0.2),(0.1,0.2))

def test_vietnamese_normalization_preserves_and_folds_accents():
    assert normalize_text("  THỜI-SỰ 19H! ")=="thời sự 19h"
    assert normalize_text("THỜI SỰ",fold_accents=True)=="thoi su"

def test_ocr_schema_validates_identity_confidence_and_bbox():
    detection=OcrDetection("Thời sự",.9,BOX)
    frame=OcrFrame("L21_V001",1,25,1.0,25.0,(detection,))
    assert frame.to_dict()["detections"][0]["text_folded"]=="thoi su"
    with pytest.raises(ValueError,match="confidence"): OcrDetection("x",1.1,BOX)
    with pytest.raises(ValueError,match="bbox"): OcrDetection("x",.5,((2,0),)*4)

def test_asr_schema_distinguishes_no_audio_and_invalid_timestamps():
    assert AsrTranscript("L21_V001","NO_AUDIO").segments==()
    with pytest.raises(ValueError,match="precede"): AsrSegment(0,2,1,"text")
    with pytest.raises(ValueError,match="must not contain"): AsrTranscript("L21_V001","NO_AUDIO",(AsrSegment(0,0,1,"text"),))

def test_temporal_mapping_inside_nearest_and_unmapped():
    refs=[SimpleNamespace(video_id="L21_V001",keyframe_id=1,frame_idx=25,pts_time=1),SimpleNamespace(video_id="L21_V001",keyframe_id=2,frame_idx=75,pts_time=3)]
    assert map_segment_to_frame("L21_V001",2.5,3.5,refs).mapping_kind=="inside_segment"
    nearest=map_segment_to_frame("L21_V001",4,4.2,refs,max_distance=2); assert (nearest.keyframe_id,nearest.mapping_kind)==(2,"nearest")
    assert map_segment_to_frame("L21_V001",20,21,refs,max_distance=2).mapping_kind=="unmapped"
