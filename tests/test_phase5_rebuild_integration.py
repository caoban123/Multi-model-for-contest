import json
import sys
from pathlib import Path

import numpy as np

from aic_retrieval.phase5_store import Phase5SearchService
from tools import build_phase5_store


def test_build_phase5_store_with_valid_index_creates_missing_output_parent(tmp_path,monkeypatch):
    index=tmp_path/"artifacts/indexes/l21_numpy"; index.mkdir(parents=True); np.save(index/"vectors.npy",np.array([[1,0]],dtype=np.float32)); (index/"refs.json").write_text(json.dumps([{"video_id":"L21_V001","group":"L21","keyframe_id":1,"frame_idx":25,"pts_time":1.0,"fps":25.0,"keyframe_path":None}]),encoding="utf-8")
    ocr=tmp_path/"evidence/ocr.jsonl"; ocr.parent.mkdir(); ocr.write_text(json.dumps({"video_id":"L21_V001","keyframe_id":1,"frame_idx":25,"pts_time":1.0,"fps":25.0,"detections":[{"text":"THỜI SỰ 19H","confidence":.9,"bbox":[[0,0],[1,0],[1,1],[0,1]]}]})+"\n",encoding="utf-8")
    output=tmp_path/"artifacts/phase5/manual/phase5.sqlite3"
    monkeypatch.setattr(sys,"argv",["build_phase5_store.py","--ocr-jsonl",str(ocr),"--index-dir",str(index),"--output",str(output)])
    assert build_phase5_store.main()==0
    assert output.is_file()
    assert Phase5SearchService(output).search_ocr("thoi su")[0]["video_id"]=="L21_V001"
