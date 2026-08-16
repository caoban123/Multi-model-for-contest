import json
import sqlite3
import threading
import urllib.request
from pathlib import Path

import numpy as np

import aic_retrieval.retrieval_ui as retrieval_ui
from aic_retrieval.color_attributes import ColorAttributeService
from aic_retrieval.hybrid_candidates import StructuredCandidateGenerator
from aic_retrieval.metadata_search import document_from_payload
from aic_retrieval.object_search import ObjectSearchService
from aic_retrieval.object_store import create_schema, load_alias_dictionary
from aic_retrieval.retrieval_ui import RetrievalUiConfig, RetrievalUiService, parse_bool, parse_float, parse_object_position
from aic_retrieval.search import FrameRef


def make_structured_service(tmp_path: Path) -> RetrievalUiService:
    refs=[FrameRef("L21_V001","L21",1,10,1,30,None),FrameRef("L21_V002","L21",1,10,1,30,None),FrameRef("L21_V003","L21",1,10,1,30,None)]
    index=np.array([[1,0],[0,1],[.5,.5]],dtype=np.float32)
    db=tmp_path/"objects.sqlite"; connection=sqlite3.connect(db); create_schema(connection)
    connection.executemany("INSERT INTO frames VALUES (?,?,?,?,?,?)", [("L21_V001",1,10,1,0,"UNKNOWN"),("L21_V002",1,10,1,1,"AVAILABLE"),("L21_V003",1,10,1,1,"AVAILABLE")])
    connection.execute("INSERT INTO detections(video_id,keyframe_id,label_id,label_raw,label_normalized,confidence,bbox_y1,bbox_x1,bbox_y2,bbox_x2,center_x,center_y,area,source,schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("L21_V002",1,"1","Mobile phone","phone",.9,0,0,.2,.2,.1,.1,.04,"test","v1"))
    connection.commit(); connection.close()
    service=RetrievalUiService.__new__(RetrievalUiService)
    service.config=RetrievalUiConfig(tmp_path,tmp_path/"registry",tmp_path/"index",tmp_path/"media",tmp_path/"web",{"L21"})
    service.index=index; service.refs=refs; service.index_metadata={}; service.assets_by_video={"L21_V001":{},"L21_V002":{},"L21_V003":{}}
    service.refs_by_video={"L21_V001":[refs[0]],"L21_V002":[refs[1]],"L21_V003":[refs[2]]}; service.metadata_by_video={}
    service.metadata_docs=[document_from_payload("L21_V001",{"title":"Morning news"}),document_from_payload("L21_V002",{"title":"Sports"}),document_from_payload("L21_V003",{"title":"Weather"})]
    aliases=load_alias_dictionary(Path(__file__).resolve().parents[1]/"config"/"object_aliases_v1.json")
    service.object_service=ObjectSearchService(db,aliases)
    service.attribute_service=ColorAttributeService(tmp_path, refs)
    service.structured_generator=StructuredCandidateGenerator(index,refs,service.object_service,service.metadata_docs,service.attribute_service)
    class Encoder:
        def encode_text(self,text): return np.array([1,0],dtype=np.float32)
    service.encoder=Encoder()
    return service


def test_structured_service_returns_complete_status_contract(tmp_path: Path) -> None:
    service=make_structured_service(tmp_path)
    payload=service.structured_search("phone",top_k=3,candidate_pool=3,enable_clip=True,enable_objects=True,object_label="điện thoại")
    assert payload["experimental"] is True and payload["default_search_unchanged"] is True
    by_video={item["video_id"]:item for item in payload["video_results"]}
    assert by_video["L21_V002"]["evidence"]["objects"]["status"] == "matched"
    assert by_video["L21_V001"]["evidence"]["objects"]["status"] == "unknown"
    assert by_video["L21_V003"]["evidence"]["objects"]["status"] == "not_matched"
    assert {item["evidence"]["metadata"]["status"] for item in payload["video_results"]} == {"disabled"}
    assert all(set(item["evidence"]) >= {"clip","objects","attributes","metadata","fusion","representative_rule"} for item in payload["video_results"])
    assert all("evidence" in item for item in payload["results"])


def test_structured_parameter_parsers() -> None:
    assert parse_bool("true") and not parse_bool("0")
    assert parse_float("0.4",0.3)==0.4
    assert parse_object_position("left:bottom") == ("left","bottom")


def test_structured_route_is_opt_in_and_search_route_remains_separate(tmp_path: Path, monkeypatch) -> None:
    calls=[]
    class FakeService:
        def __init__(self,config): self.config=config; self.index=np.zeros((1,2)); self.metadata_docs=[]; self.object_service=object(); self.translator=type("T",(),{"is_configured":False,"config":type("C",(),{"provider":"x","model":"x"})()})()
        def structured_search(self,*args,**kwargs): calls.append(("structured",args,kwargs)); return {"route":"structured"}
        def search(self,*args,**kwargs): calls.append(("search",args,kwargs)); return {"route":"search"}
    monkeypatch.setattr(retrieval_ui,"RetrievalUiService",FakeService)
    static=tmp_path/"web"; static.mkdir(); (static/"index.html").write_text("ok",encoding="utf-8")
    config=RetrievalUiConfig(tmp_path,tmp_path/"r",tmp_path/"i",tmp_path/"m",static,{"L21"})
    server=retrieval_ui.run_server(config,port=0); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    port=server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/structured-search?q=phone&enable_objects=true&object_label=phone") as response:
            assert json.load(response)["route"] == "structured"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/search?q=phone") as response:
            assert json.load(response)["route"] == "search"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    assert [item[0] for item in calls] == ["structured","search"]
