from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import aic_retrieval.retrieval_ui as retrieval_ui
from aic_retrieval.retrieval_ui import RetrievalUiConfig


def post(port: int, route: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def test_trake_routes_are_separate_and_bad_session_is_http_400(tmp_path: Path, monkeypatch) -> None:
    class FakeService:
        def __init__(self, config):
            self.config=config; self.index=np.zeros((1,2)); self.metadata_docs=[]; self.object_service=None
            self.attribute_service=type("A",(),{"available":False})(); self.translator=type("T",(),{"is_configured":False,"config":type("C",(),{"provider":"x","model":"x"})()})()
        def trake_plan(self, *_args): return {"state":{"session_id":"trake-1"}}
        def trake_search(self, session_id, *_args):
            if session_id == "missing": raise KeyError("unknown TRAKE session: missing")
            return {"state":{"session_id":session_id,"videos":[]}}
        def trake_align(self, session_id, *_args): return {"state":{"session_id":session_id,"alignments":[]}}
        def trake_session(self, session_id): return {"state":{"session_id":session_id}}
        def trake_sessions(self): return {"sessions":[]}
    monkeypatch.setattr(retrieval_ui,"RetrievalUiService",FakeService)
    static=tmp_path/"web"; static.mkdir(); (static/"index.html").write_text("ok")
    server=retrieval_ui.run_server(RetrievalUiConfig(tmp_path,tmp_path/"r",tmp_path/"i",tmp_path/"m",static,{"L21"}),port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start(); port=server.server_address[1]
    try:
        assert post(port,"/api/trake/plan",{"query":"first then second"})["state"]["session_id"] == "trake-1"
        assert post(port,"/api/trake/search",{"session_id":"trake-1"})["state"]["videos"] == []
        assert post(port,"/api/trake/align",{"session_id":"trake-1"})["state"]["alignments"] == []
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/trake/health") as response:
            assert json.load(response)["vlm_enabled"] is False
        try: post(port,"/api/trake/search",{"session_id":"missing"})
        except urllib.error.HTTPError as exc: assert exc.code == 400
        else: raise AssertionError("expected HTTP 400")
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
