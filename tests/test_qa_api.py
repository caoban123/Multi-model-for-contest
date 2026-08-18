from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np

import aic_retrieval.retrieval_ui as retrieval_ui
from aic_retrieval.retrieval_ui import RetrievalUiConfig


def test_qa_post_routes_are_separate_and_return_service_payloads(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class FakeService:
        def __init__(self, config):
            self.config = config
            self.index = np.zeros((1, 2))
            self.metadata_docs = []
            self.object_service = None
            self.attribute_service = type("Attributes", (), {"available": False})()
            self.translator = type("Translator", (), {"is_configured": False, "config": type("Config", (), {"provider": "x", "model": "x"})()})()

        def qa_prepare(self, *args):
            calls.append("prepare")
            return {"session": {"session_id": "qa-1"}}

        def qa_draft_answer(self, *args):
            calls.append("draft")
            return {"answer_draft": {"draft_id": "draft-1"}}

        def qa_review(self, *args):
            calls.append("review")
            return {"review": {"decision": "confirmed"}}

        def qa_session(self, session_id):
            calls.append("session")
            return {"session": {"session_id": session_id}}

    monkeypatch.setattr(retrieval_ui, "RetrievalUiService", FakeService)
    static = tmp_path / "web"; static.mkdir(); (static / "index.html").write_text("ok", encoding="utf-8")
    config = RetrievalUiConfig(tmp_path, tmp_path / "r", tmp_path / "i", tmp_path / "m", static, {"L21"})
    server = retrieval_ui.run_server(config, port=0); thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    port = server.server_address[1]
    try:
        for endpoint, payload, expected in (
            ("prepare", {"event_query": "event", "question": "question"}, "session"),
            ("draft-answer", {"session_id": "qa-1", "selected_evidence_ids": []}, "answer_draft"),
            ("review", {"session_id": "qa-1", "draft_id": "draft-1", "decision": "confirmed", "selected_evidence_ids": ["e1"]}, "review"),
        ):
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/qa/{endpoint}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(request) as response:
                assert expected in json.load(response)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/qa/session/qa-1") as response:
            assert json.load(response)["session"]["session_id"] == "qa-1"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    assert calls == ["prepare", "draft", "review", "session"]
