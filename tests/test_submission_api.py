from __future__ import annotations

import json
import threading
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import numpy as np
import pytest

import aic_retrieval.retrieval_ui as retrieval_ui
from aic_retrieval.retrieval_ui import RetrievalUiConfig, RetrievalUiService
from aic_retrieval.qa_schema import ReviewDecision
from aic_retrieval.submission_session import SubmissionSessionStore


def post(port: int, route: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def test_service_kis_agent_keeps_retrieval_frame_separate_from_official_frame(tmp_path: Path) -> None:
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.submission_store = SubmissionSessionStore(tmp_path / "sessions.sqlite3", tmp_path / "output")
    service.hybrid_engine = object()
    service.agent_search = lambda *args, **kwargs: {
        "query_plan": {"profile": "clip"},
        "agent_trace": {"status": "SUCCEEDED", "raw_text": "{\"routes\":[\"clip\"]}"},
        "channel_hit_counts": {"clip": 1},
        "fusion_method": "single_channel_rank",
        "structured_constraints": {"applied": False},
        "latency_ms": {"clip": 1.0, "total": 1.2},
        "video_results": [{
            "rank": 1,
            "video_id": "L21_V001",
            "keyframe_id": 7,
            "frame_idx": 210,
            "pts_time": 7.0,
            "image_url": "/keyframe?path=x.jpg",
            "raw_score": 0.9,
            "provenance": {"retriever_ranks": {"clip": 1}},
            "metadata": {"title": "News presenter", "author": "Channel"},
            "source_type": "clip",
        }],
    }
    session_id = service.submission_start()["session_id"]

    result = service.submission_agent_run(session_id, "query-1-kis", "KIS", "red shirt")

    candidate = result["candidates"][0]
    assert candidate["frame_idx"] == 210
    assert candidate["official_frame_id"] is None
    assert candidate["mapping_status"] == "REQUIRES_MANUAL_OFFICIAL_FRAME_ID"
    assert candidate["metadata"]["title"] == "News presenter"
    assert result["agent_trace"]["status"] == "SUCCEEDED"
    assert result["channel_hit_counts"] == {"clip": 1}
    assert result["fusion_method"] == "single_channel_rank"
    assert result["structured_constraints"]["applied"] is False

    confirmed = service.submission_confirm(
        session_id,
        "query-1-kis",
        "KIS",
        [{"video_id": "L21_V001", "frame_id": 999, "mapping_source": "manual_official"}],
        source={"candidate_id": candidate["candidate_id"]},
    )
    assert confirmed["queries"][0]["csv_preview"] == "L21_V001, 999\n"
    assert "210" not in confirmed["queries"][0]["csv_preview"]
    for number in range(2, 5):
        expanded_queue = service.submission_confirm(
            session_id,
            f"query-{number}-kis",
            "KIS",
            [{"video_id": f"L21_V00{number + 2}", "frame_id": number * 100, "mapping_source": "manual_official"}],
        )
    assert len(expanded_queue["queries"]) == 4
    edited = service.submission_apply_csv(
        session_id,
        "query-1-kis",
        "L21_V002,123\nL21_V003,456\n",
    )
    edited_by_id = {item["query_id"]: item for item in edited["queries"]}
    assert edited_by_id["query-1-kis"]["csv_preview"] == "L21_V002, 123\nL21_V003, 456\n"
    assert edited_by_id["query-1-kis"]["source"]["csv_edited"] is True
    assert edited_by_id["query-2-kis"]["csv_preview"] == "L21_V004, 200\n"
    assert edited_by_id["query-3-kis"]["csv_preview"] == "L21_V005, 300\n"
    assert edited_by_id["query-4-kis"]["csv_preview"] == "L21_V006, 400\n"
    try:
        service.submission_confirm(
            session_id,
            "query-9-kis",
            "KIS",
            [{"video_id": "L21_V001", "frame_id": True, "mapping_source": "manual_official"}],
        )
    except ValueError as exc:
        assert "frame_id" in str(exc)
    else:
        raise AssertionError("boolean must not be accepted as an official frame_id")


def test_agent_structured_constraints_are_opt_in_and_rrf_fused() -> None:
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.object_service = object()
    service.attribute_service = SimpleNamespace(available=True)
    service.phase5_service = SimpleNamespace(available=False)
    service.metadata_docs = []
    service.structured_search = lambda *_args, **kwargs: {
        "results": [
            {"video_id": "L21_V002", "rank": 1, "keyframe_id": 2, "frame_idx": 20, "pts_time": 2.0, "provenance": ["clip", "object"]},
            {"video_id": "L21_V001", "rank": 2, "keyframe_id": 1, "frame_idx": 10, "pts_time": 1.0},
        ],
        "channel_counts": {"object_frames": 2, "attribute_frames": 2},
    }
    plan = retrieval_ui.HybridQueryPlan(
        original_query="person wearing red",
        visual_clip_query_en="person wearing red",
        semantic_text_query="person wearing red",
        lexical_text_query="person wearing red",
        intent="visual",
        enabled_retrievers=("clip",),
        profile="clip",
        fusion_method="none",
        reasons=("visible constraint",),
        structured_filter_suggestions={
            "enable_objects": True,
            "object_label": "person",
            "enable_attributes": True,
            "attribute_color": "red",
        },
    )

    fused, diagnostics = service._apply_agent_structured_constraints(
        plan,
        [
            {"video_id": "L21_V001", "rank": 1, "keyframe_id": 1, "frame_idx": 10, "pts_time": 1.0, "provenance": {"retriever_ranks": {"clip": 1}}},
            {"video_id": "L21_V003", "rank": 2, "keyframe_id": 3, "frame_idx": 30, "pts_time": 3.0},
        ],
        3,
    )

    assert diagnostics["applied"] is True
    assert diagnostics["fusion_method"] == "rrf"
    assert fused[0]["video_id"] == "L21_V001"
    assert fused[0]["provenance"]["retriever_ranks"] == {"clip": 1, "structured": 2}
    assert next(item for item in fused if item["video_id"] == "L21_V002")["provenance"]["structured_modalities"] == ["clip", "object"]


def test_reviewed_qa_and_trake_adapters_require_manual_official_frames(tmp_path: Path) -> None:
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.submission_store = SubmissionSessionStore(tmp_path / "sessions.sqlite3", tmp_path / "output")
    session_id = service.submission_start()["session_id"]
    review = SimpleNamespace(
        review_id="review-qa",
        decision=ReviewDecision.CONFIRMED,
        final_answer="short answer",
        selected_evidence_refs=("e1",),
    )
    evidence = SimpleNamespace(evidence_id="e1", keyframe_id=7, video_id="L21_V011")
    qa_state = SimpleNamespace(
        reviews=[review],
        pack=SimpleNamespace(evidence_refs=(evidence,)),
        session=SimpleNamespace(request=SimpleNamespace(query_id="query-2-qa")),
    )
    service.qa_workflow = SimpleNamespace(get=lambda _session_id: qa_state)

    qa = service.submission_import_qa(session_id, "qa-session", "review-qa", 1450, "manual_official")
    assert qa["queries"][0]["csv_preview"] == 'L21_V011, 1450, "short answer"\n'

    service.trake_workflow = SimpleNamespace(export=lambda *_args: {"export": {
        "query_id": "query-3-trake",
        "video_id": "L21_V001",
        "human_confirmed": True,
        "same_video": True,
        "temporally_ordered": True,
        "events": [{"event_id": "e1", "frame_idx": 10}, {"event_id": "e2", "frame_idx": 20}],
    }})
    trake = service.submission_import_trake(session_id, "trake-session", "review-trake", [1200, 1850], "manual_official")
    assert trake["queries"][1]["csv_preview"] == "L21_V001, 1200, 1850\n"
    assert "10" not in trake["queries"][1]["csv_preview"]

    service.submission_confirm(
        session_id,
        "query-1-kis",
        "KIS",
        [{"video_id": "L21_V002", "frame_id": 999, "mapping_source": "manual_official"}],
        source={"score": 0.99, "provenance": {"clip": 1}},
    )
    done = service.submission_done(session_id)
    with zipfile.ZipFile(done["zip_path"]) as archive:
        names = archive.namelist()
        values = {name: archive.read(name).decode("utf-8") for name in names}
    assert names == [
        "submission/query-1-kis.csv",
        "submission/query-2-qa.csv",
        "submission/query-3-trake.csv",
    ]
    assert values["submission/query-1-kis.csv"] == "L21_V002, 999\n"
    assert values["submission/query-2-qa.csv"] == 'L21_V011, 1450, "short answer"\n'
    assert values["submission/query-3-trake.csv"] == "L21_V001, 1200, 1850\n"
    assert all("score" not in value and "provenance" not in value for value in values.values())


def test_submission_qa_import_rejects_legacy_cross_video_review(tmp_path: Path) -> None:
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.submission_store = SubmissionSessionStore(tmp_path / "sessions.sqlite3", tmp_path / "output")
    submission_session_id = service.submission_start()["session_id"]
    review = SimpleNamespace(
        review_id="review-cross-video",
        decision=ReviewDecision.CONFIRMED,
        final_answer="answer",
        selected_evidence_refs=("e1", "e2"),
    )
    qa_state = SimpleNamespace(
        reviews=[review],
        pack=SimpleNamespace(evidence_refs=(
            SimpleNamespace(evidence_id="e1", keyframe_id=1, video_id="L21_V001"),
            SimpleNamespace(evidence_id="e2", keyframe_id=2, video_id="L21_V002"),
        )),
        session=SimpleNamespace(request=SimpleNamespace(query_id="query-cross-video")),
    )
    service.qa_workflow = SimpleNamespace(get=lambda _session_id: qa_state)

    with pytest.raises(ValueError, match="exactly one video"):
        service.submission_import_qa(submission_session_id, "qa-session", "review-cross-video", 100, "manual_official")


def test_submission_http_session_queue_validate_done_and_download(tmp_path: Path, monkeypatch) -> None:
    class FakeService:
        def __init__(self, config):
            self.config = config
            self.index = np.zeros((1, 2))
            self.metadata_docs = []
            self.object_service = None
            self.attribute_service = type("A", (), {"available": False})()
            self.translator = type("T", (), {"is_configured": False, "config": type("C", (), {"provider": "x", "model": "x"})()})()
            self.store = SubmissionSessionStore(tmp_path / "http.sqlite3", tmp_path / "outputs")
            self.submission_store = self.store

        def submission_start(self): return self.store.start()
        def submission_session(self, session_id): return self.store.get(session_id)
        def submission_refine_candidate(self, session_id, query_id, query, video_id, **kwargs):
            return {
                "schema_version": "candidate-local-refinement-v1",
                "session_id": session_id,
                "query_id": query_id,
                "video_id": video_id,
                "status": "READY_FOR_REVIEW",
                "frames": [{"keyframe_id": kwargs.get("seed_keyframe_id"), "frame_idx": 120}],
            }
        def submission_confirm(self, session_id, query_id, task, predictions, event_count=None, source=None):
            rows = tuple(retrieval_ui.OfficialPrediction(row["video_id"], (int(row["frame_id"]),)) for row in predictions)
            return self.store.confirm_query(session_id, retrieval_ui.OfficialQuery(query_id, task, rows, event_count), mapping_sources=tuple(row["mapping_source"] for row in predictions), source=source)
        def submission_apply_csv(self, session_id, query_id, csv_text):
            current = self.store.get(session_id)
            item = next(row for row in current["queries"] if row["query_id"] == query_id)
            query = retrieval_ui.parse_official_csv(query_id, item["task"], csv_text, event_count=item["event_count"])
            return self.store.confirm_query(session_id, query, mapping_sources=tuple("manual_official" for _ in query.predictions), source={"csv_edited": True})
        def submission_validate(self, session_id): return self.store.validate(session_id)
        def submission_done(self, session_id): return self.store.done(session_id)
        def submission_zip_path(self, session_id): return self.store.zip_path(session_id)

    monkeypatch.setattr(retrieval_ui, "RetrievalUiService", FakeService)
    static = tmp_path / "web"; static.mkdir(); (static / "index.html").write_text("ok", encoding="utf-8")
    config = RetrievalUiConfig(tmp_path, tmp_path / "r", tmp_path / "i", tmp_path / "m", static, {"L21"}, submission_static_dir=static)
    server = retrieval_ui.run_server(config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    port = server.server_address[1]
    try:
        session = post(port, "/api/submission/session/start", {})
        session_id = session["session_id"]
        refined = post(port, "/api/submission/candidate/refine", {
            "session_id": session_id,
            "query_id": "query-1-kis",
            "query": "person entering a room",
            "video_id": "L21_V001",
            "seed_keyframe_id": 7,
            "radius_seconds": 40,
            "frame_limit": 12,
            "use_gemini": False,
        })
        assert refined["video_id"] == "L21_V001"
        assert refined["frames"][0]["keyframe_id"] == 7
        queued = post(port, "/api/submission/query/confirm", {
            "session_id": session_id,
            "query_id": "query-1-kis",
            "task": "KIS",
            "predictions": [{"video_id": "L21_V001", "frame_id": 1234, "mapping_source": "manual_official"}],
        })
        assert queued["submission_queue"][0]["prediction_count"] == 1
        edited = post(port, "/api/submission/query/csv", {
            "session_id": session_id,
            "query_id": "query-1-kis",
            "csv_text": "L21_V002,4321\n",
        })
        assert edited["queries"][0]["csv_preview"] == "L21_V002, 4321\n"
        assert post(port, "/api/submission/session/validate", {"session_id": session_id})["valid"] is True
        assert post(port, "/api/submission/session/done", {"session_id": session_id})["status"] == "DONE"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/submission/session/{session_id}/download") as response:
            assert response.headers["Content-Type"] == "application/zip"
            assert response.read(2) == b"PK"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_http_translate_and_time_to_frame_routes_are_callable(tmp_path: Path, monkeypatch) -> None:
    class FakeService:
        def __init__(self, config):
            self.config = config
            self.index = np.zeros((1, 2))
            self.metadata_docs = []
            self.object_service = None
            self.attribute_service = type("A", (), {"available": False})()
            self.translator = type("T", (), {
                "is_configured": True,
                "config": type("C", (), {"provider": "gemini", "model": "test-model"})(),
            })()

        def translate(self, text):
            return {"source_text": text, "translated_text": "red shirt", "provider": "gemini", "model": "test-model"}

        def time_to_frame(self, video_id, minutes, seconds):
            return {"video_id": video_id, "minutes": minutes, "seconds": seconds, "estimated_frame_idx": 1875}

    monkeypatch.setattr(retrieval_ui, "RetrievalUiService", FakeService)
    static = tmp_path / "web"
    static.mkdir()
    (static / "index.html").write_text("ok", encoding="utf-8")
    config = RetrievalUiConfig(tmp_path, tmp_path / "r", tmp_path / "i", tmp_path / "m", static, {"L21"})
    server = retrieval_ui.run_server(config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/translate?{urlencode({'q': 'ao do'})}") as response:
            assert json.load(response)["translated_text"] == "red shirt"
        params = urlencode({"video_id": "L21_V001", "minutes": 1, "seconds": 2.5})
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/time-to-frame?{params}") as response:
            assert json.load(response)["estimated_frame_idx"] == 1875
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
