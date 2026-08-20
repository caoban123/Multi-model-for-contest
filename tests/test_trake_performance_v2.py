from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import aic_retrieval.retrieval_ui as retrieval_ui_module
from aic_retrieval.retrieval_ui import RetrievalUiService
from aic_retrieval.trake_config import load_trake_config
from aic_retrieval.trake_schema import TrakeEvent, TrakeRequest
from aic_retrieval.trake_workflow import TrakeDomainError
from tools.phase8_profile import profile_store, render_markdown


def test_profiler_reads_store_without_labels_and_reports_p50_p95(tmp_path: Path) -> None:
    store = tmp_path / "trake.sqlite3"
    with sqlite3.connect(store) as connection:
        connection.execute("CREATE TABLE sessions(session_id TEXT, created_at REAL, state_json TEXT)")
        for index, retrieval in enumerate((10.0, 20.0, 30.0), start=1):
            state = {"algorithm": "trake_v2", "stage_timings_ms": {"planner": 1.0, "retrieval": retrieval, "alignment": 2.0}}
            connection.execute("INSERT INTO sessions VALUES(?,?,?)", (f"s{index}", float(index), json.dumps(state)))
    modified_before = store.stat().st_mtime_ns

    profile = profile_store(store, algorithm="trake_v2")

    assert profile["session_count"] == 3 and profile["warm_session_count"] == 2
    assert profile["stage_latency_ms"]["retrieval"] == {"p50": 20.0, "p95": 30.0, "sample_count": 3}
    assert profile["warm_stage_latency_ms"]["retrieval"] == {"p50": 25.0, "p95": 30.0, "sample_count": 2}
    assert profile["quality_claim"] is None
    assert "| retrieval | 20.000 | 30.000 | 3 |" in render_markdown(profile)
    assert store.stat().st_mtime_ns == modified_before


def test_profiler_missing_store_is_controlled(tmp_path: Path) -> None:
    profile = profile_store(tmp_path / "missing.sqlite3")
    assert profile["availability"] == "unavailable"
    assert profile["session_count"] == 0


def test_request_embedding_cache_batches_variants_and_reuses_vectors(monkeypatch: object) -> None:
    class Encoder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode_texts(self, texts: list[str]) -> np.ndarray:
            self.calls.append(texts)
            return np.ones((len(texts), 2), dtype="float32")

    service = RetrievalUiService.__new__(RetrievalUiService)
    service.trake_runtime_config = load_trake_config(Path(__file__).parents[1] / "configs" / "phase8_trake_v2.json")
    service.trake_text_embedding_cache = {}
    service.trake_workflow = SimpleNamespace(session_provenance={"clip_model_fingerprint": "clip-fixture"})
    service.config = SimpleNamespace(clip_model_id="clip-fixture")
    encoder = Encoder()
    service._encoder = lambda: encoder
    service.index = np.ones((1, 2), dtype="float32")
    service.refs = []
    service._trake_retrieve_event = lambda _event, _pool, _clip, timing_sink=None: []
    monkeypatch.setattr(retrieval_ui_module, "search_numpy_index_batch", lambda _index, _refs, vectors, _pool: [[] for _ in vectors])
    request = TrakeRequest(
        "q",
        "event",
        (TrakeEvent("e1", 1, "event", "event", query_variants=("event variant",)),),
    )

    first = service._trake_retrieve_request(request, 10)
    second = service._trake_retrieve_request(request, 10)

    assert encoder.calls == [["event", "event variant"]]
    assert len(service.trake_text_embedding_cache) == 2
    assert set(first.stage_timings_ms) == {"embedding", "retrieval", "fusion"}
    assert second.stage_timings_ms["embedding"] >= 0


def test_missing_local_encoder_model_is_a_typed_retrieval_failure() -> None:
    class Encoder:
        def encode_texts(self, _texts: list[str]) -> np.ndarray:
            raise OSError("private cache path and provider details")

    service = RetrievalUiService.__new__(RetrievalUiService)
    service.trake_runtime_config = load_trake_config(Path(__file__).parents[1] / "configs" / "phase8_trake_v2.json")
    service.trake_text_embedding_cache = {}
    service.trake_workflow = SimpleNamespace(session_provenance={"clip_model_fingerprint": "clip-fixture"})
    service.config = SimpleNamespace(clip_model_id="clip-fixture")
    service._encoder = lambda: Encoder()
    request = TrakeRequest("q", "event", (TrakeEvent("e1", 1, "event", "event"),))

    with pytest.raises(TrakeDomainError) as raised:
        service._trake_retrieve_request(request, 10)

    assert raised.value.code == "RETRIEVAL_UNAVAILABLE"
    assert raised.value.stage == "RETRIEVAL"
    assert "private" not in str(raised.value)
