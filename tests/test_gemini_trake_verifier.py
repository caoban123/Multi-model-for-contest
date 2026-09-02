from __future__ import annotations

import json

from aic_retrieval.gemini_trake_verifier import (
    GeminiTrakeVerifier,
    gemini_trake_verify_payload,
    parse_gemini_trake_response,
    prepare_trake_evidence,
)
from aic_retrieval.translation import TranslationConfig


def test_trake_payload_attaches_labeled_images(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    safe = {"query": "first cook then serve", "event_descriptions": ["cook", "serve"], "chains": [{
        "chain_id": "c1",
        "events": [{"event_id": "e1", "description": "cook", "pts_time": 1.0, "keyframe_path": str(image)}],
    }]}
    prepared = prepare_trake_evidence(safe, lambda value: tmp_path / value if not (tmp_path / value).is_absolute() else (tmp_path / value), 4)
    payload = gemini_trake_verify_payload(safe, prepared["images"])
    parts = payload["contents"][0]["parts"]
    assert "Do not invent chain IDs" in parts[0]["text"]
    assert "Chain ID: c1 | Event ID: e1" in parts[1]["text"]
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"


def test_trake_parser_drops_unknown_chain_event_and_duplicates() -> None:
    response = {"candidates": [{"content": {"parts": [{"text": json.dumps({"chains": [
        {"chain_id": "bad", "support": 1, "temporal_consistency": True, "event_support": []},
        {"chain_id": "c1", "support": 0.8, "temporal_consistency": True, "reason": "coherent", "event_support": [
            {"event_id": "invented", "support": 1, "reason": "bad"},
            {"event_id": "e1", "support": 0.9, "reason": "visible"},
        ]},
        {"chain_id": "c1", "support": 0.1, "temporal_consistency": False, "event_support": []},
    ]})}]}}]}
    rows, _ = parse_gemini_trake_response(response, allowed_events={"c1": {"e1"}})
    assert len(rows) == 1
    assert rows[0]["chain_id"] == "c1"
    assert rows[0]["event_support"] == [{"event_id": "e1", "support": 0.9, "reason": "visible"}]


def test_trake_verifier_calls_gemini_with_only_existing_images(tmp_path, monkeypatch) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    calls = []

    def fake_call(config, payload):
        calls.append(payload)
        return {"candidates": [{"content": {"parts": [{"text": '{"chains":[{"chain_id":"c1","support":0.7,"temporal_consistency":true,"reason":"ok","event_support":[{"event_id":"e1","support":0.8,"reason":"visible"}]}]}'}]}}]}

    monkeypatch.setattr("aic_retrieval.gemini_trake_verifier._call_gemini_visual", fake_call)
    verifier = GeminiTrakeVerifier(TranslationConfig(api_key="test", provider="gemini"))
    result = verifier({"query": "q", "chains": [{"chain_id": "c1", "events": [{"event_id": "e1", "keyframe_path": str(image)}]}]})
    assert result["status"] == "ADVISORY_READY"
    assert result["chains"][0]["support"] == 0.7
    assert len(calls) == 1
