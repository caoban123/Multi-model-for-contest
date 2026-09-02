from __future__ import annotations

import json

from aic_retrieval.gemini_visual_reranker import (
    GeminiVisualReranker,
    gemini_visual_rerank_payload,
    parse_gemini_visual_rerank_response,
)
from aic_retrieval.translation import TranslationConfig


def test_visual_rerank_payload_attaches_allowlisted_images(tmp_path) -> None:
    path = tmp_path / "one.jpg"
    path.write_bytes(b"\xff\xd8\xff\xd9")
    payload = gemini_visual_rerank_payload("person cooking", [{
        "candidate_id": "L21_V001:k2",
        "path": path,
        "mime_type": "image/jpeg",
    }])
    parts = payload["contents"][0]["parts"]
    assert "Do not invent a candidate ID" in parts[0]["text"]
    assert parts[1]["text"] == "Candidate ID: L21_V001:k2"
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"


def test_visual_rerank_parser_drops_unknown_and_duplicate_candidates() -> None:
    response = {"candidates": [{"content": {"parts": [{"text": json.dumps({"candidates": [
        {"candidate_id": "invented", "relevance": 1, "matched_events": [], "reason": "bad"},
        {"candidate_id": "L21_V001:k2", "relevance": 0.8, "matched_events": ["turning"], "reason": "match"},
        {"candidate_id": "L21_V001:k2", "relevance": 0.2, "matched_events": [], "reason": "duplicate"},
    ]})}]}}]}
    items, _ = parse_gemini_visual_rerank_response(response, allowed_ids={"L21_V001:k2"})
    assert len(items) == 1
    assert items[0].candidate_id == "L21_V001:k2"
    assert items[0].relevance == 0.8


def test_visual_reranker_caches_validated_response(tmp_path, monkeypatch) -> None:
    path = tmp_path / "one.jpg"
    path.write_bytes(b"\xff\xd8\xff\xd9")
    calls = []

    def fake_call(config, payload):
        calls.append(payload)
        return {"candidates": [{"content": {"parts": [{"text": '{"candidates":[{"candidate_id":"L21_V001:k2","relevance":0.9,"matched_events":["action"],"reason":"visible"}]}'}]}}]}

    monkeypatch.setattr("aic_retrieval.gemini_visual_reranker._call_gemini_visual", fake_call)
    reranker = GeminiVisualReranker(TranslationConfig(api_key="test", provider="gemini"), max_images=4)
    frames = [{"video_id": "L21_V001", "keyframe_id": 2, "keyframe_path": str(path)}]
    first = reranker.rerank("person cooking", frames)
    second = reranker.rerank("person cooking", frames)

    assert first["items"][0]["relevance"] == 0.9
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(calls) == 1
