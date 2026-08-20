from __future__ import annotations

import json

from aic_retrieval.hybrid_query_planner import (
    HybridQueryPlanner,
    LocalHybridQueryPlanner,
    gemini_hybrid_plan_payload,
    parse_gemini_hybrid_plan,
)
from aic_retrieval.translation import TranslationConfig


def _gemini_response(value: dict) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(value, ensure_ascii=False)}]}}]}


def test_local_planner_routes_visual_and_exact_date_differently() -> None:
    planner = LocalHybridQueryPlanner()
    visual = planner.plan("a person wearing a red shirt", "a person wearing a red shirt")
    assert visual.enabled_retrievers == ("clip",)
    assert visual.intent == "visual"
    date = planner.plan("video xuất bản ngày 31 tháng 8 năm 2024")
    assert "bm25" in date.enabled_retrievers
    assert date.lexical_text_query.endswith("2024")
    assert date.source_filters["bm25"] == ("metadata",)


def test_local_planner_marks_asr_and_ocr_coverage() -> None:
    planner = LocalHybridQueryPlanner()
    asr = planner.plan('người nói "xin chào"')
    assert set(asr.enabled_retrievers) >= {"bge", "bm25"}
    assert any("ASR" in warning for warning in asr.warnings)
    assert asr.source_filters["bm25"] == ("asr",)
    ocr = planner.plan('biển hiệu chữ "Samsung"')
    assert "bm25" in ocr.enabled_retrievers
    assert any("OCR" in warning for warning in ocr.warnings)


def test_gemini_payload_is_compact_and_parser_validates_routes() -> None:
    payload = gemini_hybrid_plan_payload("người mặc áo đỏ")
    prompt = payload["contents"][0]["parts"][0]["text"]
    assert "clip_query" in prompt and "routes" in prompt
    plan = parse_gemini_hybrid_plan(
        _gemini_response(
            {
                "clip_query": "a person wearing a red shirt",
                "semantic_query": "người mặc áo đỏ",
                "lexical_query": "áo đỏ",
                "intent": "visual",
                "routes": ["clip"],
                "reasons": ["visible clothing color"],
                "structured_filters": {
                    "object_label": "person",
                    "attribute_color": "red",
                    "enable_ocr": False,
                    "enable_asr": False,
                },
            }
        ),
        "người mặc áo đỏ",
    )
    assert plan.planner_source == "gemini"
    assert plan.profile == "clip"
    assert plan.structured_filter_suggestions["object_label"] == "person"
    assert plan.structured_filter_suggestions["attribute_color"] == "red"


def test_unconfigured_gemini_falls_back_to_local_plan() -> None:
    planner = HybridQueryPlanner(TranslationConfig(provider="gemini"))
    result = planner.plan_with_trace("a person walking outdoors")
    assert result.plan.planner_source == "local_rules"
    assert result.trace.status == "UNAVAILABLE"
    assert result.trace.fallback == "local_rules"
    assert result.trace.raw_text is None
    assert any("unavailable" in warning for warning in result.plan.warnings)


def test_configured_gemini_exposes_safe_raw_parsed_and_validated_trace(monkeypatch) -> None:
    response_payload = _gemini_response(
        {
            "clip_query": "a red car on a city street",
            "semantic_query": "xe ô tô màu đỏ trên đường phố",
            "lexical_query": "xe đỏ",
            "intent": "visual",
            "routes": ["clip"],
            "reasons": ["visible object and color"],
            "structured_filters": {"object_label": "car", "attribute_color": "red"},
        }
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(response_payload, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(
        "aic_retrieval.hybrid_query_planner.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    planner = HybridQueryPlanner(
        TranslationConfig(
            provider="gemini",
            api_key="secret-that-must-not-leak",
            api_url="https://example.invalid/v1beta",
            model="gemini-test",
        )
    )

    result = planner.plan_with_trace("xe ô tô màu đỏ trên đường phố")
    trace = result.trace.to_dict()

    assert trace["status"] == "SUCCEEDED"
    assert trace["model"] == "gemini-test"
    assert trace["parsed_output"]["routes"] == ["clip"]
    assert trace["validated_plan"]["structured_filter_suggestions"]["object_label"] == "car"
    assert "secret-that-must-not-leak" not in json.dumps(trace)
    assert "a red car on a city street" in trace["raw_text"]
