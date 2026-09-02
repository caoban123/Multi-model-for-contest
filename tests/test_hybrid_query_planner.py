from __future__ import annotations

import json

from aic_retrieval.hybrid_query_planner import (
    HybridQueryPlanner,
    HybridTextCoverage,
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
    assert date.enabled_retrievers == ("bm25",)
    assert date.fusion_method == "none"
    assert date.lexical_text_query.endswith("2024")
    assert date.source_filters["bm25"] == ("metadata",)


def test_local_planner_marks_asr_and_ocr_coverage() -> None:
    planner = LocalHybridQueryPlanner()
    asr = planner.plan('người nói "xin chào"')
    assert set(asr.enabled_retrievers) >= {"bge", "bm25"}
    assert any("ASR" in warning for warning in asr.warnings)
    assert asr.source_filters["bm25"] == ("asr",)
    ocr = planner.plan('biển hiệu chữ "Samsung"')
    assert ocr.enabled_retrievers == ("bm25",)
    assert any("OCR" in warning for warning in ocr.warnings)


def test_local_planner_uses_runtime_phase5_coverage() -> None:
    planner = LocalHybridQueryPlanner(HybridTextCoverage(
        ocr_records=10_000,
        ocr_videos=800,
        asr_segments=50_000,
        asr_videos=873,
        total_videos=873,
    ))
    assert not any("OCR" in warning for warning in planner.plan('biển hiệu chữ "Samsung"').warnings)
    assert not any("ASR" in warning for warning in planner.plan('người nói "xin chào"').warnings)


def test_local_planner_uses_bge_for_semantic_text_and_rrf_only_for_mixed_queries() -> None:
    planner = LocalHybridQueryPlanner()
    semantic = planner.plan("news report about land subsidence")
    assert semantic.enabled_retrievers == ("bge",)
    assert semantic.fusion_method == "none"

    mixed = planner.plan('người nói "xin chào"')
    assert mixed.enabled_retrievers == ("clip", "bge", "bm25")
    assert mixed.fusion_method == "rrf"


def test_gemini_payload_is_compact_and_parser_validates_routes() -> None:
    payload = gemini_hybrid_plan_payload("người mặc áo đỏ")
    prompt = payload["contents"][0]["parts"][0]["text"]
    assert "clip_query" in prompt and "routes" in prompt
    plan = parse_gemini_hybrid_plan(
        _gemini_response(
            {
                "clip_query": "a person wearing a red shirt",
                "clip_queries": [
                    "a person wearing a red shirt",
                    "red clothing in a news studio",
                ],
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
    assert plan.clip_queries() == (
        "a person wearing a red shirt",
        "red clothing in a news studio",
    )
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


def test_gemini_routes_are_normalized_by_local_intent_guardrails(monkeypatch) -> None:
    response_payload = _gemini_response(
        {
            "clip_query": "news report about land subsidence",
            "semantic_query": "news report about land subsidence",
            "lexical_query": "land subsidence",
            "intent": "mixed",
            "routes": ["clip", "bge", "bm25"],
            "reasons": ["model requested every route"],
            "structured_filters": {},
        }
    )

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self) -> bytes: return json.dumps(response_payload).encode("utf-8")

    monkeypatch.setattr("aic_retrieval.hybrid_query_planner.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    planner = HybridQueryPlanner(TranslationConfig(provider="gemini", api_key="x", api_url="https://example.invalid", model="gemini-test"))

    result = planner.plan_with_trace("news report about land subsidence")

    assert result.trace.parsed_output["routes"] == ["clip", "bge", "bm25"]
    assert result.plan.enabled_retrievers == ("bge",)
    assert result.plan.fusion_method == "none"


def test_gemini_numeric_visual_narrative_does_not_force_bm25(monkeypatch) -> None:
    response_payload = _gemini_response(
        {
            "clip_query": "five people playing beside a yellow animal and hiding a pumpkin",
            "clip_queries": [
                "five people playing beside a yellow animal",
                "a person hiding a pumpkin",
                "a man waking up a yellow animal",
            ],
            "semantic_query": "nhóm 5 người chơi cạnh con vật vàng",
            "lexical_query": "5 người bí đỏ",
            "intent": "mixed",
            "routes": ["clip", "bm25"],
            "reasons": ["visible sequence"],
            "structured_filters": {},
        }
    )

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self) -> bytes: return json.dumps(response_payload).encode("utf-8")

    monkeypatch.setattr("aic_retrieval.hybrid_query_planner.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    planner = HybridQueryPlanner(
        TranslationConfig(provider="gemini", api_key="x", api_url="https://example.invalid", model="gemini-test")
    )
    result = planner.plan_with_trace(
        "Nhóm 5 người chơi cạnh con vật màu vàng. Một người mang quả bí đỏ đi giấu."
    )

    assert result.trace.parsed_output["routes"] == ["clip", "bm25"]
    assert result.plan.enabled_retrievers == ("clip",)
    assert result.plan.fusion_method == "none"


def test_local_planner_routes_visual_ocr_question_to_clip_and_bm25() -> None:
    planner = LocalHybridQueryPlanner(HybridTextCoverage(ocr_records=845_478, ocr_videos=873))

    plan = planner.plan(
        "Cảnh quay chiếc xe màu trắng rẽ trái. Hỏi con số ghi trên hông xe là số mấy?"
    )

    assert plan.enabled_retrievers == ("clip", "bm25")
    assert plan.fusion_method == "rrf"
    assert plan.source_filters["bm25"] == ("ocr",)


def test_local_planner_routes_visual_conversation_address_question_to_all_channels() -> None:
    planner = LocalHybridQueryPlanner(HybridTextCoverage(
        ocr_records=845_478,
        ocr_videos=873,
        asr_segments=563,
        asr_videos=2,
        total_videos=873,
    ))

    plan = planner.plan(
        "Một cụ ông trò chuyện với nhóm người nước ngoài. Hỏi quán trọ nằm trên đường nào?"
    )

    assert plan.enabled_retrievers == ("clip", "bge", "bm25")
    assert plan.fusion_method == "rrf"
    assert plan.source_filters["bge"] == ("asr", "ocr")
    assert plan.source_filters["bm25"] == ("asr", "ocr")
    assert any("ASR coverage is partial" in warning for warning in plan.warnings)


def test_local_planner_does_not_enable_bge_for_generic_vietnamese_preposition() -> None:
    planner = LocalHybridQueryPlanner()

    plan = planner.plan("người mặc áo đỏ đi về phía chiếc xe")

    assert plan.enabled_retrievers == ("clip",)


def test_local_planner_decomposes_temporal_visual_narrative_without_answer_clause() -> None:
    planner = LocalHybridQueryPlanner()

    plan = planner.plan(
        "Cảnh quay bên trong xe, sau đó xe màu trắng rẽ trái, cuối cùng xe dừng lại. "
        "Hỏi số ghi trên xe là số mấy?"
    )

    assert len(plan.clip_queries()) == 4
    assert all("hỏi" not in query.casefold() for query in plan.clip_queries())
    assert any("xe màu trắng rẽ trái" in query for query in plan.clip_queries())


def test_long_visual_counts_do_not_enable_bm25_or_generic_person_constraint() -> None:
    planner = LocalHybridQueryPlanner(HybridTextCoverage(ocr_records=845_478, ocr_videos=873))

    plan = planner.plan(
        "Nhóm 5 người đang chơi đùa bên cạnh một con vật màu vàng. "
        "Một người mang một vật trông như trái bí đỏ đi giấu. "
        "Người đàn ông thức dậy và đánh thức con vật."
    )

    assert plan.enabled_retrievers == ("clip",)
    assert plan.fusion_method == "none"
    assert len(plan.clip_queries()) == 4
    assert plan.structured_filter_suggestions["enable_objects"] is False
    assert plan.structured_filter_suggestions["attribute_color"] == "vàng"


def test_visual_counts_are_distinct_from_explicit_ocr_numbers() -> None:
    planner = LocalHybridQueryPlanner(HybridTextCoverage(ocr_records=845_478, ocr_videos=873))

    people = planner.plan("Cận cảnh 3 vận động viên đua xe đạp mặc áo xanh và vàng")
    countdown = planner.plan("Nhóm xe rẽ phải cạnh đèn xanh đang đếm ngược đến 13 giây")

    assert people.enabled_retrievers == ("clip",)
    assert countdown.enabled_retrievers == ("clip", "bm25")
    assert countdown.source_filters["bm25"] == ("ocr",)


def test_pumpkin_compound_is_not_interpreted_as_red_attribute() -> None:
    plan = LocalHybridQueryPlanner().plan("Một người đang mang quả bí đỏ đi giấu")

    assert plan.structured_filter_suggestions["attribute_color"] is None


def test_four_stage_visual_recipe_keeps_every_event_query() -> None:
    plan = LocalHybridQueryPlanner().plan(
        "Đầu bếp lăn nguyên liệu xiên que qua rau xanh và đỏ. "
        "Nguyên liệu được chuyển sang đĩa bột trắng. "
        "Đầu bếp xoay que xiên nhiều lần trên bột. "
        "Cuối cùng nguyên liệu phủ bột được đặt sang đĩa."
    )

    assert len(plan.clip_queries()) == 5
    assert any("lăn nguyên liệu" in query for query in plan.clip_queries())
    assert any("chuyển sang đĩa bột" in query for query in plan.clip_queries())
    assert any("xoay que xiên" in query for query in plan.clip_queries())
    assert any("phủ bột được đặt sang đĩa" in query for query in plan.clip_queries())
