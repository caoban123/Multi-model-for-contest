from __future__ import annotations

from aic_retrieval.trake_decomposition import PLANNER_LOW_CONFIDENCE, PLANNER_V2_VERSION, TrakePlannerV2
from aic_retrieval.trake_schema import EventSource, TRAKE_V2_SCHEMA_VERSION, request_from_dict


def test_planner_v2_produces_global_context_structured_events_and_routing() -> None:
    result = TrakePlannerV2().decompose(
        "q",
        'In a football stadium, a red player kicks the ball then a screen shows "GOAL" then the crowd says "hooray"',
    )

    request = result.request
    assert request.schema_version == TRAKE_V2_SCHEMA_VERSION
    assert request.planner_version == PLANNER_V2_VERSION
    assert request.global_context == "football match / stadium"
    assert len(request.events) == 3
    assert {"clip", "attribute", "object"} <= set(request.events[0].modalities)
    assert request.events[0].attribute_constraints[0]["color"] == "red"
    assert request.events[0].object_constraints[0]["label"] == "ball"
    assert "ocr" in request.events[1].modalities and request.events[1].ocr_terms == ("GOAL",)
    assert "asr" in request.events[2].modalities and request.events[2].asr_terms == ("hooray",)
    assert all(event.raw_text == event.text for event in request.events)
    assert all(event.visual_query == event.clip_query for event in request.events)


def test_query_variants_are_bounded_and_original_is_preserved() -> None:
    event = TrakePlannerV2().decompose("q", "a man holding a trophy").request.events[0]

    assert event.raw_text == "a man holding a trophy"
    assert event.visual_query == "a man holding a trophy"
    assert 1 <= len(event.query_variants) <= 3
    assert any("person" in variant or "carrying" in variant for variant in event.query_variants)


def test_manual_override_has_highest_priority_and_round_trips() -> None:
    result = TrakePlannerV2().manual_override("q", "ambiguous", ["custom first", "custom second"])

    assert [event.text for event in result.request.events] == ["custom first", "custom second"]
    assert all(event.source is EventSource.MANUAL for event in result.request.events)
    assert all(event.planner_confidence == 1.0 for event in result.request.events)
    assert request_from_dict(result.request.to_dict()) == result.request


def test_ambiguous_rule_parse_surfaces_low_confidence_warning() -> None:
    result = TrakePlannerV2().decompose("q", "a person somewhere after the car")

    assert PLANNER_LOW_CONFIDENCE in result.warnings
    assert result.request.events[0].planner_confidence == 0.65


def test_explicit_sequence_uses_hard_default_gap_constraints() -> None:
    request = TrakePlannerV2(sequence_default_max_gap_seconds=60.0).decompose(
        "q", "a person walks in → opens a door → sits down"
    ).request

    assert request.constraints["gap_mode"] == "hard"
    assert [event.max_gap_seconds for event in request.events] == [None, 60.0, 60.0]
    assert [event.expected_gap_class for event in request.events] == [None, "short", "short"]


def test_ascii_arrow_splits_all_events_for_retrieval_and_alignment() -> None:
    request = TrakePlannerV2(60.0).decompose(
        "q",
        "a male news anchor is reading the news -> a police officer is reading a sheet of paper -> a woman is sitting at a table and crying",
    ).request

    assert [event.text for event in request.events] == [
        "a male news anchor is reading the news",
        "a police officer is reading a sheet of paper",
        "a woman is sitting at a table and crying",
    ]
    assert [event.max_gap_seconds for event in request.events] == [None, 60.0, 60.0]


def test_numbered_vietnamese_moments_become_events_without_the_introductory_question() -> None:
    request = TrakePlannerV2(60.0).decompose(
        "q",
        "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy.",
    ).request

    assert [event.text for event in request.events] == ["giậm nhảy", "bay qua xà", "tiếp đất", "đứng dậy"]
    assert request.global_context is None
    assert [event.max_gap_seconds for event in request.events] == [None, 60.0, 60.0, 60.0]
