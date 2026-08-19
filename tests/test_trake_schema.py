from __future__ import annotations

import pytest

from aic_retrieval.trake_decomposition import REVIEW_WARNING, RuleBasedTrakeDecomposer
from aic_retrieval.trake_schema import EventSource, TrakeEvent, TrakeRequest, request_from_dict


def event(order: int, event_id: str | None = None) -> TrakeEvent:
    return TrakeEvent(event_id=event_id or f"e{order}", order=order, text=f"event {order}", clip_query=f"event {order}")


@pytest.mark.parametrize("events", [(), tuple(event(i) for i in range(1, 7))])
def test_request_rejects_event_count_outside_one_to_five(events: tuple[TrakeEvent, ...]) -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        TrakeRequest("q1", "query", events)


def test_request_rejects_duplicate_and_non_contiguous_order() -> None:
    with pytest.raises(ValueError, match="duplicate event_id"):
        TrakeRequest("q1", "query", (event(1, "x"), event(2, "x")))
    with pytest.raises(ValueError, match="contiguous"):
        TrakeRequest("q1", "query", (event(1), event(3)))
    with pytest.raises(ValueError, match="unsupported group"):
        TrakeRequest("q1", "query", (event(1),), group="L22")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("người bước vào → người ngồi xuống", ("người bước vào", "người ngồi xuống")),
        ("người bước vào sau đó ngồi xuống", ("người bước vào", "ngồi xuống")),
        ("door opens then a person enters", ("door opens", "a person enters")),
        ("door opens followed by a person enters", ("door opens", "a person enters")),
        ("người đứng trước khi chạy", ("người đứng", "chạy")),
        ("the person stands before running", ("the person stands", "running")),
    ],
)
def test_parser_preserves_explicit_temporal_order(query: str, expected: tuple[str, ...]) -> None:
    result = RuleBasedTrakeDecomposer().decompose("q1", query)
    assert tuple(item.text for item in result.request.events) == expected
    assert result.warnings == ()


def test_ambiguous_temporal_hint_requires_review() -> None:
    result = RuleBasedTrakeDecomposer().decompose("q1", "người ở phía sau chiếc xe")
    assert len(result.request.events) == 1
    assert REVIEW_WARNING in result.warnings


def test_manual_override_is_auditable_and_round_trips() -> None:
    result = RuleBasedTrakeDecomposer().manual_override("q1", "ambiguous", ["first", "second"])
    assert all(item.source is EventSource.MANUAL for item in result.request.events)
    restored = request_from_dict(result.request.to_dict())
    assert restored == result.request
