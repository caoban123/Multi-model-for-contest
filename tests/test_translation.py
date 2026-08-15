import pytest

from aic_retrieval.translation import (
    DEFAULT_TRANSLATION_MODEL,
    TranslationError,
    openai_compatible_payload,
    parse_openai_compatible_translation,
)


def test_openai_compatible_payload_requests_plain_english_query() -> None:
    payload = openai_compatible_payload("người cầm điện thoại", DEFAULT_TRANSLATION_MODEL)

    assert payload["model"] == DEFAULT_TRANSLATION_MODEL
    assert payload["temperature"] == 0
    assert payload["messages"][1]["content"] == "người cầm điện thoại"
    assert "Return only the English query" in payload["messages"][0]["content"]


def test_parse_openai_compatible_translation_normalizes_text() -> None:
    payload = {"choices": [{"message": {"content": "  \"a person holding a phone\"  "}}]}

    assert parse_openai_compatible_translation(payload) == "a person holding a phone"


def test_parse_openai_compatible_translation_rejects_bad_shape() -> None:
    with pytest.raises(TranslationError):
        parse_openai_compatible_translation({"choices": []})
