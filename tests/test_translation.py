import pytest
from urllib.error import HTTPError
from io import BytesIO

from aic_retrieval.translation import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    TranslationError,
    gemini_payload,
    http_error_detail,
    openai_compatible_payload,
    parse_gemini_translation,
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


def test_gemini_payload_requests_plain_english_query() -> None:
    payload = gemini_payload("người mặc áo đỏ")

    assert payload["generationConfig"]["temperature"] == 0
    assert payload["contents"][0]["role"] == "user"
    assert "Return only the English query" in payload["contents"][0]["parts"][0]["text"]


def test_parse_gemini_translation_normalizes_text() -> None:
    payload = {"candidates": [{"content": {"parts": [{"text": "  a person wearing a red shirt  "}]}}]}

    assert DEFAULT_GEMINI_MODEL.startswith("gemini-")
    assert parse_gemini_translation(payload) == "a person wearing a red shirt"


def test_parse_gemini_translation_rejects_bad_shape() -> None:
    with pytest.raises(TranslationError):
        parse_gemini_translation({"candidates": []})


def test_http_error_detail_extracts_api_message() -> None:
    body = b'{"error":{"status":"INVALID_ARGUMENT","message":"Model is not found"}}'
    error = HTTPError("https://example.test", 400, "Bad Request", hdrs=None, fp=BytesIO(body))

    assert http_error_detail(error) == "HTTP 400 INVALID_ARGUMENT: Model is not found"
