from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib import request


DEFAULT_TRANSLATION_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_TRANSLATION_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
SUPPORTED_PROVIDERS = {"gemini", "openai-compatible"}


@dataclass(frozen=True)
class TranslationConfig:
    api_key: str | None = None
    api_url: str = DEFAULT_TRANSLATION_API_URL
    model: str = DEFAULT_TRANSLATION_MODEL
    provider: str = "openai-compatible"
    timeout_seconds: float = 30.0


class TranslationError(RuntimeError):
    pass


class ExternalTranslator:
    def __init__(self, config: TranslationConfig) -> None:
        self.config = config

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.api_url and self.config.model)

    def translate_vi_to_en(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("translation text must not be empty")
        if not self.is_configured:
            raise TranslationError("translation API is not configured")
        if self.config.provider not in SUPPORTED_PROVIDERS:
            raise TranslationError(f"unsupported translation provider: {self.config.provider}")

        if self.config.provider == "gemini":
            return self._translate_with_gemini(text)

        return self._translate_with_openai_compatible(text)

    def _translate_with_openai_compatible(self, text: str) -> str:
        payload = openai_compatible_payload(text, self.config.model)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise TranslationError(f"translation API request failed: {http_error_detail(exc)}") from exc
        except Exception as exc:
            raise TranslationError(f"translation API request failed: {exc}") from exc
        return parse_openai_compatible_translation(response_payload)

    def _translate_with_gemini(self, text: str) -> str:
        payload = gemini_payload(text)
        body = json.dumps(payload).encode("utf-8")
        base_url = self.config.api_url.rstrip("/")
        model = self.config.model
        if not model.startswith("models/"):
            model = f"models/{model}"
        url = f"{base_url}/{model}:generateContent"
        req = request.Request(
            url,
            data=body,
            headers={
                "x-goog-api-key": self.config.api_key or "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise TranslationError(f"Gemini translation API request failed: {http_error_detail(exc)}") from exc
        except Exception as exc:
            raise TranslationError(f"Gemini translation API request failed: {exc}") from exc
        return parse_gemini_translation(response_payload)


def openai_compatible_payload(text: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate Vietnamese video retrieval queries into concise English CLIP search queries. "
                    "Keep visual nouns, actions, colors, places, and objects. Return only the English query."
                ),
            },
            {"role": "user", "content": text},
        ],
    }


def gemini_payload(text: str) -> dict[str, Any]:
    prompt = (
        "Translate this Vietnamese video retrieval query into a concise English CLIP search query. "
        "Keep visual nouns, actions, colors, places, and objects. Return only the English query.\n\n"
        f"Vietnamese query: {text}"
    )
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
        },
    }


def parse_openai_compatible_translation(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError("translation API response does not match OpenAI-compatible chat format") from exc
    translated = str(content).strip().strip('"')
    if not translated:
        raise TranslationError("translation API returned an empty query")
    return " ".join(translated.split())


def parse_gemini_translation(payload: dict[str, Any]) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError("Gemini API response does not match generateContent format") from exc
    text_parts = [str(part.get("text", "")) for part in parts if isinstance(part, dict)]
    translated = " ".join(text_parts).strip().strip('"')
    if not translated:
        raise TranslationError("Gemini API returned an empty query")
    return " ".join(translated.split())


def http_error_detail(error: HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace")
    if not body:
        return f"HTTP {error.code}: {error.reason}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return f"HTTP {error.code}: {body[:500]}"
    message = payload.get("error", {}).get("message")
    status = payload.get("error", {}).get("status")
    if message:
        if status:
            return f"HTTP {error.code} {status}: {message}"
        return f"HTTP {error.code}: {message}"
    return f"HTTP {error.code}: {body[:500]}"
