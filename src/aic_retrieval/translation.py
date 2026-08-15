from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request


DEFAULT_TRANSLATION_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_TRANSLATION_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class TranslationConfig:
    api_key: str | None = None
    api_url: str = DEFAULT_TRANSLATION_API_URL
    model: str = DEFAULT_TRANSLATION_MODEL
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
        except Exception as exc:
            raise TranslationError(f"translation API request failed: {exc}") from exc
        return parse_openai_compatible_translation(response_payload)


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


def parse_openai_compatible_translation(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError("translation API response does not match OpenAI-compatible chat format") from exc
    translated = str(content).strip().strip('"')
    if not translated:
        raise TranslationError("translation API returned an empty query")
    return " ".join(translated.split())
