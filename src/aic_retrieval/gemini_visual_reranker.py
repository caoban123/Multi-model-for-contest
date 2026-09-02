"""Optional allowlisted Gemini reranking for already-retrieved keyframes."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib import request
from urllib.error import HTTPError

from aic_retrieval.translation import (
    DEFAULT_GEMINI_API_URL,
    DEFAULT_GEMINI_MODEL,
    TranslationConfig,
    TranslationError,
    http_error_detail,
)


GEMINI_VISUAL_RERANK_VERSION = "gemini-visual-rerank-v1"


@dataclass(frozen=True)
class VisualRerankItem:
    candidate_id: str
    relevance: float
    matched_events: tuple[str, ...]
    reason: str


class GeminiVisualReranker:
    def __init__(
        self,
        config: TranslationConfig,
        *,
        path_resolver: Callable[[str], Path] | None = None,
        max_images: int = 12,
        cache_size: int = 64,
    ) -> None:
        if not 1 <= max_images <= 24:
            raise ValueError("max_images must be between 1 and 24")
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self.config = config
        self.path_resolver = path_resolver or (lambda value: Path(value))
        self.max_images = max_images
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def rerank(self, query: str, frames: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        if not self.is_configured:
            raise TranslationError("Gemini visual reranker is not configured")
        prepared = _prepare_frames(frames, self.path_resolver, self.max_images)
        if not prepared:
            return {
                "version": GEMINI_VISUAL_RERANK_VERSION,
                "status": "NO_IMAGES",
                "items": [],
                "raw_text": None,
                "cache_hit": False,
            }
        cache_key = _cache_key(query, prepared, self.config.model)
        if cache_key in self._cache:
            cached = dict(self._cache.pop(cache_key))
            self._cache[cache_key] = cached
            return {**cached, "cache_hit": True}
        payload = gemini_visual_rerank_payload(query, prepared)
        response = _call_gemini_visual(self.config, payload)
        parsed, raw_text = parse_gemini_visual_rerank_response(
            response,
            allowed_ids={item["candidate_id"] for item in prepared},
        )
        result = {
            "version": GEMINI_VISUAL_RERANK_VERSION,
            "status": "APPLIED",
            "items": [
                {
                    "candidate_id": item.candidate_id,
                    "relevance": item.relevance,
                    "matched_events": list(item.matched_events),
                    "reason": item.reason,
                }
                for item in parsed
            ],
            "raw_text": raw_text,
            "model": self.config.model or DEFAULT_GEMINI_MODEL,
            "image_count": len(prepared),
            "cache_hit": False,
        }
        if self.cache_size:
            self._cache[cache_key] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result


def gemini_visual_rerank_payload(query: str, prepared_frames: list[dict[str, Any]]) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{
        "text": (
            "Rerank only the attached candidate video keyframes for the retrieval query. "
            "Do not invent a candidate ID. Judge visible evidence, action state, text, objects, "
            "and temporal clues. Return strict JSON with one key named candidates. candidates is "
            "an array of objects with candidate_id, relevance from 0 to 1, matched_events as an "
            "array of short strings, and reason. Include only candidates with useful evidence.\n\n"
            f"Query: {query.strip()}"
        )
    }]
    for item in prepared_frames:
        parts.append({"text": f"Candidate ID: {item['candidate_id']}"})
        parts.append({
            "inline_data": {
                "mime_type": item["mime_type"],
                "data": base64.b64encode(item["path"].read_bytes()).decode("ascii"),
            }
        })
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }


def parse_gemini_visual_rerank_response(
    payload: dict[str, Any],
    *,
    allowed_ids: set[str],
) -> tuple[list[VisualRerankItem], str]:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError("Gemini visual rerank response does not match generateContent format") from exc
    raw_text = " ".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    text = raw_text
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranslationError("Gemini visual rerank response was not valid JSON") from exc
    rows = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        raise TranslationError("Gemini visual rerank JSON requires a candidates array")
    items: list[VisualRerankItem] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id not in allowed_ids or candidate_id in seen:
            continue
        try:
            relevance = max(0.0, min(1.0, float(row.get("relevance", 0.0))))
        except (TypeError, ValueError):
            continue
        matched = row.get("matched_events")
        events = tuple(str(value).strip() for value in matched if str(value).strip()) if isinstance(matched, list) else ()
        items.append(VisualRerankItem(candidate_id, relevance, events, str(row.get("reason") or "").strip()))
        seen.add(candidate_id)
    items.sort(key=lambda item: (-item.relevance, item.candidate_id))
    return items, raw_text


def _prepare_frames(
    frames: Iterable[Mapping[str, Any]],
    resolver: Callable[[str], Path],
    limit: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame in frames:
        video_id = str(frame.get("video_id") or "").strip()
        keyframe_id = frame.get("keyframe_id", frame.get("best_keyframe_id"))
        path_value = frame.get("keyframe_path", frame.get("best_keyframe_path"))
        if not video_id or keyframe_id is None or not path_value:
            continue
        candidate_id = f"{video_id}:k{int(keyframe_id)}"
        if candidate_id in seen:
            continue
        try:
            path = resolver(str(path_value))
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            continue
        if not path.is_file():
            continue
        prepared.append({
            "candidate_id": candidate_id,
            "video_id": video_id,
            "keyframe_id": int(keyframe_id),
            "path": path,
            "mime_type": mimetypes.guess_type(path.name)[0] or "image/jpeg",
        })
        seen.add(candidate_id)
        if len(prepared) >= limit:
            break
    return prepared


def _cache_key(query: str, frames: list[dict[str, Any]], model: str) -> str:
    identity = {
        "query": query.strip(),
        "model": model,
        "frames": [
            {
                "candidate_id": item["candidate_id"],
                "path": str(item["path"]),
                "size": item["path"].stat().st_size,
                "mtime_ns": item["path"].stat().st_mtime_ns,
            }
            for item in frames
        ],
        "version": GEMINI_VISUAL_RERANK_VERSION,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def _call_gemini_visual(config: TranslationConfig, payload: dict[str, Any]) -> dict[str, Any]:
    base_url = (config.api_url or DEFAULT_GEMINI_API_URL).rstrip("/")
    model = config.model or DEFAULT_GEMINI_MODEL
    if not model.startswith("models/"):
        model = f"models/{model}"
    req = request.Request(
        f"{base_url}/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": config.api_key or "", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise TranslationError(f"Gemini visual rerank request failed: {http_error_detail(exc)}") from exc
    except Exception as exc:
        raise TranslationError(f"Gemini visual rerank request failed: {exc}") from exc
