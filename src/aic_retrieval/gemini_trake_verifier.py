"""Allowlisted, advisory-only Gemini verification for TRAKE chains."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Callable, Mapping

from aic_retrieval.gemini_visual_reranker import _call_gemini_visual
from aic_retrieval.translation import TranslationConfig, TranslationError


GEMINI_TRAKE_VERIFY_VERSION = "gemini-trake-verify-v1"


class GeminiTrakeVerifier:
    def __init__(
        self,
        config: TranslationConfig,
        *,
        path_resolver: Callable[[str], Path] | None = None,
        max_images: int = 18,
    ) -> None:
        if not 1 <= max_images <= 24:
            raise ValueError("max_images must be between 1 and 24")
        self.config = config
        self.path_resolver = path_resolver or (lambda value: Path(value))
        self.max_images = max_images

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def __call__(self, safe_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_configured:
            raise TranslationError("Gemini TRAKE verifier is not configured")
        prepared = prepare_trake_evidence(safe_payload, self.path_resolver, self.max_images)
        if not prepared["images"]:
            return {
                "version": GEMINI_TRAKE_VERIFY_VERSION,
                "status": "NO_IMAGES",
                "chains": [],
                "raw_text": None,
            }
        response = _call_gemini_visual(
            self.config,
            gemini_trake_verify_payload(safe_payload, prepared["images"]),
        )
        chains, raw_text = parse_gemini_trake_response(
            response,
            allowed_events=prepared["allowed_events"],
        )
        return {
            "version": GEMINI_TRAKE_VERIFY_VERSION,
            "status": "ADVISORY_READY",
            "chains": chains,
            "raw_text": raw_text,
            "model": self.config.model,
            "image_count": len(prepared["images"]),
        }


def prepare_trake_evidence(
    payload: Mapping[str, Any],
    resolver: Callable[[str], Path],
    limit: int,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    allowed_events: dict[str, set[str]] = {}
    for chain in payload.get("chains") or ():
        chain_id = str(chain.get("chain_id") or "").strip()
        if not chain_id:
            continue
        allowed_events[chain_id] = set()
        for event in chain.get("events") or ():
            event_id = str(event.get("event_id") or "").strip()
            path_value = event.get("keyframe_path")
            if not event_id or not path_value:
                continue
            try:
                path = resolver(str(path_value))
            except (FileNotFoundError, PermissionError, OSError, ValueError):
                continue
            if not path.is_file():
                continue
            allowed_events[chain_id].add(event_id)
            images.append({
                "chain_id": chain_id,
                "event_id": event_id,
                "description": str(event.get("description") or ""),
                "pts_time": float(event.get("pts_time") or 0.0),
                "path": path,
                "mime_type": mimetypes.guess_type(path.name)[0] or "image/jpeg",
            })
            if len(images) >= limit:
                return {"images": images, "allowed_events": allowed_events}
    return {"images": images, "allowed_events": allowed_events}


def gemini_trake_verify_payload(payload: Mapping[str, Any], images: list[dict[str, Any]]) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{
        "text": (
            "Review only the supplied TRAKE chains and images. Determine whether each image visibly supports its "
            "event description and whether the sequence is coherent in the stated order. Do not invent chain IDs, "
            "event IDs, videos, frames, or facts. This is advisory review; the server enforces video and time order. "
            "Return strict JSON with key chains. Each item must contain chain_id, support from 0 to 1, "
            "temporal_consistency (true or false), reason, and event_support. event_support is an array with event_id, "
            "support from 0 to 1, and reason.\n\n"
            f"Query: {str(payload.get('query') or '').strip()}\n"
            f"Events: {json.dumps(payload.get('event_descriptions') or [], ensure_ascii=False)}"
        )
    }]
    for item in images:
        parts.append({
            "text": (
                f"Chain ID: {item['chain_id']} | Event ID: {item['event_id']} | "
                f"time: {item['pts_time']:.3f}s | description: {item['description']}"
            )
        })
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


def parse_gemini_trake_response(
    payload: dict[str, Any],
    *,
    allowed_events: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]], str]:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError("Gemini TRAKE response does not match generateContent format") from exc
    raw_text = " ".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    text = raw_text
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranslationError("Gemini TRAKE response was not valid JSON") from exc
    rows = parsed.get("chains") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        raise TranslationError("Gemini TRAKE JSON requires a chains array")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        chain_id = str(row.get("chain_id") or "").strip()
        if chain_id not in allowed_events or chain_id in seen:
            continue
        event_rows: list[dict[str, Any]] = []
        event_seen: set[str] = set()
        for event in row.get("event_support") or ():
            if not isinstance(event, Mapping):
                continue
            event_id = str(event.get("event_id") or "").strip()
            if event_id not in allowed_events[chain_id] or event_id in event_seen:
                continue
            score = _score(event.get("support"))
            if score is None:
                continue
            event_rows.append({"event_id": event_id, "support": score, "reason": str(event.get("reason") or "").strip()})
            event_seen.add(event_id)
        support = _score(row.get("support"))
        if support is None:
            continue
        output.append({
            "chain_id": chain_id,
            "support": support,
            "temporal_consistency": bool(row.get("temporal_consistency")),
            "reason": str(row.get("reason") or "").strip(),
            "event_support": event_rows,
        })
        seen.add(chain_id)
    output.sort(key=lambda item: (-item["support"], item["chain_id"]))
    return output, raw_text


def _score(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
