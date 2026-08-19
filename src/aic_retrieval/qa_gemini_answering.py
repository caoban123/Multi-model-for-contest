"""Optional Gemini-backed Q&A drafting from selected evidence only."""

from __future__ import annotations

import hashlib
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError

from aic_retrieval.qa_normalization import AnswerNormalizer
from aic_retrieval.qa_question_router import QuestionRoute
from aic_retrieval.qa_schema import AnswerDraft, ConfidenceState, EvidencePack, EvidenceRef
from aic_retrieval.translation import DEFAULT_GEMINI_API_URL, DEFAULT_GEMINI_MODEL, TranslationConfig, TranslationError, http_error_detail


GEMINI_QA_STRATEGY_VERSION = "phase7-gemini-vlm-evidence-v1"
MAX_GEMINI_IMAGES = 4


class GeminiEvidenceAnswerer:
    def __init__(self, config: TranslationConfig, normalizer: AnswerNormalizer | None = None) -> None:
        self.config = config
        self.normalizer = normalizer or AnswerNormalizer()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def draft(self, pack: EvidencePack, route: QuestionRoute, selected_evidence_ids: tuple[str, ...] = ()) -> AnswerDraft:
        if not self.is_configured:
            raise TranslationError("Gemini Q&A API is not configured")
        selected = _selected_evidence(pack.evidence_refs, selected_evidence_ids)
        if not selected:
            return AnswerDraft(
                draft_id=_draft_id(route, selected),
                raw_answer=None,
                normalized_answer=None,
                alternative_answers=(),
                question_type=route.question_type,
                evidence_refs=(),
                confidence_state=ConfidenceState.NEEDS_EVIDENCE,
                strategy_version=GEMINI_QA_STRATEGY_VERSION,
                generation_method="gemini_vlm_evidence",
                warnings=("SELECT_EVIDENCE_BEFORE_GEMINI_DRAFT",),
            )
        context_refs = _context_evidence_with_images(pack, selected)
        response = _call_gemini(self.config, gemini_qa_payload(pack, route, selected, context_refs=context_refs))
        parsed = parse_gemini_qa_response(response)
        answer = str(parsed.get("answer") or "").strip()
        allowed_ids = {ref.evidence_id for ref in context_refs}
        evidence_ids = tuple(item for item in parsed.get("evidence_ids", ()) if item in allowed_ids)
        if not answer:
            return AnswerDraft(
                draft_id=_draft_id(route, selected),
                raw_answer=None,
                normalized_answer=None,
                alternative_answers=(),
                question_type=route.question_type,
                evidence_refs=evidence_ids,
                confidence_state=ConfidenceState.NEEDS_EVIDENCE,
                strategy_version=GEMINI_QA_STRATEGY_VERSION,
                generation_method="gemini_vlm_evidence",
                warnings=tuple(parsed.get("warnings") or ("GEMINI_RETURNED_NO_SUPPORTED_ANSWER",)),
            )
        normalized = self.normalizer.normalize(answer)
        return AnswerDraft(
            draft_id=_draft_id(route, selected),
            raw_answer=normalized.raw_answer,
            normalized_answer=normalized.normalized_answer,
            alternative_answers=normalized.alternative_answers,
            question_type=route.question_type,
            evidence_refs=evidence_ids or tuple(ref.evidence_id for ref in selected),
            confidence_state=ConfidenceState.REVIEW_REQUIRED,
            strategy_version=GEMINI_QA_STRATEGY_VERSION,
            answer_source="gemini_from_selected_evidence_and_images",
            generation_method="gemini_vlm_evidence",
            warnings=tuple(parsed.get("warnings") or ("HUMAN_CONFIRMATION_REQUIRED",)),
        )


def gemini_qa_payload(pack: EvidencePack, route: QuestionRoute, selected: list[EvidenceRef], context_refs: list[EvidenceRef] | None = None) -> dict[str, Any]:
    context_refs = context_refs or _context_evidence_with_images(pack, selected)
    evidence = [
        {
            "evidence_id": ref.evidence_id,
            "modality": ref.modality.value,
            "video_id": ref.video_id,
            "keyframe_id": ref.keyframe_id,
            "timestamp": ref.timestamp,
            "payload": _compact_payload(ref.payload),
        }
        for ref in context_refs
    ]
    image_parts = _image_parts(context_refs)
    image_note = f"{len(image_parts)} keyframe image(s) are attached after this text prompt." if image_parts else "No keyframe image could be attached; use text evidence only."
    prompt = (
        "You are drafting an answer for an evidence-grounded video retrieval Q&A workflow. "
        "Answer using only the selected evidence below and the attached keyframe images. If the evidence is insufficient, return an empty answer and a warning. "
        "Use the same language as the user's question when possible. Do not mention evidence IDs in the answer. "
        "Return strict JSON only with keys: answer, evidence_ids, warnings.\n\n"
        f"Event query: {pack.request.event_query}\n"
        f"Question: {pack.request.question}\n"
        f"Question type: {route.question_type.value}\n"
        f"Image evidence: {image_note}\n"
        f"Selected evidence JSON: {json.dumps(evidence, ensure_ascii=False, default=str)}"
    )
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}, *image_parts]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }


def parse_gemini_qa_response(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError("Gemini Q&A response does not match generateContent format") from exc
    text = " ".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranslationError("Gemini Q&A response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise TranslationError("Gemini Q&A response JSON must be an object")
    if "evidence_ids" in parsed and not isinstance(parsed["evidence_ids"], list):
        raise TranslationError("Gemini Q&A evidence_ids must be a list")
    if "warnings" in parsed and not isinstance(parsed["warnings"], list):
        raise TranslationError("Gemini Q&A warnings must be a list")
    return parsed


def _call_gemini(config: TranslationConfig, payload: dict[str, Any]) -> dict[str, Any]:
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
        raise TranslationError(f"Gemini Q&A API request failed: {http_error_detail(exc)}") from exc
    except Exception as exc:
        raise TranslationError(f"Gemini Q&A API request failed: {exc}") from exc


def _selected_evidence(refs: tuple[EvidenceRef, ...], ids: tuple[str, ...]) -> list[EvidenceRef]:
    selected_ids = set(ids)
    return [ref for ref in refs if ref.evidence_id in selected_ids]


def _context_evidence_with_images(pack: EvidencePack, selected: list[EvidenceRef]) -> list[EvidenceRef]:
    context = list(selected)
    known_ids = {ref.evidence_id for ref in context}
    for selected_ref in selected:
        if _keyframe_path(selected_ref):
            continue
        related = next(
            (
                ref for ref in pack.evidence_refs
                if ref.keyframe_id == selected_ref.keyframe_id
                and ref.video_id == selected_ref.video_id
                and _keyframe_path(ref)
            ),
            None,
        )
        if related is not None and related.evidence_id not in known_ids:
            context.append(related)
            known_ids.add(related.evidence_id)
    return context


def _image_parts(refs: list[EvidenceRef]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for ref in refs:
        path = _keyframe_path(ref)
        if path is None or path in seen_paths or not path.is_file():
            continue
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        })
        seen_paths.add(path)
        if len(parts) >= MAX_GEMINI_IMAGES:
            break
    return parts


def _keyframe_path(ref: EvidenceRef) -> Path | None:
    value = ref.payload.get("keyframe_path") or ref.payload.get("best_keyframe_path")
    if not value:
        return None
    return Path(str(value))


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "matched_text", "text_raw", "text", "title", "author", "channel_id", "publish_date",
        "description", "matched_labels", "labels", "label_raw", "label_normalized", "color",
        "matched_color", "value", "keyframe_path", "rank", "score", "confidence",
    )
    return {key: payload[key] for key in keep if key in payload and payload[key] not in (None, "", [])}


def _draft_id(route: QuestionRoute, refs: list[EvidenceRef]) -> str:
    identity = "|".join([route.question_type.value, *(ref.evidence_id for ref in refs), GEMINI_QA_STRATEGY_VERSION])
    return "draft-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
