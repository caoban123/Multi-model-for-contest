"""Evidence-first answer drafting with no generative or remote dependency."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from aic_retrieval.qa_question_router import QuestionRoute
from aic_retrieval.qa_normalization import AnswerNormalizer
from aic_retrieval.qa_schema import (
    AnswerDraft,
    AvailabilityStatus,
    ConfidenceState,
    EvidenceModality,
    EvidencePack,
    EvidenceRef,
    QuestionType,
)


ANSWER_STRATEGY_VERSION = "phase7-evidence-first-v1"


class EvidenceFirstAnswerer:
    def __init__(self, normalizer: AnswerNormalizer | None = None) -> None:
        self.normalizer = normalizer or AnswerNormalizer()

    def draft(self, pack: EvidencePack, route: QuestionRoute, selected_evidence_ids: tuple[str, ...] = ()) -> AnswerDraft:
        selected = _select_evidence(pack.evidence_refs, selected_evidence_ids)
        eligible = selected or _for_route(pack.evidence_refs, route.question_type)
        if route.question_type is QuestionType.ASR and pack.modality_availability[EvidenceModality.ASR] is not AvailabilityStatus.AVAILABLE:
            return _blocked(route, "ASR_UNAVAILABLE")
        if route.question_type is QuestionType.OCR and pack.modality_availability[EvidenceModality.OCR] is not AvailabilityStatus.AVAILABLE:
            return _blocked(route, "OCR_UNAVAILABLE")
        if route.question_type in {QuestionType.COUNT, QuestionType.TEMPORAL, QuestionType.VISUAL, QuestionType.UNKNOWN}:
            warning = "COUNT_REQUIRES_MULTI_FRAME_REVIEW" if route.question_type is QuestionType.COUNT else "EVIDENCE_REVIEW_REQUIRED"
            return _review_required(route, eligible, warning)
        answer, evidence = _answer_from_evidence(route.question_type, eligible, route.answer_field)
        if not answer or not evidence:
            return _blocked(route, f"NO_{route.question_type.value}_EVIDENCE")
        normalized = self.normalizer.normalize(answer)
        return AnswerDraft(
            draft_id=_draft_id(route.question_type, evidence),
            raw_answer=normalized.raw_answer,
            normalized_answer=normalized.normalized_answer,
            alternative_answers=normalized.alternative_answers,
            question_type=route.question_type,
            evidence_refs=tuple(item.evidence_id for item in evidence),
            confidence_state=ConfidenceState.REVIEW_REQUIRED,
            strategy_version=ANSWER_STRATEGY_VERSION,
            answer_source=evidence[0].source,
            warnings=("HUMAN_CONFIRMATION_REQUIRED",),
        )


def _select_evidence(refs: Iterable[EvidenceRef], ids: tuple[str, ...]) -> list[EvidenceRef]:
    selected_ids = set(ids)
    return [ref for ref in refs if ref.evidence_id in selected_ids]


def _for_route(refs: Iterable[EvidenceRef], question_type: QuestionType) -> list[EvidenceRef]:
    modality = {
        QuestionType.OCR: EvidenceModality.OCR,
        QuestionType.ASR: EvidenceModality.ASR,
        QuestionType.METADATA: EvidenceModality.METADATA,
        QuestionType.OBJECT: EvidenceModality.OBJECT,
        QuestionType.ATTRIBUTE_COLOR: EvidenceModality.ATTRIBUTE,
    }.get(question_type)
    return [ref for ref in refs if modality is None or ref.modality is modality]


def _answer_from_evidence(question_type: QuestionType, refs: list[EvidenceRef], metadata_field: str | None = None) -> tuple[str | None, list[EvidenceRef]]:
    for ref in refs:
        payload = ref.payload
        if question_type in {QuestionType.OCR, QuestionType.ASR}:
            value = payload.get("matched_text") or payload.get("text_raw") or payload.get("text")
        elif question_type is QuestionType.METADATA:
            value = _metadata_value(payload, metadata_field)
        elif question_type is QuestionType.OBJECT:
            labels = payload.get("matched_labels") or payload.get("labels")
            value = labels[0] if isinstance(labels, list) and labels else payload.get("label_normalized") or payload.get("label_raw")
        elif question_type is QuestionType.ATTRIBUTE_COLOR:
            value = payload.get("color") or payload.get("matched_color") or payload.get("value")
        else:
            value = None
        if value is not None and str(value).strip():
            return str(value).strip(), [ref]
    return None, []


def _metadata_value(payload: dict, requested_field: str | None = None) -> str | None:
    for field in ((requested_field,) if requested_field else ("author", "channel_id", "publish_date", "title")):
        if field is None:
            continue
        value = payload.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    matched_fields = payload.get("matched_fields")
    if isinstance(matched_fields, list):
        for item in matched_fields:
            if isinstance(item, dict) and item.get("value"):
                return str(item["value"]).strip()
    return None


def _blocked(route: QuestionRoute, warning: str) -> AnswerDraft:
    return AnswerDraft(
        draft_id=_draft_id(route.question_type, []), raw_answer=None, normalized_answer=None,
        alternative_answers=(), question_type=route.question_type, evidence_refs=(),
        confidence_state=ConfidenceState.UNSUPPORTED, strategy_version=ANSWER_STRATEGY_VERSION,
        warnings=(warning,),
    )


def _review_required(route: QuestionRoute, refs: list[EvidenceRef], warning: str) -> AnswerDraft:
    return AnswerDraft(
        draft_id=_draft_id(route.question_type, refs), raw_answer=None, normalized_answer=None,
        alternative_answers=(), question_type=route.question_type,
        evidence_refs=tuple(ref.evidence_id for ref in refs), confidence_state=ConfidenceState.NEEDS_EVIDENCE,
        strategy_version=ANSWER_STRATEGY_VERSION, warnings=(warning,),
    )


def _draft_id(question_type: QuestionType, refs: Iterable[EvidenceRef]) -> str:
    identity = "|".join([question_type.value, *(ref.evidence_id for ref in refs)])
    return "draft-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
