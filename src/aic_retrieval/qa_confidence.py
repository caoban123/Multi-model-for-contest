"""Server-side guards for Phase-7 review and confirmation."""

from __future__ import annotations

from dataclasses import dataclass

from aic_retrieval.qa_schema import AnswerDraft, ConfidenceState, EvidencePack, EvidenceModality, QuestionType


CONFIDENCE_POLICY_VERSION = "phase7-evidence-guard-v1"


@dataclass(frozen=True)
class ConfirmationGuard:
    allowed: bool
    reason: str
    version: str = CONFIDENCE_POLICY_VERSION


def confirmation_guard(
    draft: AnswerDraft,
    pack: EvidencePack,
    selected_evidence_ids: tuple[str, ...],
) -> ConfirmationGuard:
    selected = {item.evidence_id: item for item in pack.evidence_refs if item.evidence_id in set(selected_evidence_ids)}
    if draft.confidence_state in {ConfidenceState.NEEDS_EVIDENCE, ConfidenceState.UNSUPPORTED}:
        return ConfirmationGuard(False, f"draft confidence state is {draft.confidence_state.value}")
    if not draft.raw_answer:
        return ConfirmationGuard(False, "draft has no answer")
    if not selected:
        return ConfirmationGuard(False, "at least one selected evidence reference is required")
    if not set(draft.evidence_refs).issubset(selected):
        return ConfirmationGuard(False, "all answer evidence must be selected for confirmation")
    if draft.question_type is QuestionType.ASR and not any(ref.modality is EvidenceModality.ASR for ref in selected.values()):
        return ConfirmationGuard(False, "speech answer requires selected ASR evidence")
    return ConfirmationGuard(True, "evidence and answer satisfy Phase-7 confirmation policy")


def require_confirmable(draft: AnswerDraft, pack: EvidencePack, selected_evidence_ids: tuple[str, ...]) -> None:
    guard = confirmation_guard(draft, pack, selected_evidence_ids)
    if not guard.allowed:
        raise ValueError(guard.reason)
