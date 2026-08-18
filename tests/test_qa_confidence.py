from __future__ import annotations

from aic_retrieval.qa_answering import EvidenceFirstAnswerer
from aic_retrieval.qa_confidence import confirmation_guard
from aic_retrieval.qa_evidence import build_evidence_pack
from aic_retrieval.qa_question_router import RuleBasedQuestionRouter
from aic_retrieval.qa_schema import AvailabilityStatus, EvidenceModality, QaRequest


def _pack():
    return build_evidence_pack(
        QaRequest("q1", "a sign", "Biển hiệu ghi gì?"),
        {"video_results":[{"video_id":"L21_V001","frames":[{"video_id":"L21_V001","keyframe_id":5,"frame_idx":125,"pts_time":5.0,"evidence":{"ocr":[{"matched_text":"Samsung"}],"asr":[],"object":[],"attribute":[]}}]}]},
        modality_availability={EvidenceModality.OCR: AvailabilityStatus.AVAILABLE},
    )


def test_confirmation_requires_selected_answer_evidence() -> None:
    pack = _pack()
    draft = EvidenceFirstAnswerer().draft(pack, RuleBasedQuestionRouter().route("Biển hiệu ghi gì?"))
    assert not confirmation_guard(draft, pack, ()).allowed
    assert confirmation_guard(draft, pack, draft.evidence_refs).allowed
