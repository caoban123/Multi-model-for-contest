from __future__ import annotations

from aic_retrieval.qa_answering import EvidenceFirstAnswerer
from aic_retrieval.qa_evidence import build_evidence_pack
from aic_retrieval.qa_question_router import RuleBasedQuestionRouter
from aic_retrieval.qa_schema import AvailabilityStatus, ConfidenceState, EvidenceModality, QaRequest


def ocr_response() -> dict:
    return {"video_results":[{"video_id":"L21_V001","frames":[{"video_id":"L21_V001","keyframe_id":5,"frame_idx":125,"pts_time":5.0,"evidence":{"ocr":[{"matched_text":"Samsung","confidence":.9}],"asr":[],"object":[],"attribute":[]}}]}]}


def test_ocr_answer_is_evidence_backed_but_requires_human_review() -> None:
    pack = build_evidence_pack(
        QaRequest("q1", "a sign", "Biển hiệu ghi gì?"), ocr_response(),
        modality_availability={EvidenceModality.OCR: AvailabilityStatus.AVAILABLE},
    )
    draft = EvidenceFirstAnswerer().draft(pack, RuleBasedQuestionRouter().route("Biển hiệu ghi gì?"))
    assert draft.raw_answer == "Samsung"
    assert draft.evidence_refs
    assert draft.confidence_state is ConfidenceState.REVIEW_REQUIRED


def test_speech_question_is_blocked_when_asr_is_unavailable() -> None:
    pack = build_evidence_pack(QaRequest("q2", "a speaker", "Người đó nói gì?"), ocr_response())
    draft = EvidenceFirstAnswerer().draft(pack, RuleBasedQuestionRouter().route("Người đó nói gì?"))
    assert draft.raw_answer is None
    assert draft.confidence_state is ConfidenceState.UNSUPPORTED
    assert draft.warnings == ("ASR_UNAVAILABLE",)


def test_count_question_is_never_auto_answered_from_a_single_frame() -> None:
    pack = build_evidence_pack(QaRequest("q3", "people", "Có bao nhiêu người?"), ocr_response())
    draft = EvidenceFirstAnswerer().draft(pack, RuleBasedQuestionRouter().route("Có bao nhiêu người?"))
    assert draft.raw_answer is None
    assert draft.confidence_state is ConfidenceState.NEEDS_EVIDENCE


def test_channel_question_prefers_the_human_readable_metadata_author() -> None:
    response = {"video_results":[{"video_id":"L21_V001","frames":[]}]}
    pack = build_evidence_pack(QaRequest("q4", "a video", "Which channel is this video from?"), response, metadata_by_video={"L21_V001":{"author":"60 Giay Official","channel_id":"UC-technical-id"}})
    draft = EvidenceFirstAnswerer().draft(pack, RuleBasedQuestionRouter().route("Which channel is this video from?"))
    assert draft.raw_answer == "60 Giay Official"


def test_channel_id_question_returns_the_explicit_technical_identifier() -> None:
    response = {"video_results":[{"video_id":"L21_V001","frames":[]}]}
    pack = build_evidence_pack(QaRequest("q5", "a video", "What is the YouTube channel ID?"), response, metadata_by_video={"L21_V001":{"author":"60 Giay Official","channel_id":"UC-technical-id"}})
    draft = EvidenceFirstAnswerer().draft(pack, RuleBasedQuestionRouter().route("What is the YouTube channel ID?"))
    assert draft.raw_answer == "UC-technical-id"
