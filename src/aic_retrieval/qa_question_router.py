"""Local, explainable question routing for evidence-grounded Q&A."""

from __future__ import annotations

from dataclasses import dataclass

from aic_retrieval.qa_schema import EvidenceModality, QuestionType


QUESTION_ROUTER_VERSION = "phase7-rule-router-v1"


@dataclass(frozen=True)
class QuestionRoute:
    question_type: QuestionType
    required_modalities: tuple[EvidenceModality, ...]
    reason: str
    version: str = QUESTION_ROUTER_VERSION
    answer_field: str | None = None


class RuleBasedQuestionRouter:
    """Routes explicit question cues without introducing a new model dependency."""

    def route(self, question: str) -> QuestionRoute:
        normalized = " ".join(question.casefold().split())
        if not normalized:
            raise ValueError("question must not be empty")
        if _contains(normalized, "youtube channel id", "channel id"):
            return QuestionRoute(QuestionType.METADATA, (EvidenceModality.METADATA,), "metadata channel-id cue", answer_field="channel_id")
        if _contains(normalized, "channel"):
            return QuestionRoute(QuestionType.METADATA, (EvidenceModality.METADATA,), "metadata channel-author cue", answer_field="author")
        if _contains(normalized, "bao nhiêu", "how many", "mấy ", "số lượng", "number of"):
            return QuestionRoute(QuestionType.COUNT, (EvidenceModality.OBJECT, EvidenceModality.TEMPORAL), "count cue requires multi-frame review")
        if _contains(normalized, "trước", "sau", "tiếp theo", "before", "after", "next"):
            return QuestionRoute(QuestionType.TEMPORAL, (EvidenceModality.TEMPORAL,), "temporal cue")
        if _contains(normalized, "nói gì", "phát biểu", "nhắc đến", "lời thoại", "what does", "say", "speak"):
            return QuestionRoute(QuestionType.ASR, (EvidenceModality.ASR,), "speech cue")
        if _contains(normalized, "ghi gì", "viết gì", "chữ", "biển hiệu", "logo", "text", "sign"):
            return QuestionRoute(QuestionType.OCR, (EvidenceModality.OCR,), "visible-text cue")
        if _contains(normalized, "kênh nào", "tác giả", "đăng ngày", "tiêu đề", "channel", "author", "published", "title"):
            return QuestionRoute(QuestionType.METADATA, (EvidenceModality.METADATA,), "metadata-field cue")
        if _contains(normalized, "màu gì", "màu", "color", "colour"):
            return QuestionRoute(QuestionType.ATTRIBUTE_COLOR, (EvidenceModality.ATTRIBUTE,), "color cue")
        if _contains(normalized, "cầm gì", "vật gì", "đối tượng", "what is", "what object", "holding"):
            return QuestionRoute(QuestionType.OBJECT, (EvidenceModality.OBJECT,), "object cue")
        if _contains(normalized, "ai", "ở đâu", "mặc", "trông", "what is shown", "who is", "where is"):
            return QuestionRoute(QuestionType.VISUAL, (EvidenceModality.KEYFRAME,), "visual cue")
        return QuestionRoute(QuestionType.UNKNOWN, (), "no reliable local question-type cue")


def _contains(value: str, *markers: str) -> bool:
    return any(marker in value for marker in markers)
