from __future__ import annotations

from aic_retrieval.qa_question_router import RuleBasedQuestionRouter
from aic_retrieval.qa_schema import EvidenceModality, QuestionType


def test_router_recognizes_source_specific_question_types() -> None:
    router = RuleBasedQuestionRouter()
    assert router.route("Biển hiệu ghi gì?").question_type is QuestionType.OCR
    assert router.route("Người đó nói gì?").question_type is QuestionType.ASR
    assert router.route("Video thuộc kênh nào?").question_type is QuestionType.METADATA
    assert router.route("Có bao nhiêu người?").question_type is QuestionType.COUNT


def test_router_marks_color_as_attribute_and_unknown_as_unknown() -> None:
    router = RuleBasedQuestionRouter()
    assert router.route("Áo màu gì?").required_modalities == (EvidenceModality.ATTRIBUTE,)
    assert router.route("Hãy giải thích ý nghĩa của cảnh này").question_type is QuestionType.UNKNOWN
