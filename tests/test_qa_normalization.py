from __future__ import annotations

from pathlib import Path

from aic_retrieval.qa_normalization import AnswerNormalizer, load_normalization_policy


def test_normalization_preserves_raw_answer_and_adds_accent_color_alternatives() -> None:
    result = AnswerNormalizer().normalize("  Đỏ  ")
    assert result.raw_answer == "Đỏ"
    assert result.normalized_answer == "đỏ"
    assert set(result.alternative_answers) == {"do", "red"}


def test_number_and_unit_variants_are_non_destructive() -> None:
    number = AnswerNormalizer().normalize("Năm")
    unit = AnswerNormalizer().normalize("5 kg")
    assert number.raw_answer == "Năm"
    assert number.normalized_answer == "năm"
    assert "5" in number.alternative_answers
    assert unit.raw_answer == "5 kg"
    assert unit.normalized_answer == "5 kg"
    assert "5kg" in unit.alternative_answers


def test_proper_name_raw_case_is_preserved_and_policy_loads_from_config() -> None:
    result = AnswerNormalizer().normalize("Samsung")
    policy = load_normalization_policy(Path("configs/phase7_qa_v1.json"))
    assert result.raw_answer == "Samsung"
    assert result.normalized_answer == "samsung"
    assert policy.version == result.rules_version
