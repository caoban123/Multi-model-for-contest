"""Non-destructive, versioned answer normalization for Phase 7."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NORMALIZATION_VERSION = "phase7-nondestructive-normalization-v1"


@dataclass(frozen=True)
class NormalizationPolicy:
    version: str
    number_aliases: dict[str, tuple[str, ...]]
    color_aliases: dict[str, tuple[str, ...]]
    unit_aliases: tuple[str, ...]


@dataclass(frozen=True)
class NormalizationResult:
    raw_answer: str
    normalized_answer: str
    alternative_answers: tuple[str, ...]
    rules_version: str


class AnswerNormalizer:
    def __init__(self, policy: NormalizationPolicy | None = None) -> None:
        self.policy = policy or default_policy()

    def normalize(self, raw_answer: str) -> NormalizationResult:
        raw = raw_answer.strip()
        if not raw:
            raise ValueError("raw_answer must not be empty")
        normalized = _canonical(raw)
        alternatives: list[str] = []
        _append(alternatives, _accent_fold(normalized), normalized)
        for candidate in self.policy.number_aliases.get(normalized, ()):
            _append(alternatives, candidate, normalized)
        for candidate in self.policy.color_aliases.get(normalized, ()):
            _append(alternatives, candidate, normalized)
        for candidate in _unit_variants(normalized, self.policy.unit_aliases):
            _append(alternatives, candidate, normalized)
        return NormalizationResult(raw, normalized, tuple(alternatives), self.policy.version)


def default_policy() -> NormalizationPolicy:
    return NormalizationPolicy(
        version=NORMALIZATION_VERSION,
        number_aliases={
            "một": ("1",), "hai": ("2",), "ba": ("3",), "bốn": ("4",), "năm": ("5",),
            "sáu": ("6",), "bảy": ("7",), "tám": ("8",), "chín": ("9",), "mười": ("10",),
            "one": ("1",), "two": ("2",), "three": ("3",), "four": ("4",), "five": ("5",),
            "six": ("6",), "seven": ("7",), "eight": ("8",), "nine": ("9",), "ten": ("10",),
        },
        color_aliases={
            "đỏ": ("red",), "red": ("đỏ",), "xanh": ("blue", "green"), "blue": ("xanh",),
            "green": ("xanh",), "trắng": ("white",), "white": ("trắng",), "đen": ("black",),
            "black": ("đen",), "vàng": ("yellow",), "yellow": ("vàng",), "hồng": ("pink",), "pink": ("hồng",),
        },
        unit_aliases=("kg", "g", "km", "m", "cm", "phút", "giây", "hours", "minutes", "seconds"),
    )


def load_normalization_policy(path: Path) -> NormalizationPolicy:
    import json

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    section = payload.get("normalization", {})
    if payload.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError(f"unsupported normalization version: {payload.get('normalization_version')}")
    return NormalizationPolicy(
        version=payload["normalization_version"],
        number_aliases={key: tuple(value) for key, value in section.get("number_aliases", {}).items()},
        color_aliases={key: tuple(value) for key, value in section.get("color_aliases", {}).items()},
        unit_aliases=tuple(section.get("unit_aliases", ())),
    )


def _canonical(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = " ".join(text.split())
    return text.casefold().strip(" .,:;!?…")


def _accent_fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").replace("đ", "d")


def _unit_variants(value: str, units: tuple[str, ...]) -> tuple[str, ...]:
    if not units:
        return ()
    pattern = "|".join(re.escape(unit) for unit in units)
    compact = re.fullmatch(rf"(\d+(?:[.,]\d+)?)\s+({pattern})", value)
    if compact:
        return (f"{compact.group(1)}{compact.group(2)}",)
    spaced = re.fullmatch(rf"(\d+(?:[.,]\d+)?)({pattern})", value)
    if spaced:
        return (f"{spaced.group(1)} {spaced.group(2)}",)
    return ()


def _append(values: list[str], candidate: str, normalized: str) -> None:
    if candidate and candidate != normalized and candidate not in values:
        values.append(candidate)
