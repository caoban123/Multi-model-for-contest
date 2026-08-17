from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

PLANNER_VERSION = "phase6-rule-planner-v1"

OBJECT_TERMS = {
    "phone": ("phone", "điện thoại", "smartphone"),
    "car": ("car", "vehicle", "ô tô", "xe hơi"),
    "person": ("person", "people", "người", "man", "woman"),
    "bicycle": ("bicycle", "bike", "xe đạp"),
}
TEXT_MARKERS = ("chữ", "text", "biển hiệu", "logo", "phụ đề", "ghi", "hiển thị")
SPEECH_MARKERS = ("nói", "phát biểu", "nhắc đến", "nghe", "lời thoại", "says", "speaks")
TEMPORAL_MARKERS = ("trước", "sau", "sau đó", "tiếp theo", "before", "after", "then")
METADATA_MARKERS = ("kênh", "tác giả", "ngày", "tiêu đề", "channel", "author", "published")
COLORS = ("đỏ", "xanh", "vàng", "trắng", "đen", "hồng", "green", "red", "blue", "white", "black")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    visual_query: str
    variants: tuple[str, ...]
    events: tuple[str, ...]
    objects: tuple[str, ...]
    text_hints: tuple[str, ...]
    metadata_hints: tuple[str, ...]
    temporal_hints: tuple[str, ...]
    recommended_modalities: tuple[str, ...]
    reasons: tuple[str, ...]
    version: str = PLANNER_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RuleBasedQueryPlanner:
    """Deterministic local planner; it never calls an LLM or network API."""

    def plan(self, query: str, max_variants: int = 3) -> QueryPlan:
        original = query.strip()
        if not original:
            raise ValueError("query must not be empty")
        text = normalize(original)
        objects = tuple(label for label, aliases in OBJECT_TERMS.items() if any(contains_phrase(text, alias) for alias in aliases))
        text_hints = tuple(marker for marker in TEXT_MARKERS if contains_phrase(text, marker))
        speech_hints = tuple(marker for marker in SPEECH_MARKERS if contains_phrase(text, marker))
        temporal = tuple(marker for marker in TEMPORAL_MARKERS if contains_phrase(text, marker))
        metadata = tuple(marker for marker in METADATA_MARKERS if contains_phrase(text, marker))
        events = tuple(part.strip(" ,.;") for part in re.split(r"\b(?:sau đó|tiếp theo|then|after that)\b", text) if part.strip(" ,.;"))

        modalities = ["clip"]
        reasons = ["CLIP là fallback semantic mặc định"]
        if objects:
            modalities.append("objects"); reasons.append("query chứa object đã biết")
        if any(contains_phrase(text, color) for color in COLORS):
            modalities.append("attributes"); reasons.append("query chứa màu/thuộc tính")
        if text_hints or quoted_phrases(original):
            modalities.append("ocr"); reasons.append("query chứa dấu hiệu chữ trong hình")
        if speech_hints:
            modalities.append("asr"); reasons.append("query chứa dấu hiệu lời nói")
        if metadata:
            modalities.append("metadata"); reasons.append("query chứa dấu hiệu metadata")
        if temporal:
            reasons.append("query có quan hệ thời gian; giữ thứ tự event trong explanation")

        variants = [original]
        without_markers = text
        for marker in (*TEXT_MARKERS, *SPEECH_MARKERS, *METADATA_MARKERS):
            without_markers = re.sub(rf"(?<!\w){re.escape(marker)}(?!\w)", " ", without_markers)
        compact = " ".join(without_markers.split())
        if compact and compact != text:
            variants.append(compact)
        for quoted in quoted_phrases(original):
            if quoted not in variants:
                variants.append(quoted)
        return QueryPlan(
            original_query=original,
            visual_query=compact or original,
            variants=tuple(variants[: max(1, min(max_variants, 5))]),
            events=events,
            objects=objects,
            text_hints=tuple((*text_hints, *speech_hints)),
            metadata_hints=metadata,
            temporal_hints=temporal,
            recommended_modalities=tuple(dict.fromkeys(modalities)),
            reasons=tuple(reasons),
        )

    def select_variants(self, plan: QueryPlan, selected: tuple[str, ...]) -> QueryPlan:
        allowed=set(plan.variants); chosen=tuple(value for value in selected if value in allowed)
        if not chosen:
            chosen=(plan.original_query,)
        return QueryPlan(**{**plan.to_dict(),"variants":chosen})


def quoted_phrases(value: str) -> tuple[str, ...]:
    return tuple(match.strip() for match in re.findall(r'["“”]([^"“”]+)["“”]', value) if match.strip())
