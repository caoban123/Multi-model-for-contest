from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

PLANNER_VERSION = "phase6-rule-planner-v3"

OBJECT_TERMS = {
    "phone": ("phone", "điện thoại", "smartphone"),
    "car": ("car", "vehicle", "ô tô", "xe hơi"),
    "person": ("person", "people", "người", "man", "woman"),
    "bicycle": ("bicycle", "bike", "xe đạp"),
}
TEXT_MARKERS = (
    "chữ", "text", "biển hiệu", "biển báo", "logo", "phụ đề", "ghi", "hiển thị",
    "bảng", "ký tự", "con số", "số mấy", "địa chỉ", "đường nào", "màn hình",
    "đếm ngược", "written", "sign", "caption", "countdown",
)
SPEECH_MARKERS = (
    "nói", "phát biểu", "nhắc đến", "được nhắc đến", "nghe", "lời thoại", "trò chuyện",
    "đối thoại", "trả lời", "says", "speaks", "talks", "conversation",
)
TEMPORAL_MARKERS = ("trước", "sau", "sau đó", "tiếp theo", "cuối cùng", "before", "after", "then", "finally")
METADATA_MARKERS = ("kênh", "tác giả", "ngày", "tiêu đề", "channel", "author", "published")
COLORS = ("đỏ", "xanh", "vàng", "trắng", "đen", "hồng", "green", "red", "blue", "white", "black")
QUESTION_CLAUSE_RE = re.compile(
    r"(?:[.!?]\s*)?(?:hỏi|hãy cho biết|cho biết|question)\s*[:：]?\s*",
    flags=re.IGNORECASE,
)


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
        visual_source = _visual_clause(original)
        visual_text = normalize(visual_source)
        events = _visual_events(visual_text)

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
        without_markers = normalize(visual_source)
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


def _visual_clause(value: str) -> str:
    """Keep the scene description and remove an explicit trailing answer question."""
    parts = QUESTION_CLAUSE_RE.split(value.strip(), maxsplit=1)
    scene = parts[0].strip(" ,.;:-") if parts else ""
    return scene or value.strip()


def _visual_events(value: str) -> tuple[str, ...]:
    """Split a narrative into reviewable visual moments without inventing text."""
    parts = re.split(
        r"(?:[.!?;]+\s*|\b(?:sau đó|tiếp theo|cuối cùng|rồi|then|after that|finally)\b)",
        value,
        flags=re.IGNORECASE,
    )
    events: list[str] = []
    seen: set[str] = set()
    for part in parts:
        event = " ".join(part.strip(" ,.;:-").split())
        key = event.casefold()
        if len(event) < 8 or key in seen:
            continue
        seen.add(key)
        events.append(event)
        if len(events) >= 5:
            break
    return tuple(events) or (" ".join(value.split()),)
