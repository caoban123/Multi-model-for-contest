from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MOJIBAKE_MARKERS = (
    "\u00c3\u00a2",
    "\u00c3\u00a1",
    "\u00c3\u00a0",
    "\u00c3\u00a9",
    "\u00c3\u00aa",
    "\u00c3\u00b4",
    "\u00c6\u00b0",
    "\u00c4\u0091",
    "\u00c4\u0090",
    "\u00e1\u00bb",
    "\u00e1\u00ba",
    "\u00e2\u20ac",
    "\u00e2\u0096",
    "\u00e2\u009c",
    "\u00c2\u00a9",
    "\u00c2\u00ae",
)

VIETNAMESE_CHARS = set(
    "\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0"
    "\u00e1\u00e0\u1ea3\u00e3\u1ea1\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u1ea5\u1ea7\u1ea9\u1eab\u1ead"
    "\u00e9\u00e8\u1ebb\u1ebd\u1eb9\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7"
    "\u00ed\u00ec\u1ec9\u0129\u1ecb"
    "\u00f3\u00f2\u1ecf\u00f5\u1ecd\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u1edb\u1edd\u1edf\u1ee1\u1ee3"
    "\u00fa\u00f9\u1ee7\u0169\u1ee5\u1ee9\u1eeb\u1eed\u1eef\u1ef1"
    "\u00fd\u1ef3\u1ef7\u1ef9\u1ef5"
)


@dataclass(frozen=True)
class MetadataAudit:
    total_files: int
    json_errors: int
    files_with_mojibake_markers: int
    files_with_vietnamese_text: int
    files_with_control_chars: int
    field_presence: dict[str, int]
    top_authors: list[dict[str, Any]]
    sample_records: list[dict[str, Any]]


def audit_media_info(media_dir: Path, sample_limit: int = 5) -> MetadataAudit:
    paths = sorted(media_dir.glob("*.json"))
    json_errors = 0
    files_with_mojibake = 0
    files_with_vietnamese = 0
    files_with_control_chars = 0
    field_presence: Counter[str] = Counter()
    authors: Counter[str] = Counter()
    samples = []

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            json_errors += 1
            continue

        for key, value in payload.items():
            if value not in (None, "", []):
                field_presence[key] += 1
        text = metadata_text(payload)
        if has_mojibake_markers(text):
            files_with_mojibake += 1
        if has_vietnamese_text(text):
            files_with_vietnamese += 1
        if has_control_chars(text):
            files_with_control_chars += 1
        if payload.get("author"):
            authors[str(payload["author"])] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "video_id": path.stem,
                    "title": payload.get("title", ""),
                    "author": payload.get("author", ""),
                    "publish_date": payload.get("publish_date", ""),
                    "keywords_count": len(payload.get("keywords") or []),
                    "description_preview": str(payload.get("description", ""))[:180],
                }
            )

    return MetadataAudit(
        total_files=len(paths),
        json_errors=json_errors,
        files_with_mojibake_markers=files_with_mojibake,
        files_with_vietnamese_text=files_with_vietnamese,
        files_with_control_chars=files_with_control_chars,
        field_presence=dict(sorted(field_presence.items())),
        top_authors=[{"author": author, "count": count} for author, count in authors.most_common(10)],
        sample_records=samples,
    )


def metadata_text(payload: dict[str, Any]) -> str:
    parts = []
    for key in ("title", "author", "description"):
        value = payload.get(key)
        if value:
            parts.append(str(value))
    keywords = payload.get("keywords") or []
    parts.extend(str(item) for item in keywords)
    return "\n".join(parts)


def normalize_metadata_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_mojibake_markers(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def has_vietnamese_text(text: str) -> bool:
    lowered = text.lower()
    return any(char in lowered for char in VIETNAMESE_CHARS)


def has_control_chars(text: str) -> bool:
    return any((unicodedata.category(char) == "Cc" and char not in "\n\r\t") for char in text)


def audit_to_dict(audit: MetadataAudit) -> dict[str, Any]:
    return asdict(audit)
