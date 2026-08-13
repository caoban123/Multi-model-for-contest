from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from aic_retrieval.metadata import normalize_metadata_text

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
FIELD_WEIGHTS = {
    "video_id": 2.0,
    "title": 3.0,
    "author": 2.0,
    "publish_date": 2.0,
    "keywords": 2.0,
    "description": 1.0,
}
EXACT_TOKEN_BONUS = 0.35
EXACT_PHRASE_BONUSES = {
    "title": 3.0,
    "author": 2.5,
    "keywords": 2.0,
    "description": 0.5,
}
NORMALIZED_PHRASE_BONUSES = {
    "title": 0.8,
    "author": 0.6,
    "keywords": 0.5,
    "description": 0.1,
}


@dataclass(frozen=True)
class MetadataDocument:
    video_id: str
    title: str
    author: str
    publish_date: str
    keywords: list[str]
    description: str
    watch_url: str
    token_weights: dict[str, float]
    original_token_weights: dict[str, float]
    original_fields: dict[str, str]
    normalized_fields: dict[str, str]
    token_count: float


@dataclass(frozen=True)
class MetadataSearchResult:
    rank: int
    video_id: str
    score: float
    matched_terms: list[str]
    title: str
    author: str
    publish_date: str
    keywords: list[str]
    description_preview: str
    watch_url: str


def load_metadata_documents(media_dir: Path, groups: set[str] | None = None) -> list[MetadataDocument]:
    docs = []
    for path in sorted(media_dir.glob("*.json")):
        video_id = path.stem
        if groups and video_id.split("_", 1)[0] not in groups:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        docs.append(document_from_payload(video_id, payload))
    return docs


def document_from_payload(video_id: str, payload: dict[str, Any]) -> MetadataDocument:
    keywords = [str(item) for item in (payload.get("keywords") or [])]
    description = str(payload.get("description", "") or "")
    fields = {
        "video_id": video_id,
        "title": str(payload.get("title", "") or ""),
        "author": str(payload.get("author", "") or ""),
        "publish_date": str(payload.get("publish_date", "") or ""),
        "keywords": " ".join(keywords),
        "description": description,
    }
    token_weights: defaultdict[str, float] = defaultdict(float)
    original_token_weights: defaultdict[str, float] = defaultdict(float)
    original_fields = {field: normalize_metadata_text(text).lower() for field, text in fields.items()}
    normalized_fields = {field: normalize_for_search(text) for field, text in fields.items()}
    for field, text in fields.items():
        weight = FIELD_WEIGHTS[field]
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            token_weights[token] += count * weight
        original_counts = Counter(tokenize_original(text))
        for token, count in original_counts.items():
            original_token_weights[token] += count * weight
    return MetadataDocument(
        video_id=video_id,
        title=fields["title"],
        author=fields["author"],
        publish_date=fields["publish_date"],
        keywords=keywords,
        description=description,
        watch_url=str(payload.get("watch_url", "") or ""),
        token_weights=dict(token_weights),
        original_token_weights=dict(original_token_weights),
        original_fields=original_fields,
        normalized_fields=normalized_fields,
        token_count=sum(token_weights.values()),
    )


def search_metadata(
    docs: list[MetadataDocument],
    query: str,
    top_k: int = 10,
    min_match: int = 1,
) -> list[MetadataSearchResult]:
    query_terms = tokenize(query)
    original_query_terms = tokenize_original(query)
    if not query_terms or not docs:
        return []

    query_counts = Counter(query_terms)
    original_query_counts = Counter(original_query_terms)
    doc_freq = compute_doc_freqs(docs)
    avg_len = sum(doc.token_count for doc in docs) / max(len(docs), 1)
    scored = []
    for doc in docs:
        score, matched = score_document(
            doc,
            query_counts,
            original_query_counts,
            normalize_metadata_text(query).lower(),
            normalize_for_search(query),
            doc_freq,
            len(docs),
            avg_len,
        )
        if score > 0 and len(matched) >= min_match:
            scored.append((score, matched, doc))

    scored.sort(key=lambda item: (-item[0], item[2].video_id))
    return [
        MetadataSearchResult(
            rank=rank,
            video_id=doc.video_id,
            score=round(score, 6),
            matched_terms=matched,
            title=doc.title,
            author=doc.author,
            publish_date=doc.publish_date,
            keywords=doc.keywords[:12],
            description_preview=normalize_metadata_text(doc.description)[:240],
            watch_url=doc.watch_url,
        )
        for rank, (score, matched, doc) in enumerate(scored[:top_k], start=1)
    ]


def score_document(
    doc: MetadataDocument,
    query_counts: Counter[str],
    original_query_counts: Counter[str],
    original_query: str,
    normalized_query: str,
    doc_freq: dict[str, int],
    doc_count: int,
    avg_len: float,
) -> tuple[float, list[str]]:
    k1 = 1.2
    b = 0.75
    score = 0.0
    matched = []
    length_norm = k1 * (1 - b + b * (doc.token_count / max(avg_len, 1e-9)))
    for term, query_tf in query_counts.items():
        tf = doc.token_weights.get(term, 0.0)
        if tf <= 0:
            continue
        idf = math.log(1 + (doc_count - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
        score += query_tf * idf * ((tf * (k1 + 1)) / (tf + length_norm))
        matched.append(term)
    for term, query_tf in original_query_counts.items():
        if term in doc.original_token_weights:
            score += EXACT_TOKEN_BONUS * query_tf
    if len(original_query_counts) > 1:
        for field, bonus in EXACT_PHRASE_BONUSES.items():
            if original_query and original_query in doc.original_fields.get(field, ""):
                score += bonus
            elif normalized_query and normalized_query in doc.normalized_fields.get(field, ""):
                score += NORMALIZED_PHRASE_BONUSES[field]
    return score, matched


def compute_doc_freqs(docs: Iterable[MetadataDocument]) -> dict[str, int]:
    freqs: defaultdict[str, int] = defaultdict(int)
    for doc in docs:
        for token in doc.token_weights:
            freqs[token] += 1
    return dict(freqs)


def tokenize(text: str) -> list[str]:
    normalized = normalize_for_search(text)
    tokens = TOKEN_RE.findall(normalized)
    return [token for token in tokens if token]


def tokenize_original(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(normalize_metadata_text(text).lower()) if token]


def normalize_for_search(text: str) -> str:
    text = normalize_metadata_text(text).lower()
    text = text.replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def results_to_dict(results: list[MetadataSearchResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]
