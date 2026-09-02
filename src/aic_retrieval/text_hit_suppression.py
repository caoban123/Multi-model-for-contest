from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from aic_retrieval.phase5_schema import normalize_text
from aic_retrieval.retrievers import RetrievalHit


TEXT_HIT_SUPPRESSION_VERSION = "text-hit-suppression-v1"


@dataclass(frozen=True)
class TextHitSuppressionConfig:
    max_ocr_hits_per_video: int = 4
    max_same_ocr_text_per_video: int = 1

    def __post_init__(self) -> None:
        if self.max_ocr_hits_per_video < 1 or self.max_same_ocr_text_per_video < 1:
            raise ValueError("text hit suppression limits must be positive")


@dataclass(frozen=True)
class TextHitSuppressionResult:
    hits: tuple[RetrievalHit, ...]
    input_hits: int
    duplicate_ocr_removed: int
    video_cap_removed: int
    version: str = TEXT_HIT_SUPPRESSION_VERSION

    def stats(self) -> dict[str, int | str]:
        return {
            "version": self.version,
            "input_hits": self.input_hits,
            "output_hits": len(self.hits),
            "duplicate_ocr_removed": self.duplicate_ocr_removed,
            "video_cap_removed": self.video_cap_removed,
        }


def suppress_text_hit_noise(
    hits: list[RetrievalHit] | tuple[RetrievalHit, ...],
    *,
    limit: int,
    config: TextHitSuppressionConfig = TextHitSuppressionConfig(),
) -> TextHitSuppressionResult:
    """Diversify OCR hits without modifying the indexed documents or raw scores."""
    if limit < 1:
        raise ValueError("limit must be positive")
    ordered = sorted(hits, key=lambda hit: (hit.rank, hit.video_id, hit.document_id or ""))
    ocr_per_video: Counter[str] = Counter()
    repeated_text: Counter[tuple[str, str]] = Counter()
    output: list[RetrievalHit] = []
    duplicate_removed = 0
    cap_removed = 0
    for hit in ordered:
        if hit.source_type == "ocr":
            normalized = normalize_text(hit.matched_text or "", fold_accents=True)
            duplicate_key = (hit.video_id, normalized)
            if normalized and repeated_text[duplicate_key] >= config.max_same_ocr_text_per_video:
                duplicate_removed += 1
                continue
            if ocr_per_video[hit.video_id] >= config.max_ocr_hits_per_video:
                cap_removed += 1
                continue
            repeated_text[duplicate_key] += 1
            ocr_per_video[hit.video_id] += 1
        output.append(
            replace(
                hit,
                rank=len(output) + 1,
                provenance={
                    **dict(hit.provenance),
                    "text_hit_suppression": {
                        "version": TEXT_HIT_SUPPRESSION_VERSION,
                        "original_rank": hit.rank,
                    },
                },
            )
        )
        if len(output) >= limit:
            break
    return TextHitSuppressionResult(
        tuple(output),
        input_hits=len(ordered),
        duplicate_ocr_removed=duplicate_removed,
        video_cap_removed=cap_removed,
    )
