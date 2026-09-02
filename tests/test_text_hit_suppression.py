from __future__ import annotations

import pytest

from aic_retrieval.retrievers import RetrievalHit
from aic_retrieval.text_hit_suppression import TextHitSuppressionConfig, suppress_text_hit_noise


def _hit(rank: int, video_id: str, text: str, *, source_type: str = "ocr") -> RetrievalHit:
    return RetrievalHit(
        "bm25",
        rank,
        10.0 - rank,
        video_id,
        source_type,
        document_id=f"{source_type}:{video_id}:{rank}",
        keyframe_id=rank,
        frame_idx=rank * 30,
        pts_time=float(rank),
        matched_text=text,
    )


def test_suppression_removes_repeated_ocr_and_diversifies_videos() -> None:
    result = suppress_text_hit_noise(
        [
            _hit(1, "L21_V001", "HTV"),
            _hit(2, "L21_V001", "htv"),
            _hit(3, "L21_V001", "60 giây"),
            _hit(4, "L21_V001", "địa chỉ đường A"),
            _hit(5, "L21_V002", "địa chỉ đường B"),
            _hit(6, "L21_V003", "news title", source_type="metadata"),
        ],
        limit=5,
        config=TextHitSuppressionConfig(max_ocr_hits_per_video=2),
    )

    assert [hit.document_id for hit in result.hits] == [
        "ocr:L21_V001:1",
        "ocr:L21_V001:3",
        "ocr:L21_V002:5",
        "metadata:L21_V003:6",
    ]
    assert result.duplicate_ocr_removed == 1
    assert result.video_cap_removed == 1
    assert result.hits[0].provenance["text_hit_suppression"]["original_rank"] == 1


def test_suppression_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        TextHitSuppressionConfig(max_ocr_hits_per_video=0)
    with pytest.raises(ValueError, match="positive"):
        suppress_text_hit_noise([], limit=0)
