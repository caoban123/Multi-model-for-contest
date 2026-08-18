from __future__ import annotations

from aic_retrieval.qa_evidence import build_evidence_pack
from aic_retrieval.qa_schema import AvailabilityStatus, EvidenceModality, QaRequest


def response() -> dict:
    return {
        "mode": "structured",
        "query": "a person holding a phone",
        "index_schema_version": "2.0",
        "video_results": [{
            "video_id": "L21_V001",
            "rank": 1,
            "frames": [{
                "video_id": "L21_V001", "keyframe_id": 5, "frame_idx": 125,
                "pts_time": 5.0, "keyframe_path": "data/keyframes/L21_V001/005.jpg",
                "clip_rank": 1, "clip_score": 0.9,
                "evidence": {
                    "object": [{"matched_labels": ["phone"], "object_rank": 1}],
                    "attribute": [],
                    "ocr": [],
                    "asr": [],
                },
            }],
            "evidence": {"metadata": {"matched_fields": [{"field": "author", "value": "News"}]}},
        }],
    }


def test_builder_keeps_metadata_video_level_and_marks_missing_phase5_unavailable() -> None:
    pack = build_evidence_pack(
        QaRequest("q1", "person holding a phone", "Which channel is this video from?"),
        response(),
        metadata_by_video={"L21_V001": {"channel_id": "news", "title": "Example"}},
    )
    metadata = next(item for item in pack.evidence_refs if item.modality is EvidenceModality.METADATA)
    assert metadata.keyframe_id is None
    assert metadata.payload["channel_id"] == "news"
    assert pack.modality_availability[EvidenceModality.OCR] is AvailabilityStatus.UNAVAILABLE
    assert pack.modality_availability[EvidenceModality.ASR] is AvailabilityStatus.UNAVAILABLE


def test_builder_preserves_frame_evidence_and_provenance() -> None:
    pack = build_evidence_pack(QaRequest("q2", "person", "What is held?"), response())
    object_ref = next(item for item in pack.evidence_refs if item.modality is EvidenceModality.OBJECT)
    assert object_ref.video_id == "L21_V001"
    assert object_ref.keyframe_id == 5
    assert object_ref.source == "object_store"
    assert object_ref.evidence_id.startswith("object:L21_V001:")


def test_builder_is_deterministic_for_the_same_retrieval_response() -> None:
    request = QaRequest("q3", "person", "What is held?")
    first = build_evidence_pack(request, response())
    second = build_evidence_pack(request, response())
    assert [item.evidence_id for item in first.evidence_refs] == [item.evidence_id for item in second.evidence_refs]
