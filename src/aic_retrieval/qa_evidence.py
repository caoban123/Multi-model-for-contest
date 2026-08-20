"""Build auditable Phase-7 evidence packs from retrieval outputs.

The builder is deliberately retrieval-model agnostic.  It accepts the existing
visual or structured response contract and converts only source-backed values
into typed evidence.  In particular, metadata stays video-level and absent
OCR/ASR stores are represented as unavailable capabilities, never as empty
transcripts or negative evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from aic_retrieval.qa_schema import (
    AvailabilityStatus,
    EvidenceModality,
    EvidencePack,
    EvidenceRef,
    QaRequest,
)


DEFAULT_AVAILABILITY: dict[EvidenceModality, AvailabilityStatus] = {
    EvidenceModality.KEYFRAME: AvailabilityStatus.AVAILABLE,
    EvidenceModality.CLIP: AvailabilityStatus.AVAILABLE,
    EvidenceModality.OBJECT: AvailabilityStatus.UNKNOWN,
    EvidenceModality.METADATA: AvailabilityStatus.UNKNOWN,
    EvidenceModality.OCR: AvailabilityStatus.UNAVAILABLE,
    EvidenceModality.ASR: AvailabilityStatus.UNAVAILABLE,
    EvidenceModality.ATTRIBUTE: AvailabilityStatus.UNKNOWN,
    EvidenceModality.TEMPORAL: AvailabilityStatus.UNKNOWN,
}


def build_evidence_pack(
    request: QaRequest,
    retrieval_response: Mapping[str, Any],
    *,
    metadata_by_video: Mapping[str, Mapping[str, Any]] | None = None,
    modality_availability: Mapping[EvidenceModality, AvailabilityStatus] | None = None,
    max_evidence_per_modality: int = 8,
) -> EvidencePack:
    """Create a deterministic evidence pack without generating an answer."""
    if max_evidence_per_modality <= 0:
        raise ValueError("max_evidence_per_modality must be positive")
    availability = dict(DEFAULT_AVAILABILITY)
    if modality_availability:
        availability.update(modality_availability)
    metadata_by_video = metadata_by_video or {}
    candidates = _candidate_rows(retrieval_response)
    refs: list[EvidenceRef] = []
    counts: dict[EvidenceModality, int] = {modality: 0 for modality in EvidenceModality}

    for candidate in candidates:
        video_id = str(candidate.get("video_id", "")).strip()
        if not video_id:
            continue
        for frame in _candidate_frames(candidate):
            refs.extend(_frame_evidence(frame, video_id, retrieval_response, availability, counts, max_evidence_per_modality))
        metadata = metadata_by_video.get(video_id)
        if metadata and counts[EvidenceModality.METADATA] < max_evidence_per_modality:
            refs.append(_ref(
                EvidenceModality.METADATA,
                video_id,
                None,
                None,
                None,
                _safe_metadata_payload(metadata),
                "metadata_documents",
                _version(retrieval_response, "metadata_version"),
            ))
            counts[EvidenceModality.METADATA] += 1
            availability[EvidenceModality.METADATA] = AvailabilityStatus.AVAILABLE
        elif _matched_metadata(candidate) and counts[EvidenceModality.METADATA] < max_evidence_per_modality:
            refs.append(_ref(
                EvidenceModality.METADATA,
                video_id,
                None,
                None,
                None,
                {"matched_fields": _matched_metadata(candidate)},
                "structured_retrieval",
                _version(retrieval_response, "metadata_version"),
            ))
            counts[EvidenceModality.METADATA] += 1

    context = _retrieval_context(retrieval_response)
    return EvidencePack(
        request=request,
        retrieval_context=context,
        candidates=tuple(candidates),
        evidence_refs=tuple(refs),
        modality_availability=availability,
    )


def _candidate_rows(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = response.get("video_results") or response.get("video_groups") or response.get("results") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _candidate_frames(candidate: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    frames = candidate.get("frames")
    if isinstance(frames, list) and frames:
        for frame in frames:
            if isinstance(frame, Mapping):
                yield {**candidate, **dict(frame)}
        return
    yield dict(candidate)


def _frame_evidence(
    frame: Mapping[str, Any],
    video_id: str,
    response: Mapping[str, Any],
    availability: dict[EvidenceModality, AvailabilityStatus],
    counts: dict[EvidenceModality, int],
    limit: int,
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    keyframe_id = _optional_int(frame.get("keyframe_id", frame.get("best_keyframe_id")))
    frame_idx = _optional_int(frame.get("frame_idx", frame.get("best_frame_idx")))
    timestamp = _optional_float(frame.get("pts_time", frame.get("best_pts_time")))
    if keyframe_id is not None and counts[EvidenceModality.KEYFRAME] < limit:
        refs.append(_ref(
            EvidenceModality.KEYFRAME,
            video_id,
            keyframe_id,
            frame_idx,
            timestamp,
            {"keyframe_path": frame.get("keyframe_path", frame.get("best_keyframe_path"))},
            "retrieval_frame",
            _version(response, "index_schema_version"),
        ))
        counts[EvidenceModality.KEYFRAME] += 1
    if (frame.get("clip_score") is not None or frame.get("clip_rank") is not None) and counts[EvidenceModality.CLIP] < limit:
        refs.append(_ref(
            EvidenceModality.CLIP, video_id, keyframe_id, frame_idx, timestamp,
            {"rank": frame.get("clip_rank"), "score": frame.get("clip_score")},
            "numpy_index", _version(response, "index_schema_version"),
        ))
        counts[EvidenceModality.CLIP] += 1

    evidence = frame.get("evidence") if isinstance(frame.get("evidence"), Mapping) else {}
    refs.extend(_modality_matches(EvidenceModality.OBJECT, evidence.get("object", []), video_id, keyframe_id, frame_idx, timestamp, "object_store", _version(response, "object_store_version"), availability, counts, limit))
    refs.extend(_modality_matches(EvidenceModality.ATTRIBUTE, evidence.get("attribute", []), video_id, keyframe_id, frame_idx, timestamp, "attribute_service", _version(response, "attribute_version"), availability, counts, limit))
    refs.extend(_modality_matches(EvidenceModality.OCR, evidence.get("ocr", []), video_id, keyframe_id, frame_idx, timestamp, "phase5_store", _version(response, "phase5_store_version"), availability, counts, limit))
    refs.extend(_modality_matches(EvidenceModality.ASR, evidence.get("asr", []), video_id, keyframe_id, frame_idx, timestamp, "phase5_store", _version(response, "phase5_store_version"), availability, counts, limit))
    refs.extend(_hybrid_retriever_evidence(frame, video_id, keyframe_id, frame_idx, timestamp, availability, counts, limit))
    return refs


def _hybrid_retriever_evidence(
    frame: Mapping[str, Any],
    video_id: str,
    keyframe_id: int | None,
    frame_idx: int | None,
    timestamp: float | None,
    availability: dict[EvidenceModality, AvailabilityStatus],
    counts: dict[EvidenceModality, int],
    limit: int,
) -> list[EvidenceRef]:
    provenance = frame.get("provenance")
    channel_evidence = provenance.get("evidence") if isinstance(provenance, Mapping) else None
    if not isinstance(channel_evidence, Mapping):
        return []
    modality_by_source = {
        "clip": EvidenceModality.CLIP,
        "asr": EvidenceModality.ASR,
        "ocr": EvidenceModality.OCR,
        "object": EvidenceModality.OBJECT,
        "metadata": EvidenceModality.METADATA,
    }
    refs: list[EvidenceRef] = []
    for retriever_name, value in channel_evidence.items():
        if not isinstance(value, Mapping):
            continue
        source_type = str(value.get("source_type", ""))
        modality = modality_by_source.get(source_type)
        if modality is None or counts[modality] >= limit:
            continue
        payload = {
            "retriever": str(retriever_name),
            "rank": value.get("rank"),
            "score": value.get("raw_score"),
            "matched_text": value.get("matched_text"),
            "document_id": value.get("document_id"),
            "provenance": value.get("provenance", {}),
        }
        refs.append(
            _ref(
                modality,
                video_id,
                _optional_int(value.get("keyframe_id")) if value.get("keyframe_id") is not None else keyframe_id,
                _optional_int(value.get("frame_idx")) if value.get("frame_idx") is not None else frame_idx,
                _optional_float(value.get("pts_time")) if value.get("pts_time") is not None else timestamp,
                payload,
                f"hybrid_{retriever_name}",
                str(provenance.get("fusion_version") or "generic-video-rrf-v1"),
            )
        )
        counts[modality] += 1
        availability[modality] = AvailabilityStatus.AVAILABLE
    return refs


def _modality_matches(
    modality: EvidenceModality,
    values: Any,
    video_id: str,
    keyframe_id: int | None,
    frame_idx: int | None,
    timestamp: float | None,
    source: str,
    source_version: str | None,
    availability: dict[EvidenceModality, AvailabilityStatus],
    counts: dict[EvidenceModality, int],
    limit: int,
) -> list[EvidenceRef]:
    if not isinstance(values, list):
        return []
    refs: list[EvidenceRef] = []
    for value in values:
        if not isinstance(value, Mapping) or counts[modality] >= limit:
            continue
        refs.append(_ref(modality, video_id, keyframe_id, frame_idx, timestamp, dict(value), source, source_version))
        counts[modality] += 1
        availability[modality] = AvailabilityStatus.AVAILABLE
    return refs


def _matched_metadata(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, Mapping):
        return []
    metadata = evidence.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    fields = metadata.get("matched_fields", metadata.get("evidence", []))
    return [dict(item) for item in fields if isinstance(item, Mapping)] if isinstance(fields, list) else []


def _safe_metadata_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("title", "author", "channel_id", "publish_date", "duration", "keywords", "description", "watch_url")
    return {field: metadata[field] for field in fields if metadata.get(field) not in (None, "", [])}


def _retrieval_context(response: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "mode", "query", "groups", "top_k", "candidate_pool", "candidate_pool_size",
        "structured_query", "query_plan", "reranker", "fusion_config", "aggregation_method",
        "index_schema_version", "index_fingerprint", "phase5_store_version", "phase6_config_version",
        "qa_event_query", "qa_retrieval_query",
        "qa_hybrid_retrieval", "profile", "health", "failures", "channel_hit_counts", "latency_ms",
    )
    return {key: response[key] for key in keys if key in response}


def _version(response: Mapping[str, Any], key: str) -> str | None:
    value = response.get(key)
    return str(value) if value not in (None, "") else None


def _ref(
    modality: EvidenceModality,
    video_id: str,
    keyframe_id: int | None,
    frame_idx: int | None,
    timestamp: float | None,
    payload: dict[str, Any],
    source: str,
    source_version: str | None,
) -> EvidenceRef:
    identity = json.dumps(
        {"modality": modality.value, "video_id": video_id, "keyframe_id": keyframe_id, "frame_idx": frame_idx, "timestamp": timestamp, "payload": payload, "source": source, "source_version": source_version},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return EvidenceRef(
        evidence_id=f"{modality.value}:{video_id}:{digest}",
        modality=modality,
        video_id=video_id,
        keyframe_id=keyframe_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        payload=payload,
        source=source,
        source_version=source_version,
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
