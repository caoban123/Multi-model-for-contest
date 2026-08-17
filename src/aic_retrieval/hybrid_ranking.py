from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aic_retrieval.structured_query import StructuredQuery


@dataclass(frozen=True)
class RrfConfig:
    rrf_k: int = 60
    weight_clip: float = 1.0
    weight_object: float = 0.5
    weight_attribute: float = 0.35
    weight_metadata: float = 0.3
    weight_ocr: float = 0.6
    weight_asr: float = 0.55

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if min(self.weight_clip, self.weight_object, self.weight_attribute, self.weight_metadata, self.weight_ocr, self.weight_asr) < 0:
            raise ValueError("RRF weights must not be negative")


def contribution(rank: int | None, weight: float, k: int) -> float:
    return weight / (k + rank) if rank is not None and weight else 0.0


def modality_video_ranks(candidates: list[dict[str, Any]], rank_field: str) -> dict[str, int]:
    best: dict[str, int] = {}
    for item in candidates:
        rank = item.get(rank_field)
        if rank is not None:
            best[item["video_id"]] = min(rank, best.get(item["video_id"], rank))
    ordered = sorted(best, key=lambda video_id: (best[video_id], video_id))
    return {video_id: rank for rank, video_id in enumerate(ordered, start=1)}


def frame_fusion_score(item: dict[str, Any], config: RrfConfig) -> float:
    return (
        contribution(item.get("clip_rank"), config.weight_clip, config.rrf_k)
        + contribution(item.get("object_rank"), config.weight_object, config.rrf_k)
        + contribution(item.get("attribute_rank"), config.weight_attribute, config.rrf_k)
        + contribution(item.get("metadata_rank"), config.weight_metadata, config.rrf_k)
        + contribution(item.get("ocr_rank"), config.weight_ocr, config.rrf_k)
        + contribution(item.get("asr_rank"), config.weight_asr, config.rrf_k)
    )


def choose_representative(frames: list[dict[str, Any]], query: StructuredQuery, config: RrfConfig) -> tuple[dict[str, Any], str]:
    hard_object = query.enable_objects and any(item.filter_mode == "hard" for item in query.object_constraints)
    hard_attribute = query.enable_attributes and any(item.filter_mode == "hard" for item in query.attribute_constraints)
    object_active = query.enable_objects and any(item.filter_mode != "disabled" for item in query.object_constraints)
    attribute_active = query.enable_attributes and query.attribute_mode != "disabled" and bool(query.attribute_constraints)
    metadata_active = query.enable_metadata and query.metadata_mode != "disabled"
    clip_active = query.enable_clip and query.clip_mode != "disabled"
    if hard_attribute:
        satisfying = [item for item in frames if item.get("attribute_rank") is not None]
        with_clip = [item for item in satisfying if item.get("clip_score") is not None]
        pool = with_clip or satisfying
        rule = "highest_clip_among_hard_attribute_matches" if with_clip else "strongest_attribute_among_hard_matches"
        key = lambda item: (-(item.get("clip_score") if with_clip else item.get("attribute_score") or 0), item.get("attribute_rank") or 10**9, item.get("clip_rank") or 10**9, item["keyframe_id"])
    elif hard_object:
        satisfying = [item for item in frames if item.get("object_rank") is not None]
        with_clip = [item for item in satisfying if item.get("clip_score") is not None]
        pool = with_clip or satisfying
        rule = "highest_clip_among_hard_object_matches" if with_clip else "strongest_object_among_hard_matches"
        key = lambda item: (-(item.get("clip_score") if with_clip else item.get("object_score") or 0), item.get("clip_rank") or 10**9, item.get("object_rank") or 10**9, item["keyframe_id"])
    elif clip_active and not object_active and not metadata_active:
        pool, rule = frames, "highest_clip_score_visual_only"
        key = lambda item: (-(item.get("clip_score") if item.get("clip_score") is not None else float("-inf")), item.get("clip_rank") or 10**9, item["keyframe_id"])
    elif object_active and not clip_active and not attribute_active and not metadata_active:
        pool, rule = frames, "strongest_object_evidence_object_only"
        key = lambda item: (-(item.get("object_score") or 0), item.get("object_rank") or 10**9, item["keyframe_id"])
    elif attribute_active and not clip_active and not object_active and not metadata_active:
        pool, rule = frames, "strongest_attribute_evidence_attribute_only"
        key = lambda item: (-(item.get("attribute_score") or 0), item.get("attribute_rank") or 10**9, item["keyframe_id"])
    elif metadata_active and not clip_active and not object_active and not attribute_active:
        pool, rule = frames, "highest_clip_within_metadata_video"
        key = lambda item: (-(item.get("clip_score") if item.get("clip_score") is not None else float("-inf")), item["keyframe_id"])
    else:
        pool, rule = frames, "highest_frame_level_rrf_hybrid"
        key = lambda item: (-frame_fusion_score(item, config), item.get("clip_rank") or 10**9, item.get("object_rank") or 10**9, item.get("attribute_rank") or 10**9, item.get("metadata_rank") or 10**9, item["keyframe_id"])
    if not pool:
        raise ValueError("cannot choose representative without a real frame candidate")
    return sorted(pool, key=key)[0], rule


def rank_video_candidates(candidate_payload: dict[str, Any], query: StructuredQuery, config: RrfConfig = RrfConfig(), top_k: int = 12, matched_frames_per_video: int = 5) -> dict[str, Any]:
    if top_k <= 0 or matched_frames_per_video <= 0:
        raise ValueError("top_k and matched_frames_per_video must be positive")
    candidates = candidate_payload["candidates"]
    clip_ranks = modality_video_ranks(candidates, "clip_rank")
    object_ranks = modality_video_ranks(candidates, "object_rank")
    attribute_ranks = modality_video_ranks(candidates, "attribute_rank")
    metadata_ranks = modality_video_ranks(candidates, "metadata_rank")
    ocr_ranks = modality_video_ranks(candidates, "ocr_rank")
    asr_ranks = modality_video_ranks(candidates, "asr_rank")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        if item["keyframe_id"] is not None:
            grouped.setdefault(item["video_id"], []).append(item)
    unranked = []
    for video_id, frames in grouped.items():
        ranks = {"clip": clip_ranks.get(video_id), "object": object_ranks.get(video_id), "attribute": attribute_ranks.get(video_id), "metadata": metadata_ranks.get(video_id), "ocr":ocr_ranks.get(video_id), "asr":asr_ranks.get(video_id)}
        score = contribution(ranks["clip"], config.weight_clip, config.rrf_k) + contribution(ranks["object"], config.weight_object, config.rrf_k) + contribution(ranks["attribute"], config.weight_attribute, config.rrf_k) + contribution(ranks["metadata"], config.weight_metadata, config.rrf_k) + contribution(ranks["ocr"],config.weight_ocr,config.rrf_k) + contribution(ranks["asr"],config.weight_asr,config.rrf_k)
        representative, rule = choose_representative(frames, query, config)
        ordered_frames = sorted(frames, key=lambda item: (-frame_fusion_score(item, config), item.get("clip_rank") or 10**9, item.get("object_rank") or 10**9, item.get("attribute_rank") or 10**9, item.get("metadata_rank") or 10**9, item["keyframe_id"]))
        selected_frames = [representative, *(item for item in ordered_frames if item is not representative)][:matched_frames_per_video]
        frame_payloads = []
        for frame_rank, frame in enumerate(selected_frames, start=1):
            frame_payloads.append({**frame, "rank": frame_rank, "score": frame_fusion_score(frame, config), "frame_fusion_score": frame_fusion_score(frame, config), "is_representative": frame is representative})
        unranked.append({
            "video_id": video_id, "group": representative["group"], "fusion_score": score, "video_score": score,
            "best_score": frame_fusion_score(representative, config), "best_keyframe_id": representative["keyframe_id"],
            "best_frame_idx": representative["frame_idx"], "best_pts_time": representative["pts_time"], "best_keyframe_path": representative["keyframe_path"],
            "frame_count": len(frames), "matched_frame_count": len(frame_payloads), "aggregation_method": "rrf",
            "modality_ranks": ranks, "representative_rule": rule, "frames": frame_payloads,
        })
    unranked.sort(key=lambda item: (-item["fusion_score"], item["modality_ranks"]["clip"] or 10**9, item["modality_ranks"]["object"] or 10**9, item["modality_ranks"]["attribute"] or 10**9, item["modality_ranks"]["metadata"] or 10**9, item["video_id"]))
    results = []
    for rank, item in enumerate(unranked[:top_k], start=1): results.append({"rank": rank, **item})
    ordered_raw = sorted(candidates, key=lambda item: (-frame_fusion_score(item, config), item.get("clip_rank") or 10**9, item.get("object_rank") or 10**9, item.get("attribute_rank") or 10**9, item.get("metadata_rank") or 10**9, item["video_id"], item["keyframe_id"] if item["keyframe_id"] is not None else 10**9))
    raw_results = [{**item, "rank": rank, "score": frame_fusion_score(item, config), "frame_fusion_score": frame_fusion_score(item, config)} for rank, item in enumerate(ordered_raw, start=1) if item["keyframe_id"] is not None]
    representative_results = []
    for item in results:
        representative = next(frame for frame in item["frames"] if frame["is_representative"])
        representative_results.append({**representative, "rank": item["rank"], "score": item["fusion_score"], "fusion_score": item["fusion_score"], "representative_rule": item["representative_rule"]})
    return {"fusion_method":"rrf", "fusion_config":asdict(config), "top_k":top_k, "matched_frames_per_video":matched_frames_per_video,
            "raw_candidates":candidates, "raw_results":raw_results, "results":representative_results, "video_results":results, "video_groups":results}
