from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from aic_retrieval.color_attributes import ColorAttributeService
from aic_retrieval.metadata_search import MetadataDocument, filter_metadata_documents, has_metadata_constraints
from aic_retrieval.object_search import ObjectPredicate, ObjectSearchConfig, ObjectSearchService
from aic_retrieval.search import FrameRef, search_numpy_index
from aic_retrieval.structured_query import StructuredQuery


def _empty_candidate(video_id: str, keyframe_id: int | None, ref: FrameRef | None = None) -> dict[str, Any]:
    return {
        "video_id": video_id, "keyframe_id": keyframe_id,
        "group": ref.group if ref else video_id.split("_", 1)[0],
        "frame_idx": ref.frame_idx if ref else None, "pts_time": ref.pts_time if ref else None,
        "fps": ref.fps if ref else None, "keyframe_path": ref.keyframe_path if ref else None,
        "clip_rank": None, "clip_score": None, "object_rank": None, "object_score": None,
        "attribute_rank": None, "attribute_score": None,
        "metadata_rank": None, "metadata_score": None,
        "provenance": [], "evidence": {"clip": None, "object": [], "attribute": [], "metadata": None},
    }


class StructuredCandidateGenerator:
    def __init__(
        self,
        index: np.ndarray,
        refs: list[FrameRef],
        object_service: ObjectSearchService | None,
        metadata_docs: list[MetadataDocument],
        attribute_service: ColorAttributeService | None = None,
    ) -> None:
        self.index, self.refs = index, refs
        self.object_service, self.metadata_docs = object_service, metadata_docs
        self.attribute_service = attribute_service
        self.ref_by_key = {(ref.video_id, ref.keyframe_id): ref for ref in refs}
        self.positions_by_video: dict[str, list[int]] = {}
        for position, ref in enumerate(refs): self.positions_by_video.setdefault(ref.video_id, []).append(position)

    def best_clip_frame_for_video(self, video_id: str, query_vector: np.ndarray) -> dict[str, Any] | None:
        positions = self.positions_by_video.get(video_id, [])
        if not positions:
            return None
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self.index[positions] @ vector
        local = sorted(range(len(positions)), key=lambda idx: (-float(scores[idx]), self.refs[positions[idx]].keyframe_id, self.refs[positions[idx]].frame_idx))[0]
        ref = self.refs[positions[local]]
        return {"ref": ref, "score": float(scores[local])}

    def generate(self, query: StructuredQuery, query_vector: np.ndarray | None) -> dict[str, Any]:
        candidates: dict[tuple[str, int], dict[str, Any]] = {}
        channel_counts = {"clip_frames": 0, "object_frames": 0, "attribute_frames": 0, "metadata_videos": 0}
        hard_object_sets: list[set[tuple[str, int]]] = []
        hard_attribute_sets: list[set[tuple[str, int]]] = []
        hard_clip_keys: set[tuple[str, int]] | None = None
        unknown_object_keys: set[tuple[str, int]] = set()
        unknown_attribute_keys: set[tuple[str, int]] = set()

        if query.enable_clip and query.clip_mode != "disabled":
            if query_vector is None or not query.visual_text.strip():
                raise ValueError("enabled CLIP requires visual_text and query_vector")
            clip_results = search_numpy_index(self.index, self.refs, query_vector, top_k=query.clip_candidate_pool)
            channel_counts["clip_frames"] = len(clip_results)
            if query.clip_mode == "hard":
                hard_clip_keys = {(result.video_id, result.keyframe_id) for result in clip_results}
            for result in clip_results:
                key = (result.video_id, result.keyframe_id)
                item = candidates.setdefault(key, _empty_candidate(result.video_id, result.keyframe_id, self.ref_by_key[key]))
                item.update({"clip_rank": result.rank, "clip_score": result.score})
                item["provenance"].append("clip")
                item["evidence"]["clip"] = {"rank": result.rank, "score": result.score}

        if query.enable_objects:
            if self.object_service is None:
                raise ValueError("object modality enabled but object store is not configured")
            for constraint_index, constraint in enumerate(query.object_constraints):
                if constraint.filter_mode == "disabled": continue
                payload = self.object_service.search(
                    ObjectPredicate(constraint.labels, constraint.count_operator, constraint.count, constraint.horizontal, constraint.vertical),
                    ObjectSearchConfig(constraint.min_confidence, constraint.nms_iou_threshold),
                )
                matched_keys = {(item["video_id"], item["keyframe_id"]) for item in payload["results"]}
                unknown_object_keys.update((item["video_id"], item["keyframe_id"]) for item in payload["unknown_frames"])
                if constraint.filter_mode == "hard": hard_object_sets.append(matched_keys)
                for result in payload["results"]:
                    key = (result["video_id"], result["keyframe_id"])
                    item = candidates.setdefault(key, _empty_candidate(result["video_id"], result["keyframe_id"], self.ref_by_key.get(key)))
                    rank = result["object_rank"]
                    if item["object_rank"] is None or rank < item["object_rank"]:
                        item.update({"object_rank": rank, "object_score": result["object_score"]})
                    if "object" not in item["provenance"]: item["provenance"].append("object")
                    item["evidence"]["object"].append({"constraint_index": constraint_index, **result})
            channel_counts["object_frames"] = sum("object" in item["provenance"] for item in candidates.values())

        if query.enable_attributes and query.attribute_mode != "disabled":
            if self.attribute_service is None:
                raise ValueError("attribute modality enabled but attribute service is not configured")
            for constraint_index, constraint in enumerate(query.attribute_constraints):
                if constraint.filter_mode == "disabled":
                    continue
                candidate_keys = {(video_id, keyframe_id) for video_id, keyframe_id in candidates.keys() if keyframe_id != -1}
                payload = self.attribute_service.search(constraint, candidate_keys or None)
                matched_keys = {(item["video_id"], item["keyframe_id"]) for item in payload["results"]}
                unknown_attribute_keys.update(
                    (ref.video_id, ref.keyframe_id)
                    for ref in self.refs
                    if ref.keyframe_path is None or not (self.attribute_service.repo_root / ref.keyframe_path).is_file()
                )
                if constraint.filter_mode == "hard":
                    hard_attribute_sets.append(matched_keys)
                for result in payload["results"]:
                    key = (result["video_id"], result["keyframe_id"])
                    item = candidates.setdefault(key, _empty_candidate(result["video_id"], result["keyframe_id"], self.ref_by_key.get(key)))
                    rank = result["attribute_rank"]
                    if item["attribute_rank"] is None or rank < item["attribute_rank"]:
                        item.update({"attribute_rank": rank, "attribute_score": result["attribute_score"]})
                    if "attribute" not in item["provenance"]:
                        item["provenance"].append("attribute")
                    item["evidence"]["attribute"].append({"constraint_index": constraint_index, **result})
            channel_counts["attribute_frames"] = sum("attribute" in item["provenance"] for item in candidates.values())

        metadata_results = []
        if query.enable_metadata and query.metadata_mode != "disabled" and has_metadata_constraints(query.metadata_constraints):
            metadata_results = filter_metadata_documents(self.metadata_docs, query.metadata_constraints)
            channel_counts["metadata_videos"] = len(metadata_results)
            for metadata in metadata_results:
                video_candidates = [item for item in candidates.values() if item["video_id"] == metadata["video_id"]]
                representative = min(video_candidates, key=lambda item: (item["clip_rank"] is None, item["clip_rank"] or 10**9, item["object_rank"] is None, item["object_rank"] or 10**9, item["keyframe_id"] or 10**9)) if video_candidates else None
                if representative is None and query_vector is not None:
                    best = self.best_clip_frame_for_video(metadata["video_id"], query_vector)
                    if best:
                        ref = best["ref"]; key = (ref.video_id, ref.keyframe_id)
                        representative = candidates.setdefault(key, _empty_candidate(ref.video_id, ref.keyframe_id, ref))
                        representative["clip_score"] = best["score"]
                        representative["provenance"].append("clip_representative_only")
                        representative["evidence"]["clip"] = {"rank": None, "score": best["score"], "representative_only": True}
                if representative is None:
                    # Preserve a genuine video candidate without inventing frame or score.
                    representative = _empty_candidate(metadata["video_id"], None, None)
                    candidates[(metadata["video_id"], -1)] = representative
                representative["metadata_rank"] = metadata["metadata_rank"]
                representative["metadata_score"] = None
                representative["provenance"].append("metadata")
                representative["evidence"]["metadata"] = metadata

        hard_metadata_videos = {item["video_id"] for item in metadata_results} if query.enable_metadata and query.metadata_mode == "hard" else None
        if hard_clip_keys is not None:
            candidates = {key: value for key, value in candidates.items() if key in hard_clip_keys}
        for matched in hard_object_sets:
            candidates = {key: value for key, value in candidates.items() if key in matched}
        for matched in hard_attribute_sets:
            candidates = {key: value for key, value in candidates.items() if key in matched}
        if hard_metadata_videos is not None:
            candidates = {key: value for key, value in candidates.items() if value["video_id"] in hard_metadata_videos}
        ordered = sorted(candidates.values(), key=lambda item: (item["clip_rank"] is None, item["clip_rank"] or 10**9, item["object_rank"] is None, item["object_rank"] or 10**9, item["attribute_rank"] is None, item["attribute_rank"] or 10**9, item["metadata_rank"] is None, item["metadata_rank"] or 10**9, item["video_id"], item["keyframe_id"] if item["keyframe_id"] is not None else 10**9))
        return {"query": asdict(query), "channel_counts": channel_counts, "candidate_count": len(ordered),
                "unknown_object_frame_count": len(unknown_object_keys), "unknown_attribute_frame_count": len(unknown_attribute_keys), "candidates": ordered,
                "score_contract": "Scores remain modality-specific; missing scores are null and metadata_score is never fabricated."}
