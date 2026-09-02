from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from aic_retrieval.hybrid_query_planner import HybridQueryPlan
from aic_retrieval.retrievers import Retriever, RetrievalHit, RetrievalRequest
from aic_retrieval.rrf_fusion import RrfFusionConfig, fuse_video_hits
from aic_retrieval.text_hit_suppression import TextHitSuppressionConfig, suppress_text_hit_noise


class HybridRetrievalEngine:
    def __init__(
        self,
        factories: Mapping[str, Callable[[], Retriever]],
        config: RrfFusionConfig = RrfFusionConfig(),
    ) -> None:
        self.factories = dict(factories)
        self.config = config
        self.instances: dict[str, Retriever] = {}

    def _retriever(self, name: str) -> Retriever:
        if name not in self.factories:
            raise KeyError(f"retriever is not configured: {name}")
        if name not in self.instances:
            self.instances[name] = self.factories[name]()
        return self.instances[name]

    def search_channel(self, name: str, request: RetrievalRequest) -> list[RetrievalHit]:
        """Run one configured channel without applying video-level fusion."""
        hits, _ = self._search_channel(name, request)
        return hits

    def _search_channel(self, name: str, request: RetrievalRequest) -> tuple[list[RetrievalHit], dict[str, Any]]:
        retriever = self._retriever(name)
        if name not in {"bge", "bm25"}:
            return retriever.search(request), {}
        expanded_top_k = min(1000, max(request.top_k, request.top_k * 3))
        expanded_request = RetrievalRequest(
            request.query_id,
            request.query_text,
            groups=request.groups,
            top_k=expanded_top_k,
            filters=request.filters,
            trace_id=request.trace_id,
        )
        raw_hits = retriever.search(expanded_request)
        video_filter = tuple(str(item) for item in request.filters.get("video_ids", ()))
        suppression_config = (
            TextHitSuppressionConfig(max_ocr_hits_per_video=request.top_k)
            if video_filter
            else TextHitSuppressionConfig()
        )
        suppressed = suppress_text_hit_noise(raw_hits, limit=request.top_k, config=suppression_config)
        return list(suppressed.hits), suppressed.stats()

    def _search_clip_queries(
        self,
        retriever: Retriever,
        query_id: str,
        queries: tuple[str, ...],
        *,
        groups: tuple[str, ...],
        top_k: int,
        review_top_k: int,
        source_types: tuple[str, ...],
    ) -> tuple[list[RetrievalHit], dict[str, Any]]:
        requests = [
            RetrievalRequest(
                    f"{query_id}:{channel}",
                    query,
                    groups=groups,
                    top_k=top_k,
                    filters={"source_types": source_types} if source_types else {},
            )
            for index, query in enumerate(queries, 1)
            for channel in (f"clip:q{index}",)
        ]
        search_many = getattr(retriever, "search_many", None)
        result_lists = (
            search_many(requests)
            if callable(search_many)
            else [retriever.search(request) for request in requests]
        )
        if len(result_lists) != len(requests):
            raise ValueError("CLIP multi-query search returned an unexpected result count")
        variants: dict[str, list[RetrievalHit]] = {}
        for index, hits in enumerate(result_lists, 1):
            channel = f"clip:q{index}"
            variants[channel] = list(hits)
        fused = fuse_video_hits(variants, self.config, top_k=top_k)
        fused, rescue_trace = _inject_variant_rescues(
            fused,
            variants,
            review_top_k=review_top_k,
        )
        sequence_by_video = {
            candidate["video_id"]: candidate
            for candidate in rescue_trace.get("temporal_sequence", {}).get("candidates", ())
        }
        output = [
            RetrievalHit(
                retriever="clip",
                rank=rank,
                raw_score=hit.raw_score,
                video_id=hit.video_id,
                source_type="clip_multi_query",
                document_id=hit.document_id,
                keyframe_id=hit.keyframe_id,
                frame_idx=hit.frame_idx,
                pts_time=hit.pts_time,
                matched_text=None,
                provenance={
                    "multi_query": True,
                    "queries": list(queries),
                    "variant_fusion": dict(hit.provenance),
                    "temporal_sequence": sequence_by_video.get(hit.video_id),
                },
            )
            for rank, hit in enumerate(fused, 1)
        ]
        return output, {
            "mode": "multi_query_rrf_with_variant_rescue",
            "queries": list(queries),
            "variant_hit_counts": {name: len(hits) for name, hits in variants.items()},
            "variant_rescue": rescue_trace,
        }

    def search(
        self,
        query_id: str,
        plan: HybridQueryPlan,
        *,
        groups: tuple[str, ...] = ("L21",),
        top_k: int = 12,
        strict: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        channels = {}
        failures: dict[str, str] = {}
        timings: dict[str, float] = {}
        health = {}
        postprocessing: dict[str, dict[str, Any]] = {}
        pool = min(1000, max(top_k, self.config.candidate_pool))
        for name in plan.enabled_retrievers:
            channel_started = time.perf_counter()
            try:
                retriever = self._retriever(name)
                health[name] = retriever.health().to_dict()
                source_types = tuple(plan.source_filters.get(name, ()))
                if name == "clip" and len(plan.clip_queries()) > 1:
                    channels[name], postprocessing[name] = self._search_clip_queries(
                        retriever,
                        query_id,
                        plan.clip_queries(),
                        groups=groups,
                        top_k=pool,
                        review_top_k=top_k,
                        source_types=source_types,
                    )
                else:
                    channels[name], postprocessing[name] = self._search_channel(
                        name,
                        RetrievalRequest(
                            query_id,
                            plan.query_for(name),
                            groups=groups,
                            top_k=pool,
                            filters={"source_types": source_types} if source_types else {},
                        ),
                    )
            except Exception as exc:
                if strict:
                    raise
                failures[name] = f"{type(exc).__name__}: {exc}"
            timings[name] = round((time.perf_counter() - channel_started) * 1000, 3)
        if len(channels) == 1:
            name, hits = next(iter(channels.items()))
            fused = fuse_video_hits({name: hits}, self.config, top_k=top_k, failures=failures)
        else:
            fused = fuse_video_hits(channels, self.config, top_k=top_k, failures=failures)
        return {
            "schema_version": "hybrid-retrieval-response-v1",
            "query_id": query_id,
            "query_plan": plan.to_dict(),
            "profile": plan.profile,
            "fusion_method": (
                "rrf" if len(channels) > 1 else
                "multi_query_rrf_with_variant_rescue"
                if postprocessing.get("clip", {}).get("mode") == "multi_query_rrf_with_variant_rescue" else
                "single_channel_rank"
            ),
            "health": health,
            "failures": failures,
            "channel_hit_counts": {name: len(hits) for name, hits in channels.items()},
            "channel_postprocessing": postprocessing,
            "latency_ms": {**timings, "total": round((time.perf_counter() - started) * 1000, 3)},
            "results": [hit.to_dict() for hit in fused],
        }


def _inject_variant_rescues(
    fused: list[RetrievalHit],
    variants: Mapping[str, list[RetrievalHit]],
    *,
    review_top_k: int,
    consensus_head: int = 3,
    sequence_rescues: int = 2,
    rescues_per_query: int = 2,
    lane_size: int = 8,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    """Keep RRF leaders, then surface temporal and single-event review candidates."""
    if review_top_k <= 0 or not fused:
        return fused, {
            "enabled": False,
            "reason": "empty review window",
            "lanes": [],
            "temporal_sequence": {"enabled": False, "candidates": []},
        }
    by_video = {hit.video_id: hit for hit in fused}
    lanes: list[dict[str, Any]] = []
    for channel, hits in variants.items():
        video_ids: list[str] = []
        seen: set[str] = set()
        for hit in sorted(hits, key=lambda item: (item.rank, item.video_id, item.document_id or "")):
            if hit.video_id in seen:
                continue
            seen.add(hit.video_id)
            video_ids.append(hit.video_id)
            if len(video_ids) >= lane_size:
                break
        lanes.append({"channel": channel, "video_ids": video_ids})

    head_count = min(consensus_head, review_top_k, len(fused))
    selected: list[RetrievalHit] = list(fused[:head_count])
    selected_ids = {hit.video_id for hit in selected}
    rescued: list[dict[str, Any]] = []
    temporal_sequence = _temporal_sequence_candidates(variants)
    for candidate in temporal_sequence["candidates"][:sequence_rescues]:
        video_id = candidate["video_id"]
        hit = by_video.get(video_id)
        if hit is None or video_id in selected_ids:
            continue
        selected.append(hit)
        selected_ids.add(video_id)
        rescued.append({
            "channel": "clip:sequence",
            "video_id": video_id,
            "lane_rank": len([item for item in rescued if item["channel"] == "clip:sequence"]) + 1,
        })
        if len(selected) >= review_top_k:
            break
    for depth in range(rescues_per_query):
        if len(selected) >= review_top_k:
            break
        for lane in lanes:
            video_ids = lane["video_ids"]
            if depth >= len(video_ids):
                continue
            video_id = video_ids[depth]
            hit = by_video.get(video_id)
            if hit is None or video_id in selected_ids:
                continue
            selected.append(hit)
            selected_ids.add(video_id)
            rescued.append({"channel": lane["channel"], "video_id": video_id, "lane_rank": depth + 1})
            if len(selected) >= review_top_k:
                break
        if len(selected) >= review_top_k:
            break
    selected.extend(hit for hit in fused if hit.video_id not in selected_ids)
    return selected[: len(fused)], {
        "enabled": True,
        "policy": "rrf-head-sequence-then-round-robin-v2",
        "review_top_k": review_top_k,
        "consensus_head": head_count,
        "sequence_rescues": sequence_rescues,
        "rescues_per_query": rescues_per_query,
        "rescued": rescued,
        "lanes": lanes,
        "temporal_sequence": temporal_sequence,
    }


def _temporal_sequence_candidates(
    variants: Mapping[str, list[RetrievalHit]],
    *,
    hits_per_event_per_video: int = 6,
    candidate_limit: int = 16,
) -> dict[str, Any]:
    """Find compact, monotonic frame chains across visual event queries."""
    ordered_channels = sorted(
        variants,
        key=lambda channel: (
            int(channel.rsplit("q", 1)[-1]) if channel.rsplit("q", 1)[-1].isdigit() else 10**9,
            channel,
        ),
    )
    # Q1 is normally the broad scene prompt; later lanes are the concrete events.
    event_channels = ordered_channels[1:] if len(ordered_channels) >= 3 else ordered_channels
    if len(event_channels) < 2:
        return {
            "enabled": False,
            "reason": "fewer than two event channels",
            "event_channels": event_channels,
            "candidates": [],
        }

    by_video: dict[str, dict[int, list[RetrievalHit]]] = {}
    for event_index, channel in enumerate(event_channels, 1):
        per_video: dict[str, int] = {}
        for hit in sorted(variants[channel], key=lambda item: (item.rank, item.video_id, item.document_id or "")):
            if hit.pts_time is None:
                continue
            count = per_video.get(hit.video_id, 0)
            if count >= hits_per_event_per_video:
                continue
            per_video[hit.video_id] = count + 1
            by_video.setdefault(hit.video_id, {}).setdefault(event_index, []).append(hit)

    candidates: list[dict[str, Any]] = []
    event_count = len(event_channels)
    for video_id, event_hits in by_video.items():
        states: list[dict[str, Any]] = []
        for event_index in sorted(event_hits):
            channel = event_channels[event_index - 1]
            for hit in sorted(event_hits[event_index], key=lambda item: (float(item.pts_time or 0), item.rank)):
                pts_time = float(hit.pts_time or 0)
                frame = {
                    "channel": channel,
                    "event_index": event_index,
                    "rank": hit.rank,
                    "raw_score": hit.raw_score,
                    "keyframe_id": hit.keyframe_id,
                    "frame_idx": hit.frame_idx,
                    "pts_time": pts_time,
                }
                best = {
                    "last_event": event_index,
                    "start": pts_time,
                    "end": pts_time,
                    "rank_score": 1.0 / (60.0 + hit.rank),
                    "frames": [frame],
                }
                for previous in states:
                    if previous["last_event"] >= event_index or previous["end"] > pts_time + 0.5:
                        continue
                    proposal = {
                        "last_event": event_index,
                        "start": previous["start"],
                        "end": pts_time,
                        "rank_score": previous["rank_score"] + 1.0 / (60.0 + hit.rank),
                        "frames": [*previous["frames"], frame],
                    }
                    if _sequence_state_key(proposal) > _sequence_state_key(best):
                        best = proposal
                states.append(best)
        eligible = [state for state in states if len(state["frames"]) >= 2]
        if not eligible:
            continue
        best = max(eligible, key=_sequence_state_key)
        matched_events = len(best["frames"])
        span_seconds = max(0.0, best["end"] - best["start"])
        coverage_ratio = matched_events / event_count
        compactness = 1.0 / (1.0 + span_seconds / 60.0)
        score = coverage_ratio + best["rank_score"] + 0.05 * compactness
        candidates.append({
            "video_id": video_id,
            "matched_events": matched_events,
            "event_count": event_count,
            "coverage_ratio": round(coverage_ratio, 6),
            "span_seconds": round(span_seconds, 3),
            "rank_score": round(best["rank_score"], 8),
            "compactness": round(compactness, 6),
            "score": round(score, 8),
            "frames": best["frames"],
        })
    candidates.sort(
        key=lambda item: (
            -item["matched_events"],
            -item["score"],
            item["span_seconds"],
            item["video_id"],
        )
    )
    return {
        "enabled": bool(candidates),
        "policy": "monotonic-compact-event-chain-v1",
        "event_channels": event_channels,
        "hits_per_event_per_video": hits_per_event_per_video,
        "candidates": candidates[:candidate_limit],
    }


def _sequence_state_key(state: Mapping[str, Any]) -> tuple[int, float, float, float]:
    span = float(state["end"]) - float(state["start"])
    return (len(state["frames"]), float(state["rank_score"]), -span, -float(state["end"]))
