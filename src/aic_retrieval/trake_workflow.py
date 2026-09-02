"""Local-only orchestration for TRAKE planning, retrieval and alignment."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from aic_retrieval.trake_alignment import AlignmentResult, CandidateWindow, ChainEvent, DanteInspiredAligner, TemporalAligner, TemporalFeasibilityResult, TrakeChain, TrakeV2FoundationAligner, build_candidate_windows, fast_temporal_feasibility, validate_chain
from aic_retrieval.trake_candidates import EventCandidatePool, TrakeCandidate, TrakeRetrievalAdapter, VideoCandidate, group_candidates_by_video, select_anchor_event, with_distinctiveness
from aic_retrieval.trake_config import TrakeRuntimeConfig
from aic_retrieval.trake_decomposition import DecompositionResult, RuleBasedTrakeDecomposer, TrakePlannerV2
from aic_retrieval.trake_schema import Availability, TrakeEvent, TrakeRequest, request_from_dict
from aic_retrieval.trake_refinement import DenseWindowExpansion, dense_expansion_from_dict
from aic_retrieval.trake_scoring import ScoreBreakdown, linkage_gate_reason, rerank_alignments, score_chain
from aic_retrieval.trake_store import TrakeStore


LEGACY_WORKFLOW_VERSION = "phase8-workflow-v1"
V2_WORKFLOW_VERSION = "trake-workflow-v2"


class TrakeDomainError(ValueError):
    def __init__(self, code: str, message: str, *, stage: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.stage = stage
        self.user_message = message


@dataclass
class TrakeWorkflowState:
    session_id: str
    request: TrakeRequest
    decomposition_warnings: tuple[str, ...]
    created_at_epoch: float
    pools: tuple[EventCandidatePool, ...] = ()
    videos: tuple[VideoCandidate, ...] = ()
    alignments: tuple[AlignmentResult, ...] = ()
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    plan_revision: int = 1
    locked_event_ids: tuple[str, ...] = ()
    manual_chain: TrakeChain | None = None
    algorithm: str = "legacy"
    algorithm_version: str = "phase8-temporal-dp-v1"
    config_fingerprint: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    temporal_feasibility: tuple[TemporalFeasibilityResult, ...] = ()
    candidate_windows: tuple[CandidateWindow, ...] = ()
    dense_expansions: tuple[DenseWindowExpansion, ...] = ()
    # TRAKE has a two-stage contract: retrieval selects one video, then
    # alignment emits semantic keyframes within that same video only.
    retrieved_video_id: str | None = None
    trake_answer: dict[str, Any] | None = None


class TrakeWorkflow:
    def __init__(
        self,
        event_retriever: Callable[[TrakeEvent, int], list[dict[str, Any]]],
        *,
        modality_availability: dict[str, Availability] | None = None,
        store: TrakeStore | None = None,
        runtime_config: TrakeRuntimeConfig | None = None,
        session_provenance: dict[str, Any] | None = None,
        vlm_verifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        request_retriever: Callable[[TrakeRequest, int], dict[str, list[dict[str, Any]]]] | None = None,
        window_expander: Any | None = None,
    ) -> None:
        self.runtime_config = runtime_config or TrakeRuntimeConfig.legacy()
        self.decomposer = (
            TrakePlannerV2(self.runtime_config.temporal.sequence_default_max_gap_seconds)
            if self.runtime_config.features.planner_v2
            else RuleBasedTrakeDecomposer()
        )
        self.adapter = TrakeRetrievalAdapter(event_retriever)
        self.modality_availability = dict(modality_availability or {"clip": Availability.AVAILABLE})
        if self.runtime_config.algorithm == "legacy":
            self.aligner = TemporalAligner(self.runtime_config.legacy_scoring)
        elif self.runtime_config.temporal.alignment == "dante" and self.runtime_config.features.dante_coarse:
            self.aligner = DanteInspiredAligner(
                self.runtime_config.legacy_scoring,
                self.runtime_config.dante,
                self.modality_availability,
                self.runtime_config.reranking,
            )
        else:
            self.aligner = TrakeV2FoundationAligner(self.runtime_config.legacy_scoring)
        self.sessions: dict[str, TrakeWorkflowState] = {}
        self.store = store
        self.session_provenance = dict(session_provenance or {})
        self.vlm_verifier = vlm_verifier
        self.request_retriever = request_retriever
        self.window_expander = window_expander

    def plan(
        self,
        query_id: str,
        query: str,
        *,
        manual_events: list[str] | None = None,
        constraints: dict[str, Any] | None = None,
        group: str = "L21",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        effective_constraints = dict(constraints or {})
        effective_constraints.setdefault("allow_same_frame", self.runtime_config.temporal.allow_same_frame)
        result = (
            self.decomposer.manual_override(query_id, query, manual_events, group=group, constraints=effective_constraints)
            if manual_events is not None
            else self.decomposer.decompose(query_id, query, group=group)
        )
        request = result.request
        if manual_events is None:
            payload = request.to_dict(); payload["constraints"] = {**request.constraints, **effective_constraints}
            payload["source_versions"] = {
                **(payload.get("source_versions") or {}),
                "algorithm": self.aligner.alignment_version,
                "config_fingerprint": self.runtime_config.fingerprint,
            }
            request = request_from_dict(payload)
        session_id = f"trake-{uuid4().hex}"
        provenance = {
            "config_fingerprint": self.runtime_config.fingerprint,
            "algorithm_version": self.aligner.alignment_version,
            "index_fingerprint": None,
            "clip_model_fingerprint": None,
            "siglip_model_fingerprint": None,
            "ocr_store_version": None,
            "asr_store_version": None,
            "object_store_version": None,
            "mapping_version": None,
            **self.session_provenance,
        }
        state = TrakeWorkflowState(
            session_id,
            request,
            result.warnings,
            time.time(),
            algorithm=self.runtime_config.algorithm,
            algorithm_version=self.aligner.alignment_version,
            config_fingerprint=self.runtime_config.fingerprint,
            provenance=provenance,
            diagnostics={
                "planner": {
                    "planner_version": request.planner_version or result.version,
                    "warnings": list(result.warnings),
                    "global_context": request.global_context,
                    "events": [
                        {
                            "event_id": event.event_id,
                            "raw_text": event.raw_text,
                            "visual_query": event.visual_query,
                            "query_variants": list(event.query_variants),
                            "modalities": list(event.modalities),
                            "planner_confidence": event.planner_confidence,
                            "required": event.required,
                        }
                        for event in request.events
                    ],
                }
            },
        )
        plan_timing_key = "planner" if self.runtime_config.algorithm == "trake_v2" else "plan"
        state.stage_timings_ms[plan_timing_key] = round((time.perf_counter() - started) * 1000, 3)
        self.sessions[session_id] = state
        return self._persist(state)

    def search(self, session_id: str, *, event_pool_size: int | None = None, max_per_video: int | None = None, video_pool_size: int | None = None) -> dict[str, Any]:
        state = self._state(session_id)
        explicit_pool_size = event_pool_size is not None
        event_pool_size = self.runtime_config.retrieval.topk_default if event_pool_size is None else event_pool_size
        max_per_video = self.runtime_config.retrieval.max_candidates_per_video_event if max_per_video is None else max_per_video
        video_pool_size = self.runtime_config.temporal.max_candidate_videos if video_pool_size is None else video_pool_size
        if event_pool_size > self.runtime_config.retrieval.topk_max:
            raise ValueError("event_pool_size exceeds configured retrieval.topk_max")
        started = time.perf_counter()
        raw_by_event: dict[str, list[dict[str, Any]]] | None = None
        request_stage_timings: dict[str, float] = {}
        request_query_plans: dict[str, dict[str, Any]] = {}
        request_query_traces: dict[str, dict[str, Any]] = {}
        hybrid_failures: dict[str, dict[str, str]] = {}
        skipped_without_frame: dict[str, dict[str, int]] = {}
        pool_sizes = {event.event_id: event_pool_size for event in state.request.events}
        if self.request_retriever is not None and self.runtime_config.algorithm == "trake_v2":
            retrieval_limit = self.runtime_config.retrieval.topk_max if self.runtime_config.retrieval.adaptive_topk and self.runtime_config.features.adaptive_topk and not explicit_pool_size else event_pool_size
            raw_by_event = self.request_retriever(state.request, retrieval_limit)
            request_stage_timings = dict(getattr(raw_by_event, "stage_timings_ms", {}))
            request_query_plans = dict(getattr(raw_by_event, "query_plans", {}))
            request_query_traces = dict(getattr(raw_by_event, "query_traces", {}))
            hybrid_failures = dict(getattr(raw_by_event, "hybrid_failures", {}))
            skipped_without_frame = dict(getattr(raw_by_event, "skipped_without_frame", {}))
            if self.runtime_config.retrieval.adaptive_topk and self.runtime_config.features.adaptive_topk and not explicit_pool_size:
                pool_sizes = {
                    event.event_id: self._adaptive_pool_size(event, raw_by_event.get(event.event_id, []))
                    for event in state.request.events
                }
        state.pools = self.adapter.retrieve_request(
            state.request,
            pool_size=event_pool_size,
            max_per_video=max_per_video,
            availability=self.modality_availability,
            pool_sizes=pool_sizes,
            raw_by_event=raw_by_event,
        )
        if self.runtime_config.algorithm == "trake_v2":
            state.pools = tuple(with_distinctiveness(pool, self.runtime_config.distinctiveness) for pool in state.pools)
        anchor_event_id, anchor_diagnostics = select_anchor_event(state.request, state.pools) if self.runtime_config.algorithm == "trake_v2" and self.runtime_config.features.anchor else (None, {"reason": "FEATURE_DISABLED", "scores": {}})
        state.videos = group_candidates_by_video(
            state.request,
            state.pools,
            video_pool_size=video_pool_size,
            anchor_event_id=anchor_event_id,
        )
        seed_gate_diagnostics: dict[str, Any] = {}
        if self.runtime_config.algorithm == "trake_v2":
            state.videos, seed_gate_diagnostics = _gate_alignment_seeds(
                state.request,
                state.pools,
                state.videos,
                self.runtime_config.temporal.min_alignment_seed_relative_score,
            )
        state.alignments = ()
        state.retrieved_video_id = None
        state.trake_answer = None
        timing_key = "retrieval" if self.runtime_config.algorithm == "trake_v2" else "search"
        search_elapsed_ms = (time.perf_counter() - started) * 1000
        if self.runtime_config.algorithm == "trake_v2" and request_stage_timings:
            state.stage_timings_ms["embedding"] = float(request_stage_timings.get("embedding", 0.0))
            state.stage_timings_ms["fusion"] = float(request_stage_timings.get("fusion", 0.0))
            state.stage_timings_ms[timing_key] = round(max(0.0, search_elapsed_ms - state.stage_timings_ms["embedding"] - state.stage_timings_ms["fusion"]), 3)
        else:
            state.stage_timings_ms[timing_key] = round(search_elapsed_ms, 3)
        state.diagnostics["retrieval"] = {
            "hybrid_requested": bool(state.request.constraints.get("hybrid_retrieval")),
            "hybrid_query_plans": request_query_plans,
            "hybrid_query_traces": request_query_traces,
            "hybrid_failures": hybrid_failures,
            "hybrid_skipped_without_frame": skipped_without_frame,
            "adaptive_topk": self.runtime_config.retrieval.adaptive_topk and self.runtime_config.features.adaptive_topk and not explicit_pool_size,
            "topk_by_event": pool_sizes,
            "anchor_event_id": anchor_event_id,
            "anchor": anchor_diagnostics,
            "events": [
                {
                    "event_id": pool.event.event_id,
                    "topk_used": pool.topk_used,
                    "selected_modalities": list(pool.event.modalities),
                    "top_score": pool.candidates[0].local_score if pool.candidates else None,
                    "score_margin": (
                        pool.candidates[0].local_score - pool.candidates[1].local_score
                        if len(pool.candidates) > 1
                        else None
                    ),
                    "distinctiveness": pool.distinctiveness,
                    "distinctiveness_components": dict(pool.distinctiveness_components),
                    "warnings": list(pool.warnings),
                }
                for pool in state.pools
            ],
            "candidate_videos": [
                {
                    "video_id": video.video_id,
                    "required_coverage": video.required_coverage,
                    "required_total": video.required_total,
                    "complete": video.complete,
                    "warnings": list(video.warnings),
                }
                for video in state.videos
            ],
            "semantic_seed_gate": seed_gate_diagnostics,
        }
        state.temporal_feasibility = ()
        state.candidate_windows = ()
        state.dense_expansions = ()
        if self.runtime_config.algorithm == "trake_v2" and self.runtime_config.temporal.fast_feasibility_filter and self.runtime_config.features.temporal_feasibility:
            filter_started = time.perf_counter()
            state.temporal_feasibility = tuple(fast_temporal_feasibility(state.request, video) for video in state.videos)
            state.stage_timings_ms["temporal_filter"] = round((time.perf_counter() - filter_started) * 1000, 3)
            feasible_ids = [item.video_id for item in state.temporal_feasibility if item.feasible]
            state.diagnostics["temporal_feasibility"] = {
                "before_video_ids": [video.video_id for video in state.videos],
                "after_video_ids": feasible_ids,
                "before_count": len(state.videos),
                "after_count": len(feasible_ids),
                "removed": [
                    {"video_id": item.video_id, "failure_stage": item.failure_stage, "failure_reason": item.failure_reason}
                    for item in state.temporal_feasibility
                    if not item.feasible
                ],
            }
            state.retrieved_video_id = _select_retrieved_video(state.videos, state.temporal_feasibility)
            state.diagnostics["retrieval"]["selected_video_id"] = state.retrieved_video_id
            state.diagnostics["retrieval"]["contract"] = "ONE_VIDEO_THEN_SEMANTIC_KEYFRAME_ALIGNMENT"
        if self.runtime_config.algorithm == "trake_v2" and self.runtime_config.temporal.candidate_windows and self.runtime_config.features.candidate_windows:
            window_started = time.perf_counter()
            videos_for_windows = (
                tuple(video for video in state.videos if video.video_id == state.retrieved_video_id)
                if state.retrieved_video_id is not None
                else state.videos
            )
            state.candidate_windows = build_candidate_windows(
                state.request,
                videos_for_windows,
                state.temporal_feasibility,
                anchor_event_id=anchor_event_id,
                padding_seconds=self.runtime_config.refinement.coarse_padding_sec,
                max_windows_per_video=self.runtime_config.temporal.max_windows_per_video,
                max_total_windows=self.runtime_config.temporal.max_total_windows,
            )
            state.stage_timings_ms["window_build"] = round((time.perf_counter() - window_started) * 1000, 3)
            state.diagnostics["candidate_windows"] = {
                "count": len(state.candidate_windows),
                "max_per_video": self.runtime_config.temporal.max_windows_per_video,
                "max_total": self.runtime_config.temporal.max_total_windows,
            }
        return self._persist(state)

    def _adaptive_pool_size(self, event: TrakeEvent, raw: list[dict[str, Any]]) -> int:
        settings = self.runtime_config.retrieval
        token_count = len((event.visual_query or event.clip_query).split())
        if token_count <= settings.generic_token_threshold or (event.planner_confidence or event.confidence) < 0.7:
            return settings.topk_max
        if len(raw) < 2:
            return settings.topk_max
        first = float(raw[0].get("score") or 0.0)
        second = float(raw[1].get("score") or 0.0)
        margin = max(0.0, (first - second) / max(abs(first), 1e-9))
        if margin < settings.adaptive_margin_low:
            return settings.topk_max
        structured = bool(event.object_constraints or event.attribute_constraints or event.ocr_terms or event.asr_terms)
        if margin >= settings.adaptive_margin_high and structured:
            return settings.topk_min
        return settings.topk_default

    def align(self, session_id: str, *, top_k_per_video: int | None = None, top_videos: int | None = None) -> dict[str, Any]:
        state = self._state(session_id)
        if not state.pools:
            raise TrakeDomainError("SEARCH_REQUIRED", "TRAKE search must run before alignment", stage="RETRIEVAL")
        if not state.videos:
            self._domain_failure(state, "NO_CANDIDATES", "no candidate video covers any planned event", "EVENT_RECALL_FAILURE")
        top_k_per_video = self.runtime_config.temporal.top_chains_per_video if top_k_per_video is None else top_k_per_video
        top_videos = self.runtime_config.temporal.max_candidate_videos if top_videos is None else top_videos
        started = time.perf_counter()
        alignment_elapsed_ms = 0.0
        videos_for_alignment = state.videos
        if self.runtime_config.algorithm == "trake_v2" and self.runtime_config.temporal.fast_feasibility_filter and self.runtime_config.features.temporal_feasibility:
            feasible_ids = {item.video_id for item in state.temporal_feasibility if item.feasible}
            videos_for_alignment = tuple(video for video in state.videos if video.video_id in feasible_ids)
            if not videos_for_alignment:
                self._domain_failure(state, "NO_FEASIBLE_VIDEO", "fast temporal feasibility rejected all candidate videos", "TEMPORAL_FEASIBILITY_FAILURE")
        if self.runtime_config.algorithm == "trake_v2" and state.retrieved_video_id is not None:
            videos_for_alignment = tuple(video for video in videos_for_alignment if video.video_id == state.retrieved_video_id)
            if not videos_for_alignment:
                self._domain_failure(state, "RETRIEVED_VIDEO_UNAVAILABLE", "the retrieved video is no longer available for alignment", "RETRIEVAL_CONTRACT_FAILURE")
            # Do not turn the alignment stage into a second video search. TRAKE
            # answers one video plus one semantic frame per requested event.
            top_k_per_video = 1
            top_videos = 1
        if (
            self.runtime_config.algorithm == "trake_v2"
            and self.runtime_config.temporal.candidate_windows
            and self.runtime_config.features.candidate_windows
            and not state.candidate_windows
        ):
            self._domain_failure(state, "NO_CANDIDATE_WINDOW", "no bounded temporal window can cover the required events", "WINDOW_BUILD_FAILURE")
        if isinstance(self.aligner, DanteInspiredAligner) and state.candidate_windows:
            candidate_by_id = {candidate.candidate_id: candidate for pool in state.pools for candidate in pool.candidates}
            video_by_id = {video.video_id: video for video in videos_for_alignment}
            expansions: list[DenseWindowExpansion] = []
            window_results: list[AlignmentResult] = []
            selected_windows = state.candidate_windows[: self.runtime_config.refinement.max_dense_windows]
            for window in selected_windows:
                video = video_by_id.get(window.video_id)
                if video is None:
                    continue
                coarse_expansion = (
                    self.window_expander.expand(window, state.request, candidate_by_id)
                    if self.window_expander is not None and self.runtime_config.refinement.automatic and self.runtime_config.features.dense_refinement
                    else None
                )
                if coarse_expansion is not None:
                    expansions.append(coarse_expansion)
                coarse_align_started = time.perf_counter()
                coarse_result = self.aligner.align_window(
                    state.request,
                    video,
                    window,
                    coarse_expansion,
                    top_k=self.runtime_config.dante.coarse_top_k,
                    beam_size=self.runtime_config.temporal.beam_size,
                )
                alignment_elapsed_ms += (time.perf_counter() - coarse_align_started) * 1000
                final_result = coarse_result
                if (
                    self.window_expander is not None
                    and self.runtime_config.refinement.automatic
                    and self.runtime_config.features.dense_refinement
                    and self.runtime_config.features.dante_fine
                    and coarse_result.chains
                ):
                    fine_expansion = self.window_expander.expand(
                        window,
                        state.request,
                        candidate_by_id,
                        predicted_chain=coarse_result.chains[0],
                    )
                    expansions.append(fine_expansion)
                    if fine_expansion.status == "AVAILABLE":
                        fine_align_started = time.perf_counter()
                        final_result = self.aligner.align_window(
                            state.request,
                            video,
                            window,
                            fine_expansion,
                            top_k=self.runtime_config.dante.fine_top_k,
                            beam_size=self.runtime_config.temporal.beam_size,
                        )
                        alignment_elapsed_ms += (time.perf_counter() - fine_align_started) * 1000
                window_results.append(final_result)
            state.dense_expansions = tuple(expansions)
            state.stage_timings_ms["dense_decode"] = round(sum(item.latency_ms for item in expansions), 3)
            state.diagnostics["dense_refinement"] = {
                "automatic": self.runtime_config.refinement.automatic,
                "window_limit": self.runtime_config.refinement.max_dense_windows,
                "selected_window_ids": [window.window_id for window in selected_windows],
                "coarse_statuses": [item.status for item in expansions if item.sampling.get("stage") == "coarse"],
                "fine_statuses": [item.status for item in expansions if item.sampling.get("stage") == "fine"],
                "decoded_frame_count": sum(item.decoded_frame_count for item in expansions),
                "quality_claim": None,
                "quality_reason": "timestamp labels are required to measure precision improvement",
            }
            state.alignments = _merge_window_results(window_results, top_k_per_video, top_videos)
        else:
            align_started = time.perf_counter()
            state.alignments = self.aligner.align_videos(
                state.request,
                videos_for_alignment,
                top_k_per_video=top_k_per_video,
                top_videos=top_videos,
                beam_size=self.runtime_config.temporal.beam_size,
            )
            alignment_elapsed_ms = (time.perf_counter() - align_started) * 1000
        align_timing_key = "alignment" if self.runtime_config.algorithm == "trake_v2" else "align"
        state.stage_timings_ms[align_timing_key] = round(alignment_elapsed_ms, 3)
        if self.runtime_config.algorithm == "trake_v2":
            rerank_started = time.perf_counter()
            before_chains = [chain for result in state.alignments for chain in result.chains]
            before = [chain.chain_id for chain in before_chains]
            linkage_rejections = [
                {"chain_id": chain.chain_id, "video_id": chain.video_id, "reason": reason}
                for chain in before_chains
                if (reason := linkage_gate_reason(chain, self.runtime_config.reranking, self.modality_availability)) is not None
            ]
            state.alignments = rerank_alignments(
                state.request,
                state.alignments,
                self.runtime_config.reranking,
                self.modality_availability,
            )
            state.alignments = _single_video_answer(state.alignments, state.retrieved_video_id)
            state.stage_timings_ms["rerank"] = round((time.perf_counter() - rerank_started) * 1000, 3)
            after = [chain.chain_id for result in state.alignments for chain in result.chains]
            state.diagnostics["final_reranking"] = {
                "before_chain_ids": before,
                "after_chain_ids": after,
                "weights": dict(self.runtime_config.reranking.weights),
                "context_enabled": self.runtime_config.reranking.context_enabled,
                "linkage_enabled": self.runtime_config.reranking.linkage_enabled,
                "linkage_hard_gate": self.runtime_config.reranking.linkage_hard_gate,
                "linkage_gate_rejections": linkage_rejections,
                "version": "trake-final-rerank-v1",
            }
            state.trake_answer = _trake_answer_payload(state.alignments)
        state.diagnostics["alignment"] = {
            "algorithm_version": self.aligner.alignment_version,
            "paths": [
                {
                    "chain_id": chain.chain_id,
                    "video_id": chain.video_id,
                    "events": [
                        {
                            "event_id": entry.event_id,
                            "candidate_id": entry.candidate.candidate_id if entry.candidate else None,
                            "pts_time": entry.candidate.pts_time if entry.candidate else None,
                        }
                        for entry in chain.events
                    ],
                    "score_breakdown": dict(chain.score_components or {}),
                    "warnings": list(chain.warnings),
                }
                for result in state.alignments
                for chain in result.chains
            ],
            "failures": [
                {"video_id": result.video_id, "reasons": list(result.failures)}
                for result in state.alignments
                if result.failures
            ],
        }
        if self.runtime_config.algorithm == "trake_v2":
            state.stage_timings_ms["align_total"] = round((time.perf_counter() - started) * 1000, 3)
            measured_stages = ("planner", "embedding", "retrieval", "fusion", "temporal_filter", "window_build", "dense_decode", "alignment", "rerank")
            state.stage_timings_ms["total"] = round(sum(float(state.stage_timings_ms.get(stage, 0.0)) for stage in measured_stages), 3)
        return self._persist(state)

    def update_plan(self, session_id: str, events: list[dict[str, Any]], constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self._state(session_id)
        payload = state.request.to_dict()
        payload["events"] = events
        if constraints is not None: payload["constraints"] = constraints
        state.request = request_from_dict(payload)
        state.plan_revision += 1
        state.pools = (); state.videos = (); state.alignments = (); state.manual_chain = None; state.locked_event_ids = ()
        state.temporal_feasibility = (); state.candidate_windows = (); state.dense_expansions = (); state.diagnostics = {}
        state.retrieved_video_id = None; state.trake_answer = None
        return self._persist(state)

    def replace_candidate(self, session_id: str, chain_id: str, event_id: str, candidate_id: str, *, lock_event: bool = True) -> dict[str, Any]:
        state = self._state(session_id)
        source = state.manual_chain if state.manual_chain and state.manual_chain.chain_id == chain_id else next((chain for result in state.alignments for chain in result.chains if chain.chain_id == chain_id), None)
        if source is None: raise KeyError(f"unknown TRAKE chain: {chain_id}")
        replacement = next((item for pool in state.pools for item in pool.candidates if item.candidate_id == candidate_id and item.event_id == event_id), None)
        if replacement is None: raise KeyError(f"unknown candidate for event {event_id}: {candidate_id}")
        selected: list[TrakeCandidate | None] = []
        events: list[ChainEvent] = []
        for entry in source.events:
            candidate = replacement if entry.event_id == event_id else entry.candidate
            selected.append(candidate)
            events.append(ChainEvent(entry.event_id, entry.required, candidate, "user_replacement" if entry.event_id == event_id else entry.selection))
        required = tuple(event.required for event in state.request.events)
        breakdown = score_chain(tuple(selected), required, 0, self.aligner.score_config)
        manual = TrakeChain(
            f"{source.chain_id}:manual:{uuid4().hex[:8]}",
            source.video_id,
            tuple(events),
            breakdown,
            (),
            manual=True,
            version=self.aligner.alignment_version,
        )
        valid, failures = validate_chain(state.request, manual, allow_same_frame=bool(state.request.constraints.get("allow_same_frame", False)))
        if not valid: raise ValueError("manual replacement violates TRAKE invariants: " + ", ".join(failures))
        state.manual_chain = manual
        if lock_event:
            state.locked_event_ids = tuple(dict.fromkeys((*state.locked_event_ids, event_id)))
        return self._persist(state)

    def review(self, session_id: str, chain_id: str, decision: str, reviewer: str) -> dict[str, Any]:
        state = self._state(session_id)
        if decision not in {"confirmed", "rejected"}: raise ValueError("decision must be confirmed or rejected")
        chain = state.manual_chain if state.manual_chain and state.manual_chain.chain_id == chain_id else next((item for result in state.alignments for item in result.chains if item.chain_id == chain_id), None)
        if chain is None: raise KeyError(f"unknown TRAKE chain: {chain_id}")
        valid, failures = validate_chain(state.request, chain, allow_same_frame=bool(state.request.constraints.get("allow_same_frame", False)))
        if decision == "confirmed" and not valid: raise ValueError("cannot confirm invalid chain: " + ", ".join(failures))
        review = {"review_id": f"trake-review-{uuid4().hex}", "session_id": session_id, "chain_id": chain_id, "decision": decision, "reviewer": reviewer.strip() or "local-reviewer", "created_at": _now(), "plan_revision": state.plan_revision, "valid_at_review": valid}
        if self.store: self.store.append_review(review)
        return {"review": review, **self._persist(state)}

    def export(self, session_id: str, review_id: str) -> dict[str, Any]:
        state = self._state(session_id)
        reviews = self.store.reviews(session_id) if self.store else []
        review = next((item for item in reviews if item["review_id"] == review_id), None)
        if not review or review["decision"] != "confirmed": raise ValueError("internal export requires a confirmed review")
        chain = state.manual_chain if state.manual_chain and state.manual_chain.chain_id == review["chain_id"] else next((item for result in state.alignments for item in result.chains if item.chain_id == review["chain_id"]), None)
        if chain is None: raise ValueError("reviewed chain is unavailable")
        valid, failures = validate_chain(state.request, chain, allow_same_frame=bool(state.request.constraints.get("allow_same_frame", False)))
        if not valid: raise ValueError("cannot export invalid chain: " + ", ".join(failures))
        if not self.store: raise ValueError("TRAKE persistence is required for export")
        algorithm_chain_id=chain.chain_id.split(":manual:",1)[0]
        algorithm_chain=next((item for result in state.alignments for item in result.chains if item.chain_id==algorithm_chain_id),chain)
        algorithm_by_event={item.event_id:item.candidate for item in algorithm_chain.events}
        refinements=self.store.refinements(session_id)
        event_records=[]
        for item in chain.events:
            if item.candidate is None: continue
            algorithm=algorithm_by_event.get(item.event_id)
            dense_mapping = bool(item.candidate.evidence.get("dense_refinement"))
            mapping = {
                "source": "dense_video_seek" if dense_mapping else "validated_index_ref",
                "frame_idx": item.candidate.frame_idx,
                "pts_time": item.candidate.pts_time,
                "fps": item.candidate.evidence.get("fps") or item.candidate.evidence.get("raw_result",{}).get("fps"),
            }
            if dense_mapping:
                mapping.update({
                    "source_keyframe_id": item.candidate.evidence.get("source_keyframe_id"),
                    "requested_pts": item.candidate.evidence.get("requested_pts"),
                    "decoded_pts": item.candidate.evidence.get("decoded_pts"),
                    "frame_index_if_known": item.candidate.evidence.get("frame_index_if_known"),
                    "frame_index_is_estimate": item.candidate.evidence.get("frame_index_is_estimate"),
                    "seek_error_estimate": item.candidate.evidence.get("seek_error_estimate"),
                })
            event_records.append({
                "event_id":item.event_id,"candidate_id":item.candidate.candidate_id,"keyframe_id":item.candidate.keyframe_id,
                "frame_idx":item.candidate.frame_idx,"pts_time":item.candidate.pts_time,"keyframe_path":item.candidate.keyframe_path,
                "mapping":mapping,
                "evidence":item.candidate.evidence,"provenance":item.candidate.provenance,"retrieval_method":item.candidate.retrieval_method,
                "algorithm_selection":algorithm.to_dict() if algorithm else None,"final_selection":item.candidate.to_dict(),
                "manually_replaced":bool(algorithm and algorithm.candidate_id!=item.candidate.candidate_id),
                "refinements":[value for value in refinements if value.get("event_id")==item.event_id],
            })
        record={"export_id":f"trake-export-{uuid4().hex}","session_id":session_id,"review_id":review_id,"created_at":_now(),"schema":"phase8-internal-record-v1","not_final_btc_format":True,"query_id":state.request.query_id,"plan_revision":state.plan_revision,"video_id":chain.video_id,"same_video":all(item.candidate is None or item.candidate.video_id==chain.video_id for item in chain.events),"temporally_ordered":valid,"score":jsonable(chain.score),"events":event_records,"confirmed_by":review["reviewer"],"confirmed_at":review["created_at"],"human_confirmed":True}
        self.store.append_export(record)
        return {"export": record}

    def get(self, session_id: str) -> dict[str, Any]:
        payload = state_payload(self._state(session_id))
        payload["reviews"] = self.store.reviews(session_id) if self.store else []
        return payload

    def maybe_verify_with_vlm(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Optionally send only low-confidence Top-N evidence for advisory review."""
        if not self.runtime_config.vlm.enabled:
            return {"status": "DISABLED", "called": False}
        if not self.runtime_config.features.conditional_vlm:
            return {"status": "DISABLED", "called": False, "reason": "conditional_vlm feature gate is disabled"}
        if not self.runtime_config.vlm.network_calls_allowed:
            return {"status": "UNAVAILABLE", "called": False, "reason": "network calls are disabled"}
        if not self.runtime_config.vlm.decision_log_approved:
            return {"status": "UNAVAILABLE", "called": False, "reason": "project decision-log approval is absent"}
        if self.vlm_verifier is None:
            return {"status": "UNAVAILABLE", "called": False, "reason": "no verifier is configured"}
        reasons = _vlm_trigger_reasons(payload, self.runtime_config.vlm)
        if not reasons:
            return {"status": "SKIPPED_HIGH_CONFIDENCE", "called": False, "trigger_reasons": []}
        safe_payload = _vlm_safe_payload(payload, self.runtime_config.vlm.top_n)
        if not safe_payload["chains"]:
            return {"status": "SKIPPED_INVALID", "called": False, "reason": "no server-valid chain can be sent"}
        try:
            result = self.vlm_verifier(safe_payload)
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "called": True,
                "trigger_reasons": reasons,
                "reason": f"VLM verifier failed: {exc}",
                "advisory_only": True,
                "server_selection_unchanged": True,
            }
        return {
            "status": "AVAILABLE",
            "called": True,
            "trigger_reasons": reasons,
            "result": result,
            "advisory_only": True,
            "server_selection_unchanged": True,
        }

    def list_sessions(self) -> dict[str, Any]:
        if self.store: return {"sessions": self.store.list_sessions()}
        items = sorted(self.sessions.values(), key=lambda item: (-item.created_at_epoch, item.session_id))
        return {"sessions": [{"session_id": item.session_id, "query_id": item.request.query_id, "original_query": item.request.original_query, "plan_revision": item.plan_revision} for item in items]}

    def _state(self, session_id: str) -> TrakeWorkflowState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            payload = self.store.load_state(session_id) if self.store else None
            if payload is not None:
                state = _state_from_payload(payload); self.sessions[session_id] = state; return state
            raise KeyError(f"unknown TRAKE session: {session_id}") from exc

    def _persist(self, state: TrakeWorkflowState) -> dict[str, Any]:
        state.diagnostics["latency_ms"] = dict(state.stage_timings_ms)
        payload = state_payload(state)
        if self.store: self.store.save_state(payload)
        return payload

    def _domain_failure(self, state: TrakeWorkflowState, code: str, message: str, stage: str) -> None:
        state.diagnostics["failure"] = {"code": code, "failure_stage": stage, "message": message}
        self._persist(state)
        raise TrakeDomainError(code, message, stage=stage)


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def state_payload(state: TrakeWorkflowState) -> dict[str, Any]:
    version = LEGACY_WORKFLOW_VERSION if state.algorithm == "legacy" else V2_WORKFLOW_VERSION
    return {"workflow_version": version, "state": jsonable(state)}


def _candidate(payload: dict[str, Any]) -> TrakeCandidate:
    values=dict(payload)
    for name in ("provenance","warnings"): values[name]=tuple(values.get(name,()))
    return TrakeCandidate(**values)


def _chain(payload: dict[str, Any]) -> TrakeChain:
    events=tuple(ChainEvent(item["event_id"],item["required"],_candidate(item["candidate"]) if item.get("candidate") else None,item.get("selection","algorithm")) for item in payload["events"])
    score=ScoreBreakdown(**payload["score"])
    return TrakeChain(payload["chain_id"],payload["video_id"],events,score,tuple(payload.get("warnings",())),payload.get("valid",True),payload.get("manual",False),payload.get("version","phase8-temporal-dp-v1"),payload.get("score_components"))


def _state_from_payload(payload: dict[str, Any]) -> TrakeWorkflowState:
    request=request_from_dict(payload["request"]); event_by_id={item.event_id:item for item in request.events}
    pools=tuple(
        EventCandidatePool(
            event_by_id[item["event"]["event_id"]],
            tuple(_candidate(candidate) for candidate in item["candidates"]),
            {key: Availability(value) for key, value in item.get("modality_availability", {}).items()},
            tuple(item.get("warnings", ())),
            item.get("topk_used"),
            item.get("distinctiveness"),
            dict(item.get("distinctiveness_components", {})),
        )
        for item in payload.get("pools", ())
    )
    videos=tuple(VideoCandidate(item["video_id"],{key:tuple(_candidate(candidate) for candidate in values) for key,values in item["event_candidates"].items()},item["required_coverage"],item["required_total"],item["optional_coverage"],item["optional_total"],item["complete"],tuple(item.get("warnings",()))) for item in payload.get("videos",()))
    alignments=tuple(AlignmentResult(item["video_id"],tuple(_chain(chain) for chain in item.get("chains",())),tuple(item.get("failures",()))) for item in payload.get("alignments",()))
    return TrakeWorkflowState(
        session_id=payload["session_id"],
        request=request,
        decomposition_warnings=tuple(payload.get("decomposition_warnings", ())),
        created_at_epoch=payload["created_at_epoch"],
        pools=pools,
        videos=videos,
        alignments=alignments,
        stage_timings_ms=dict(payload.get("stage_timings_ms", {})),
        plan_revision=payload.get("plan_revision", 1),
        locked_event_ids=tuple(payload.get("locked_event_ids", ())),
        manual_chain=_chain(payload["manual_chain"]) if payload.get("manual_chain") else None,
        algorithm=payload.get("algorithm", "legacy"),
        algorithm_version=payload.get("algorithm_version", "phase8-temporal-dp-v1"),
        config_fingerprint=payload.get("config_fingerprint"),
        provenance=dict(payload.get("provenance", {})),
        diagnostics=dict(payload.get("diagnostics", {})),
        temporal_feasibility=tuple(
            TemporalFeasibilityResult(
                video_id=item["video_id"],
                feasible=bool(item["feasible"]),
                witness_candidate_ids=tuple(item.get("witness_candidate_ids", ())),
                event_intervals={key: tuple(value) for key, value in (item.get("event_intervals") or {}).items()},
                failure_stage=item.get("failure_stage"),
                failure_reason=item.get("failure_reason"),
                version=item.get("version", "k-pointer-inspired-feasibility-v1"),
            )
            for item in payload.get("temporal_feasibility", ())
        ),
        candidate_windows=tuple(
            CandidateWindow(
                window_id=item["window_id"],
                video_id=item["video_id"],
                start_pts=float(item["start_pts"]),
                end_pts=float(item["end_pts"]),
                anchor_event_id=item.get("anchor_event_id"),
                event_candidate_ids=dict(item.get("event_candidate_ids", {})),
                score=float(item.get("score", 0.0)),
                source=item.get("source", "anchor-monotonic-window-v1"),
                warnings=tuple(item.get("warnings", ())),
            )
            for item in payload.get("candidate_windows", ())
        ),
        dense_expansions=tuple(dense_expansion_from_dict(item) for item in payload.get("dense_expansions", ())),
        retrieved_video_id=payload.get("retrieved_video_id"),
        trake_answer=dict(payload["trake_answer"]) if payload.get("trake_answer") else None,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_window_results(
    results: list[AlignmentResult],
    top_k_per_video: int,
    top_videos: int,
) -> tuple[AlignmentResult, ...]:
    by_video: dict[str, list[TrakeChain]] = {}
    failures: dict[str, set[str]] = {}
    for result in results:
        by_video.setdefault(result.video_id, []).extend(result.chains)
        failures.setdefault(result.video_id, set()).update(result.failures)
    merged: list[AlignmentResult] = []
    for video_id, chains in by_video.items():
        chains.sort(
            key=lambda chain: (
                -float((chain.score_components or {}).get("dante_objective", float("-inf"))),
                chain.chain_id,
            )
        )
        merged.append(AlignmentResult(video_id, tuple(chains[:top_k_per_video]), tuple(sorted(failures.get(video_id, set())))))
    merged.sort(
        key=lambda result: (
            not bool(result.chains),
            -float((result.chains[0].score_components or {}).get("dante_objective", float("-inf"))) if result.chains else float("inf"),
            result.video_id,
        )
    )
    return tuple(merged[:top_videos])


def _select_retrieved_video(
    videos: tuple[VideoCandidate, ...],
    feasibility: tuple[TemporalFeasibilityResult, ...],
) -> str | None:
    """Select the one video for TRAKE stage 1 after same-video feasibility.

    ``videos`` is already deterministically ranked by whole-sequence coverage,
    anchor evidence, and local event evidence.  The temporal witness is a hard
    prerequisite: no video is selected merely because it has independently
    similar frames for several events.
    """
    feasible_ids = {item.video_id for item in feasibility if item.feasible}
    return next((video.video_id for video in videos if video.complete and video.video_id in feasible_ids), None)


def _gate_alignment_seeds(
    request: TrakeRequest,
    pools: tuple[EventCandidatePool, ...],
    videos: tuple[VideoCandidate, ...],
    min_relative_score: float,
) -> tuple[tuple[VideoCandidate, ...], dict[str, Any]]:
    """Fail closed when temporal ordering would require a weak semantic seed.

    Retrieval ranks frames for each event independently.  Without this gate,
    the temporal path builder can replace a high-ranked event frame with an
    unrelated low-ranked one solely because the latter is earlier/later in the
    video.  A candidate remains eligible only when its rank-normalized score
    is close enough to the best candidate for that event in the retrieval
    pool.  Dense refinement is then constrained around this eligible seed.
    """
    best_by_event = {
        pool.event.event_id: max((candidate.local_score for candidate in pool.candidates), default=0.0)
        for pool in pools
    }
    required_ids = {event.event_id for event in request.events if event.required}
    optional_ids = {event.event_id for event in request.events if not event.required}
    filtered: list[VideoCandidate] = []
    rejected_by_video: dict[str, dict[str, list[str]]] = {}
    for video in videos:
        accepted: dict[str, tuple[TrakeCandidate, ...]] = {}
        rejected: dict[str, list[str]] = {}
        for event_id, candidates in video.event_candidates.items():
            best = best_by_event.get(event_id, 0.0)
            cutoff = best * min_relative_score
            kept = tuple(candidate for candidate in candidates if candidate.local_score + 1e-12 >= cutoff)
            accepted[event_id] = kept
            rejected_ids = [candidate.candidate_id for candidate in candidates if candidate not in kept]
            if rejected_ids:
                rejected[event_id] = rejected_ids
        required_coverage = sum(bool(accepted.get(event_id)) for event_id in required_ids)
        optional_coverage = sum(bool(accepted.get(event_id)) for event_id in optional_ids)
        complete = required_coverage == len(required_ids)
        warnings = tuple(dict.fromkeys((*video.warnings, *(() if not rejected else ("LOW_SEMANTIC_SEED_REJECTED",)), *(() if complete else ("REQUIRED_EVENT_MISSING_AFTER_SEMANTIC_GATE",)))))
        filtered.append(VideoCandidate(
            video.video_id,
            accepted,
            required_coverage,
            video.required_total,
            optional_coverage,
            video.optional_total,
            complete,
            warnings,
        ))
        if rejected:
            rejected_by_video[video.video_id] = rejected
    filtered.sort(
        key=lambda item: (
            not item.complete,
            -item.required_coverage,
            -item.optional_coverage,
            item.video_id,
        )
    )
    return tuple(filtered), {
        "enabled": True,
        "min_relative_score": min_relative_score,
        "rejected_candidate_ids_by_video": rejected_by_video,
    }


def _single_video_answer(
    alignments: tuple[AlignmentResult, ...],
    retrieved_video_id: str | None,
) -> tuple[AlignmentResult, ...]:
    """Keep exactly one chain from the Stage-1 retrieved video for TRAKE V2."""
    if retrieved_video_id is None:
        return ()
    result = next((item for item in alignments if item.video_id == retrieved_video_id), None)
    if result is None or not result.chains:
        return (AlignmentResult(retrieved_video_id, (), result.failures if result else ("NO_ALIGNED_CHAIN",)),)
    return (AlignmentResult(retrieved_video_id, (result.chains[0],), result.failures),)


def _trake_answer_payload(alignments: tuple[AlignmentResult, ...]) -> dict[str, Any] | None:
    """Expose the contest-shaped answer, not a gallery of unrelated hits."""
    if not alignments or not alignments[0].chains:
        return None
    chain = alignments[0].chains[0]
    frame_ids = [entry.candidate.frame_idx if entry.candidate is not None else None for entry in chain.events]
    if any(frame_id is None for frame_id in frame_ids):
        return None
    return {
        "format": "<video_id>, <frame_id_1>, ..., <frame_id_N>",
        "video_id": chain.video_id,
        "frame_ids": frame_ids,
        "answer_text": ", ".join([chain.video_id, *(str(frame_id) for frame_id in frame_ids)]),
        "semantic_keyframes": [
            {
                "event_id": entry.event_id,
                "frame_id": entry.candidate.frame_idx,
                "pts_time": entry.candidate.pts_time,
                "evidence_path": entry.candidate.keyframe_path,
            }
            for entry in chain.events
            if entry.candidate is not None
        ],
    }


def _vlm_trigger_reasons(payload: dict[str, Any], settings: Any) -> list[str]:
    reasons: list[str] = []
    chains = list(payload.get("chains") or ())
    scores = [
        float((chain.get("score_components") or {}).get("final_rerank_score"))
        for chain in chains[:2]
        if (chain.get("score_components") or {}).get("final_rerank_score") is not None
    ]
    if len(scores) >= 2 and scores[0] - scores[1] <= settings.low_confidence_margin:
        reasons.append("SMALL_TOP1_TOP2_MARGIN")
    linkages = [
        (chain.get("score_components") or {}).get("linkage_score", {}).get("score")
        for chain in chains[: settings.top_n]
    ]
    if any(value is not None and float(value) < settings.low_linkage_threshold for value in linkages):
        reasons.append("LOW_LINKAGE")
    if any(float(value) < 0.7 for value in payload.get("event_confidences", ())):
        reasons.append("LOW_EVENT_CONFIDENCE")
    if payload.get("ood_event") is True:
        reasons.append("OOD_EVENT")
    return reasons


def _vlm_safe_payload(payload: dict[str, Any], top_n: int) -> dict[str, Any]:
    safe_chains: list[dict[str, Any]] = []
    for chain in list(payload.get("chains") or ())[:top_n]:
        events = list(chain.get("events") or ())
        selected = [event for event in events if event.get("candidate")]
        video_ids = {event["candidate"].get("video_id") for event in selected}
        times = [float(event["candidate"].get("pts_time", 0.0)) for event in selected]
        required_missing = any(event.get("required") and not event.get("candidate") for event in events)
        if chain.get("valid") is False or len(video_ids) != 1 or required_missing or any(left >= right for left, right in zip(times, times[1:])):
            continue
        safe_chains.append({
            "chain_id": chain.get("chain_id"),
            "video_id": chain.get("video_id"),
            "events": [
                {
                    "event_id": event.get("event_id"),
                    "description": event.get("description"),
                    "pts_time": event.get("candidate", {}).get("pts_time"),
                    "keyframe_path": event.get("candidate", {}).get("keyframe_path"),
                }
                for event in selected
            ],
        })
    return {
        "query": payload.get("query"),
        "event_descriptions": list(payload.get("event_descriptions") or ()),
        "chains": safe_chains,
    }
