from __future__ import annotations

import json
import mimetypes
import os
import sqlite3
import time
import traceback
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from aic_retrieval.search import (
    aggregate_results_by_video,
    diversify_results_by_video,
    load_numpy_index,
    load_registry,
    search_numpy_index,
    search_numpy_index_batch,
    validate_numpy_index,
)
from aic_retrieval.metadata_search import load_metadata_documents, results_to_dict, search_metadata
from aic_retrieval.metadata_search import MetadataConstraints
from aic_retrieval.hybrid_candidates import StructuredCandidateGenerator
from aic_retrieval.hybrid_ranking import RrfConfig, rank_video_candidates
from aic_retrieval.color_attributes import ColorAttributeService, parse_color_constraint
from aic_retrieval.object_search import ObjectPredicate, ObjectSearchConfig, ObjectSearchService
from aic_retrieval.object_store import load_alias_dictionary
from aic_retrieval.structured_query import ObjectConstraint, StructuredQuery
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder
from aic_retrieval.translation import DEFAULT_GEMINI_API_URL, ExternalTranslator, TranslationConfig
from aic_retrieval.phase5_store import Phase5SearchService
from aic_retrieval.query_planner import RuleBasedQueryPlanner
from aic_retrieval.hybrid_query_planner import HybridQueryPlan, HybridQueryPlanner
from aic_retrieval.hybrid_engine import HybridRetrievalEngine
from aic_retrieval.rrf_fusion import RrfFusionConfig
from aic_retrieval.clip_retriever import ClipRetriever
from aic_retrieval.bge_retriever import BgeEncoder, BgeRetriever
from aic_retrieval.bm25_retriever import Bm25Retriever
from aic_retrieval.reranking import RerankerConfig, load_reranker_config, rerank_video_results, reranker_metadata, with_top_n
from aic_retrieval.qa_schema import AvailabilityStatus, EvidenceModality, QaRequest, ReviewDecision
from aic_retrieval.qa_workflow import QaWorkflow, jsonable, state_payload
from aic_retrieval.qa_store import QaStore
from aic_retrieval.qa_gemini_answering import GeminiEvidenceAnswerer
from aic_retrieval.trake_schema import Availability, TrakeEvent, TrakeRequest
from aic_retrieval.trake_workflow import TrakeDomainError, TrakeWorkflow
from aic_retrieval.trake_store import TrakeStore
from aic_retrieval.trake_refinement import DenseRefiner, DenseWindowExpander
from aic_retrieval.trake_config import TrakeRuntimeConfig, load_trake_config
from aic_retrieval.trake_hybrid import retrieve_trake_hybrid_channels
from aic_retrieval.submission_official import OfficialPrediction, OfficialQuery
from aic_retrieval.submission_session import SubmissionSessionStore
from aic_retrieval.provenance import fingerprint_json


DEFAULT_TOP_K_VIDEOS = 12
DEFAULT_CANDIDATE_POOL_SIZE = 40
DEFAULT_MAX_FRAMES_PER_VIDEO = 1
DEFAULT_MATCHED_FRAMES_PER_VIDEO = 5
DEFAULT_NEIGHBOR_RADIUS = 3
DEFAULT_AGGREGATION_METHOD = "max"
DEFAULT_MEAN_TOP_N = 3


@dataclass(frozen=True)
class RetrievalUiConfig:
    repo_root: Path
    registry_path: Path
    index_dir: Path
    metadata_dir: Path
    static_dir: Path
    groups: set[str]
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_cache_dir: Path | None = None
    clip_local_files_only: bool = False
    require_keyframes: bool = False
    allow_stale_index: bool = False
    translation: TranslationConfig | None = None
    object_store_path: Path | None = None
    object_aliases_path: Path | None = None
    phase5_store_path: Path | None = None
    phase6_config_path: Path | None = None
    qa_store_path: Path | None = None
    trake_store_path: Path | None = None
    trake_static_dir: Path | None = None
    trake_refinement_dir: Path | None = None
    trake_config_path: Path | None = None
    hybrid_enabled: bool = False
    bge_index_dir: Path | None = None
    bge_model_id: str = "BAAI/bge-m3"
    bge_model_path: Path | None = None
    bge_device: str | None = None
    bm25_index_dir: Path | None = None
    submission_store_path: Path | None = None
    submission_output_dir: Path | None = None
    submission_static_dir: Path | None = None


class TimedEventResults(dict[str, list[dict[str, Any]]]):
    def __init__(
        self,
        values: dict[str, list[dict[str, Any]]],
        stage_timings_ms: dict[str, float],
        *,
        query_plans: dict[str, dict[str, Any]] | None = None,
        query_traces: dict[str, dict[str, Any]] | None = None,
        hybrid_failures: dict[str, dict[str, str]] | None = None,
        skipped_without_frame: dict[str, dict[str, int]] | None = None,
    ) -> None:
        super().__init__(values)
        self.stage_timings_ms = stage_timings_ms
        self.query_plans = query_plans or {}
        self.query_traces = query_traces or {}
        self.hybrid_failures = hybrid_failures or {}
        self.skipped_without_frame = skipped_without_frame or {}


class RetrievalUiService:
    def __init__(self, config: RetrievalUiConfig) -> None:
        self.config = config
        self.registry = load_registry(config.registry_path)
        self.metadata_docs = load_metadata_documents(config.metadata_dir, groups=config.groups)
        self.index, self.refs, self.index_metadata = load_numpy_index(config.index_dir)
        validate_numpy_index(
            self.index,
            self.refs,
            self.index_metadata,
            requested_groups=config.groups,
            require_keyframes=config.require_keyframes,
            registry_path=config.registry_path,
            repo_root=config.repo_root,
            allow_stale_index=config.allow_stale_index,
        )
        self.assets_by_video = {asset["video_id"]: asset for asset in self.registry.get("videos", [])}
        self.refs_by_video: dict[str, list[Any]] = {}
        for ref in self.refs:
            self.refs_by_video.setdefault(ref.video_id, []).append(ref)
        for video_refs in self.refs_by_video.values():
            video_refs.sort(key=lambda ref: (ref.pts_time, ref.keyframe_id))
        self.metadata_by_video = {
            doc.video_id: {
                "title": doc.title, "author": doc.author, "channel_id": doc.channel_id,
                "publish_date": doc.publish_date, "duration": doc.duration, "keywords": doc.keywords,
                "description": doc.description, "watch_url": doc.watch_url,
            }
            for doc in self.metadata_docs
        }
        self.encoder: ClipTextEncoder | None = None
        self.trake_text_embedding_cache: dict[tuple[str, str], np.ndarray] = {}
        self.translator = ExternalTranslator(config.translation or TranslationConfig())
        self.object_service: ObjectSearchService | None = None
        if config.object_store_path is not None and config.object_store_path.is_file():
            aliases_path = config.object_aliases_path or config.repo_root / "config" / "object_aliases_v1.json"
            self.object_service = ObjectSearchService(
                config.object_store_path,
                load_alias_dictionary(aliases_path),
                config.repo_root / "data" / "objects",
            )
        self.attribute_service = ColorAttributeService(config.repo_root, self.refs)
        self.phase5_service = Phase5SearchService(config.phase5_store_path, self.refs) if config.phase5_store_path else None
        self.query_planner = RuleBasedQueryPlanner()
        self.hybrid_query_planner = HybridQueryPlanner(config.translation)
        self.hybrid_engine = self._build_hybrid_engine() if config.hybrid_enabled else None
        self.reranker_config = load_reranker_config(config.phase6_config_path) if config.phase6_config_path and config.phase6_config_path.is_file() else RerankerConfig()
        phase6_payload = json.loads(config.phase6_config_path.read_text(encoding="utf-8")) if config.phase6_config_path and config.phase6_config_path.is_file() else {}
        self.rrf_config = RrfConfig(**phase6_payload.get("fusion", {}))
        self.trake_runtime_config = (
            load_trake_config(config.trake_config_path)
            if config.trake_config_path and config.trake_config_path.is_file()
            else TrakeRuntimeConfig.legacy()
        )
        self.structured_generator = StructuredCandidateGenerator(self.index, self.refs, self.object_service, self.metadata_docs, self.attribute_service, self.phase5_service)
        self.qa_workflow = QaWorkflow(store=QaStore(config.qa_store_path) if config.qa_store_path else None)
        self.submission_store = (
            SubmissionSessionStore(config.submission_store_path, config.submission_output_dir or config.repo_root / "artifacts" / "submissions")
            if config.submission_store_path else None
        )
        retrieval_settings = self.trake_runtime_config.retrieval
        modality_availability = {
            "clip": Availability.AVAILABLE if retrieval_settings.clip_enabled else Availability.UNAVAILABLE,
            "object": Availability.AVAILABLE if retrieval_settings.object_enabled and self.object_service else Availability.UNAVAILABLE,
            "attribute": Availability.AVAILABLE if retrieval_settings.attribute_enabled and self.attribute_service.available else Availability.UNAVAILABLE,
            "ocr": Availability.UNKNOWN if retrieval_settings.ocr_enabled and self.phase5_service and self.phase5_service.available else Availability.UNAVAILABLE,
            "asr": Availability.UNKNOWN if retrieval_settings.asr_enabled and self.phase5_service and self.phase5_service.available else Availability.UNAVAILABLE,
            "metadata": Availability.AVAILABLE if retrieval_settings.metadata_enabled and self.metadata_docs else Availability.UNAVAILABLE,
        }
        phase5_version = _sqlite_metadata_value(config.phase5_store_path, "store_version")
        clip_model_fingerprint = fingerprint_json({"model_id": config.clip_model_id})
        index_fingerprint = fingerprint_json(self.index_metadata)
        session_provenance = {
            "index_fingerprint": index_fingerprint,
            "clip_model_fingerprint": clip_model_fingerprint,
            "siglip_model_fingerprint": None,
            "ocr_store_version": phase5_version if _sqlite_row_count(config.phase5_store_path, "ocr") > 0 else None,
            "asr_store_version": phase5_version if _sqlite_row_count(config.phase5_store_path, "asr") > 0 else None,
            "object_store_version": _sqlite_metadata_value(config.object_store_path, "store_version"),
            "mapping_version": self.index_metadata.get("mapping_source_fingerprint"),
        }
        refinement_dir = config.trake_refinement_dir or (config.repo_root / "artifacts" / "trake" / "refinement")
        self.trake_window_expander = DenseWindowExpander(
            config.repo_root,
            self.assets_by_video,
            refinement_dir,
            self._trake_frame_scores,
            config=self.trake_runtime_config.refinement.to_refinement_config(),
            model_fingerprint=clip_model_fingerprint,
            config_fingerprint=self.trake_runtime_config.fingerprint,
            index_fingerprint=index_fingerprint,
            candidates_per_event=self.trake_runtime_config.dante.dense_candidates_per_event,
        )
        self.trake_workflow = TrakeWorkflow(
            self._trake_retrieve_event,
            modality_availability=modality_availability,
            store=TrakeStore(config.trake_store_path) if config.trake_store_path else None,
            runtime_config=self.trake_runtime_config,
            session_provenance=session_provenance,
            request_retriever=self._trake_retrieve_request,
            window_expander=self.trake_window_expander,
        )
        self.trake_refiner = DenseRefiner(
            config.repo_root, self.assets_by_video,
            refinement_dir,
            self._trake_frame_scores,
            config=self.trake_runtime_config.refinement.to_refinement_config(),
        )

    def _trake_retrieve_request(self, request: TrakeRequest, pool_size: int) -> dict[str, list[dict[str, Any]]]:
        query_entries: list[tuple[str, str]] = []
        event_by_id = {event.event_id: event for event in request.events}
        hybrid_requested = bool(request.constraints.get("hybrid_retrieval"))
        hybrid_active = hybrid_requested and self.hybrid_engine is not None
        use_gemini = bool(request.constraints.get("hybrid_use_gemini", True))
        hybrid_planning: dict[str, Any] = {}
        hybrid_plans: dict[str, HybridQueryPlan] = {}
        if hybrid_active:
            for event in request.events:
                if hasattr(self.hybrid_query_planner, "plan_with_trace"):
                    planning = self.hybrid_query_planner.plan_with_trace(event.text, use_gemini=use_gemini)
                    hybrid_planning[event.event_id] = planning
                    hybrid_plans[event.event_id] = planning.plan
                else:
                    hybrid_plans[event.event_id] = self.hybrid_query_planner.plan(event.text, use_gemini=use_gemini)
        hybrid_failures: dict[str, dict[str, str]] = {}
        skipped_without_frame: dict[str, dict[str, int]] = {}
        if hybrid_requested and not hybrid_active:
            hybrid_failures["request"] = {"hybrid": "Hybrid retrieval is not enabled; TRAKE used its baseline retriever."}
        if self.trake_runtime_config.retrieval.clip_enabled:
            for event in request.events:
                if "clip" not in event.modalities:
                    continue
                hybrid_plan = hybrid_plans.get(event.event_id)
                if hybrid_plan is not None:
                    if "clip" not in hybrid_plan.enabled_retrievers:
                        continue
                    variants = (hybrid_plan.visual_clip_query_en,)
                else:
                    variants = (
                        tuple(dict.fromkeys((event.visual_query or event.clip_query, *event.query_variants)))
                        if self.trake_runtime_config.features.query_variants
                        else (event.visual_query or event.clip_query,)
                    )
                query_entries.extend((event.event_id, variant) for variant in variants if variant.strip())
        clip_by_event: dict[str, list[tuple[str, list[Any]]]] = {event.event_id: [] for event in request.events}
        embedding_ms = 0.0
        vector_search_ms = 0.0
        if query_entries:
            cache_size = self.trake_runtime_config.retrieval.text_embedding_cache_size
            model_key = str(self.trake_workflow.session_provenance.get("clip_model_fingerprint") or self.config.clip_model_id)
            unique_variants = list(dict.fromkeys(variant for _, variant in query_entries))
            missing = [variant for variant in unique_variants if (model_key, variant) not in self.trake_text_embedding_cache]
            vectors_by_variant = {
                variant: self.trake_text_embedding_cache[(model_key, variant)]
                for variant in unique_variants
                if (model_key, variant) in self.trake_text_embedding_cache
            }
            embedding_started = time.perf_counter()
            if missing:
                try:
                    encoded = self._encoder().encode_texts(missing)
                except (ImportError, OSError, RuntimeError) as exc:
                    raise TrakeDomainError(
                        "RETRIEVAL_UNAVAILABLE",
                        "local text encoder is unavailable; preload the configured model or allow its download",
                        stage="RETRIEVAL",
                    ) from exc
                for variant, vector in zip(missing, encoded):
                    vectors_by_variant[variant] = vector
                    if cache_size > 0:
                        while len(self.trake_text_embedding_cache) >= cache_size:
                            self.trake_text_embedding_cache.pop(next(iter(self.trake_text_embedding_cache)))
                        self.trake_text_embedding_cache[(model_key, variant)] = vector
            vectors = np.stack([vectors_by_variant[variant] for _, variant in query_entries])
            embedding_ms = (time.perf_counter() - embedding_started) * 1000
            vector_started = time.perf_counter()
            result_batches = search_numpy_index_batch(self.index, self.refs, vectors, pool_size)
            vector_search_ms = (time.perf_counter() - vector_started) * 1000
            for (event_id, variant), results in zip(query_entries, result_batches):
                clip_by_event[event_id].append((variant, [result for result in results if result.group == request.group]))
        timing_sink = {"fusion": 0.0}
        channel_started = time.perf_counter()
        if hybrid_active:
            values = {
                event_id: self._trake_retrieve_event(
                    event,
                    pool_size,
                    clip_by_event.get(event_id, []),
                    timing_sink=timing_sink,
                    hybrid_plan=hybrid_plans.get(event_id),
                    hybrid_failure_sink=hybrid_failures,
                    skipped_without_frame_sink=skipped_without_frame,
                    query_id=request.query_id,
                    groups=(request.group,),
                )
                for event_id, event in event_by_id.items()
            }
        else:
            values = {
                event_id: self._trake_retrieve_event(
                    event,
                    pool_size,
                    clip_by_event.get(event_id, []),
                    timing_sink=timing_sink,
                )
                for event_id, event in event_by_id.items()
            }
        channel_ms = (time.perf_counter() - channel_started) * 1000
        return TimedEventResults(values, {
            "embedding": round(embedding_ms, 3),
            "retrieval": round(vector_search_ms + max(0.0, channel_ms - timing_sink["fusion"]), 3),
            "fusion": round(timing_sink["fusion"], 3),
        }, query_plans={key: value.to_dict() for key, value in hybrid_plans.items()}, query_traces={key: value.trace.to_dict() for key, value in hybrid_planning.items()}, hybrid_failures=hybrid_failures, skipped_without_frame=skipped_without_frame)

    def _trake_retrieve_event(
        self,
        event: TrakeEvent,
        pool_size: int,
        clip_results: list[tuple[str, list[Any]]] | None = None,
        timing_sink: dict[str, float] | None = None,
        *,
        hybrid_plan: Any | None = None,
        hybrid_failure_sink: dict[str, dict[str, str]] | None = None,
        skipped_without_frame_sink: dict[str, dict[str, int]] | None = None,
        query_id: str = "trake-event",
        groups: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        fused: dict[tuple[str, int], dict[str, Any]] = {}
        ref_by_key = self.structured_generator.ref_by_key

        def add(video_id: str, keyframe_id: int, modality: str, rank: int, score: float | None, evidence: dict[str, Any], *, source_key: str | None = None, query_variant: str | None = None) -> None:
            ref = ref_by_key.get((video_id, keyframe_id))
            if ref is None: return
            if groups and ref.group not in groups: return
            item = fused.setdefault((video_id,keyframe_id), {**asdict(ref),"provenance":[],"evidence":{"fps":ref.fps},"raw_modality_scores":{},"raw_modality_ranks":{},"rrf_score":0.0})
            if modality not in item["provenance"]: item["provenance"].append(modality)
            evidence_row = {**evidence, "query_variant": query_variant} if query_variant else evidence
            item["evidence"].setdefault(modality,[]).append(evidence_row)
            source_key = source_key or modality
            item["raw_modality_scores"][source_key]=score
            item["raw_modality_ranks"][source_key]=rank
            weight = float(self.trake_runtime_config.fusion.weights.get(modality, 1.0))
            item["rrf_score"] += weight / (float(self.trake_runtime_config.fusion.rrf_k) + rank)

        query_vector = None
        settings = self.trake_runtime_config.retrieval
        if "clip" in event.modalities and settings.clip_enabled:
            if clip_results is None:
                query_vector = self._encoder().encode_text(event.visual_query or event.clip_query)
                clip_results = [(event.visual_query or event.clip_query, search_numpy_index(self.index,self.refs,query_vector,top_k=pool_size))]
            for variant_index, (variant, results) in enumerate(clip_results):
                for result in results:
                    add(
                        result.video_id,
                        result.keyframe_id,
                        "clip",
                        result.rank,
                        result.score,
                        {"rank":result.rank,"score":result.score},
                        source_key=f"clip_variant_{variant_index}",
                        query_variant=variant,
                    )
        routing_enabled = self.trake_runtime_config.features.modality_routing
        if routing_enabled and "object" in event.modalities and settings.object_enabled and self.object_service:
            for constraint in event.object_constraints:
                labels=tuple(str(value) for value in constraint.get("labels",()) if str(value))
                if not labels and constraint.get("label"): labels=(str(constraint["label"]),)
                if not labels: continue
                result=self.object_service.search(ObjectPredicate(labels,str(constraint.get("count_operator",">=")),int(constraint.get("count",1)),str(constraint.get("horizontal","any")),str(constraint.get("vertical","any"))),ObjectSearchConfig(float(constraint.get("min_confidence",.3)),float(constraint.get("nms_iou_threshold",.5))))
                for row in result["results"][:pool_size]: add(row["video_id"],row["keyframe_id"],"object",int(row["object_rank"]),float(row["object_score"]),row)
        if routing_enabled and "attribute" in event.modalities and settings.attribute_enabled and self.attribute_service.available:
            for raw_constraint in event.attribute_constraints:
                constraint=parse_color_constraint(event.clip_query,str(raw_constraint.get("color", "")),str(raw_constraint.get("filter_mode","soft")))
                if constraint is None: continue
                candidate_keys=set(fused) or None
                for row in self.attribute_service.search(constraint,candidate_keys)["results"][:pool_size]: add(row["video_id"],row["keyframe_id"],"attribute",int(row["attribute_rank"]),float(row["attribute_score"]),row)
        if self.phase5_service and self.phase5_service.available:
            if routing_enabled and "ocr" in event.modalities and settings.ocr_enabled:
                rows=self.phase5_service.search_ocr(event.clip_query,pool_size,float((event.ocr_constraints or ({},))[0].get("min_confidence",0)))
                for row in rows:
                    if row.get("keyframe_id") is not None: add(row["video_id"],int(row["keyframe_id"]),"ocr",int(row["ocr_rank"]),float(row["ocr_score"]),row)
            if routing_enabled and "asr" in event.modalities and settings.asr_enabled:
                for row in self.phase5_service.search_asr(event.clip_query,pool_size):
                    if row.get("keyframe_id") is not None: add(row["video_id"],int(row["keyframe_id"]),"asr",int(row["asr_rank"]),float(row["asr_score"]),row)
        if hybrid_plan is not None and self.hybrid_engine is not None:
            hybrid = retrieve_trake_hybrid_channels(
                self.hybrid_engine,
                hybrid_plan,
                query_id=f"{query_id}:{event.event_id}",
                groups=groups or tuple(sorted(self.config.groups)),
                top_k=pool_size,
            )
            if hybrid_failure_sink is not None and hybrid.failures:
                hybrid_failure_sink[event.event_id] = dict(hybrid.failures)
            if skipped_without_frame_sink is not None:
                skipped_without_frame_sink[event.event_id] = dict(hybrid.skipped_without_frame)
            for retriever_name, hits in hybrid.hits.items():
                seen_frames: set[tuple[str, int]] = set()
                for hit in hits:
                    frame_key = (hit.video_id, int(hit.keyframe_id))
                    if frame_key in seen_frames:
                        continue
                    seen_frames.add(frame_key)
                    add(
                        hit.video_id,
                        int(hit.keyframe_id),
                        retriever_name,
                        int(hit.rank),
                        float(hit.raw_score),
                        {**hit.to_dict(), "query_plan": hybrid_plan.to_dict()},
                    )
        fusion_started = time.perf_counter()
        ordered=sorted(fused.values(),key=lambda item:(-item["rrf_score"],item["video_id"],item["keyframe_id"]))[:pool_size]
        for rank,item in enumerate(ordered,1):
            if hybrid_plan is not None:
                method = "rank_fusion_hybrid_event_v1"
            else:
                method = "rank_fusion_existing_channels_v2" if self.trake_runtime_config.algorithm == "trake_v2" else "rank_fusion_existing_channels_v1"
            item.update({"rank":rank,"score":item["rrf_score"],"retrieval_method":method})
            for modality, raw_score in item["raw_modality_scores"].items():
                item[f"{modality}_score"] = raw_score
        if timing_sink is not None:
            timing_sink["fusion"] = timing_sink.get("fusion", 0.0) + ((time.perf_counter() - fusion_started) * 1000)
        return ordered

    def trake_plan(self, query_id: str, query: str, manual_events: list[str] | None = None, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.trake_workflow.plan(query_id, query, manual_events=manual_events, constraints=constraints)

    def trake_search(self, session_id: str, event_pool_size: int | None = None, max_per_video: int | None = None, video_pool_size: int | None = None) -> dict[str, Any]:
        return self.trake_workflow.search(session_id, event_pool_size=event_pool_size, max_per_video=max_per_video, video_pool_size=video_pool_size)

    def trake_align(self, session_id: str, top_k_per_video: int | None = None, top_videos: int | None = None) -> dict[str, Any]:
        return self.trake_workflow.align(session_id, top_k_per_video=top_k_per_video, top_videos=top_videos)

    def trake_session(self, session_id: str) -> dict[str, Any]:
        return self.trake_workflow.get(session_id)

    def trake_sessions(self) -> dict[str, Any]:
        return self.trake_workflow.list_sessions()

    def trake_update_plan(self, session_id: str, events: list[dict[str, Any]], constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.trake_workflow.update_plan(session_id, events, constraints)

    def trake_replace(self, session_id: str, chain_id: str, event_id: str, candidate_id: str, lock_event: bool = True) -> dict[str, Any]:
        return self.trake_workflow.replace_candidate(session_id, chain_id, event_id, candidate_id, lock_event=lock_event)

    def trake_review(self, session_id: str, chain_id: str, decision: str, reviewer: str) -> dict[str, Any]:
        return self.trake_workflow.review(session_id, chain_id, decision, reviewer)

    def trake_export(self, session_id: str, review_id: str) -> dict[str, Any]:
        return self.trake_workflow.export(session_id, review_id)

    def _trake_frame_scores(self, text: str, images: list[Any]) -> list[float]:
        text_vector = self._encoder().encode_text(text)
        image_vectors = self._encoder().encode_images(images)
        return [float(value) for value in image_vectors @ text_vector]

    def trake_refine(self, session_id: str, chain_id: str, event_id: str) -> dict[str, Any]:
        state = self.trake_workflow._state(session_id)
        chain = state.manual_chain if state.manual_chain and state.manual_chain.chain_id == chain_id else next((item for result in state.alignments for item in result.chains if item.chain_id == chain_id), None)
        if chain is None: raise KeyError(f"unknown TRAKE chain: {chain_id}")
        entry = next((item for item in chain.events if item.event_id == event_id), None)
        event = next((item for item in state.request.events if item.event_id == event_id), None)
        if entry is None or entry.candidate is None or event is None: raise ValueError("selected event candidate is required for refinement")
        fps = float(entry.candidate.evidence.get("fps") or entry.candidate.evidence.get("raw_result",{}).get("fps") or 25.0)
        result = self.trake_refiner.refine(video_id=chain.video_id,event_id=event_id,event_text=event.clip_query,source_keyframe_id=entry.candidate.keyframe_id,source_frame_idx=entry.candidate.frame_idx,source_pts_time=entry.candidate.pts_time,fps=fps)
        payload=result.to_dict(); payload.update({"refinement_id":f"refine-{int(time.time()*1000)}-{event_id}","session_id":session_id,"chain_id":chain_id,"created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
        if self.trake_workflow.store: self.trake_workflow.store.append_refinement(payload)
        if payload.get("refined_image_path"): payload["image_url"]="/keyframe?path="+payload["refined_image_path"]
        return {"refinement":payload}

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_VIDEOS,
        candidate_pool: int | None = None,
        max_frames_per_video: int = DEFAULT_MAX_FRAMES_PER_VIDEO,
        matched_frames_per_video: int = DEFAULT_MATCHED_FRAMES_PER_VIDEO,
        aggregation_method: str = DEFAULT_AGGREGATION_METHOD,
        mean_top_n: int = DEFAULT_MEAN_TOP_N,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        _require_positive("top_k", top_k)
        candidate_pool = candidate_pool if candidate_pool is not None else DEFAULT_CANDIDATE_POOL_SIZE
        _require_positive("candidate_pool", candidate_pool)
        _require_positive("max_frames_per_video", max_frames_per_video)
        _require_positive("matched_frames_per_video", matched_frames_per_video)
        _require_positive("mean_top_n", mean_top_n)
        top_k = min(top_k, 50)
        candidate_pool = max(candidate_pool, top_k)
        max_frames_per_video = min(max_frames_per_video, 10)
        matched_frames_per_video = min(matched_frames_per_video, 50)

        started = time.perf_counter()
        encode_started = time.perf_counter()
        query_vector = self._encoder().encode_text(query)
        encode_ms = (time.perf_counter() - encode_started) * 1000
        retrieval_started = time.perf_counter()
        raw_results = search_numpy_index(self.index, self.refs, query_vector, top_k=candidate_pool)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        aggregation_started = time.perf_counter()
        results = diversify_results_by_video(raw_results, max_frames_per_video=max_frames_per_video)[:top_k]
        video_results = aggregate_results_by_video(
            raw_results,
            max_frames_per_video=matched_frames_per_video,
            aggregation_method=aggregation_method,
            mean_top_n=mean_top_n,
        )[:top_k]
        aggregation_ms = (time.perf_counter() - aggregation_started) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "mode": "visual",
            "query": query,
            "groups": sorted(self.config.groups),
            "top_k": top_k,
            "top_k_videos": top_k,
            "candidate_pool": candidate_pool,
            "candidate_pool_size": candidate_pool,
            "max_frames_per_video": max_frames_per_video,
            "matched_frames_per_video": matched_frames_per_video,
            "aggregation_method": aggregation_method,
            "mean_top_n": mean_top_n,
            "encode_ms": round(encode_ms, 3),
            "retrieval_ms": round(retrieval_ms, 3),
            "aggregation_ms": round(aggregation_ms, 3),
            "elapsed_ms": round(elapsed_ms, 3),
            "index_vectors": int(self.index.shape[0]),
            "index_dim": int(self.index.shape[1]),
            "raw_results": [self.enrich_result(asdict(result)) for result in raw_results],
            "results": [self.enrich_result(asdict(result)) for result in results],
            "video_results": [self.enrich_video_result(asdict(result)) for result in video_results],
            "video_groups": [self.enrich_video_result(asdict(result)) for result in video_results],
        }

    def metadata_search(self, query: str, top_k: int = 12, min_match: int = 1) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("metadata query must not be empty")
        top_k = max(1, min(top_k, 50))
        min_match = max(1, min(min_match, 10))
        started = time.perf_counter()
        results = search_metadata(self.metadata_docs, query, top_k=top_k, min_match=min_match)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload_results = results_to_dict(results)
        for result in payload_results:
            result["video_url"] = self.video_url(result["video_id"])
        return {
            "mode": "metadata",
            "query": query,
            "groups": sorted(self.config.groups),
            "top_k": top_k,
            "min_match": min_match,
            "elapsed_ms": round(elapsed_ms, 3),
            "total_documents": len(self.metadata_docs),
            "results": payload_results,
        }

    def structured_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_VIDEOS,
        candidate_pool: int = 100,
        enable_clip: bool = True,
        enable_objects: bool = False,
        enable_metadata: bool = False,
        enable_attributes: bool = False,
        enable_ocr: bool = False,
        enable_asr: bool = False,
        object_label: str = "",
        object_min_count: int = 1,
        object_position: str = "any",
        object_min_confidence: float = 0.3,
        attribute_color: str = "",
        attribute_filter_mode: str = "soft",
        ocr_filter_mode: str = "soft",
        asr_filter_mode: str = "soft",
        ocr_min_confidence: float = 0.0,
        metadata_author: str | None = None,
        metadata_date: str | None = None,
        metadata_title: str | None = None,
        fusion_method: str = "rrf",
        object_filter_mode: str = "soft",
        metadata_filter_mode: str = "soft",
        matched_frames_per_video: int = DEFAULT_MATCHED_FRAMES_PER_VIDEO,
        enable_query_planner: bool = False,
        enable_reranker: bool = False,
        reranker_top_n: int = 20,
        query_variants: tuple[str, ...] | None = None,
        restrict_structured_to_clip_candidates: bool = False,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        _require_positive("top_k", top_k); _require_positive("candidate_pool", candidate_pool)
        _require_positive("object_min_count", object_min_count); _require_positive("matched_frames_per_video", matched_frames_per_video)
        if enable_reranker:
            _require_positive("reranker_top_n", reranker_top_n)
        if enable_objects and not object_label.strip():
            raise ValueError("object_label is required when objects are enabled")
        horizontal, vertical = parse_object_position(object_position)
        object_constraints = (
            ObjectConstraint((object_label.strip(),), ">=", object_min_count, horizontal, vertical, object_min_confidence, 0.5, object_filter_mode),
        ) if enable_objects else ()
        attribute_constraint = parse_color_constraint(query, attribute_color, attribute_filter_mode) if enable_attributes else None
        attribute_constraints = (attribute_constraint,) if attribute_constraint else ()
        if enable_attributes and not attribute_constraints:
            raise ValueError("attribute_color is required when attributes are enabled and no color is found in the query")
        structured_query = StructuredQuery(
            visual_text=query, enable_clip=enable_clip, enable_objects=enable_objects, enable_metadata=enable_metadata, enable_attributes=enable_attributes,
            enable_ocr=enable_ocr, enable_asr=enable_asr,
            clip_mode="soft" if enable_clip else "disabled", object_constraints=object_constraints, attribute_constraints=attribute_constraints,
            attribute_mode=attribute_filter_mode if enable_attributes else "disabled",
            metadata_constraints=MetadataConstraints(channel=metadata_author or None, publish_date=metadata_date or None, title_phrase=metadata_title or None),
            metadata_mode=metadata_filter_mode if enable_metadata else "disabled", clip_candidate_pool=max(candidate_pool, top_k), fusion_method=fusion_method,
            ocr_mode=ocr_filter_mode if enable_ocr else "disabled", asr_mode=asr_filter_mode if enable_asr else "disabled", ocr_min_confidence=ocr_min_confidence,
            restrict_structured_to_clip_candidates=restrict_structured_to_clip_candidates,
        )
        started = time.perf_counter()
        query_plan = self.query_planner.plan(query) if enable_query_planner or enable_reranker else None
        if query_plan is not None and query_variants is not None:
            query_plan = self.query_planner.select_variants(query_plan, query_variants)
        encode_started = time.perf_counter()
        query_vector = self._encoder().encode_text(query) if enable_clip or enable_metadata else None
        encode_ms = (time.perf_counter() - encode_started) * 1000
        candidate_started = time.perf_counter()
        candidate_payload = self.structured_generator.generate(structured_query, query_vector)
        candidate_ms = (time.perf_counter() - candidate_started) * 1000
        fusion_started = time.perf_counter()
        ranking_limit = min(max(top_k, reranker_top_n if enable_reranker else top_k), 50)
        ranked = rank_video_candidates(candidate_payload, structured_query, self.rrf_config, ranking_limit, min(matched_frames_per_video, 50))
        reranker_config = with_top_n(self.reranker_config, min(reranker_top_n, ranking_limit)) if enable_reranker else None
        if enable_reranker and query_plan is not None:
            assert reranker_config is not None
            ranked["video_results"] = rerank_video_results(ranked["video_results"], query_plan, reranker_config)[:top_k]
            ranked["video_groups"] = ranked["video_results"]
            ranked["results"] = [
                {**next(frame for frame in item["frames"] if frame["is_representative"]), "rank":item["rank"], "score":item["rerank_score"], "rerank_score":item["rerank_score"], "pre_rerank_rank":item["pre_rerank_rank"]}
                for item in ranked["video_results"]
            ]
        else:
            ranked["video_results"] = ranked["video_results"][:top_k]
            ranked["video_groups"] = ranked["video_results"]
            ranked["results"] = ranked["results"][:top_k]
        ranked["top_k"] = top_k
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        ranked["video_results"] = [self.enrich_video_result(item) for item in ranked["video_results"]]
        ranked["video_groups"] = ranked["video_results"]
        ranked["results"] = [self.enrich_result(item) for item in ranked["results"]]
        ranked["raw_results"] = [self.enrich_result(item) for item in ranked["raw_results"]]
        for result in ranked["video_results"]:
            result["evidence"] = self.structured_evidence(result, structured_query, ranked["fusion_config"])
        evidence_by_video = {item["video_id"]: item["evidence"] for item in ranked["video_results"]}
        for result in ranked["results"]:
            result["evidence"] = evidence_by_video[result["video_id"]]
        for result in ranked["raw_results"]:
            result["evidence"] = evidence_by_video.get(result["video_id"], result.get("evidence", {}))
        response = {
            "mode":"structured", "experimental":True, "default_search_unchanged":True, "query":query,
            "structured_query":asdict(structured_query), "candidate_pool":candidate_pool, "top_k":top_k,
            "encode_ms":round(encode_ms,3), "candidate_generation_ms":round(candidate_ms,3), "fusion_ms":round(fusion_ms,3),
            "elapsed_ms":round((time.perf_counter()-started)*1000,3), "channel_counts":candidate_payload["channel_counts"],
            "unknown_object_frame_count":candidate_payload["unknown_object_frame_count"], "unknown_attribute_frame_count":candidate_payload["unknown_attribute_frame_count"], **ranked,
        }
        if query_plan is not None:
            response["query_plan"] = query_plan.to_dict()
        if enable_reranker:
            assert reranker_config is not None
            response["reranker"] = reranker_metadata(reranker_config)
        return response

    def query_plan(self, query: str) -> dict[str, Any]:
        return self.query_planner.plan(query).to_dict()

    def agent_query_plan(self, query: str, use_gemini: bool = True) -> dict[str, Any]:
        return self.hybrid_query_planner.plan(query, use_gemini=use_gemini).to_dict()

    def agent_search(
        self,
        query_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K_VIDEOS,
        use_gemini: bool = True,
        strict: bool = False,
    ) -> dict[str, Any]:
        if self.hybrid_engine is None:
            raise ValueError("hybrid retrieval is disabled; start the UI with --enable-hybrid-retrieval")
        query_id = query_id.strip()
        if not query_id:
            raise ValueError("query_id must not be empty")
        top_k = max(1, min(top_k, 50))
        planning = self.hybrid_query_planner.plan_with_trace(query, use_gemini=use_gemini)
        plan = planning.plan
        response = self.hybrid_engine.search(
            query_id,
            plan,
            groups=tuple(sorted(self.config.groups)),
            top_k=top_k,
            strict=strict,
        )
        enriched = []
        for result in response["results"]:
            keyframe_path = self._hybrid_keyframe_path(result["video_id"], result.get("keyframe_id"))
            enriched.append(self.enrich_result({**result, "score": result["raw_score"], "keyframe_path": keyframe_path}))
        enriched, structured = self._apply_agent_structured_constraints(plan, enriched, top_k)
        failures = dict(response.get("failures", {}))
        if structured.get("error"):
            failures["structured"] = str(structured["error"])
        return {
            **response,
            "mode": "agent_hybrid",
            "agent_workspace_version": "agent-workspace-v2",
            "agent_trace": planning.trace.to_dict(),
            "default_search_unchanged": True,
            "top_k": top_k,
            "fusion_method": "rrf" if structured.get("applied") else response.get("fusion_method"),
            "failures": failures,
            "structured_constraints": structured,
            "results": enriched,
            "video_results": enriched,
        }

    def _apply_agent_structured_constraints(
        self,
        plan: HybridQueryPlan,
        hybrid_results: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        suggestions = dict(plan.structured_filter_suggestions)
        object_service = getattr(self, "object_service", None)
        attribute_service = getattr(self, "attribute_service", None)
        phase5_service = getattr(self, "phase5_service", None)
        enable_objects = bool(suggestions.get("enable_objects") and suggestions.get("object_label") and object_service)
        enable_attributes = bool(suggestions.get("enable_attributes") and suggestions.get("attribute_color") and attribute_service and attribute_service.available)
        enable_ocr = bool(suggestions.get("enable_ocr") and phase5_service and phase5_service.available)
        enable_asr = bool(suggestions.get("enable_asr") and phase5_service and phase5_service.available)
        enable_metadata = bool(
            suggestions.get("enable_metadata")
            and getattr(self, "metadata_docs", None)
            and any(suggestions.get(key) for key in ("metadata_author", "metadata_date", "metadata_title"))
        )
        enabled = {
            "objects": enable_objects,
            "attributes": enable_attributes,
            "ocr": enable_ocr,
            "asr": enable_asr,
            "metadata": enable_metadata,
        }
        if not any(enabled.values()):
            return hybrid_results, {"applied": False, "enabled": enabled, "fusion_method": "none"}
        try:
            structured = self.structured_search(
                plan.visual_clip_query_en,
                top_k=max(top_k, 30),
                candidate_pool=max(60, top_k * 2),
                enable_clip=True,
                enable_objects=enable_objects,
                enable_attributes=enable_attributes,
                enable_ocr=enable_ocr,
                enable_asr=enable_asr,
                enable_metadata=enable_metadata,
                object_label=str(suggestions.get("object_label") or ""),
                attribute_color=str(suggestions.get("attribute_color") or ""),
                metadata_author=suggestions.get("metadata_author"),
                metadata_date=suggestions.get("metadata_date"),
                metadata_title=suggestions.get("metadata_title"),
                object_filter_mode="soft",
                attribute_filter_mode="soft",
                ocr_filter_mode="soft",
                asr_filter_mode="soft",
                metadata_filter_mode="soft",
                fusion_method="rrf",
                restrict_structured_to_clip_candidates=True,
            )
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            return hybrid_results, {
                "applied": False,
                "enabled": enabled,
                "fusion_method": "none",
                "error": f"{type(exc).__name__}: {exc}",
            }

        structured_results = list(structured.get("results") or [])
        channel_counts = dict(structured.get("channel_counts") or {})
        support_count = sum(
            int(channel_counts.get(key, 0) or 0)
            for key, enabled_now in (
                ("object_frames", enable_objects),
                ("attribute_frames", enable_attributes),
                ("ocr_frames", enable_ocr),
                ("asr_segments", enable_asr),
                ("metadata_videos", enable_metadata),
            )
            if enabled_now
        )
        if support_count == 0:
            return hybrid_results, {
                "applied": False,
                "enabled": enabled,
                "fusion_method": "none",
                "structured_hit_count": 0,
                "channel_counts": channel_counts,
                "warning": "structured constraints found no supporting evidence; base retrieval was preserved",
            }
        if not structured_results:
            return hybrid_results, {
                "applied": False,
                "enabled": enabled,
                "fusion_method": "none",
                "structured_hit_count": 0,
                "warning": "structured constraints returned no supporting result; base retrieval was preserved",
            }
        by_video: dict[str, dict[str, Any]] = {}
        rank_by_video: dict[str, dict[str, int]] = {}
        for channel, rows in (("retrieval", hybrid_results), ("structured", structured_results)):
            for rank, row in enumerate(rows, 1):
                video_id = str(row.get("video_id") or "")
                if not video_id:
                    continue
                rank_by_video.setdefault(video_id, {})[channel] = rank
                by_video.setdefault(video_id, dict(row))
        ordered = sorted(
            by_video,
            key=lambda video_id: (
                -sum(1.0 / (60.0 + rank) for rank in rank_by_video[video_id].values()),
                video_id,
            ),
        )[:top_k]
        fused: list[dict[str, Any]] = []
        for rank, video_id in enumerate(ordered, 1):
            row = dict(by_video[video_id])
            channel_ranks = rank_by_video[video_id]
            score = sum(1.0 / (60.0 + value) for value in channel_ranks.values())
            raw_provenance = row.get("provenance")
            provenance = dict(raw_provenance) if isinstance(raw_provenance, dict) else {
                "structured_modalities": list(raw_provenance or []),
            }
            retriever_ranks = dict(provenance.get("retriever_ranks") or {})
            contributions = dict(provenance.get("contributions") or {})
            if "structured" in channel_ranks:
                retriever_ranks["structured"] = channel_ranks["structured"]
                contributions["structured"] = 1.0 / (60.0 + channel_ranks["structured"])
            provenance.update({
                "retriever_ranks": retriever_ranks,
                "contributions": contributions,
                "agent_constraint_fusion": "rrf-k60-v1",
            })
            row.update({"rank": rank, "score": score, "raw_score": score, "provenance": provenance})
            fused.append(row)
        return fused, {
            "applied": True,
            "enabled": enabled,
            "fusion_method": "rrf",
            "structured_hit_count": len(structured_results),
            "channel_counts": channel_counts,
        }

    def qa_prepare(
        self,
        query_id: str,
        event_query: str,
        question: str,
        selected_video_id: str | None = None,
        selected_frame_id: int | None = None,
        retrieval_query: str | None = None,
        use_hybrid_retrieval: bool = False,
        use_gemini_planner: bool = True,
    ) -> dict[str, Any]:
        request = QaRequest(query_id, event_query, question, selected_video_id, selected_frame_id)
        search_query = (retrieval_query or event_query).strip()
        if selected_video_id:
            response = self._qa_selected_response(request)
        elif use_hybrid_retrieval:
            response = self.agent_search(query_id, event_query, top_k=12, use_gemini=use_gemini_planner)
            search_query = str(response.get("query_plan", {}).get("visual_clip_query_en") or event_query)
        else:
            response = self.search(search_query, top_k=12, candidate_pool=40)
        response = {
            **response,
            "qa_event_query": event_query,
            "qa_retrieval_query": search_query,
            "qa_hybrid_retrieval": use_hybrid_retrieval,
            "index_schema_version": self.index_metadata.get("index_schema_version"),
            "index_fingerprint": self.index_metadata.get("feature_source_fingerprint"),
            "phase5_store_version": "phase5-store-v1" if getattr(self, "phase5_service", None) and self.phase5_service.available else None,
            "phase6_config_version": self.reranker_config.__class__.__name__,
        }
        state = self.qa_workflow.prepare(
            request,
            response,
            metadata_by_video=self.metadata_by_video,
            modality_availability=self.qa_modality_availability(),
            max_evidence_per_modality=8,
        )
        return self.qa_workflow.payload(state)

    def qa_draft_answer(self, session_id: str, selected_evidence_ids: tuple[str, ...], answer_method: str = "evidence_first") -> dict[str, Any]:
        answerer = None
        if answer_method == "gemini":
            gemini_config = self.translator.config
            if gemini_config.provider != "gemini":
                gemini_config = TranslationConfig(
                    api_key=gemini_config.api_key,
                    api_url=DEFAULT_GEMINI_API_URL,
                    model=gemini_config.model,
                    provider="gemini",
                    timeout_seconds=gemini_config.timeout_seconds,
                )
            answerer = GeminiEvidenceAnswerer(gemini_config)
        elif answer_method != "evidence_first":
            raise ValueError(f"unsupported Q&A answer method: {answer_method}")
        state, draft = self.qa_workflow.draft(session_id, selected_evidence_ids, answerer=answerer)
        return {**self.qa_workflow.payload(state), "answer_draft": jsonable(draft)}

    def qa_review(
        self,
        session_id: str,
        draft_id: str,
        decision: str,
        final_answer: str | None,
        selected_evidence_ids: tuple[str, ...],
        reviewer: str,
    ) -> dict[str, Any]:
        state, review = self.qa_workflow.review(
            session_id, draft_id, ReviewDecision(decision), final_answer, selected_evidence_ids, reviewer,
        )
        return {**self.qa_workflow.payload(state), "review": jsonable(review)}

    def qa_session(self, session_id: str) -> dict[str, Any]:
        return self.qa_workflow.payload(self.qa_workflow.get(session_id))

    def qa_export(self, session_id: str, review_id: str) -> dict[str, Any]:
        return {"export_record": jsonable(self.qa_workflow.create_export(session_id, review_id))}

    def qa_sessions(self) -> dict[str, Any]:
        return {"sessions": self.qa_workflow.list_sessions(), "persistence": "sqlite" if self.qa_workflow.store else "memory_until_p7_5"}

    def qa_modality_availability(self) -> dict[EvidenceModality, AvailabilityStatus]:
        return {
            EvidenceModality.KEYFRAME: AvailabilityStatus.AVAILABLE if self.refs else AvailabilityStatus.UNAVAILABLE,
            EvidenceModality.CLIP: AvailabilityStatus.AVAILABLE if len(self.index) else AvailabilityStatus.UNAVAILABLE,
            EvidenceModality.OBJECT: AvailabilityStatus.AVAILABLE if self.object_service is not None else AvailabilityStatus.UNAVAILABLE,
            EvidenceModality.METADATA: AvailabilityStatus.AVAILABLE if self.metadata_docs else AvailabilityStatus.UNAVAILABLE,
            EvidenceModality.OCR: AvailabilityStatus.AVAILABLE if getattr(self, "phase5_service", None) and self.phase5_service.available else AvailabilityStatus.UNAVAILABLE,
            EvidenceModality.ASR: AvailabilityStatus.AVAILABLE if getattr(self, "phase5_service", None) and self.phase5_service.available else AvailabilityStatus.UNAVAILABLE,
            EvidenceModality.ATTRIBUTE: AvailabilityStatus.AVAILABLE if self.attribute_service.available else AvailabilityStatus.UNAVAILABLE,
            EvidenceModality.TEMPORAL: AvailabilityStatus.AVAILABLE if self.refs_by_video else AvailabilityStatus.UNAVAILABLE,
        }

    def _qa_selected_response(self, request: QaRequest) -> dict[str, Any]:
        refs = self.refs_by_video.get(request.selected_video_id or "", [])
        ref = next((item for item in refs if request.selected_frame_id is None or item.keyframe_id == request.selected_frame_id), None)
        if ref is None:
            raise ValueError("selected video/frame is not present in the current index")
        frame = asdict(ref)
        frame["evidence"] = {"object": [], "attribute": [], "ocr": [], "asr": []}
        return {
            "mode": "selected_candidate", "query": request.event_query, "groups": sorted(self.config.groups),
            "top_k": 1, "candidate_pool": 1,
            "video_results": [{"video_id": ref.video_id, "rank": 1, "frames": [frame]}],
        }

    def structured_evidence(self, result: dict[str, Any], query: StructuredQuery, fusion_config: dict[str, Any]) -> dict[str, Any]:
        ranks = result["modality_ranks"]
        clip_enabled = query.enable_clip and query.clip_mode != "disabled"
        object_enabled = query.enable_objects and any(item.filter_mode != "disabled" for item in query.object_constraints)
        attribute_enabled = query.enable_attributes and query.attribute_mode != "disabled" and bool(query.attribute_constraints)
        metadata_enabled = query.enable_metadata and query.metadata_mode != "disabled"
        ocr_enabled = query.enable_ocr and query.ocr_mode != "disabled"
        asr_enabled = query.enable_asr and query.asr_mode != "disabled"
        video_frames = self.refs_by_video.get(result["video_id"], [])
        clip_status = "matched" if ranks["clip"] is not None else ("not_matched" if clip_enabled and video_frames else ("unknown" if clip_enabled else "disabled"))
        object_data = self.object_service.video_data_status(result["video_id"]) if object_enabled and self.object_service else {"status":"UNKNOWN","available_frames":0,"unknown_frames":len(video_frames)}
        if not object_enabled: object_status = "disabled"
        elif ranks["object"] is not None: object_status = "matched"
        elif object_data["unknown_frames"]: object_status = "unknown"
        else: object_status = "not_matched"
        metadata_known = any(doc.video_id == result["video_id"] for doc in self.metadata_docs)
        metadata_status = "matched" if ranks["metadata"] is not None else ("not_matched" if metadata_enabled and metadata_known else ("unknown" if metadata_enabled else "disabled"))
        object_matches = []
        attribute_matches = []
        ocr_matches = []
        asr_matches = []
        metadata_match = None
        clip_scores = []
        for frame in result["frames"]:
            if frame.get("clip_score") is not None: clip_scores.append(frame["clip_score"])
            object_matches.extend(frame.get("evidence", {}).get("object", []))
            attribute_matches.extend(frame.get("evidence", {}).get("attribute", []))
            ocr_matches.extend(frame.get("evidence", {}).get("ocr", []))
            asr_matches.extend(frame.get("evidence", {}).get("asr", []))
            metadata_match = metadata_match or frame.get("evidence", {}).get("metadata")
        attribute_status = "matched" if ranks["attribute"] is not None else ("not_matched" if attribute_enabled else "disabled")
        return {
            "clip":{"enabled":clip_enabled,"status":clip_status,"rank":ranks["clip"],"score":max(clip_scores) if clip_scores else None},
            "objects":{"enabled":object_enabled,"status":object_status,"rank":ranks["object"],"data_coverage":object_data,"matches":object_matches},
            "attributes":{"enabled":attribute_enabled,"status":attribute_status,"rank":ranks["attribute"],"matches":attribute_matches},
            "metadata":{"enabled":metadata_enabled,"status":metadata_status,"rank":ranks["metadata"],"level":"video","matched_fields":metadata_match["evidence"] if metadata_match else []},
            "ocr":{"enabled":ocr_enabled,"status":"matched" if ranks.get("ocr") is not None else ("not_matched" if ocr_enabled and getattr(self,"phase5_service",None) and self.phase5_service.available else ("unknown" if ocr_enabled else "disabled")),"rank":ranks.get("ocr"),"matches":ocr_matches},
            "asr":{"enabled":asr_enabled,"status":"matched" if ranks.get("asr") is not None else ("not_matched" if asr_enabled and getattr(self,"phase5_service",None) and self.phase5_service.available else ("unknown" if asr_enabled else "disabled")),"rank":ranks.get("asr"),"matches":asr_matches},
            "fusion":{"method":"rrf","config":fusion_config}, "representative_rule":result["representative_rule"],
        }

    def keyframe_neighborhood(self, video_id: str, keyframe_id: int, radius: int = 3) -> dict[str, Any]:
        video_id = video_id.strip()
        if not video_id:
            raise ValueError("video_id must not be empty")
        _require_positive("radius", radius)
        radius = min(radius, 12)
        refs = self.refs_by_video.get(video_id)
        if not refs:
            raise ValueError(f"video_id not found in index: {video_id}")

        center_index = None
        for index, ref in enumerate(refs):
            if ref.keyframe_id == keyframe_id:
                center_index = index
                break
        if center_index is None:
            raise ValueError(f"{video_id} has no keyframe_id {keyframe_id}")

        start = max(0, center_index - radius)
        end = min(len(refs), center_index + radius + 1)
        frames = []
        for ref in refs[start:end]:
            item = asdict(ref)
            item["is_center"] = ref.keyframe_id == keyframe_id
            item["image_url"] = f"/keyframe?path={ref.keyframe_path}" if ref.keyframe_path else None
            phase5_service = getattr(self, "phase5_service", None)
            item["phase5_evidence"] = phase5_service.evidence_for_frame(video_id, ref.keyframe_id) if phase5_service else {"ocr": [], "asr": []}
            frames.append(item)

        return {
            "video_id": video_id,
            "keyframe_id": keyframe_id,
            "radius": radius,
            "ordering": "pts_time_ascending",
            "total_frames": len(refs),
            "start_keyframe_id": frames[0]["keyframe_id"] if frames else None,
            "end_keyframe_id": frames[-1]["keyframe_id"] if frames else None,
            "frames": frames,
        }

    def translate(self, text: str) -> dict[str, Any]:
        source_text = text.strip()
        if not source_text:
            raise ValueError("translation query must not be empty")
        translated = self.translator.translate_vi_to_en(source_text)
        return {
            "source_text": source_text,
            "translated_text": translated,
            "provider": self.translator.config.provider,
            "model": self.translator.config.model,
        }

    def enrich_result(self, result: dict[str, Any]) -> dict[str, Any]:
        video_id = result["video_id"]
        metadata = self.video_metadata(video_id)
        keyframe_path = result.get("keyframe_path")
        result["image_url"] = f"/keyframe?path={keyframe_path}" if keyframe_path else None
        result["video_url"] = self.video_url(video_id)
        result["metadata"] = {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "publish_date": metadata.get("publish_date", ""),
            "watch_url": metadata.get("watch_url", ""),
            "keywords": (metadata.get("keywords") or [])[:8],
        }
        return result

    def enrich_video_result(self, result: dict[str, Any]) -> dict[str, Any]:
        video_id = result["video_id"]
        keyframe_path = result.get("best_keyframe_path")
        result["image_url"] = f"/keyframe?path={keyframe_path}" if keyframe_path else None
        result["video_url"] = self.video_url(video_id)
        result["metadata"] = self.enrich_result(
            {
                "video_id": video_id,
                "keyframe_path": keyframe_path,
            }
        )["metadata"]
        for frame in result.get("frames", []):
            frame_path = frame.get("keyframe_path")
            frame["image_url"] = f"/keyframe?path={frame_path}" if frame_path else None
            frame["is_representative"] = (
                frame.get("keyframe_id") == result.get("best_keyframe_id")
                and frame.get("frame_idx") == result.get("best_frame_idx")
            )
        return result

    def video_url(self, video_id: str) -> str | None:
        asset = self.assets_by_video.get(video_id, {})
        return f"/video?video_id={video_id}" if asset.get("video_path") else None

    def video_metadata(self, video_id: str) -> dict[str, Any]:
        if video_id in self.metadata_by_video:
            return self.metadata_by_video[video_id]
        asset = self.assets_by_video.get(video_id, {})
        metadata_path = asset.get("media_info_path")
        payload: dict[str, Any] = {}
        if metadata_path:
            path = self.config.repo_root / metadata_path
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
        self.metadata_by_video[video_id] = payload
        return payload

    def resolve_keyframe_path(self, value: str) -> Path:
        if not value:
            raise FileNotFoundError("empty keyframe path")
        path = self._resolve_media_path(value)
        if not path.is_file():
            raise FileNotFoundError(value)
        return path

    def resolve_video_path(self, video_id: str) -> Path:
        video_id = video_id.strip()
        if not video_id:
            raise FileNotFoundError("empty video_id")
        asset = self.assets_by_video.get(video_id)
        if not asset or not asset.get("video_path"):
            raise FileNotFoundError(video_id)
        path = self._resolve_media_path(str(asset["video_path"]))
        if not path.is_file():
            raise FileNotFoundError(video_id)
        return path

    def _resolve_media_path(self, value: str) -> Path:
        raw_path = Path(value)
        path = (raw_path if raw_path.is_absolute() else self.config.repo_root / raw_path).resolve()
        registry = getattr(self, "registry", {})
        raw_data_root = Path(str(registry.get("data_root") or "data"))
        data_root = (
            raw_data_root if raw_data_root.is_absolute() else self.config.repo_root / raw_data_root
        ).resolve()
        allowed_roots = (self.config.repo_root.resolve(), data_root)
        for allowed_root in allowed_roots:
            try:
                path.relative_to(allowed_root)
                return path
            except ValueError:
                continue
        raise PermissionError("media path is outside the repository and configured data root")

    def _encoder(self) -> ClipTextEncoder:
        if self.encoder is None:
            self.encoder = ClipTextEncoder(
                model_id=self.config.clip_model_id,
                cache_dir=self.config.clip_cache_dir,
                local_files_only=self.config.clip_local_files_only,
            )
        return self.encoder

    def _hybrid_keyframe_path(self, video_id: str, keyframe_id: int | None) -> str | None:
        if keyframe_id is None:
            return None
        ref = next((item for item in self.refs_by_video.get(video_id, ()) if item.keyframe_id == keyframe_id), None)
        return ref.keyframe_path if ref else None

    def _build_hybrid_engine(self) -> HybridRetrievalEngine:
        factories = {
            "clip": lambda: ClipRetriever(
                self.config.repo_root,
                self.config.index_dir,
                self.config.registry_path,
                self._encoder(),
                groups=tuple(sorted(self.config.groups)),
                require_keyframes=self.config.require_keyframes,
                allow_stale_index=self.config.allow_stale_index,
            )
        }
        if self.config.bm25_index_dir and (self.config.bm25_index_dir / "manifest.json").is_file():
            bm25_index_dir = self.config.bm25_index_dir
            factories["bm25"] = lambda: Bm25Retriever(self.config.repo_root, bm25_index_dir)
        if self.config.bge_index_dir and (self.config.bge_index_dir / "manifest.json").is_file():
            bge_index_dir = self.config.bge_index_dir
            factories["bge"] = lambda: BgeRetriever(
                self.config.repo_root,
                bge_index_dir,
                BgeEncoder(
                    self.config.bge_model_id,
                    model_path=self.config.bge_model_path,
                    local_files_only=True,
                    device=self.config.bge_device,
                ),
            )
        return HybridRetrievalEngine(factories, RrfFusionConfig(rrf_k=60, candidate_pool=200))

    def submission_start(self) -> dict[str, Any]:
        return self._submission_store().start()

    def submission_session(self, session_id: str) -> dict[str, Any]:
        return self._submission_store().get(session_id)

    def submission_agent_run(
        self,
        session_id: str,
        query_id: str,
        task: str,
        query: str,
        *,
        use_hybrid: bool = True,
        use_gemini: bool = True,
    ) -> dict[str, Any]:
        session = self._submission_store().get(session_id)
        if session["status"] != "ACTIVE":
            raise ValueError("submission session is not ACTIVE")
        normalized_task = task.strip().upper()
        if normalized_task != "KIS":
            raise ValueError("submission Agent run currently supports KIS; Q&A and TRAKE require their reviewed workflow adapters")
        if use_hybrid and self.hybrid_engine is not None:
            response = self.agent_search(query_id, query, top_k=30, use_gemini=use_gemini)
        else:
            response = self.search(query, top_k=30, candidate_pool=100)
        candidates = []
        for result in response.get("video_results") or response.get("results") or []:
            keyframe_id = result.get("keyframe_id", result.get("best_keyframe_id"))
            frame_idx = result.get("frame_idx", result.get("best_frame_idx"))
            pts_time = result.get("pts_time", result.get("best_pts_time"))
            keyframe_path = result.get("keyframe_path", result.get("best_keyframe_path"))
            candidates.append({
                "candidate_id": f"{query_id}:r{result.get('rank', len(candidates) + 1)}",
                "rank": result.get("rank", len(candidates) + 1),
                "video_id": result.get("video_id"),
                "keyframe_id": keyframe_id,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "image_url": result.get("image_url") or (f"/keyframe?path={keyframe_path}" if keyframe_path else None),
                "video_url": result.get("video_url"),
                "metadata": result.get("metadata", {}),
                "source_type": result.get("source_type"),
                "matched_text": result.get("matched_text"),
                "official_frame_id": None,
                "mapping_status": "REQUIRES_MANUAL_OFFICIAL_FRAME_ID",
                "retrieval": {
                    "score": result.get("score", result.get("raw_score")),
                    "provenance": result.get("provenance", {}),
                },
            })
        return {
            "schema_version": "submission-agent-result-v1",
            "session_id": session_id,
            "query_id": query_id,
            "task": "KIS",
            "query": query,
            "status": "READY_FOR_REVIEW" if candidates else "NO_CANDIDATES",
            "mapping_warning": "frame_idx/keyframe_id are retrieval diagnostics, not confirmed official frame_id values",
            "agent_plan": response.get("query_plan"),
            "agent_trace": response.get("agent_trace"),
            "fusion_method": response.get("fusion_method"),
            "structured_constraints": response.get("structured_constraints", {"applied": False}),
            "channel_hit_counts": response.get("channel_hit_counts", {}),
            "latency_ms": response.get("latency_ms", {}),
            "health": response.get("health", {}),
            "failures": response.get("failures", {}),
            "candidates": candidates,
        }

    def submission_confirm(
        self,
        session_id: str,
        query_id: str,
        task: str,
        predictions: list[dict[str, Any]],
        *,
        event_count: int | None = None,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        official_predictions = []
        mapping_sources = []
        for row in predictions:
            frame_values = row.get("frame_ids")
            if frame_values is None and row.get("frame_id") is not None:
                frame_values = [row["frame_id"]]
            if not isinstance(frame_values, list):
                raise ValueError("prediction frame_ids must be a list")
            official_predictions.append(OfficialPrediction(
                str(row.get("video_id") or ""),
                tuple(parse_official_frame_id(value) for value in frame_values),
                str(row["answer"]) if row.get("answer") is not None else None,
            ))
            mapping_sources.append(str(row.get("mapping_source") or ""))
        official_query = OfficialQuery(query_id, task, tuple(official_predictions), event_count)
        return self._submission_store().confirm_query(
            session_id,
            official_query,
            mapping_sources=tuple(mapping_sources),
            source=source,
        )

    def submission_remove(self, session_id: str, query_id: str) -> dict[str, Any]:
        return self._submission_store().remove_query(session_id, query_id)

    def submission_import_qa(
        self,
        submission_session_id: str,
        qa_session_id: str,
        review_id: str,
        official_frame_id: int,
        mapping_source: str,
    ) -> dict[str, Any]:
        state = self.qa_workflow.get(qa_session_id)
        review = next((item for item in state.reviews if item.review_id == review_id), None)
        if review is None:
            raise ValueError("Q&A review_id was not created in this session")
        if review.decision not in {ReviewDecision.CONFIRMED, ReviewDecision.EDITED}:
            raise ValueError("Q&A submission import requires a confirmed or edited review")
        if not review.final_answer:
            raise ValueError("Q&A confirmed review has no final answer")
        evidence_by_id = {item.evidence_id: item for item in state.pack.evidence_refs}
        frame_ref = next(
            (
                evidence_by_id[item]
                for item in review.selected_evidence_refs
                if item in evidence_by_id and evidence_by_id[item].keyframe_id is not None
            ),
            None,
        )
        if frame_ref is None:
            raise ValueError("Q&A submission import requires selected frame-level evidence")
        query = OfficialQuery(
            state.session.request.query_id,
            "QA",
            (OfficialPrediction(frame_ref.video_id, (official_frame_id,), review.final_answer),),
        )
        return self._submission_store().confirm_query(
            submission_session_id,
            query,
            mapping_sources=(mapping_source,),
            source={
                "workflow": "phase7_qa",
                "qa_session_id": qa_session_id,
                "review_id": review_id,
                "selected_evidence_ids": list(review.selected_evidence_refs),
            },
        )

    def submission_import_trake(
        self,
        submission_session_id: str,
        trake_session_id: str,
        review_id: str,
        official_frame_ids: list[int],
        mapping_source: str,
    ) -> dict[str, Any]:
        internal = self.trake_workflow.export(trake_session_id, review_id)["export"]
        events = list(internal.get("events") or [])
        if not internal.get("human_confirmed") or not internal.get("same_video") or not internal.get("temporally_ordered"):
            raise ValueError("TRAKE submission import requires a confirmed valid same-video chain")
        if len(official_frame_ids) != len(events):
            raise ValueError(f"TRAKE requires exactly {len(events)} official frame_ids")
        query = OfficialQuery(
            str(internal["query_id"]),
            "TRAKE",
            (OfficialPrediction(str(internal["video_id"]), tuple(official_frame_ids)),),
            event_count=len(events),
        )
        return self._submission_store().confirm_query(
            submission_session_id,
            query,
            mapping_sources=(mapping_source,),
            source={
                "workflow": "phase8_trake",
                "trake_session_id": trake_session_id,
                "review_id": review_id,
                "event_ids": [item.get("event_id") for item in events],
            },
        )

    def submission_validate(self, session_id: str) -> dict[str, Any]:
        return self._submission_store().validate(session_id)

    def submission_done(self, session_id: str) -> dict[str, Any]:
        return self._submission_store().done(session_id)

    def submission_zip_path(self, session_id: str) -> Path:
        return self._submission_store().zip_path(session_id)

    def _submission_store(self) -> SubmissionSessionStore:
        if self.submission_store is None:
            raise ValueError("submission persistence is not configured")
        return self.submission_store


def run_server(config: RetrievalUiConfig, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    service = RetrievalUiService(config)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AICRetrievalUI/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._serve_static("index.html")
            elif parsed.path == "/submission" or parsed.path == "/submission/":
                self._serve_submission_static("index.html")
            elif parsed.path.startswith("/submission/static/"):
                self._serve_submission_static(parsed.path.removeprefix("/submission/static/"))
            elif parsed.path == "/trake" or parsed.path == "/trake/":
                self._serve_trake_static("index.html")
            elif parsed.path.startswith("/trake/static/"):
                self._serve_trake_static(parsed.path.removeprefix("/trake/static/"))
            elif parsed.path.startswith("/static/"):
                self._serve_static(parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/health":
                self._json(
                    {
                        "ok": True,
                        "groups": sorted(service.config.groups),
                        "index_vectors": int(service.index.shape[0]),
                        "index_dim": int(service.index.shape[1]),
                        "metadata_documents": len(service.metadata_docs),
                        "translation_configured": service.translator.is_configured,
                        "translation_provider": service.translator.config.provider,
                        "translation_model": service.translator.config.model,
                        "default_ranking_mode": "video",
                        "default_aggregation_method": DEFAULT_AGGREGATION_METHOD,
                        "default_candidate_pool_size": DEFAULT_CANDIDATE_POOL_SIZE,
                        "structured_search_available": service.object_service is not None,
                        "attribute_search_available": service.attribute_service.available,
                        "ocr_search_available": bool(getattr(service,"phase5_service",None) and service.phase5_service.available),
                        "asr_search_available": bool(getattr(service,"phase5_service",None) and service.phase5_service.available),
                        "query_planner_available": True,
                        "hybrid_retrieval_enabled": service.hybrid_engine is not None,
                        "hybrid_planner_gemini_configured": service.hybrid_query_planner.gemini_configured,
                        "hybrid_retrievers_configured": sorted(service.hybrid_engine.factories) if service.hybrid_engine else [],
                        "reranker_available": True,
                        "qa_available": hasattr(service, "qa_prepare"),
                        "qa_persistence": "sqlite" if getattr(service, "qa_workflow", None) and service.qa_workflow.store else "memory_until_p7_5",
                        "trake_available": hasattr(service, "trake_plan"),
                        "trake_scope": "L21",
                        "trake_vlm_enabled": bool(getattr(getattr(service, "trake_runtime_config", None), "vlm", None) and service.trake_runtime_config.vlm.enabled),
                        "submission_available": getattr(service, "submission_store", None) is not None,
                    }
                )
            elif parsed.path.startswith("/api/submission/session/") and parsed.path.endswith("/download"):
                session_id = parsed.path.removeprefix("/api/submission/session/").removesuffix("/download").strip("/")
                self._serve_submission_zip(session_id)
            elif parsed.path.startswith("/api/submission/session/"):
                session_id = parsed.path.removeprefix("/api/submission/session/").strip("/")
                try:
                    self._json(service.submission_session(session_id))
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                except ValueError as exc:
                    self._error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/api/search":
                self._handle_search(parsed.query)
            elif parsed.path == "/api/structured-search":
                self._handle_structured_search(parsed.query)
            elif parsed.path == "/api/metadata-search":
                self._handle_metadata_search(parsed.query)
            elif parsed.path == "/api/neighborhood":
                self._handle_neighborhood(parsed.query)
            elif parsed.path == "/api/translate":
                self._handle_translate(parsed.query)
            elif parsed.path == "/api/query-plan":
                self._handle_query_plan(parsed.query)
            elif parsed.path == "/api/agent-plan":
                self._handle_agent_plan(parsed.query)
            elif parsed.path == "/api/agent-search":
                self._handle_agent_search(parsed.query)
            elif parsed.path.startswith("/api/qa/session/"):
                self._handle_qa_session(parsed.path.removeprefix("/api/qa/session/"))
            elif parsed.path == "/api/qa/sessions":
                self._handle_qa_sessions()
            elif parsed.path == "/api/trake/health":
                runtime = getattr(service, "trake_runtime_config", None)
                self._json({
                    "ok": True,
                    "available": hasattr(service, "trake_plan"),
                    "group": "L21",
                    "vlm_enabled": bool(runtime and runtime.vlm.enabled),
                    "hybrid_retrieval_enabled": getattr(service, "hybrid_engine", None) is not None,
                    "hybrid_retrievers": sorted(getattr(getattr(service, "hybrid_engine", None), "factories", {})),
                    "gemini_configured": bool(getattr(getattr(service, "translator", None), "is_configured", False)),
                    "algorithm": getattr(runtime, "algorithm", None),
                    "config_fingerprint": getattr(runtime, "fingerprint", None),
                })
            elif parsed.path.startswith("/api/trake/session/"):
                self._handle_trake_session(parsed.path.removeprefix("/api/trake/session/"))
            elif parsed.path == "/api/trake/sessions":
                self._json(service.trake_sessions())
            elif parsed.path == "/keyframe":
                self._serve_keyframe(parsed.query)
            elif parsed.path == "/video":
                self._serve_video(parsed.query)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:
            try:
                payload = self._request_json()
                if self.path == "/api/submission/session/start":
                    result = service.submission_start()
                elif self.path == "/api/submission/agent/run":
                    result = service.submission_agent_run(
                        str(payload.get("session_id") or ""),
                        str(payload.get("query_id") or ""),
                        str(payload.get("task") or ""),
                        str(payload.get("query") or ""),
                        use_hybrid=bool(payload.get("use_hybrid", True)),
                        use_gemini=bool(payload.get("use_gemini", True)),
                    )
                elif self.path in {"/api/submission/query/confirm", "/api/submission/query/update"}:
                    result = service.submission_confirm(
                        str(payload.get("session_id") or ""),
                        str(payload.get("query_id") or ""),
                        str(payload.get("task") or ""),
                        list(payload.get("predictions") or []),
                        event_count=int(payload["event_count"]) if payload.get("event_count") is not None else None,
                        source=dict(payload.get("source") or {}),
                    )
                elif self.path == "/api/submission/query/remove":
                    result = service.submission_remove(str(payload.get("session_id") or ""), str(payload.get("query_id") or ""))
                elif self.path == "/api/submission/import/qa":
                    if payload.get("official_frame_id") is None:
                        raise ValueError("official_frame_id is required")
                    result = service.submission_import_qa(
                        str(payload.get("session_id") or ""),
                        str(payload.get("qa_session_id") or ""),
                        str(payload.get("review_id") or ""),
                        parse_official_frame_id(payload.get("official_frame_id")),
                        str(payload.get("mapping_source") or ""),
                    )
                elif self.path == "/api/submission/import/trake":
                    result = service.submission_import_trake(
                        str(payload.get("session_id") or ""),
                        str(payload.get("trake_session_id") or ""),
                        str(payload.get("review_id") or ""),
                        [parse_official_frame_id(value) for value in payload.get("official_frame_ids") or []],
                        str(payload.get("mapping_source") or ""),
                    )
                elif self.path == "/api/submission/session/validate":
                    result = service.submission_validate(str(payload.get("session_id") or ""))
                elif self.path == "/api/submission/session/done":
                    result = service.submission_done(str(payload.get("session_id") or ""))
                elif self.path == "/api/qa/prepare":
                    result = service.qa_prepare(
                        str(payload.get("query_id") or f"qa-{int(time.time() * 1000)}"),
                        str(payload.get("event_query") or ""),
                        str(payload.get("question") or ""),
                        str(payload["selected_video_id"]) if payload.get("selected_video_id") else None,
                        int(payload["selected_frame_id"]) if payload.get("selected_frame_id") is not None else None,
                        str(payload.get("retrieval_query") or "") or None,
                        bool(payload.get("use_hybrid_retrieval", False)),
                        bool(payload.get("use_gemini_planner", True)),
                    )
                elif self.path == "/api/qa/draft-answer":
                    result = service.qa_draft_answer(
                        str(payload.get("session_id") or ""),
                        tuple(str(item) for item in payload.get("selected_evidence_ids", []) if str(item)),
                        str(payload.get("answer_method") or "evidence_first"),
                    )
                elif self.path == "/api/qa/review":
                    result = service.qa_review(
                        str(payload.get("session_id") or ""), str(payload.get("draft_id") or ""),
                        str(payload.get("decision") or ""),
                        str(payload["final_answer"]) if payload.get("final_answer") is not None else None,
                        tuple(str(item) for item in payload.get("selected_evidence_ids", []) if str(item)),
                        str(payload.get("reviewer") or "local-reviewer"),
                    )
                elif self.path == "/api/qa/export":
                    result = service.qa_export(str(payload.get("session_id") or ""), str(payload.get("review_id") or ""))
                elif self.path == "/api/trake/plan":
                    result = service.trake_plan(
                        str(payload.get("query_id") or f"trake-{int(time.time() * 1000)}"),
                        str(payload.get("query") or ""),
                        [str(item) for item in payload["manual_events"]] if payload.get("manual_events") is not None else None,
                        dict(payload.get("constraints") or {}),
                    )
                elif self.path == "/api/trake/search":
                    result = service.trake_search(
                        str(payload.get("session_id") or ""),
                        int(payload["event_pool_size"]) if payload.get("event_pool_size") is not None else None,
                        int(payload["max_per_video"]) if payload.get("max_per_video") is not None else None,
                        int(payload["video_pool_size"]) if payload.get("video_pool_size") is not None else None,
                    )
                elif self.path == "/api/trake/align":
                    result = service.trake_align(
                        str(payload.get("session_id") or ""),
                        int(payload["top_k_per_video"]) if payload.get("top_k_per_video") is not None else None,
                        int(payload["top_videos"]) if payload.get("top_videos") is not None else None,
                    )
                elif self.path == "/api/trake/update-plan":
                    result = service.trake_update_plan(str(payload.get("session_id") or ""), list(payload.get("events") or []), dict(payload["constraints"]) if payload.get("constraints") is not None else None)
                elif self.path == "/api/trake/replace":
                    result = service.trake_replace(str(payload.get("session_id") or ""), str(payload.get("chain_id") or ""), str(payload.get("event_id") or ""), str(payload.get("candidate_id") or ""), bool(payload.get("lock_event", True)))
                elif self.path == "/api/trake/review":
                    result = service.trake_review(str(payload.get("session_id") or ""), str(payload.get("chain_id") or ""), str(payload.get("decision") or ""), str(payload.get("reviewer") or "local-reviewer"))
                elif self.path == "/api/trake/export":
                    result = service.trake_export(str(payload.get("session_id") or ""), str(payload.get("review_id") or ""))
                elif self.path == "/api/trake/refine":
                    result = service.trake_refine(str(payload.get("session_id") or ""), str(payload.get("chain_id") or ""), str(payload.get("event_id") or ""))
                elif self.path == "/api/trake/save":
                    result = service.trake_session(str(payload.get("session_id") or ""))
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
            except TrakeDomainError as exc:
                self._domain_error(HTTPStatus.UNPROCESSABLE_ENTITY, exc.code, exc.user_message, exc.stage)
                return
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
                return
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception:
                traceback.print_exc()
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")
                return
            self._json(result)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _handle_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.search(
                    first(params, "q"),
                    top_k=parse_int(
                        first(params, "top_k", str(DEFAULT_TOP_K_VIDEOS)),
                        DEFAULT_TOP_K_VIDEOS,
                    ),
                    candidate_pool=parse_int(
                        first(params, "candidate_pool", str(DEFAULT_CANDIDATE_POOL_SIZE)),
                        DEFAULT_CANDIDATE_POOL_SIZE,
                    ),
                    max_frames_per_video=parse_int(
                        first(params, "max_frames_per_video", str(DEFAULT_MAX_FRAMES_PER_VIDEO)),
                        DEFAULT_MAX_FRAMES_PER_VIDEO,
                    ),
                    matched_frames_per_video=parse_int(
                        first(
                            params,
                            "matched_frames_per_video",
                            str(DEFAULT_MATCHED_FRAMES_PER_VIDEO),
                        ),
                        DEFAULT_MATCHED_FRAMES_PER_VIDEO,
                    ),
                    aggregation_method=first(
                        params, "aggregation_method", DEFAULT_AGGREGATION_METHOD
                    ),
                    mean_top_n=parse_int(
                        first(params, "mean_top_n", str(DEFAULT_MEAN_TOP_N)),
                        DEFAULT_MEAN_TOP_N,
                    ),
                )
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _handle_metadata_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.metadata_search(
                    first(params, "q"),
                    top_k=parse_int(first(params, "top_k", "12"), 12),
                    min_match=parse_int(first(params, "min_match", "1"), 1),
                )
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _handle_structured_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.structured_search(
                    first(params,"q"), top_k=parse_int(first(params,"top_k","12"),12), candidate_pool=parse_int(first(params,"candidate_pool","100"),100),
                    enable_clip=parse_bool(first(params,"enable_clip","true")), enable_objects=parse_bool(first(params,"enable_objects","false")),
                    enable_metadata=parse_bool(first(params,"enable_metadata","false")), enable_attributes=parse_bool(first(params,"enable_attributes","false")), object_label=first(params,"object_label"),
                    enable_ocr=parse_bool(first(params,"enable_ocr","false")), enable_asr=parse_bool(first(params,"enable_asr","false")),
                    object_min_count=parse_int(first(params,"object_min_count","1"),1), object_position=first(params,"object_position","any"),
                    object_min_confidence=parse_float(first(params,"object_min_confidence","0.3"),0.3), attribute_color=first(params,"attribute_color"),
                    attribute_filter_mode=first(params,"attribute_filter_mode","soft"), metadata_author=first(params,"metadata_author") or None,
                    metadata_date=first(params,"metadata_date") or None, metadata_title=first(params,"metadata_title") or None,
                    fusion_method=first(params,"fusion_method","rrf"), object_filter_mode=first(params,"object_filter_mode","soft"),
                    metadata_filter_mode=first(params,"metadata_filter_mode","soft"), matched_frames_per_video=parse_int(first(params,"matched_frames_per_video","5"),5),
                    ocr_filter_mode=first(params,"ocr_filter_mode","soft"), asr_filter_mode=first(params,"asr_filter_mode","soft"), ocr_min_confidence=parse_float(first(params,"ocr_min_confidence","0"),0),
                    enable_query_planner=parse_bool(first(params,"enable_query_planner","false")), enable_reranker=parse_bool(first(params,"enable_reranker","false")), reranker_top_n=parse_int(first(params,"reranker_top_n","20"),20),
                    query_variants=tuple(item for item in first(params,"query_variants").split("||") if item) if "query_variants" in params else None,
                )
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc)); return
            self._json(payload)

        def _handle_query_plan(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.query_plan(first(params, "q"))
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc)); return
            self._json(payload)

        def _handle_agent_plan(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.agent_query_plan(
                    first(params, "q"),
                    use_gemini=parse_bool(first(params, "use_gemini", "true")),
                )
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc)); return
            self._json(payload)

        def _handle_agent_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.agent_search(
                    first(params, "query_id"),
                    first(params, "q"),
                    top_k=parse_int(first(params, "top_k", str(DEFAULT_TOP_K_VIDEOS)), DEFAULT_TOP_K_VIDEOS),
                    use_gemini=parse_bool(first(params, "use_gemini", "true")),
                    strict=parse_bool(first(params, "strict", "false")),
                )
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc)); return
            self._json(payload)

        def _handle_qa_session(self, session_id: str) -> None:
            try:
                payload = service.qa_session(session_id)
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc)); return
            self._json(payload)

        def _handle_qa_sessions(self) -> None:
            try:
                payload = service.qa_sessions()
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc)); return
            self._json(payload)

        def _handle_trake_session(self, session_id: str) -> None:
            try:
                payload = service.trake_session(session_id)
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc)); return
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc)); return
            except Exception:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error"); return
            self._json(payload)

        def _handle_neighborhood(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.keyframe_neighborhood(
                    first(params, "video_id"),
                    keyframe_id=parse_int(first(params, "keyframe_id", "0"), 0),
                    radius=parse_int(first(params, "radius", "3"), 3),
                )
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _handle_translate(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.translate(first(params, "q"))
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _serve_keyframe(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                path = service.resolve_keyframe_path(first(params, "path"))
            except PermissionError as exc:
                self._error(HTTPStatus.FORBIDDEN, str(exc))
                return
            except FileNotFoundError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

        def _serve_video(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                path = service.resolve_video_path(first(params, "video_id"))
            except PermissionError as exc:
                self._error(HTTPStatus.FORBIDDEN, str(exc))
                return
            except FileNotFoundError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
                return
            self._serve_binary_file(path, supports_range=True)

        def _serve_static(self, relative_path: str) -> None:
            path = (config.static_dir / relative_path).resolve()
            try:
                path.relative_to(config.static_dir.resolve())
            except ValueError:
                self._error(HTTPStatus.FORBIDDEN, "static path is outside static dir")
                return
            if not path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "static file not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

        def _serve_trake_static(self, relative_path: str) -> None:
            static_dir = config.trake_static_dir or (config.repo_root / "web" / "trake_ui")
            path = (static_dir / relative_path).resolve()
            try: path.relative_to(static_dir.resolve())
            except ValueError:
                self._error(HTTPStatus.FORBIDDEN, "TRAKE static path is outside static dir"); return
            if not path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "TRAKE static file not found"); return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _serve_submission_static(self, relative_path: str) -> None:
            static_dir = config.submission_static_dir or (config.repo_root / "web" / "submission_ui")
            path = (static_dir / relative_path).resolve()
            try:
                path.relative_to(static_dir.resolve())
            except ValueError:
                self._error(HTTPStatus.FORBIDDEN, "submission static path is outside static dir")
                return
            if not path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "submission static file not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_submission_zip(self, session_id: str) -> None:
            try:
                path = service.submission_zip_path(session_id)
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
                return
            except (ValueError, FileNotFoundError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="submission.zip"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_binary_file(self, path: Path, supports_range: bool = False) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            file_size = path.stat().st_size
            range_header = self.headers.get("Range") if supports_range else None
            if range_header and range_header.startswith("bytes="):
                start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
                try:
                    start = int(start_text) if start_text else 0
                    end = int(end_text) if end_text else file_size - 1
                except ValueError:
                    self._error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "invalid range")
                    return
                start = max(0, start)
                end = min(file_size - 1, end)
                if start > end:
                    self._error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "invalid range")
                    return
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.end_headers()
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            if supports_range:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

        def _json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _request_json(self) -> dict[str, Any]:
            content_length = parse_int(self.headers.get("Content-Length", "0"), 0)
            if content_length <= 0:
                raise ValueError("JSON request body is required")
            if content_length > 1_000_000:
                raise ValueError("JSON request body is too large")
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("request body must be valid UTF-8 JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON request must be an object")
            return payload

        def _error(self, status: HTTPStatus, message: str) -> None:
            body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _domain_error(self, status: HTTPStatus, code: str, message: str, stage: str) -> None:
            body = json.dumps({"ok": False, "error": message, "code": code, "failure_stage": stage}, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def first(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key)
    return values[0] if values else default


def parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def parse_official_frame_id(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("official frame_id must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("official frame_id must be a non-negative integer") from exc
    if parsed < 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError("official frame_id must be a non-negative integer")
    return parsed


def parse_float(value: str, default: float) -> float:
    try: return float(value)
    except ValueError: return default


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1","true","yes","on"}: return True
    if normalized in {"0","false","no","off"}: return False
    raise ValueError(f"invalid boolean: {value}")


def _sqlite_metadata_value(path: Path | None, key: str) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None
    except sqlite3.Error:
        return None


def _sqlite_row_count(path: Path | None, table: str) -> int:
    if path is None or not path.is_file() or table not in {"ocr", "asr"}:
        return 0
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def parse_object_position(value: str) -> tuple[str, str]:
    value = value.strip().lower() or "any"
    if value in {"any","left","center","right"}: return value, "any"
    if value in {"top","middle","bottom"}: return "any", value
    if ":" in value:
        horizontal, vertical = value.split(":",1)
        if horizontal in {"any","left","center","right"} and vertical in {"any","top","middle","bottom"}: return horizontal, vertical
    raise ValueError(f"invalid object_position: {value}")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
