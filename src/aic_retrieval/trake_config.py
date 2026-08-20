"""Typed, fingerprinted runtime configuration for Phase-8 TRAKE.

The legacy defaults intentionally reproduce the pre-V2 Python defaults.  A
V2 JSON config must be loaded explicitly by the runtime/CLI so configuration
changes are observable, persistable and testable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aic_retrieval.provenance import fingerprint_json
from aic_retrieval.trake_refinement import RefinementConfig
from aic_retrieval.trake_scoring import ScoreConfig


CONFIG_SCHEMA_VERSION = "trake-v2"
SUPPORTED_ALGORITHMS = {"legacy", "trake_v2"}


@dataclass(frozen=True)
class RetrievalSettings:
    clip_enabled: bool = True
    siglip_enabled: bool = False
    object_enabled: bool = True
    attribute_enabled: bool = True
    ocr_enabled: bool = False
    asr_enabled: bool = True
    metadata_enabled: bool = True
    adaptive_topk: bool = True
    topk_min: int = 50
    topk_default: int = 100
    topk_max: int = 200
    max_candidates_per_video_event: int = 12
    adaptive_margin_low: float = 0.05
    adaptive_margin_high: float = 0.20
    generic_token_threshold: int = 3
    text_embedding_cache_size: int = 2048

    def __post_init__(self) -> None:
        if not 1 <= self.topk_min <= self.topk_default <= self.topk_max <= 1000:
            raise ValueError("retrieval Top-K must satisfy 1 <= min <= default <= max <= 1000")
        if not 1 <= self.max_candidates_per_video_event <= 64:
            raise ValueError("max_candidates_per_video_event must be between 1 and 64")
        if not 0 <= self.adaptive_margin_low <= self.adaptive_margin_high <= 1:
            raise ValueError("adaptive margins must satisfy 0 <= low <= high <= 1")
        if self.generic_token_threshold < 1:
            raise ValueError("generic_token_threshold must be positive")
        if not 0 <= self.text_embedding_cache_size <= 10000:
            raise ValueError("text_embedding_cache_size must be between 0 and 10000")


@dataclass(frozen=True)
class FusionSettings:
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("fusion.rrf_k must be positive")
        if any(float(value) < 0 for value in self.weights.values()):
            raise ValueError("fusion weights must be non-negative")


@dataclass(frozen=True)
class TemporalSettings:
    allow_same_frame: bool = False
    fast_feasibility_filter: bool = True
    candidate_windows: bool = True
    max_candidate_videos: int = 30
    max_windows_per_video: int = 5
    max_total_windows: int = 50
    alignment: str = "dante"
    legacy_alignment_enabled: bool = True
    top_chains_per_video: int = 5
    beam_size: int = 100
    sequence_default_max_gap_seconds: float | None = 60.0
    # A temporal path must not silently fall from a strong retrieval hit to a
    # much weaker frame merely to make timestamps monotonic.
    min_alignment_seed_relative_score: float = 0.85

    def __post_init__(self) -> None:
        if self.alignment not in {"legacy", "legacy_beam", "dante"}:
            raise ValueError("temporal.alignment must be legacy, legacy_beam, or dante")
        if not 1 <= self.max_candidate_videos <= 100:
            raise ValueError("max_candidate_videos must be between 1 and 100")
        if not 1 <= self.max_windows_per_video <= 20:
            raise ValueError("max_windows_per_video must be between 1 and 20")
        if not 1 <= self.max_total_windows <= 500:
            raise ValueError("max_total_windows must be between 1 and 500")
        if not 1 <= self.top_chains_per_video <= 20:
            raise ValueError("top_chains_per_video must be between 1 and 20")
        if self.beam_size < self.top_chains_per_video:
            raise ValueError("beam_size must be at least top_chains_per_video")
        if self.sequence_default_max_gap_seconds is not None and self.sequence_default_max_gap_seconds <= 0:
            raise ValueError("sequence_default_max_gap_seconds must be positive when set")
        if not 0.0 < self.min_alignment_seed_relative_score <= 1.0:
            raise ValueError("min_alignment_seed_relative_score must be in (0, 1]")


@dataclass(frozen=True)
class RefinementSettings:
    automatic: bool = True
    coarse_fps: float = 3.0
    coarse_padding_sec: float = 5.0
    fine_fps: float = 12.0
    fine_padding_sec: float = 1.0
    max_frames_per_window: int = 120
    max_dense_windows: int = 10
    max_refinement_shift_seconds: float = 10.0

    def to_refinement_config(self) -> RefinementConfig:
        return RefinementConfig(
            coarse_window_seconds=self.coarse_padding_sec,
            coarse_fps=self.coarse_fps,
            fine_window_seconds=self.fine_padding_sec,
            fine_fps=self.fine_fps,
            max_frames_per_refinement=self.max_frames_per_window,
            max_refinement_shift_seconds=self.max_refinement_shift_seconds,
        )

    def __post_init__(self) -> None:
        self.to_refinement_config()
        if not 1 <= self.max_dense_windows <= 50:
            raise ValueError("max_dense_windows must be between 1 and 50")
        if self.max_refinement_shift_seconds <= 0:
            raise ValueError("max_refinement_shift_seconds must be positive")


@dataclass(frozen=True)
class RerankingSettings:
    context_enabled: bool = True
    linkage_enabled: bool = True
    weights: dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.35,
        "temporal": 0.25,
        "linkage": 0.15,
        "context": 0.15,
        "multimodal": 0.05,
        "coverage": 0.05,
    })
    multimodal_weights: dict[str, float] = field(default_factory=lambda: {
        "clip": 1.0,
        "siglip": 1.0,
        "object": 2.0,
        "attribute": 2.0,
        "ocr": 3.0,
        "asr": 3.0,
        "metadata": 1.0,
    })
    temporal_link_scale_seconds: float = 60.0
    linkage_hard_gate: bool = False
    min_visual_continuity: float = 0.65

    def __post_init__(self) -> None:
        if any(float(value) < 0 for value in self.weights.values()):
            raise ValueError("reranking weights must be non-negative")
        if not self.weights or sum(float(value) for value in self.weights.values()) <= 0:
            raise ValueError("reranking weights must have a positive sum")
        if any(float(value) < 0 for value in self.multimodal_weights.values()):
            raise ValueError("multimodal weights must be non-negative")
        if self.temporal_link_scale_seconds <= 0:
            raise ValueError("temporal_link_scale_seconds must be positive")
        if not 0 <= self.min_visual_continuity <= 1:
            raise ValueError("min_visual_continuity must be between 0 and 1")


@dataclass(frozen=True)
class DistinctivenessSettings:
    margin_weight: float = 0.35
    video_concentration_weight: float = 0.25
    constraint_weight: float = 0.25
    lexical_weight: float = 0.15

    def __post_init__(self) -> None:
        values = (self.margin_weight, self.video_concentration_weight, self.constraint_weight, self.lexical_weight)
        if min(values) < 0 or sum(values) <= 0:
            raise ValueError("distinctiveness weights must be non-negative with a positive sum")


@dataclass(frozen=True)
class DanteSettings:
    semantic_weight: float = 1.0
    transition_gap_penalty: float = 0.25
    optional_skip_penalty: float = 0.15
    coarse_top_k: int = 5
    fine_top_k: int = 5
    dense_candidates_per_event: int = 12

    def __post_init__(self) -> None:
        if min(self.semantic_weight, self.transition_gap_penalty, self.optional_skip_penalty) < 0:
            raise ValueError("DANTE-inspired costs must be non-negative")
        if not 1 <= self.coarse_top_k <= 20 or not 1 <= self.fine_top_k <= 20:
            raise ValueError("DANTE-inspired Top-K values must be between 1 and 20")
        if not 1 <= self.dense_candidates_per_event <= 64:
            raise ValueError("dense_candidates_per_event must be between 1 and 64")


@dataclass(frozen=True)
class VlmSettings:
    enabled: bool = False
    provider: str | None = None
    model: str | None = None
    network_calls_allowed: bool = False
    decision_log_approved: bool = False
    top_n: int = 3
    low_confidence_margin: float = 0.05
    low_linkage_threshold: float = 0.40

    def __post_init__(self) -> None:
        if self.enabled and not self.network_calls_allowed:
            raise ValueError("VLM cannot be enabled while network_calls_allowed is false")
        if self.enabled and not self.decision_log_approved:
            raise ValueError("VLM cannot be enabled without decision_log_approved=true")
        if self.enabled and (not self.provider or not self.model):
            raise ValueError("VLM provider and model are required when enabled")
        if self.top_n < 1:
            raise ValueError("vlm.top_n must be positive")
        if not 0 <= self.low_confidence_margin <= 1 or not 0 <= self.low_linkage_threshold <= 1:
            raise ValueError("VLM confidence thresholds must be between 0 and 1")


@dataclass(frozen=True)
class BenchmarkSettings:
    development_config_frozen: bool = False
    holdout_tuning_forbidden: bool = True


@dataclass(frozen=True)
class FeatureSettings:
    planner_v2: bool = True
    query_variants: bool = True
    second_visual_encoder: bool = False
    modality_routing: bool = True
    adaptive_topk: bool = True
    anchor: bool = True
    temporal_feasibility: bool = True
    candidate_windows: bool = True
    dante_coarse: bool = True
    dense_refinement: bool = True
    dante_fine: bool = True
    global_context: bool = True
    linkage: bool = True
    conditional_vlm: bool = False


@dataclass(frozen=True)
class TrakeRuntimeConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    algorithm: str = "trake_v2"
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    fusion: FusionSettings = field(default_factory=FusionSettings)
    temporal: TemporalSettings = field(default_factory=TemporalSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    reranking: RerankingSettings = field(default_factory=RerankingSettings)
    distinctiveness: DistinctivenessSettings = field(default_factory=DistinctivenessSettings)
    dante: DanteSettings = field(default_factory=DanteSettings)
    vlm: VlmSettings = field(default_factory=VlmSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    legacy_scoring: ScoreConfig = field(default_factory=ScoreConfig)

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported TRAKE config schema: {self.schema_version}")
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"unsupported TRAKE algorithm: {self.algorithm}")
        if self.algorithm == "legacy" and not self.temporal.legacy_alignment_enabled:
            raise ValueError("legacy algorithm requires temporal.legacy_alignment_enabled=true")

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    @property
    def algorithm_version(self) -> str:
        if self.algorithm == "legacy":
            return "phase8-temporal-dp-v1"
        return "dante-inspired-coarse-fine-v1" if self.temporal.alignment == "dante" and self.features.dante_coarse else "trake-v2-foundation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def legacy(cls) -> "TrakeRuntimeConfig":
        return cls(
            algorithm="legacy",
            retrieval=RetrievalSettings(
                clip_enabled=True,
                siglip_enabled=False,
                object_enabled=True,
                attribute_enabled=True,
                ocr_enabled=True,
                asr_enabled=True,
                metadata_enabled=True,
                adaptive_topk=False,
                topk_min=30,
                topk_default=60,
                topk_max=100,
                max_candidates_per_video_event=8,
                adaptive_margin_low=0.05,
                adaptive_margin_high=0.20,
                generic_token_threshold=3,
                text_embedding_cache_size=0,
            ),
            temporal=TemporalSettings(
                allow_same_frame=False,
                fast_feasibility_filter=False,
                candidate_windows=False,
                max_candidate_videos=20,
                max_windows_per_video=1,
                max_total_windows=20,
                alignment="legacy",
                legacy_alignment_enabled=True,
                top_chains_per_video=5,
                beam_size=100,
            ),
            refinement=RefinementSettings(
                automatic=False,
                coarse_fps=3.0,
                coarse_padding_sec=5.0,
                fine_fps=12.0,
                fine_padding_sec=1.0,
                max_frames_per_window=60,
                max_dense_windows=1,
            ),
            reranking=RerankingSettings(context_enabled=False, linkage_enabled=False),
            benchmark=BenchmarkSettings(development_config_frozen=True, holdout_tuning_forbidden=True),
            features=FeatureSettings(
                planner_v2=False,
                query_variants=False,
                second_visual_encoder=False,
                modality_routing=False,
                adaptive_topk=False,
                anchor=False,
                temporal_feasibility=False,
                candidate_windows=False,
                dante_coarse=False,
                dense_refinement=False,
                dante_fine=False,
                global_context=False,
                linkage=False,
                conditional_vlm=False,
            ),
        )


def load_trake_config(path: Path) -> TrakeRuntimeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload)
    values["retrieval"] = RetrievalSettings(**values.get("retrieval", {}))
    values["fusion"] = FusionSettings(**values.get("fusion", {}))
    values["temporal"] = TemporalSettings(**values.get("temporal", {}))
    values["refinement"] = RefinementSettings(**values.get("refinement", {}))
    values["reranking"] = RerankingSettings(**values.get("reranking", {}))
    values["distinctiveness"] = DistinctivenessSettings(**values.get("distinctiveness", {}))
    values["dante"] = DanteSettings(**values.get("dante", {}))
    values["vlm"] = VlmSettings(**values.get("vlm", {}))
    values["benchmark"] = BenchmarkSettings(**values.get("benchmark", {}))
    values["features"] = FeatureSettings(**values.get("features", {}))
    values["legacy_scoring"] = ScoreConfig(**values.get("legacy_scoring", {}))
    return TrakeRuntimeConfig(**values)
