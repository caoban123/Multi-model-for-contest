from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from aic_retrieval.metadata_search import MetadataConstraints
from aic_retrieval.color_attributes import ColorAttributeConstraint

FilterMode = Literal["hard", "soft", "disabled"]


def validate_mode(mode: str) -> None:
    if mode not in {"hard", "soft", "disabled"}:
        raise ValueError(f"unsupported modality mode: {mode}")


@dataclass(frozen=True)
class ObjectConstraint:
    labels: tuple[str, ...]
    count_operator: str = ">="
    count: int = 1
    horizontal: str = "any"
    vertical: str = "any"
    min_confidence: float = 0.3
    nms_iou_threshold: float = 0.5
    filter_mode: FilterMode = "soft"

    def __post_init__(self) -> None:
        validate_mode(self.filter_mode)


@dataclass(frozen=True)
class StructuredQuery:
    visual_text: str
    enable_clip: bool = True
    enable_objects: bool = False
    enable_metadata: bool = False
    enable_attributes: bool = False
    enable_ocr: bool = False
    enable_asr: bool = False
    clip_mode: FilterMode = "soft"
    object_constraints: tuple[ObjectConstraint, ...] = ()
    attribute_constraints: tuple[ColorAttributeConstraint, ...] = ()
    attribute_mode: FilterMode = "disabled"
    metadata_constraints: MetadataConstraints = field(default_factory=MetadataConstraints)
    metadata_mode: FilterMode = "disabled"
    ocr_mode: FilterMode = "disabled"
    asr_mode: FilterMode = "disabled"
    ocr_min_confidence: float = 0.0
    clip_candidate_pool: int = 100
    fusion_method: str = "rrf"

    def __post_init__(self) -> None:
        validate_mode(self.clip_mode)
        validate_mode(self.attribute_mode)
        validate_mode(self.metadata_mode)
        validate_mode(self.ocr_mode)
        validate_mode(self.asr_mode)
        if not 0 <= self.ocr_min_confidence <= 1:
            raise ValueError("ocr_min_confidence must be between 0 and 1")
        if self.clip_candidate_pool <= 0:
            raise ValueError("clip_candidate_pool must be positive")
        if self.fusion_method != "rrf":
            raise ValueError("Phase 4 only supports rrf fusion")
