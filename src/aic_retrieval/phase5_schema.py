from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

SCHEMA_VERSION = "phase5-v1"


def normalize_text(value: str, *, fold_accents: bool = False) -> str:
    text = unicodedata.normalize("NFC", value).casefold()
    if fold_accents:
        text = "".join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")
        text = text.replace("đ", "d")
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def _finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class OcrDetection:
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...]
    text_normalized: str = ""
    text_folded: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("OCR text must not be empty")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("OCR confidence must be between 0 and 1")
        if len(self.bbox) != 4 or any(len(point) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in point) for point in self.bbox):
            raise ValueError("OCR bbox must contain four normalized (x, y) points")
        object.__setattr__(self, "text_normalized", self.text_normalized or normalize_text(self.text))
        object.__setattr__(self, "text_folded", self.text_folded or normalize_text(self.text, fold_accents=True))


@dataclass(frozen=True)
class OcrFrame:
    video_id: str
    keyframe_id: int
    frame_idx: int
    pts_time: float
    fps: float
    detections: tuple[OcrDetection, ...] = ()
    schema_version: str = SCHEMA_VERSION
    run_id: str = "unknown"

    def __post_init__(self) -> None:
        if not self.video_id or self.keyframe_id < 0 or self.frame_idx < 0:
            raise ValueError("invalid OCR frame identity")
        _finite_non_negative("pts_time", self.pts_time)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsrSegment:
    segment_id: int
    start_time: float
    end_time: float
    text: str
    confidence: float | None = None
    text_normalized: str = ""
    text_folded: str = ""

    def __post_init__(self) -> None:
        if self.segment_id < 0 or not self.text.strip():
            raise ValueError("invalid ASR segment")
        _finite_non_negative("start_time", self.start_time)
        _finite_non_negative("end_time", self.end_time)
        if self.end_time < self.start_time:
            raise ValueError("ASR end_time must not precede start_time")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("ASR confidence must be between 0 and 1")
        object.__setattr__(self, "text_normalized", self.text_normalized or normalize_text(self.text))
        object.__setattr__(self, "text_folded", self.text_folded or normalize_text(self.text, fold_accents=True))


@dataclass(frozen=True)
class AsrTranscript:
    video_id: str
    status: str
    segments: tuple[AsrSegment, ...] = ()
    language: str | None = None
    duration: float | None = None
    schema_version: str = SCHEMA_VERSION
    run_id: str = "unknown"

    def __post_init__(self) -> None:
        if not self.video_id or self.status not in {"AVAILABLE", "NO_AUDIO", "ERROR"}:
            raise ValueError("invalid ASR transcript status")
        if self.status != "AVAILABLE" and self.segments:
            raise ValueError("unavailable audio must not contain ASR segments")
        if self.duration is not None:
            _finite_non_negative("duration", self.duration)
            if any(segment.end_time > self.duration + 0.1 for segment in self.segments):
                raise ValueError("ASR segment exceeds media duration")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalMatch:
    video_id: str
    keyframe_id: int | None
    frame_idx: int | None
    pts_time: float | None
    mapping_kind: str
    distance_seconds: float | None


def map_segment_to_frame(video_id: str, start_time: float, end_time: float, refs: Iterable[Any], max_distance: float = 3.0) -> TemporalMatch:
    refs = sorted((ref for ref in refs if ref.video_id == video_id), key=lambda ref: (ref.pts_time, ref.keyframe_id))
    if not refs:
        return TemporalMatch(video_id, None, None, None, "unmapped", None)
    inside = [ref for ref in refs if start_time <= ref.pts_time <= end_time]
    midpoint = (start_time + end_time) / 2
    selected = min(inside or refs, key=lambda ref: (abs(ref.pts_time - midpoint), ref.keyframe_id))
    distance = 0.0 if inside else min(abs(selected.pts_time - start_time), abs(selected.pts_time - end_time))
    if not inside and distance > max_distance:
        return TemporalMatch(video_id, None, None, None, "unmapped", distance)
    return TemporalMatch(video_id, selected.keyframe_id, selected.frame_idx, selected.pts_time, "inside_segment" if inside else "nearest", distance)
