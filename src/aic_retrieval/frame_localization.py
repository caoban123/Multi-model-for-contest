"""Bounded coarse-to-fine keyframe localization inside retrieved videos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from aic_retrieval.search import FrameRef, normalize_query


LOCALIZATION_VERSION = "frame-localization-v1"


@dataclass(frozen=True)
class FrameLocalizationConfig:
    video_top_k: int = 12
    radius_seconds: float = 40.0
    max_frames_per_video: int = 12
    min_gap_seconds: float = 1.5
    search_full_video_without_seed: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.video_top_k <= 50:
            raise ValueError("video_top_k must be between 1 and 50")
        if self.radius_seconds <= 0:
            raise ValueError("radius_seconds must be positive")
        if not 1 <= self.max_frames_per_video <= 50:
            raise ValueError("max_frames_per_video must be between 1 and 50")
        if self.min_gap_seconds < 0:
            raise ValueError("min_gap_seconds must not be negative")


class FrameLocalizer:
    """Re-score keyframes only inside the videos selected by Stage A."""

    def __init__(self, index: np.ndarray, refs: list[FrameRef]) -> None:
        if index.ndim != 2 or len(index) != len(refs):
            raise ValueError("frame localizer requires one 2-D index row per ref")
        self.index = index
        self.refs = refs
        self.positions_by_video: dict[str, list[int]] = {}
        for position, ref in enumerate(refs):
            self.positions_by_video.setdefault(ref.video_id, []).append(position)

    def localize(
        self,
        candidates: Iterable[Mapping[str, Any]],
        query_vector: np.ndarray,
        *,
        config: FrameLocalizationConfig | None = None,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        settings = config or FrameLocalizationConfig()
        vector = normalize_query(query_vector, expected_dim=self.index.shape[1])
        output: list[dict[str, Any]] = []
        for candidate_position, source in enumerate(candidates):
            candidate = dict(source)
            if candidate_position >= settings.video_top_k:
                output.append(candidate)
                continue
            video_id = str(candidate.get("video_id") or "").strip()
            positions = self.positions_by_video.get(video_id, [])
            if not positions:
                candidate["frame_localization"] = {
                    "version": LOCALIZATION_VERSION,
                    "status": "UNAVAILABLE",
                    "reason": "video has no indexed keyframes",
                }
                output.append(candidate)
                continue
            seed_times = _seed_times(candidate)
            eligible = _eligible_positions(
                positions,
                self.refs,
                seed_times,
                settings.radius_seconds,
                settings.search_full_video_without_seed,
            )
            if not eligible:
                candidate["frame_localization"] = {
                    "version": LOCALIZATION_VERSION,
                    "status": "NO_CANDIDATES",
                    "seed_times": seed_times,
                }
                output.append(candidate)
                continue
            scores = self.index[np.asarray(eligible, dtype=np.int64)] @ vector
            ranked = sorted(
                zip(eligible, scores.tolist()),
                key=lambda item: (-float(item[1]), self.refs[item[0]].pts_time, item[0]),
            )
            selected = _select_temporally_diverse(
                ranked,
                self.refs,
                settings.max_frames_per_video,
                settings.min_gap_seconds,
            )
            original_by_key = {
                int(frame["keyframe_id"]): dict(frame)
                for frame in candidate.get("frames", [])
                if isinstance(frame, Mapping) and frame.get("keyframe_id") is not None
            }
            frames: list[dict[str, Any]] = []
            for rank, (position, score) in enumerate(selected, 1):
                ref = self.refs[position]
                frame = {**original_by_key.get(ref.keyframe_id, {}), **asdict(ref)}
                frame.update({
                    "localization_rank": rank,
                    "localization_score": float(score),
                    "localization_source": "clip_within_retrieved_video",
                    "is_representative": rank == 1,
                })
                frames.append(frame)
            best = frames[0]
            candidate.update({
                "frames": frames,
                "best_keyframe_id": best["keyframe_id"],
                "best_frame_idx": best["frame_idx"],
                "best_pts_time": best["pts_time"],
                "best_keyframe_path": best.get("keyframe_path"),
                "frame_localization": {
                    "version": LOCALIZATION_VERSION,
                    "status": "APPLIED",
                    "query": query_text,
                    "seed_times": seed_times,
                    "radius_seconds": settings.radius_seconds,
                    "eligible_frames": len(eligible),
                    "returned_frames": len(frames),
                    "min_gap_seconds": settings.min_gap_seconds,
                    "video_rank_preserved": True,
                },
            })
            output.append(candidate)
        return output


def _seed_times(candidate: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    frames = candidate.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, Mapping) and frame.get("pts_time") is not None:
                values.append(float(frame["pts_time"]))
    for key in ("pts_time", "best_pts_time"):
        if candidate.get(key) is not None:
            values.append(float(candidate[key]))
    return sorted(set(values))


def _eligible_positions(
    positions: list[int],
    refs: list[FrameRef],
    seed_times: list[float],
    radius_seconds: float,
    full_video_without_seed: bool,
) -> list[int]:
    if not seed_times:
        return list(positions) if full_video_without_seed else []
    return [
        position
        for position in positions
        if any(abs(float(refs[position].pts_time) - seed) <= radius_seconds for seed in seed_times)
    ]


def _select_temporally_diverse(
    ranked: list[tuple[int, float]],
    refs: list[FrameRef],
    limit: int,
    min_gap_seconds: float,
) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    selected_positions: set[int] = set()
    for position, score in ranked:
        pts_time = float(refs[position].pts_time)
        if all(abs(pts_time - float(refs[item[0]].pts_time)) >= min_gap_seconds for item in selected):
            selected.append((position, score))
            selected_positions.add(position)
            if len(selected) >= limit:
                return selected
    for position, score in ranked:
        if position in selected_positions:
            continue
        selected.append((position, score))
        if len(selected) >= limit:
            break
    return selected
