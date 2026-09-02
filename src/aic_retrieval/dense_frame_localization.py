"""Bounded two-stage localization over decoded video frames."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from aic_retrieval.trake_refinement import OpenCvDecoder, SampledFrame


DENSE_FRAME_LOCALIZATION_VERSION = "dense-frame-localization-v2"


@dataclass(frozen=True)
class DenseFrameLocalizationConfig:
    radius_seconds: float = 30.0
    coarse_fps: float = 1.0
    coarse_frame_limit: int = 72
    fine_window_seconds: float = 2.0
    fine_fps: float = 8.0
    fine_anchor_limit: int = 3
    fine_frame_limit: int = 96
    result_limit: int = 16
    min_result_gap_seconds: float = 0.2

    def __post_init__(self) -> None:
        if not 1.0 <= self.radius_seconds <= 180.0:
            raise ValueError("radius_seconds must be between 1 and 180")
        if not 0.25 <= self.coarse_fps <= 4.0:
            raise ValueError("coarse_fps must be between 0.25 and 4")
        if not 1 <= self.coarse_frame_limit <= 240:
            raise ValueError("coarse_frame_limit must be between 1 and 240")
        if not 0.25 <= self.fine_window_seconds <= 10.0:
            raise ValueError("fine_window_seconds must be between 0.25 and 10")
        if not 2.0 <= self.fine_fps <= 15.0:
            raise ValueError("fine_fps must be between 2 and 15")
        if not 1 <= self.fine_anchor_limit <= 8:
            raise ValueError("fine_anchor_limit must be between 1 and 8")
        if not 1 <= self.fine_frame_limit <= 240:
            raise ValueError("fine_frame_limit must be between 1 and 240")
        if not 1 <= self.result_limit <= 30:
            raise ValueError("result_limit must be between 1 and 30")
        if self.min_result_gap_seconds < 0:
            raise ValueError("min_result_gap_seconds must not be negative")


class DenseFrameLocalizer:
    def __init__(
        self,
        video_resolver: Callable[[str], Path],
        scorer: Callable[[str, list[Any]], list[float]],
        cache_dir: Path,
        *,
        decoder: Any | None = None,
        model_fingerprint: str | None = None,
    ) -> None:
        self.video_resolver = video_resolver
        self.scorer = scorer
        self.cache_dir = cache_dir.resolve()
        self.decoder = decoder or OpenCvDecoder()
        self.model_fingerprint = model_fingerprint

    def localize(
        self,
        *,
        video_id: str,
        query_text: str,
        seed_times: Iterable[float],
        fps: float,
        config: DenseFrameLocalizationConfig | None = None,
    ) -> dict[str, Any]:
        settings = config or DenseFrameLocalizationConfig()
        query_text = query_text.strip()
        seeds = sorted({max(0.0, float(value)) for value in seed_times})
        if not query_text:
            raise ValueError("query_text must not be empty")
        if not seeds:
            return self._unavailable(video_id, query_text, settings, "dense localization requires at least one seed timestamp")
        if fps <= 0:
            return self._unavailable(video_id, query_text, settings, "video FPS is unavailable")
        try:
            video_path = self.video_resolver(video_id).resolve()
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return self._unavailable(video_id, query_text, settings, f"video is unavailable: {exc}")

        started = time.perf_counter()
        try:
            cache_key = _cache_key(video_id, video_path, query_text, seeds, settings, self.model_fingerprint)
        except (OSError, ValueError) as exc:
            return self._unavailable(
                video_id,
                query_text,
                settings,
                f"video cache identity is unavailable: {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        metadata_path = self.cache_dir / cache_key / "result.json"
        cached = self._read_cache(metadata_path)
        if cached is not None:
            cached["cache_hit"] = True
            cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return cached

        try:
            coarse_times = _bounded_times(
                seeds,
                settings.radius_seconds,
                settings.coarse_fps,
                settings.coarse_frame_limit,
            )
            coarse = self.decoder.sample(video_path, coarse_times)
            if not coarse:
                raise RuntimeError("coarse decoder returned no frames")
            coarse_scores = _score(self.scorer, query_text, coarse)
            anchors = _select_anchors(coarse, coarse_scores, settings.fine_anchor_limit)
            fine_times = _bounded_times(
                [frame.timestamp for frame in anchors],
                settings.fine_window_seconds,
                settings.fine_fps,
                settings.fine_frame_limit,
            )
            fine = self.decoder.sample(video_path, fine_times)
            fine_scores = _score(self.scorer, query_text, fine) if fine else []
            ranked = _rank_unique_frames((*zip(coarse, coarse_scores), *zip(fine, fine_scores)))
            selected = _select_diverse(ranked, settings.result_limit, settings.min_result_gap_seconds)
            if not selected:
                raise RuntimeError("dense scoring returned no frame")

            frames: list[dict[str, Any]] = []
            warnings: list[str] = []
            for rank, (frame, score) in enumerate(selected, 1):
                image_path = self.cache_dir / cache_key / "frames" / f"rank_{rank:02d}_f{_frame_index(frame, fps):09d}.jpg"
                stored_path: str | None = None
                try:
                    self.decoder.write(image_path, frame.image)
                    stored_path = str(image_path.resolve())
                except (OSError, RuntimeError, ValueError) as exc:
                    warnings.append(f"frame {rank} cache write failed: {exc}")
                requested = float(frame.requested_pts if frame.requested_pts is not None else frame.timestamp)
                frames.append({
                    "dense_rank": rank,
                    "dense_score": float(score),
                    "frame_idx": _frame_index(frame, fps),
                    "pts_time": float(frame.timestamp),
                    "requested_pts": requested,
                    "seek_error_seconds": abs(float(frame.timestamp) - requested),
                    "fps": float(fps),
                    "keyframe_path": stored_path,
                    "dense_frame": True,
                    "localization_source": "decoded_video_clip",
                    "mapping_status": "DECODED_FRAME_REQUIRES_VISUAL_CONFIRMATION",
                })
            result = {
                "status": "APPLIED",
                "version": DENSE_FRAME_LOCALIZATION_VERSION,
                "video_id": video_id,
                "query": query_text,
                "seed_times": seeds,
                "video_path": str(video_path),
                "cache_key": cache_key,
                "cache_hit": False,
                "sampling": {
                    **asdict(settings),
                    "coarse_requested": len(coarse_times),
                    "coarse_decoded": len(coarse),
                    "fine_requested": len(fine_times),
                    "fine_decoded": len(fine),
                },
                "frames": frames,
                "warnings": warnings,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            self._write_cache(metadata_path, result)
            return result
        except Exception as exc:
            return self._unavailable(
                video_id,
                query_text,
                settings,
                f"{type(exc).__name__}: {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )

    def _read_cache(self, metadata_path: Path) -> dict[str, Any] | None:
        if not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("version") != DENSE_FRAME_LOCALIZATION_VERSION:
                return None
            if not payload.get("frames") or any(
                frame.get("keyframe_path") and not Path(str(frame["keyframe_path"])).is_file()
                for frame in payload["frames"]
            ):
                return None
            return payload
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _write_cache(metadata_path: Path, payload: dict[str, Any]) -> None:
        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = metadata_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(metadata_path)
        except OSError:
            return

    @staticmethod
    def _unavailable(
        video_id: str,
        query_text: str,
        config: DenseFrameLocalizationConfig,
        reason: str,
        *,
        latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "status": "FALLBACK",
            "version": DENSE_FRAME_LOCALIZATION_VERSION,
            "video_id": video_id,
            "query": query_text,
            "sampling": asdict(config),
            "frames": [],
            "reason": reason,
            "latency_ms": latency_ms,
        }


def _score(
    scorer: Callable[[str, list[Any]], list[float]],
    query_text: str,
    frames: list[SampledFrame],
) -> list[float]:
    values = [float(value) for value in scorer(query_text, [frame.image for frame in frames])]
    if len(values) != len(frames):
        raise ValueError("dense frame scorer returned an unexpected score count")
    return values


def _bounded_times(seeds: Iterable[float], radius: float, sample_fps: float, limit: int) -> list[float]:
    step = 1.0 / sample_fps
    values: set[float] = set()
    for seed in seeds:
        start = max(0.0, float(seed) - radius)
        stop = max(start, float(seed) + radius)
        count = int((stop - start) / step) + 1
        values.update(round(start + index * step, 6) for index in range(count))
        values.add(round(float(seed), 6))
    ordered = sorted(value for value in values if value >= 0)
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    positions = {round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)}
    return [ordered[index] for index in sorted(positions)]


def _select_anchors(
    frames: list[SampledFrame],
    scores: list[float],
    limit: int,
) -> list[SampledFrame]:
    ranked = sorted(zip(frames, scores), key=lambda item: (-item[1], item[0].timestamp))
    selected: list[SampledFrame] = []
    for frame, _score_value in ranked:
        if all(abs(frame.timestamp - existing.timestamp) >= 1.0 for existing in selected):
            selected.append(frame)
            if len(selected) >= limit:
                break
    return selected or [ranked[0][0]]


def _rank_unique_frames(
    values: Iterable[tuple[SampledFrame, float]],
) -> list[tuple[SampledFrame, float]]:
    unique: dict[tuple[str, int], tuple[SampledFrame, float]] = {}
    for frame, score in values:
        key = (
            "frame" if frame.frame_index is not None else "pts",
            int(frame.frame_index) if frame.frame_index is not None else int(round(frame.timestamp * 1000)),
        )
        previous = unique.get(key)
        if previous is None or score > previous[1]:
            unique[key] = (frame, score)
    return sorted(unique.values(), key=lambda item: (-item[1], item[0].timestamp))


def _select_diverse(
    ranked: list[tuple[SampledFrame, float]],
    limit: int,
    minimum_gap: float,
) -> list[tuple[SampledFrame, float]]:
    selected: list[tuple[SampledFrame, float]] = []
    selected_keys: set[tuple[int | None, int]] = set()
    for item in ranked:
        frame = item[0]
        if all(abs(frame.timestamp - existing[0].timestamp) >= minimum_gap for existing in selected):
            selected.append(item)
            selected_keys.add((frame.frame_index, int(round(frame.timestamp * 1000))))
            if len(selected) >= limit:
                return selected
    for item in ranked:
        key = (item[0].frame_index, int(round(item[0].timestamp * 1000)))
        if key in selected_keys:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _frame_index(frame: SampledFrame, fps: float) -> int:
    return int(frame.frame_index) if frame.frame_index is not None else max(0, int(round(frame.timestamp * fps)))


def _cache_key(
    video_id: str,
    video_path: Path,
    query_text: str,
    seed_times: list[float],
    config: DenseFrameLocalizationConfig,
    model_fingerprint: str | None,
) -> str:
    stat = video_path.stat()
    payload = {
        "version": DENSE_FRAME_LOCALIZATION_VERSION,
        "video_id": video_id,
        "video_path": str(video_path),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "query": query_text,
        "seed_times": seed_times,
        "config": asdict(config),
        "model_fingerprint": model_fingerprint,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
