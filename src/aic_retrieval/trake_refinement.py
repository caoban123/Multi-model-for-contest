"""Bounded on-demand two-stage video refinement for selected TRAKE events."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


REFINEMENT_VERSION = "phase8-refinement-v1"


@dataclass(frozen=True)
class RefinementConfig:
    coarse_window_seconds: float = 5.0
    coarse_fps: float = 3.0
    fine_window_seconds: float = 1.0
    fine_fps: float = 12.0
    max_frames_per_refinement: int = 60
    max_refinement_shift_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.coarse_window_seconds <= 0 or self.fine_window_seconds <= 0:
            raise ValueError("refinement windows must be positive")
        if not 0 < self.coarse_fps <= 5:
            raise ValueError("coarse_fps must be in (0, 5]")
        if not 8 <= self.fine_fps <= 15:
            raise ValueError("fine_fps must be between 8 and 15")
        if not 1 <= self.max_frames_per_refinement <= 120:
            raise ValueError("max_frames_per_refinement must be between 1 and 120")
        if self.max_refinement_shift_seconds <= 0:
            raise ValueError("max_refinement_shift_seconds must be positive")


@dataclass(frozen=True)
class SampledFrame:
    timestamp: float
    image: Any
    requested_pts: float | None = None
    frame_index: int | None = None


@dataclass(frozen=True)
class RefinementResult:
    status: str
    video_id: str
    event_id: str
    source_keyframe_id: int
    source_frame_idx: int
    source_pts_time: float
    refined_frame_idx: int | None
    refined_pts_time: float | None
    refined_image_path: str | None
    sampling: dict[str, Any]
    reason: str
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()
    version: str = REFINEMENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DenseFrameCandidate:
    event_id: str
    source_candidate_id: str
    source_keyframe_id: int
    frame_idx: int | None
    requested_pts: float
    decoded_pts: float
    fps: float
    seek_error_estimate: float
    pts_time: float
    semantic_score: float
    stage: str
    rank: int
    image_path: str | None = None
    visual_signature: tuple[float, ...] = ()


@dataclass(frozen=True)
class DenseWindowExpansion:
    status: str
    window_id: str
    video_id: str
    event_frames: dict[str, tuple[DenseFrameCandidate, ...]]
    decoded_frame_count: int
    sampling: dict[str, Any]
    latency_ms: float
    cache_key: str
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()
    version: str = "trake-dense-window-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenCvDecoder:
    def sample(self, video_path: Path, timestamps: list[float]) -> list[SampledFrame]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed") from exc
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release(); raise RuntimeError("video cannot be opened")
        frames: list[SampledFrame] = []
        try:
            for timestamp in timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
                ok, image = capture.read()
                if ok and image is not None:
                    decoded_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    frame_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    frames.append(
                        SampledFrame(
                            decoded_pts if decoded_pts >= 0 else timestamp,
                            image[:, :, ::-1],
                            requested_pts=timestamp,
                            frame_index=frame_position if frame_position >= 0 else None,
                        )
                    )
        finally:
            capture.release()
        return frames

    def write(self, path: Path, image: Any) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image[:, :, ::-1]):
            raise RuntimeError("refined image could not be written")


class DenseRefiner:
    def __init__(
        self,
        repo_root: Path,
        assets_by_video: dict[str, dict[str, Any]],
        cache_dir: Path,
        scorer: Callable[[str, list[Any]], list[float]] | None,
        *,
        decoder: Any | None = None,
        config: RefinementConfig | None = None,
    ) -> None:
        self.repo_root=repo_root.resolve(); self.assets_by_video=assets_by_video; self.cache_dir=cache_dir
        self.scorer=scorer; self.decoder=decoder or OpenCvDecoder(); self.config=config or RefinementConfig()

    def refine(self, *, video_id: str, event_id: str, event_text: str, source_keyframe_id: int, source_frame_idx: int, source_pts_time: float, fps: float) -> RefinementResult:
        source = self._video_path(video_id)
        sampling={"coarse_window_seconds":self.config.coarse_window_seconds,"coarse_fps":self.config.coarse_fps,"fine_window_seconds":self.config.fine_window_seconds,"fine_fps":self.config.fine_fps,"max_frames":self.config.max_frames_per_refinement}
        provenance={"source_video":str(source) if source else None,"selection":"selected_event_only","decoder":type(self.decoder).__name__,"scorer":"local_clip" if self.scorer else None}
        if source is None:
            return RefinementResult("REFINEMENT_UNAVAILABLE",video_id,event_id,source_keyframe_id,source_frame_idx,source_pts_time,None,None,None,sampling,"source video is missing or outside repository",provenance,("REFINEMENT_UNAVAILABLE",))
        if self.scorer is None:
            return RefinementResult("REFINEMENT_UNAVAILABLE",video_id,event_id,source_keyframe_id,source_frame_idx,source_pts_time,None,None,None,sampling,"local frame scorer is unavailable; keyframe chain remains valid",provenance,("REFINEMENT_UNAVAILABLE",))
        cache_key=hashlib.sha256(f"{video_id}|{event_id}|{event_text}|{source_pts_time}|{asdict(self.config)}".encode()).hexdigest()[:20]
        metadata_path=self.cache_dir/f"{cache_key}.json"
        if metadata_path.is_file():
            return RefinementResult(**json.loads(metadata_path.read_text(encoding="utf-8")))
        try:
            coarse_times=_times(source_pts_time,self.config.coarse_window_seconds,self.config.coarse_fps,self.config.max_frames_per_refinement)
            coarse=self.decoder.sample(source,coarse_times)
            if not coarse: raise RuntimeError("coarse decoder returned no frames")
            coarse_scores=self.scorer(event_text,[item.image for item in coarse]); coarse_best=_best(coarse,coarse_scores)
            remaining=max(1,self.config.max_frames_per_refinement-len(coarse))
            fine_times=_times(coarse_best.timestamp,self.config.fine_window_seconds,self.config.fine_fps,remaining)
            fine=self.decoder.sample(source,fine_times)
            if not fine: raise RuntimeError("fine decoder returned no frames")
            fine_scores=self.scorer(event_text,[item.image for item in fine]); best=_best(fine,fine_scores)
            image_path=self.cache_dir/f"{cache_key}.jpg"; self.decoder.write(image_path,best.image)
            try: relative=image_path.resolve().relative_to(self.repo_root).as_posix()
            except ValueError: relative=str(image_path.resolve())
            result=RefinementResult("AVAILABLE",video_id,event_id,source_keyframe_id,source_frame_idx,source_pts_time,round(best.timestamp*fps),best.timestamp,relative,sampling,"best local CLIP score in bounded coarse-to-fine samples",provenance)
            metadata_path.parent.mkdir(parents=True,exist_ok=True); metadata_path.write_text(json.dumps(result.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8")
            return result
        except Exception as exc:
            return RefinementResult("REFINEMENT_UNAVAILABLE",video_id,event_id,source_keyframe_id,source_frame_idx,source_pts_time,None,None,None,sampling,str(exc),provenance,("REFINEMENT_UNAVAILABLE",))

    def _video_path(self, video_id: str) -> Path | None:
        value=self.assets_by_video.get(video_id,{}).get("video_path")
        if not value: return None
        path=(self.repo_root/value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        try: path.relative_to(self.repo_root)
        except ValueError: return None
        return path if path.is_file() else None


class DenseWindowExpander:
    """Bounded automatic coarse/fine expansion for a few candidate windows."""

    def __init__(
        self,
        repo_root: Path,
        assets_by_video: dict[str, dict[str, Any]],
        cache_dir: Path,
        scorer: Callable[[str, list[Any]], list[float]] | None,
        *,
        decoder: Any | None = None,
        config: RefinementConfig | None = None,
        cache_fingerprint: str | None = None,
        model_fingerprint: str | None = None,
        config_fingerprint: str | None = None,
        index_fingerprint: str | None = None,
        candidates_per_event: int = 12,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.assets_by_video = assets_by_video
        self.cache_dir = cache_dir
        self.scorer = scorer
        self.decoder = decoder or OpenCvDecoder()
        self.config = config or RefinementConfig(max_frames_per_refinement=120)
        self.cache_fingerprint = cache_fingerprint
        self.model_fingerprint = model_fingerprint
        self.config_fingerprint = config_fingerprint
        self.index_fingerprint = index_fingerprint
        self.candidates_per_event = candidates_per_event

    def expand(
        self,
        window: Any,
        request: Any,
        candidates: dict[str, Any],
        predicted_chain: Any | None = None,
    ) -> DenseWindowExpansion:
        """Decode either the coarse window or fine event-local neighborhoods.

        Passing no ``predicted_chain`` performs only the coarse pass.  Passing
        the coarse DANTE chain performs the fine pass around its event times;
        this keeps the required window -> coarse alignment -> fine order.
        """
        started = time.perf_counter()
        video_path = self._video_path(window.video_id)
        stage = "fine" if predicted_chain is not None else "coarse"
        source_candidates = {
            event_id: candidates[candidate_id]
            for event_id, candidate_id in window.event_candidate_ids.items()
            if candidate_id in candidates
        }
        fps_values = {
            float(candidate.evidence.get("fps") or candidate.evidence.get("raw_result", {}).get("fps"))
            for candidate in source_candidates.values()
            if candidate.evidence.get("fps") or candidate.evidence.get("raw_result", {}).get("fps")
        }
        candidates_with_fps = sum(
            1
            for candidate in source_candidates.values()
            if candidate.evidence.get("fps") or candidate.evidence.get("raw_result", {}).get("fps")
        )
        sampling = {
            "stage": stage,
            "coarse_fps": self.config.coarse_fps,
            "fine_fps": self.config.fine_fps,
            "fine_padding_seconds": self.config.fine_window_seconds,
            "max_frames": self.config.max_frames_per_refinement,
            "max_refinement_shift_seconds": self.config.max_refinement_shift_seconds,
        }
        video_fingerprint = _video_fingerprint(video_path, self.repo_root) if video_path else None
        predicted = {
            entry.event_id: entry.candidate.pts_time
            for entry in getattr(predicted_chain, "events", ())
            if entry.candidate is not None
        }
        cache_payload = {
            "window": asdict(window),
            "events": [(event.event_id, event.visual_query or event.clip_query) for event in request.events],
            "config": asdict(self.config),
            "stage": stage,
            "predicted_event_pts": predicted,
            "video_fingerprint": video_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "legacy_cache_fingerprint": self.cache_fingerprint,
            "frame_evidence_version": 2,
        }
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        metadata_path = self.cache_dir / "dense" / f"{cache_key}.json"
        provenance = {
            "source_video": str(video_path) if video_path else None,
            "selection": "candidate_windows_only",
            "decoder": type(self.decoder).__name__,
            "scorer": "local_clip" if self.scorer else None,
            "video_fingerprint": video_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "index_fingerprint": self.index_fingerprint,
        }
        if metadata_path.is_file():
            return dense_expansion_from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
        if video_path is None or self.scorer is None or not source_candidates or len(fps_values) != 1 or candidates_with_fps != len(source_candidates):
            reason = "source video/scorer/candidate FPS is unavailable or inconsistent"
            return DenseWindowExpansion("REFINEMENT_UNAVAILABLE", window.window_id, window.video_id, {}, 0, sampling, round((time.perf_counter() - started) * 1000, 3), cache_key, provenance, (reason,))
        fps = next(iter(fps_values))
        try:
            event_by_id = {event.event_id: event for event in request.events}
            if stage == "coarse":
                # A candidate window can span minutes when independent event
                # retrieval is inconsistent.  Sample around each event's own
                # coarse candidate, never from the start of that broad span.
                per_event = max(1, self.config.max_frames_per_refinement // len(source_candidates))
                frames_by_event = {
                    event_id: self.decoder.sample(
                        video_path,
                        _times(source.pts_time, self.config.coarse_window_seconds, self.config.coarse_fps, per_event),
                    )
                    for event_id, source in source_candidates.items()
                }
                frames = [frame for values in frames_by_event.values() for frame in values]
            else:
                if not predicted:
                    raise RuntimeError("fine expansion requires a non-empty coarse alignment")
                per_event = max(1, self.config.max_frames_per_refinement // len(predicted))
                times_by_event = {
                    event_id: _times(
                            predicted[event_id],
                            self.config.fine_window_seconds,
                            self.config.fine_fps,
                            per_event,
                        )
                    for event_id in source_candidates
                    if event_id in predicted
                }
                frames_by_event = {
                    event_id: self.decoder.sample(video_path, requested_times)
                    for event_id, requested_times in times_by_event.items()
                }
                frames = [frame for values in frames_by_event.values() for frame in values]
            if not frames or any(not values for values in frames_by_event.values()):
                raise RuntimeError(f"{stage} decoder returned no frames")
            event_frames: dict[str, tuple[DenseFrameCandidate, ...]] = {}
            for event_id, source in source_candidates.items():
                if stage == "fine" and event_id not in predicted:
                    continue
                event = event_by_id[event_id]
                event_samples = frames_by_event[event_id]
                scores = [float(value) for value in self.scorer(event.visual_query or event.clip_query, [frame.image for frame in event_samples])]
                ranked = [
                    item for item in sorted(zip(event_samples, scores), key=lambda item: (-item[1], item[0].timestamp))
                    if abs(float(item[0].timestamp) - float(source.pts_time)) <= self.config.max_refinement_shift_seconds
                ][: self.candidates_per_event]
                dense_candidates: list[DenseFrameCandidate] = []
                for rank, (frame, score) in enumerate(ranked, start=1):
                    image_path = self._write_dense_frame_asset(cache_key, event_id, stage, rank, frame.image)
                    dense_candidates.append(DenseFrameCandidate(
                        event_id=event_id,
                        source_candidate_id=source.candidate_id,
                        source_keyframe_id=source.keyframe_id,
                        frame_idx=frame.frame_index,
                        requested_pts=float(frame.requested_pts if frame.requested_pts is not None else frame.timestamp),
                        decoded_pts=float(frame.timestamp),
                        fps=fps,
                        seek_error_estimate=abs(float(frame.timestamp) - float(frame.requested_pts if frame.requested_pts is not None else frame.timestamp)),
                        pts_time=frame.timestamp,
                        semantic_score=score,
                        stage=stage,
                        rank=rank,
                        image_path=image_path,
                        visual_signature=_visual_signature(frame.image),
                    ))
                event_frames[event_id] = tuple(dense_candidates)
            result = DenseWindowExpansion(
                "AVAILABLE",
                window.window_id,
                window.video_id,
                event_frames,
                len(frames),
                sampling,
                round((time.perf_counter() - started) * 1000, 3),
                cache_key,
                provenance,
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            return DenseWindowExpansion("REFINEMENT_UNAVAILABLE", window.window_id, window.video_id, {}, 0, sampling, round((time.perf_counter() - started) * 1000, 3), cache_key, provenance, (str(exc),))

    def _video_path(self, video_id: str) -> Path | None:
        value = self.assets_by_video.get(video_id, {}).get("video_path")
        if not value:
            return None
        path = (self.repo_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError:
            return None
        return path if path.is_file() else None

    def _write_dense_frame_asset(self, cache_key: str, event_id: str, stage: str, rank: int, image: Any) -> str | None:
        if not hasattr(self.decoder, "write"):
            return None
        path = self.cache_dir / "dense" / "frames" / cache_key / f"{event_id}_{stage}_{rank:02d}.jpg"
        try:
            self.decoder.write(path, image)
            return path.resolve().relative_to(self.repo_root).as_posix()
        except Exception:
            return None


def _times(center: float, radius: float, fps: float, limit: int) -> list[float]:
    start=max(0.0,center-radius); end=center+radius; step=1.0/fps; count=min(limit,int((end-start)/step)+1)
    return [round(start+(index*step),6) for index in range(count)]


def _interval_times(start: float, end: float, fps: float, limit: int) -> list[float]:
    start = max(0.0, float(start)); end = max(start, float(end)); step = 1.0 / fps
    count = min(limit, int((end - start) / step) + 1)
    return [round(start + (index * step), 6) for index in range(count)]


def _visual_signature(image: Any) -> tuple[float, ...]:
    """Small persisted visual-continuity descriptor; it is not an identity claim."""
    try:
        import numpy as np

        values = np.asarray(image, dtype="float32")
        if values.ndim < 3 or values.shape[-1] < 3:
            scalar = float(values.mean()) if values.size else 0.0
            return (scalar,)
        channels = values[..., :3]
        histograms = [np.histogram(channels[..., channel], bins=8, range=(0, 255), density=True)[0] for channel in range(3)]
        signature = np.concatenate(histograms)
        norm = float(np.linalg.norm(signature))
        return tuple(float(value / norm) for value in signature) if norm else ()
    except Exception:
        return ()


def _best(frames: list[SampledFrame], scores: list[float]) -> SampledFrame:
    if len(frames)!=len(scores): raise ValueError("frame scorer result count mismatch")
    return min(zip(frames,scores),key=lambda item:(-float(item[1]),item[0].timestamp))[0]


def dense_expansion_from_dict(payload: dict[str, Any]) -> DenseWindowExpansion:
    return DenseWindowExpansion(
        status=payload["status"],
        window_id=payload["window_id"],
        video_id=payload["video_id"],
        event_frames={key: tuple(DenseFrameCandidate(**item) for item in values) for key, values in payload.get("event_frames", {}).items()},
        decoded_frame_count=int(payload.get("decoded_frame_count", 0)),
        sampling=dict(payload.get("sampling", {})),
        latency_ms=float(payload.get("latency_ms", 0.0)),
        cache_key=payload["cache_key"],
        provenance=dict(payload.get("provenance", {})),
        warnings=tuple(payload.get("warnings", ())),
        version=payload.get("version", "trake-dense-window-v1"),
    )


def _video_fingerprint(path: Path, repo_root: Path) -> str:
    stat = path.stat()
    try:
        display_path = path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        display_path = str(path.resolve())
    payload = f"{display_path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
