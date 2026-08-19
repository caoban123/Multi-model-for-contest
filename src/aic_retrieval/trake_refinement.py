"""Bounded on-demand two-stage video refinement for selected TRAKE events."""

from __future__ import annotations

import hashlib
import json
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

    def __post_init__(self) -> None:
        if self.coarse_window_seconds <= 0 or self.fine_window_seconds <= 0:
            raise ValueError("refinement windows must be positive")
        if not 0 < self.coarse_fps <= 5:
            raise ValueError("coarse_fps must be in (0, 5]")
        if not 8 <= self.fine_fps <= 15:
            raise ValueError("fine_fps must be between 8 and 15")
        if not 1 <= self.max_frames_per_refinement <= 120:
            raise ValueError("max_frames_per_refinement must be between 1 and 120")


@dataclass(frozen=True)
class SampledFrame:
    timestamp: float
    image: Any


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
                    frames.append(SampledFrame(timestamp, image[:, :, ::-1]))
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


def _times(center: float, radius: float, fps: float, limit: int) -> list[float]:
    start=max(0.0,center-radius); end=center+radius; step=1.0/fps; count=min(limit,int((end-start)/step)+1)
    return [round(start+(index*step),6) for index in range(count)]


def _best(frames: list[SampledFrame], scores: list[float]) -> SampledFrame:
    if len(frames)!=len(scores): raise ValueError("frame scorer result count mismatch")
    return min(zip(frames,scores),key=lambda item:(-float(item[1]),item[0].timestamp))[0]
