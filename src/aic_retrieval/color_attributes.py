from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aic_retrieval.search import FrameRef

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    Image = None


COLOR_ALIASES = {
    "red": {"red", "đỏ", "do"},
    "blue": {"blue", "xanh duong", "xanh dương"},
    "green": {"green", "xanh la", "xanh lá"},
    "yellow": {"yellow", "vang", "vàng"},
    "white": {"white", "trang", "trắng"},
    "black": {"black", "den", "đen"},
    "orange": {"orange", "cam"},
    "pink": {"pink", "hong", "hồng"},
    "purple": {"purple", "tim", "tím"},
    "brown": {"brown", "nau", "nâu"},
}

CLOTHING_TERMS = {
    "shirt",
    "t-shirt",
    "tee",
    "clothes",
    "clothing",
    "dress",
    "jacket",
    "áo",
    "ao",
}


@dataclass(frozen=True)
class ColorAttributeConstraint:
    color: str
    target: str = "any"
    min_ratio: float = 0.04
    filter_mode: str = "soft"
    source: str = "manual"


def normalize_color(value: str) -> str:
    normalized = value.strip().lower()
    for color, aliases in COLOR_ALIASES.items():
        if normalized == color or normalized in aliases:
            return color
    return normalized


def parse_color_constraint(query: str, fallback_color: str = "", filter_mode: str = "soft") -> ColorAttributeConstraint | None:
    color = normalize_color(fallback_color) if fallback_color.strip() else ""
    source = "manual" if color else "query"
    normalized_query = query.lower()
    if not color:
        for candidate, aliases in COLOR_ALIASES.items():
            if any(alias in normalized_query for alias in aliases):
                color = candidate
                break
    if not color:
        return None
    target = "clothing" if any(term in normalized_query for term in CLOTHING_TERMS) else "any"
    return ColorAttributeConstraint(color=color, target=target, filter_mode=filter_mode, source=source)


def color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    brightness = red + green + blue
    if color == "red":
        return (red > 95) & (red > green * 1.35) & (red > blue * 1.35)
    if color == "blue":
        return (blue > 85) & (blue > red * 1.25) & (blue > green * 1.15)
    if color == "green":
        return (green > 80) & (green > red * 1.15) & (green > blue * 1.15)
    if color == "yellow":
        return (red > 120) & (green > 105) & (blue < 110) & (abs(red - green) < 80)
    if color == "white":
        return (red > 190) & (green > 190) & (blue > 190)
    if color == "black":
        return brightness < 135
    if color == "orange":
        return (red > 130) & (green > 55) & (green < 170) & (blue < 100) & (red > green * 1.15)
    if color == "pink":
        return (red > 150) & (blue > 95) & (green < 150) & (red > green * 1.2)
    if color == "purple":
        return (red > 80) & (blue > 100) & (green < 120)
    if color == "brown":
        return (red > 70) & (green > 35) & (blue < 95) & (red > blue * 1.3) & (green > blue * 1.1)
    return np.zeros(rgb.shape[:2], dtype=bool)


def target_crop(rgb: np.ndarray, target: str) -> np.ndarray:
    if target != "clothing":
        return rgb
    height, width = rgb.shape[:2]
    y0, y1 = int(height * 0.20), int(height * 0.88)
    x0, x1 = int(width * 0.18), int(width * 0.82)
    return rgb[y0:y1, x0:x1]


def score_color_image(path: Path, constraint: ColorAttributeConstraint) -> dict[str, Any] | None:
    if Image is None or not path.is_file():
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((192, 192))
        rgb = np.asarray(image)
    cropped = target_crop(rgb, constraint.target)
    mask = color_mask(cropped, constraint.color)
    ratio = float(mask.mean()) if mask.size else 0.0
    score = min(1.0, ratio / max(constraint.min_ratio, 0.001))
    return {
        "color": constraint.color,
        "target": constraint.target,
        "color_ratio": round(ratio, 6),
        "attribute_score": round(score, 6),
        "source": constraint.source,
    }


class ColorAttributeService:
    def __init__(self, repo_root: Path, refs: Iterable[FrameRef]) -> None:
        self.repo_root = repo_root
        self.refs = list(refs)

    @property
    def available(self) -> bool:
        return Image is not None

    def search(
        self,
        constraint: ColorAttributeConstraint,
        candidate_keys: set[tuple[str, int]] | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            return {"results": [], "unavailable_frames": len(self.refs), "constraint": asdict(constraint)}
        results = []
        unavailable = 0
        for ref in self.refs:
            key = (ref.video_id, ref.keyframe_id)
            if candidate_keys is not None and key not in candidate_keys:
                continue
            if not ref.keyframe_path:
                unavailable += 1
                continue
            score = score_color_image(self.repo_root / ref.keyframe_path, constraint)
            if score is None:
                unavailable += 1
                continue
            if score["color_ratio"] >= constraint.min_ratio:
                results.append(
                    {
                        "video_id": ref.video_id,
                        "keyframe_id": ref.keyframe_id,
                        "frame_idx": ref.frame_idx,
                        "pts_time": ref.pts_time,
                        "fps": ref.fps,
                        "keyframe_path": ref.keyframe_path,
                        **score,
                    }
                )
        results.sort(key=lambda item: (-item["attribute_score"], -item["color_ratio"], item["video_id"], item["keyframe_id"]))
        for rank, item in enumerate(results, start=1):
            item["attribute_rank"] = rank
        return {"results": results, "unavailable_frames": unavailable, "constraint": asdict(constraint)}
