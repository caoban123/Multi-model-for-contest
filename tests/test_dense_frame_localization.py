from __future__ import annotations

from pathlib import Path

from aic_retrieval.dense_frame_localization import (
    DenseFrameLocalizationConfig,
    DenseFrameLocalizer,
)
from aic_retrieval.trake_refinement import SampledFrame


class FakeDecoder:
    def __init__(self) -> None:
        self.sample_calls = 0
        self.write_calls = 0

    def sample(self, _video_path: Path, timestamps: list[float]) -> list[SampledFrame]:
        self.sample_calls += 1
        return [
            SampledFrame(
                timestamp=value,
                image=value,
                requested_pts=value,
                frame_index=round(value * 25),
            )
            for value in timestamps
        ]

    def write(self, path: Path, image: object) -> None:
        self.write_calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(image), encoding="ascii")


def test_dense_localizer_decodes_scores_and_reuses_cache(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture video")
    decoder = FakeDecoder()
    localizer = DenseFrameLocalizer(
        lambda _video_id: video,
        lambda _query, images: [1.0 - abs(float(image) - 10.5) / 100.0 for image in images],
        tmp_path / "cache",
        decoder=decoder,
        model_fingerprint="clip-fixture",
    )
    config = DenseFrameLocalizationConfig(
        radius_seconds=3,
        coarse_fps=1,
        coarse_frame_limit=20,
        fine_window_seconds=1,
        fine_fps=4,
        fine_anchor_limit=2,
        fine_frame_limit=20,
        result_limit=5,
        min_result_gap_seconds=0.2,
    )

    first = localizer.localize(
        video_id="L21_V001",
        query_text="person opens a door",
        seed_times=[10.0],
        fps=25.0,
        config=config,
    )

    assert first["status"] == "APPLIED"
    assert first["cache_hit"] is False
    assert len(first["frames"]) == 5
    assert first["frames"][0]["frame_idx"] == round(first["frames"][0]["pts_time"] * 25)
    assert first["frames"][0]["dense_frame"] is True
    assert Path(first["frames"][0]["keyframe_path"]).is_file()
    calls_after_first = decoder.sample_calls

    second = localizer.localize(
        video_id="L21_V001",
        query_text="person opens a door",
        seed_times=[10.0],
        fps=25.0,
        config=config,
    )

    assert second["status"] == "APPLIED"
    assert second["cache_hit"] is True
    assert decoder.sample_calls == calls_after_first


def test_dense_localizer_falls_back_without_seed_or_video(tmp_path: Path) -> None:
    localizer = DenseFrameLocalizer(
        lambda _video_id: (_ for _ in ()).throw(FileNotFoundError("missing")),
        lambda _query, _images: [],
        tmp_path / "cache",
        decoder=FakeDecoder(),
    )

    no_seed = localizer.localize(
        video_id="L21_V001",
        query_text="scene",
        seed_times=[],
        fps=25.0,
    )
    missing_video = localizer.localize(
        video_id="L21_V001",
        query_text="scene",
        seed_times=[1.0],
        fps=25.0,
    )

    assert no_seed["status"] == "FALLBACK"
    assert "seed" in no_seed["reason"]
    assert missing_video["status"] == "FALLBACK"
    assert "missing" in missing_video["reason"]
