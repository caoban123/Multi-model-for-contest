from pathlib import Path

import pytest

from aic_retrieval.registry import count_mapping_rows, read_feature_info, scan_data_root, validate_assets


def test_l21_v001_feature_shape_matches_known_sample() -> None:
    path = Path("data/clip-features-32/L21_V001.npy")
    if not path.exists():
        pytest.skip("local contest data is not available")

    shape, dtype = read_feature_info(path)

    assert shape == (307, 512)
    assert dtype == "float16"


def test_l21_v001_mapping_rows_matches_known_sample() -> None:
    path = Path("data/map-keyframes/L21_V001.csv")
    if not path.exists():
        pytest.skip("local contest data is not available")

    assert count_mapping_rows(path) == 307


def test_scan_data_root_records_optional_raw_video(tmp_path: Path) -> None:
    video_dir = tmp_path / "data" / "videos"
    video_dir.mkdir(parents=True)
    video = video_dir / "L21_V001.mp4"
    video.write_bytes(b"fake-video")

    assets = scan_data_root(tmp_path / "data")
    validation = validate_assets(assets)

    assert len(assets) == 1
    assert assets[0].video_id == "L21_V001"
    assert assets[0].video_path == "data/videos/L21_V001.mp4"
    assert assets[0].video_file_size_bytes == len(b"fake-video")
    assert validation.raw_video_group_counts == {"L21": 1}
