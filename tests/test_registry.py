from pathlib import Path

import pytest

from aic_retrieval.registry import count_mapping_rows, read_feature_info


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
