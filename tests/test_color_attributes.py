from pathlib import Path

import pytest

from aic_retrieval.color_attributes import ColorAttributeConstraint, ColorAttributeService, parse_color_constraint
from aic_retrieval.search import FrameRef


Image = pytest.importorskip("PIL.Image")


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (24, 24), color)
    image.save(path)


def test_parse_color_constraint_extracts_query_color_and_clothing_target() -> None:
    constraint = parse_color_constraint("person wearing a red shirt")

    assert constraint is not None
    assert constraint.color == "red"
    assert constraint.target == "clothing"
    assert constraint.source == "query"


def test_color_attribute_service_ranks_matching_keyframes(tmp_path: Path) -> None:
    red = tmp_path / "data" / "keyframes" / "L21_V001" / "001.jpg"
    blue = tmp_path / "data" / "keyframes" / "L21_V002" / "001.jpg"
    make_image(red, (220, 20, 20))
    make_image(blue, (20, 20, 220))
    refs = [
        FrameRef("L21_V001", "L21", 1, 0, 0.0, 30.0, "data/keyframes/L21_V001/001.jpg"),
        FrameRef("L21_V002", "L21", 1, 0, 0.0, 30.0, "data/keyframes/L21_V002/001.jpg"),
    ]
    service = ColorAttributeService(tmp_path, refs)

    payload = service.search(ColorAttributeConstraint("red", min_ratio=0.1))

    assert [(item["video_id"], item["keyframe_id"]) for item in payload["results"]] == [("L21_V001", 1)]
    assert payload["results"][0]["attribute_rank"] == 1
    assert payload["results"][0]["attribute_score"] > 0.9
