import json
from pathlib import Path

import numpy as np
import pytest

from aic_retrieval.metadata_search import load_metadata_documents
from aic_retrieval.retrieval_ui import RetrievalUiConfig, RetrievalUiService, parse_int
from aic_retrieval.search import FrameRef, SearchResult, aggregate_results_by_video


def make_service(tmp_path: Path) -> RetrievalUiService:
    media_dir = tmp_path / "data" / "media-info"
    media_dir.mkdir(parents=True)
    video_dir = tmp_path / "data" / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "L21_V001.mp4").write_bytes(b"fake video")
    (media_dir / "L21_V001.json").write_text(
        json.dumps(
            {
                "title": "60 Giây Sáng",
                "author": "60 Giây Official",
                "publish_date": "01/08/2024",
                "keywords": ["tin tức", "HTV"],
                "watch_url": "https://example.com",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.config = RetrievalUiConfig(
        repo_root=tmp_path,
        registry_path=tmp_path / "registry.json",
        index_dir=tmp_path / "index",
        metadata_dir=tmp_path / "data" / "media-info",
        static_dir=tmp_path / "web",
        groups={"L21"},
    )
    service.assets_by_video = {
        "L21_V001": {
            "media_info_path": "data/media-info/L21_V001.json",
            "video_path": "data/videos/L21_V001.mp4",
        }
    }
    service.refs_by_video = {
        "L21_V001": [
            FrameRef("L21_V001", "L21", 1, 0, 0.0, 30.0, "data/keyframes/L21_V001/001.jpg"),
            FrameRef("L21_V001", "L21", 2, 30, 1.0, 30.0, "data/keyframes/L21_V001/002.jpg"),
            FrameRef("L21_V001", "L21", 3, 60, 2.0, 30.0, "data/keyframes/L21_V001/003.jpg"),
            FrameRef("L21_V001", "L21", 4, 90, 3.0, 30.0, "data/keyframes/L21_V001/004.jpg"),
        ]
    }
    service.metadata_by_video = {}
    return service


def test_enrich_result_adds_metadata_and_image_url(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    result = service.enrich_result(
        {
            "rank": 1,
            "score": 1.0,
            "video_id": "L21_V001",
            "keyframe_id": 1,
            "keyframe_path": "data/keyframes/L21_V001/001.jpg",
        }
    )

    assert result["image_url"] == "/keyframe?path=data/keyframes/L21_V001/001.jpg"
    assert result["video_url"] == "/video?video_id=L21_V001"
    assert result["metadata"]["title"] == "60 Giây Sáng"
    assert result["metadata"]["keywords"] == ["tin tức", "HTV"]


def test_resolve_keyframe_path_rejects_outside_repo(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    with pytest.raises(PermissionError):
        service.resolve_keyframe_path("../outside.jpg")


def test_resolve_keyframe_path_accepts_existing_repo_file(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    image = tmp_path / "data" / "keyframes" / "L21_V001" / "001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")

    assert service.resolve_keyframe_path("data/keyframes/L21_V001/001.jpg") == image.resolve()


def test_resolve_video_path_accepts_registered_repo_video(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    video = tmp_path / "data" / "videos" / "L21_V001.mp4"

    assert service.video_url("L21_V001") == "/video?video_id=L21_V001"
    assert service.resolve_video_path("L21_V001") == video.resolve()


def test_resolve_video_path_rejects_missing_or_outside_video(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    with pytest.raises(FileNotFoundError):
        service.resolve_video_path("L21_V999")

    service.assets_by_video["L21_BAD"] = {"video_path": "../outside.mp4"}
    with pytest.raises(PermissionError):
        service.resolve_video_path("L21_BAD")


def test_parse_int_falls_back_for_bad_values() -> None:
    assert parse_int("12", 1) == 12
    assert parse_int("bad", 7) == 7


def test_service_translate_uses_configured_translator(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    class FakeTranslator:
        config = type("Config", (), {"model": "fake-model", "provider": "gemini"})()

        def translate_vi_to_en(self, text: str) -> str:
            assert text == "người cầm điện thoại"
            return "a person holding a phone"

    service.translator = FakeTranslator()

    assert service.translate(" người cầm điện thoại ") == {
        "source_text": "người cầm điện thoại",
        "translated_text": "a person holding a phone",
        "provider": "gemini",
        "model": "fake-model",
    }


def test_service_metadata_search_returns_video_results(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.metadata_docs = load_metadata_documents(tmp_path / "data" / "media-info", groups={"L21"})

    payload = service.metadata_search("60 giay sang", top_k=5)

    assert payload["total_documents"] == 1
    assert payload["results"][0]["video_id"] == "L21_V001"
    assert payload["results"][0]["video_url"] == "/video?video_id=L21_V001"


def test_service_keyframe_neighborhood_returns_surrounding_frames(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    payload = service.keyframe_neighborhood("L21_V001", keyframe_id=2, radius=1)

    assert payload["video_id"] == "L21_V001"
    assert payload["total_frames"] == 4
    assert [frame["keyframe_id"] for frame in payload["frames"]] == [1, 2, 3]
    assert [frame["is_center"] for frame in payload["frames"]] == [False, True, False]
    assert payload["frames"][1]["image_url"] == "/keyframe?path=data/keyframes/L21_V001/002.jpg"
    assert payload["ordering"] == "pts_time_ascending"


def test_service_keyframe_neighborhood_handles_boundaries(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    start = service.keyframe_neighborhood("L21_V001", keyframe_id=1, radius=3)
    end = service.keyframe_neighborhood("L21_V001", keyframe_id=4, radius=3)

    assert [frame["keyframe_id"] for frame in start["frames"]] == [1, 2, 3, 4]
    assert [frame["keyframe_id"] for frame in end["frames"]] == [1, 2, 3, 4]
    assert sum(frame["is_center"] for frame in start["frames"]) == 1
    assert sum(frame["is_center"] for frame in end["frames"]) == 1


def test_service_search_exposes_raw_and_video_results_from_candidate_pool(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.index = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.95, 0.05],
            [0.80, 0.20],
        ],
        dtype=np.float32,
    )
    service.index /= np.linalg.norm(service.index, axis=1, keepdims=True)
    service.refs = [
        FrameRef("L21_V001", "L21", 1, 0, 0.0, 30.0, None),
        FrameRef("L21_V001", "L21", 2, 30, 1.0, 30.0, None),
        FrameRef("L21_V002", "L21", 1, 0, 0.0, 30.0, None),
        FrameRef("L21_V003", "L21", 1, 0, 0.0, 30.0, None),
    ]
    service.assets_by_video.update({"L21_V002": {}, "L21_V003": {}})

    class FakeEncoder:
        def encode_text(self, text):
            assert text == "person"
            return np.array([1.0, 0.0], dtype=np.float32)

    service.encoder = FakeEncoder()

    payload = service.search(
        "person",
        top_k=2,
        candidate_pool=4,
        max_frames_per_video=1,
        matched_frames_per_video=2,
    )

    assert payload["aggregation_method"] == "max"
    assert payload["candidate_pool_size"] == 4
    assert len(payload["raw_results"]) == 4
    assert [item["rank"] for item in payload["raw_results"]] == [1, 2, 3, 4]
    assert [item["video_id"] for item in payload["video_results"]] == ["L21_V001", "L21_V002"]
    assert payload["video_results"][0]["frame_count"] == 2
    assert payload["video_results"][0]["matched_frame_count"] == 2
    assert len(payload["video_results"][0]["frames"]) == 2
    assert len(payload["results"]) == 2
    assert payload["aggregation_ms"] >= 0


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("top_k", 0),
        ("candidate_pool", 0),
        ("max_frames_per_video", 0),
        ("matched_frames_per_video", 0),
        ("mean_top_n", 0),
    ],
)
def test_service_search_rejects_non_positive_configuration(tmp_path: Path, argument: str, value: int) -> None:
    service = make_service(tmp_path)
    kwargs = {argument: value}

    with pytest.raises(ValueError, match=argument):
        service.search("person", **kwargs)


def test_enrich_video_result_preserves_non_l21_frame_without_image(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.assets_by_video["L23_V001"] = {}
    raw = [
        SearchResult(1, 0.9, "L23_V001", "L23", 7, 31, 1.25, 25.0, None),
    ]
    video = aggregate_results_by_video(raw, max_frames_per_video=5)[0]

    payload = service.enrich_video_result(__import__("dataclasses").asdict(video))

    assert payload["best_keyframe_path"] is None
    assert payload["image_url"] is None
    assert payload["frames"][0]["keyframe_path"] is None
    assert payload["frames"][0]["image_url"] is None
