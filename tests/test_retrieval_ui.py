import json
from pathlib import Path

import pytest

from aic_retrieval.metadata_search import load_metadata_documents
from aic_retrieval.retrieval_ui import RetrievalUiConfig, RetrievalUiService, parse_int


def make_service(tmp_path: Path) -> RetrievalUiService:
    media_dir = tmp_path / "data" / "media-info"
    media_dir.mkdir(parents=True)
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
    service.assets_by_video = {"L21_V001": {"media_info_path": "data/media-info/L21_V001.json"}}
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
