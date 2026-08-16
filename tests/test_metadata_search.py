import json

from aic_retrieval.metadata_search import MetadataConstraints, filter_metadata_documents, load_metadata_documents, search_metadata, tokenize, tokenize_original


def write_media(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_tokenize_is_accent_insensitive() -> None:
    assert tokenize("60 Giây Sáng") == ["60", "giay", "sang"]
    assert tokenize("Báo Tuổi Trẻ") == ["bao", "tuoi", "tre"]
    assert tokenize_original("Bóng đá") == ["bóng", "đá"]


def test_metadata_search_ranks_title_and_date(tmp_path) -> None:
    write_media(
        tmp_path / "L21_V001.json",
        {
            "title": "60 Giây Sáng - Ngày 01082024",
            "author": "60 Giây Official",
            "publish_date": "01/08/2024",
            "description": "Tin tức thời sự",
            "keywords": ["tin tức"],
            "watch_url": "https://example.com/1",
        },
    )
    write_media(
        tmp_path / "L21_V002.json",
        {
            "title": "Bản tin thể thao",
            "author": "HTV Sports",
            "publish_date": "02/08/2024",
            "description": "Bóng đá và tennis",
            "keywords": ["thể thao"],
            "watch_url": "https://example.com/2",
        },
    )

    docs = load_metadata_documents(tmp_path)
    results = search_metadata(docs, "giay sang 01082024", top_k=2)

    assert [result.video_id for result in results] == ["L21_V001"]
    assert {"giay", "sang", "01082024"}.issubset(set(results[0].matched_terms))


def test_metadata_search_filters_groups(tmp_path) -> None:
    write_media(tmp_path / "L21_V001.json", {"title": "Tin sáng", "author": "HTV"})
    write_media(tmp_path / "L22_V001.json", {"title": "Tin sáng", "author": "HTV"})

    docs = load_metadata_documents(tmp_path, groups={"L22"})
    results = search_metadata(docs, "tin sang", top_k=5)

    assert [result.video_id for result in results] == ["L22_V001"]


def test_metadata_search_can_require_multiple_matches(tmp_path) -> None:
    write_media(tmp_path / "L21_V001.json", {"title": "Tin sáng", "author": "HTV"})
    write_media(tmp_path / "L21_V002.json", {"title": "Tin tối", "author": "HTV"})

    docs = load_metadata_documents(tmp_path)
    results = search_metadata(docs, "tin sang htv", top_k=5, min_match=3)

    assert [result.video_id for result in results] == ["L21_V001"]


def test_metadata_search_prioritizes_accented_exact_phrase_over_collision(tmp_path) -> None:
    write_media(tmp_path / "L21_V001.json", {"title": "Bóng đá hôm nay", "author": "HTV Sports"})
    write_media(tmp_path / "L21_V002.json", {"title": "Bông đá trang trí", "author": "Kênh thủ công"})

    docs = load_metadata_documents(tmp_path)

    accented = search_metadata(docs, "bóng đá", top_k=2)
    unaccented = search_metadata(docs, "bong da", top_k=2)
    flower = search_metadata(docs, "bông", top_k=2)

    assert accented[0].video_id == "L21_V001"
    assert unaccented[0].video_id == "L21_V001"
    assert flower[0].video_id == "L21_V002"


def test_structured_metadata_filters_video_channel_date_duration_title_keywords(tmp_path) -> None:
    write_media(tmp_path / "L21_V001.json", {"title":"Bóng đá hôm nay", "author":"HTV Tin Tức", "channel_id":"chan-1", "publish_date":"01/08/2024", "length":120, "keywords":["thể thao", "bóng đá"]})
    write_media(tmp_path / "L22_V002.json", {"title":"Bông đá trang trí", "author":"Kênh khác", "channel_id":"chan-2", "publish_date":"10/08/2024", "length":300, "keywords":["thủ công"]})
    docs = load_metadata_documents(tmp_path)
    constraints = MetadataConstraints(groups=("L21",), video_ids=("L21_V001",), channel="HTV Tin Tức", date_from="2024-08-01", date_to="2024-08-05", duration_min=100, duration_max=150, title_phrase="Bóng đá", keywords=("thể thao",))
    results = filter_metadata_documents(docs, constraints)
    assert [item["video_id"] for item in results] == ["L21_V001"]
    assert results[0]["metadata_level"] == "video"
    assert results[0]["metadata_score"] is None
    assert {item["field"] for item in results[0]["evidence"]} >= {"group","video_id","author","publish_date","duration","title","keywords"}


def test_structured_metadata_exact_accent_evidence_ranks_before_normalized_collision(tmp_path) -> None:
    write_media(tmp_path / "L21_V001.json", {"title":"Bóng đá hôm nay"})
    write_media(tmp_path / "L21_V002.json", {"title":"Bông đá trang trí"})
    results = filter_metadata_documents(load_metadata_documents(tmp_path), MetadataConstraints(title_phrase="Bóng đá"))
    assert [item["video_id"] for item in results] == ["L21_V001", "L21_V002"]
    assert results[0]["evidence"][0]["match_kind"] == "exact_original_phrase"
    assert results[1]["evidence"][0]["match_kind"] == "normalized_accent_insensitive_phrase"


def test_structured_metadata_exact_date_and_channel_id(tmp_path) -> None:
    write_media(tmp_path / "L21_V001.json", {"title":"News", "channel_id":"abc", "publish_date":"01/08/2024", "length":10})
    docs = load_metadata_documents(tmp_path)
    assert len(filter_metadata_documents(docs, MetadataConstraints(channel="abc", publish_date="01-08-2024"))) == 1
    assert filter_metadata_documents(docs, MetadataConstraints(duration_min=11)) == []
