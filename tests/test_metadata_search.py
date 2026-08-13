import json

from aic_retrieval.metadata_search import load_metadata_documents, search_metadata, tokenize, tokenize_original


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
