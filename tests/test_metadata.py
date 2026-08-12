import json

from aic_retrieval.metadata import (
    audit_media_info,
    has_mojibake_markers,
    has_vietnamese_text,
    normalize_metadata_text,
)


def test_metadata_text_checks_encoding_markers() -> None:
    assert has_vietnamese_text("60 Giây Sáng")
    assert not has_mojibake_markers("60 Giây Sáng")
    assert has_mojibake_markers("60 Gi\u00c3\u00a2y")


def test_normalize_metadata_text_collapses_spaces() -> None:
    assert normalize_metadata_text("  Tin   tức\n\n\nmới  ") == "Tin tức\n\nmới"


def test_audit_media_info_counts_fields(tmp_path) -> None:
    (tmp_path / "V1.json").write_text(
        json.dumps(
            {
                "title": "60 Giây Sáng",
                "author": "HTV",
                "description": "Tin tức",
                "keywords": ["Tin tức"],
                "publish_date": "01/08/2024",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit = audit_media_info(tmp_path)

    assert audit.total_files == 1
    assert audit.json_errors == 0
    assert audit.files_with_mojibake_markers == 0
    assert audit.files_with_vietnamese_text == 1
    assert audit.field_presence["title"] == 1
