from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest

from aic_retrieval.submission_official import (
    OfficialPrediction,
    OfficialQuery,
    build_submission_zip,
    validate_official_query,
    validate_submission_zip,
    write_official_csv,
)


def codes(query: OfficialQuery) -> set[str]:
    return {issue.code for issue in validate_official_query(query)}


def test_kis_writer_has_no_header_and_no_debug_fields(tmp_path: Path) -> None:
    query = OfficialQuery("query-1-kis", "KIS", (OfficialPrediction("L21_V001", (1234,)),))
    path = tmp_path / query.filename
    write_official_csv(query, path)
    assert path.read_text(encoding="utf-8") == "L21_V001,1234\n"


def test_qa_writer_uses_standard_csv_quoting_and_character_limit(tmp_path: Path) -> None:
    query = OfficialQuery("query-3-qa", "QA", (OfficialPrediction("L21_V011", (1450,), 'Co 3 nguoi, noi "xin chao"'),))
    path = tmp_path / query.filename
    write_official_csv(query, path)
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [["L21_V011", "1450", 'Co 3 nguoi, noi "xin chao"']]
    assert "ANSWER_TOO_LONG" in codes(OfficialQuery("query-3-qa", "QA", (OfficialPrediction("L21_V011", (1,), "x" * 101),)))


def test_trake_requires_exact_event_count_and_warns_on_frame_order() -> None:
    wrong_count = OfficialQuery("query-4-trake", "TRAKE", (OfficialPrediction("L21_V001", (10, 20)),), event_count=3)
    assert "INVALID_COLUMN_COUNT" in codes(wrong_count)
    order = OfficialQuery("query-4-trake", "TRAKE", (OfficialPrediction("L21_V001", (20, 10)),), event_count=2)
    assert "FRAME_ORDER_FALLBACK" in codes(order)


def test_invalid_query_video_frame_and_duplicate_are_rejected() -> None:
    query = OfficialQuery("bad", "KIS", (
        OfficialPrediction("L21_V001.mp4", (-1,)),
        OfficialPrediction("L21_V001.mp4", (-1,)),
    ))
    assert {"INVALID_QUERY_ID", "INVALID_VIDEO_ID", "INVALID_FRAME_ID", "DUPLICATE_PREDICTION"} <= codes(query)


def test_zip_contains_submission_directory_and_exact_files(tmp_path: Path) -> None:
    queries = (
        OfficialQuery("query-1-kis", "KIS", (OfficialPrediction("L21_V001", (1234,)),)),
        OfficialQuery("query-3-qa", "QA", (OfficialPrediction("L21_V011", (1450,), "answer"),)),
    )
    path = build_submission_zip(queries, tmp_path)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["submission/query-1-kis.csv", "submission/query-3-qa.csv"]
    assert validate_submission_zip(path, [query.filename for query in queries]) == ()


def test_writer_refuses_invalid_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ANSWER_REQUIRED"):
        write_official_csv(
            OfficialQuery("query-2-qa", "QA", (OfficialPrediction("L21_V001", (1,), None),)),
            tmp_path / "bad.csv",
        )
