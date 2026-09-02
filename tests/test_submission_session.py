from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from aic_retrieval.submission_official import OfficialPrediction, OfficialQuery
from aic_retrieval.submission_session import SubmissionSessionStore


def store(tmp_path: Path) -> SubmissionSessionStore:
    return SubmissionSessionStore(tmp_path / "sessions.sqlite3", tmp_path / "outputs")


def test_session_persists_replace_remove_validate_and_finalize(tmp_path: Path) -> None:
    first = store(tmp_path)
    session_id = first.start()["session_id"]
    kis = OfficialQuery("query-1-kis", "KIS", (OfficialPrediction("L21_V001", (1234,)),))
    state = first.confirm_query(session_id, kis, mapping_sources=("manual_official",), source={"candidate_id": "r1"})
    assert state["submission_queue"][0]["output_file"] == "query-1-kis.csv"
    assert state["queries"][0]["csv_preview"] == "L21_V001, 1234\n"

    replacement = OfficialQuery("query-1-kis", "KIS", (OfficialPrediction("L21_V002", (999,)),))
    first.confirm_query(session_id, replacement, mapping_sources=("btc_mapping",))
    reopened = store(tmp_path).get(session_id)
    assert reopened["queries"][0]["predictions"][0]["video_id"] == "L21_V002"
    assert first.validate(session_id)["valid"] is True

    qa = OfficialQuery("query-2-qa", "QA", (OfficialPrediction("L21_V003", (88,), "short answer"),))
    first.confirm_query(session_id, qa, mapping_sources=("manual_official",))
    done = first.done(session_id)
    assert done["status"] == "DONE"
    with zipfile.ZipFile(done["zip_path"]) as archive:
        assert archive.namelist() == ["submission/query-1-kis.csv", "submission/query-2-qa.csv"]


def test_session_rejects_uncertified_mapping_and_invalid_query(tmp_path: Path) -> None:
    value = store(tmp_path)
    session_id = value.start()["session_id"]
    query = OfficialQuery("query-1-kis", "KIS", (OfficialPrediction("L21_V001", (1,)),))
    with pytest.raises(ValueError, match="uncertified"):
        value.confirm_query(session_id, query, mapping_sources=("guessed_from_keyframe",))
    with pytest.raises(ValueError, match="mapping source"):
        value.confirm_query(session_id, query, mapping_sources=())
    assert value.validate(session_id)["issues"][0]["code"] == "EMPTY_SUBMISSION"


def test_remove_and_done_session_guards(tmp_path: Path) -> None:
    value = store(tmp_path)
    session_id = value.start()["session_id"]
    query = OfficialQuery("query-4-trake", "TRAKE", (OfficialPrediction("L21_V001", (10, 20)),), event_count=2)
    value.confirm_query(session_id, query, mapping_sources=("manual_official",))
    emptied = value.remove_query(session_id, "query-4-trake")
    assert emptied["queries"] == []
    with pytest.raises(ValueError, match="validation failed"):
        value.done(session_id)
