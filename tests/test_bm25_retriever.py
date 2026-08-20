from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aic_retrieval.bm25_retriever import Bm25Retriever, _fts_query, build_bm25_index
from aic_retrieval.retrievers import RetrievalRequest


def _write_corpus(root: Path) -> tuple[Path, Path]:
    corpus_dir = root / "artifacts" / "corpora" / "fixture"
    corpus_dir.mkdir(parents=True)
    documents = [
        {
            "document_id": "metadata:one",
            "source_type": "metadata",
            "video_id": "L21_V001",
            "text": "60 Giây Sáng ngày 31/08/2024",
            "text_normalized": "60 giây sáng ngày 31/08/2024",
            "text_folded": "60 giay sang ngay 31/08/2024",
            "keyframe_id": None,
            "frame_idx": None,
            "pts_time": None,
            "source_field": "title",
            "provenance": {"field": "title"},
        },
        {
            "document_id": "asr:two",
            "source_type": "asr",
            "video_id": "L21_V002",
            "text": "Trung tâm điều phối ghép tạng quốc gia",
            "text_normalized": "trung tâm điều phối ghép tạng quốc gia",
            "text_folded": "trung tam dieu phoi ghep tang quoc gia",
            "keyframe_id": 5,
            "frame_idx": 42,
            "pts_time": 1.4,
            "source_field": "transcript_segment",
            "provenance": {"segment_id": 2},
        },
    ]
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in documents)
    corpus_path = corpus_dir / "documents.jsonl"
    corpus_path.write_text(payload, encoding="utf-8", newline="\n")
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "group": "L21",
                "corpus_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "document_count": 2,
                "input_fingerprint": "fixture",
            }
        ),
        encoding="utf-8",
    )
    return corpus_path, manifest_path


def test_fts_query_normalizes_and_folds_vietnamese() -> None:
    query = _fts_query("Ghép tạng")
    assert '"ghép"' in query
    assert '"ghep"' in query
    assert '"31082024"' in _fts_query("ngày 31 tháng 8 năm 2024")
    assert '"31082024"' in _fts_query("31/08/2024")
    with pytest.raises(ValueError):
        _fts_query("---")


def test_build_search_filter_and_health(tmp_path: Path) -> None:
    corpus_path, manifest_path = _write_corpus(tmp_path)
    output_dir = tmp_path / "artifacts" / "indexes" / "fixture_bm25"
    manifest = build_bm25_index(
        root=tmp_path,
        corpus_path=corpus_path,
        corpus_manifest_path=manifest_path,
        output_dir=output_dir,
    )
    assert manifest["database"]["integrity_check"] == "ok"
    assert manifest["database"]["document_count"] == 2

    retriever = Bm25Retriever(tmp_path, output_dir)
    hits = retriever.search(RetrievalRequest("q1", "ghep tang", top_k=5))
    assert [hit.video_id for hit in hits] == ["L21_V002"]
    assert hits[0].rank == 1
    assert hits[0].keyframe_id == 5
    assert hits[0].raw_score > 0

    filtered = retriever.search(
        RetrievalRequest("q2", "ngay 31 08 2024", top_k=5, filters={"source_types": ("asr",)})
    )
    assert filtered == []
    assert retriever.search(RetrievalRequest("q3", "ghep tang", groups=("L22",), top_k=5)) == []
    assert retriever.health().status == "DEGRADED"


def test_retriever_rejects_corrupted_database(tmp_path: Path) -> None:
    corpus_path, manifest_path = _write_corpus(tmp_path)
    output_dir = tmp_path / "index"
    build_bm25_index(root=tmp_path, corpus_path=corpus_path, corpus_manifest_path=manifest_path, output_dir=output_dir)
    with (output_dir / "documents.sqlite3").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        Bm25Retriever(tmp_path, output_dir)
