import json

import tools.benchmark_metadata as benchmark_metadata


def test_metadata_benchmark_reports_recall_mrr_and_query_types(tmp_path, monkeypatch) -> None:
    media_dir = tmp_path / "metadata"
    media_dir.mkdir()
    (media_dir / "L21_V001.json").write_text(json.dumps({"title": "Bóng đá hôm nay", "author": "HTV Sports"}, ensure_ascii=False), encoding="utf-8")
    (media_dir / "L21_V002.json").write_text(json.dumps({"title": "Bông đá trang trí", "author": "Thủ công"}, ensure_ascii=False), encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"benchmark_version": "fixture", "queries": [{"id": "m1", "query": "bóng đá", "query_type": "keyword", "expected_video_ids": ["L21_V001"]}]}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", ["benchmark_metadata.py", "--queries", str(queries), "--media-dir", str(media_dir), "--output", str(output)])

    assert benchmark_metadata.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["video_recall_at_1"] == 1.0
    assert payload["metrics"]["mrr"] == 1.0
    assert payload["query_type_breakdown"]["keyword"]["video_recall_at_5"] == 1.0
