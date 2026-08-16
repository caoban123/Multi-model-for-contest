import json

import numpy as np
import pytest

import tools.text_query_benchmark as text_query_benchmark
from aic_retrieval.search import FrameRef, save_numpy_index


def test_load_queries_validates_unique_ids(tmp_path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {"id": "q001", "text": "first query"},
                {"id": "q001", "text": "second query"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate query id"):
        text_query_benchmark.load_queries(path)


def test_text_query_benchmark_writes_json_and_csv(tmp_path, monkeypatch, capsys) -> None:
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps([{"id": "q001", "text": "red car"}]), encoding="utf-8")

    index_dir = tmp_path / "index"
    index = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    refs = [
        FrameRef("V1", "L21", 1, 10, 0.0, 30.0, None),
        FrameRef("V2", "L21", 1, 20, 0.0, 30.0, "data/keyframes/V2/001.jpg"),
        FrameRef("V3", "L21", 1, 30, 0.0, 30.0, "data/keyframes/V3/001.jpg"),
    ]
    save_numpy_index(index_dir, index, refs, {"groups": ["L21"]})

    class FakeEncoder:
        def __init__(self, model_id, cache_dir, local_files_only):
            assert model_id == "mock-model"
            assert cache_dir is None
            assert local_files_only is True

        def encode_texts(self, texts):
            assert texts == ["red car"]
            return np.array([[0.0, 1.0]], dtype=np.float32)

    output = tmp_path / "benchmark.json"
    csv_output = tmp_path / "benchmark.csv"
    monkeypatch.setattr(text_query_benchmark, "ClipTextEncoder", FakeEncoder)
    monkeypatch.setattr(
        "sys.argv",
        [
            "text_query_benchmark.py",
            "--queries",
            str(queries_path),
            "--index-dir",
            str(index_dir),
            "--allow-stale-index",
            "--clip-model-id",
            "mock-model",
            "--clip-local-files-only",
            "--output",
            str(output),
            "--csv-output",
            str(csv_output),
            "--candidate-pool",
            "2",
            "--top-k",
            "1",
        ],
    )

    assert text_query_benchmark.main() == 0
    assert '"query_count": 1' in capsys.readouterr().out

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["queries"][0]["results"][0]["video_id"] == "V2"
    assert len(payload["queries"][0]["raw_results"]) == 2
    assert payload["queries"][0]["video_results"][0]["video_id"] == "V2"
    assert payload["queries"][0]["video_results"][0]["aggregation_method"] == "max"
    assert payload["aggregation_method"] == "max"
    assert payload["retrieval_latency_ms"]["mean"] >= 0
    assert payload["aggregation_latency_ms"]["mean"] >= 0
    csv_text = csv_output.read_text(encoding="utf-8")
    assert "manual_judgement" in csv_text
    assert "data/keyframes/V2/001.jpg" in csv_text


def test_text_query_benchmark_can_export_raw_results_to_csv(tmp_path, monkeypatch) -> None:
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps([{"id": "q001", "text": "red car"}]), encoding="utf-8")

    index_dir = tmp_path / "index"
    index = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    refs = [
        FrameRef("V1", "L21", 1, 10, 0.0, 30.0, None),
        FrameRef("V2", "L21", 1, 20, 0.0, 30.0, "data/keyframes/V2/001.jpg"),
        FrameRef("V3", "L21", 1, 30, 0.0, 30.0, "data/keyframes/V3/001.jpg"),
    ]
    save_numpy_index(index_dir, index, refs, {"groups": ["L21"]})

    class FakeEncoder:
        def __init__(self, model_id, cache_dir, local_files_only):
            pass

        def encode_texts(self, texts):
            return np.array([[0.0, 1.0]], dtype=np.float32)

    output = tmp_path / "benchmark.json"
    csv_output = tmp_path / "benchmark.csv"
    monkeypatch.setattr(text_query_benchmark, "ClipTextEncoder", FakeEncoder)
    monkeypatch.setattr(
        "sys.argv",
        [
            "text_query_benchmark.py",
            "--queries",
            str(queries_path),
            "--index-dir",
            str(index_dir),
            "--allow-stale-index",
            "--clip-model-id",
            "mock-model",
            "--clip-local-files-only",
            "--top-k",
            "1",
            "--candidate-pool",
            "3",
            "--csv-result-set",
            "raw_results",
            "--output",
            str(output),
            "--csv-output",
            str(csv_output),
        ],
    )

    assert text_query_benchmark.main() == 0
    csv_text = csv_output.read_text(encoding="utf-8")
    assert csv_text.count("\n") == 4
    assert "V3" in csv_text
