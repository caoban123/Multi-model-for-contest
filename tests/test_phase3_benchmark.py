import csv
import json

import pytest

import tools.phase3_benchmark as phase3_benchmark


def test_phase3_benchmark_compares_frame_and_video_diversity(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "benchmark_version": "3.0",
                "candidate_pool": 4,
                "queries": [
                    {
                        "id": "q001",
                        "text": "person",
                        "query_type": "action",
                        "language": "en",
                        "retrieval_ms": 2.5,
                        "raw_results": [
                            _frame(1, "V1", 1, 0.90),
                            _frame(2, "V1", 2, 0.85),
                            _frame(3, "V2", 1, 0.80),
                            _frame(4, "V3", 1, 0.70),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    judgements = tmp_path / "judgements.csv"
    with judgements.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "video_id", "keyframe_id", "manual_judgement"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"query_id": "q001", "video_id": "V1", "keyframe_id": 1, "manual_judgement": "good"},
                {"query_id": "q001", "video_id": "V2", "keyframe_id": 1, "manual_judgement": "bad"},
                {"query_id": "q001", "video_id": "V3", "keyframe_id": 1, "manual_judgement": "partial"},
            ]
        )
    output = tmp_path / "phase3.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "phase3_benchmark.py",
            "--input",
            str(source),
            "--judgements-csv",
            str(judgements),
            "--top-k",
            "3",
            "--output",
            str(output),
        ],
    )

    assert phase3_benchmark.main() == 0
    assert '"top_frame_video_preservation_rate": 1.0' in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["quality_status"] == "AVAILABLE"
    assert payload["overall"]["frame_ranking"]["mean_unique_video_at_k"] == pytest.approx(2 / 3)
    assert payload["overall"]["video_ranking"]["mean_unique_video_at_k"] == pytest.approx(1.0)
    assert payload["overall"]["frame_ranking"]["mean_duplicate_result_count"] == 1
    assert payload["overall"]["video_ranking"]["mean_duplicate_result_count"] == 0
    assert payload["overall"]["video_quality"]["judged_result_count"] == 3
    assert payload["by_query_type"]["action"]["top_frame_video_preservation_rate"] == 1.0


def test_phase3_benchmark_rejects_missing_raw_results(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"queries": [{"id": "q001"}]}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["phase3_benchmark.py", "--input", str(source)])

    with pytest.raises(ValueError, match="raw_results"):
        phase3_benchmark.main()


def _frame(rank: int, video_id: str, keyframe_id: int, score: float) -> dict:
    return {
        "rank": rank,
        "score": score,
        "video_id": video_id,
        "group": "L21",
        "keyframe_id": keyframe_id,
        "frame_idx": keyframe_id * 10,
        "pts_time": float(keyframe_id),
        "fps": 30.0,
        "keyframe_path": None,
    }
