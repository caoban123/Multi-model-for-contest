import csv
import json

from aic_retrieval.judgements import load_judgement_rows, summarize_judgements
import tools.inspect_judged_objects as inspect_judged_objects


def test_summarize_judgements_counts_top1_and_query_metrics() -> None:
    rows = [
        {"query_id": "q001", "query_text": "car", "rank": "1", "manual_judgement": "good"},
        {"query_id": "q001", "query_text": "car", "rank": "2", "manual_judgement": "bad"},
        {"query_id": "q002", "query_text": "phone", "rank": "1", "manual_judgement": "bad"},
        {"query_id": "q002", "query_text": "phone", "rank": "2", "manual_judgement": "partial"},
    ]

    summary = summarize_judgements(rows)

    assert summary["rows"] == 4
    assert summary["totals"] == {"good": 1, "partial": 1, "bad": 2, "blank": 0}
    assert summary["top1"] == {"good": 1, "partial": 0, "bad": 1, "blank": 0}
    assert summary["queries"][0]["precision_good_at_k"] == 0.5


def test_load_judgement_rows_reads_csv(tmp_path) -> None:
    path = tmp_path / "judged.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rank", "manual_judgement"])
        writer.writeheader()
        writer.writerow({"query_id": "q001", "rank": "1", "manual_judgement": "good"})

    assert load_judgement_rows(path)[0]["manual_judgement"] == "good"


def test_inspect_judged_objects_loads_entities(tmp_path) -> None:
    object_dir = tmp_path / "objects" / "V1"
    object_dir.mkdir(parents=True)
    (object_dir / "001.json").write_text(
        json.dumps(
            {
                "detection_class_entities": ["Mobile phone", "Human hand"],
                "detection_scores": ["0.9", "0.5"],
            }
        ),
        encoding="utf-8",
    )

    entities = inspect_judged_objects.load_top_entities(tmp_path / "objects", "V1", "1", 1)

    assert entities == [{"entity": "Mobile phone", "score": 0.9}]
