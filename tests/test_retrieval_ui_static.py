from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_ui_contains_review_controls() -> None:
    html = (ROOT / "web" / "retrieval_ui" / "index.html").read_text(encoding="utf-8")

    assert 'id="export-button"' in html
    assert 'data-judgement="good"' in html
    assert 'data-judgement="partial"' in html
    assert 'data-judgement="bad"' in html
    assert 'class="note-input"' in html


def test_retrieval_ui_exports_benchmark_compatible_csv_fields() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")

    for field in (
        "query_id",
        "query_text",
        "clip_query",
        "rank",
        "video_id",
        "keyframe_id",
        "score",
        "manual_judgement",
        "manual_notes",
    ):
        assert f'"{field}"' in script
    assert "csvEscape" in script
