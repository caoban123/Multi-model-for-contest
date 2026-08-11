import json

import tools.benchmark_review_html as benchmark_review_html


def test_render_review_html_includes_images_and_export_button(tmp_path) -> None:
    output = tmp_path / "review.html"
    payload = {
        "query_count": 1,
        "top_k": 1,
        "index_vectors": 2,
        "queries": [
            {
                "id": "q001",
                "text": "red car",
                "notes": "vehicle",
                "results": [
                    {
                        "rank": 1,
                        "score": 0.5,
                        "video_id": "L21_V001",
                        "keyframe_id": 1,
                        "frame_idx": 10,
                        "pts_time": 1.0,
                        "keyframe_path": "data/keyframes/L21_V001/001.jpg",
                    }
                ],
            }
        ],
    }

    html = benchmark_review_html.render_review_html(payload, output)

    assert "L21 Text Benchmark Review" in html
    assert "Export CSV" in html
    assert "red car" in html
    assert "manual_judgement" in html
    assert "data/keyframes/L21_V001/001.jpg" in html
    assert '<input type="radio"' in html


def test_render_review_html_can_use_raw_results(tmp_path) -> None:
    output = tmp_path / "review.html"
    payload = {
        "_review_result_set": "raw_results",
        "queries": [
            {
                "id": "q001",
                "text": "red car",
                "results": [],
                "raw_results": [
                    {
                        "rank": 1,
                        "score": 0.5,
                        "video_id": "L21_V099",
                        "keyframe_id": 7,
                        "frame_idx": 70,
                        "pts_time": 7.0,
                        "keyframe_path": "data/keyframes/L21_V099/007.jpg",
                    }
                ],
            }
        ],
    }

    html = benchmark_review_html.render_review_html(payload, output)

    assert "L21_V099" in html
    assert "Review set: raw_results" in html


def test_review_html_cli_writes_file(tmp_path, monkeypatch, capsys) -> None:
    payload_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "review.html"
    payload_path.write_text(
        json.dumps(
            {
                "query_count": 1,
                "queries": [{"id": "q001", "text": "red car", "results": []}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_review_html.py",
            "--input",
            str(payload_path),
            "--output",
            str(output_path),
        ],
    )

    assert benchmark_review_html.main() == 0
    assert output_path.exists()
    assert '"query_count": 1' in capsys.readouterr().out
