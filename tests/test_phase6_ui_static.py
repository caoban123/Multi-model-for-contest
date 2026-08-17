from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_phase6_controls_and_preview_are_opt_in():
    html=(ROOT/"web/retrieval_ui/index.html").read_text(encoding="utf-8")
    app=(ROOT/"web/retrieval_ui/app.js").read_text(encoding="utf-8")
    assert 'id="enable-query-planner" type="checkbox"' in html
    assert 'id="enable-reranker" type="checkbox"' in html
    assert 'id="reranker-top-n"' in html
    assert "/api/query-plan" in app
    assert "enable_query_planner: enableQueryPlannerInput.checked" in app
    assert "enable_reranker: enableRerankerInput.checked" in app
    assert "data-query-variant" in app
