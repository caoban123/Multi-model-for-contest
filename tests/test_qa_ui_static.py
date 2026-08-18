from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qa_workspace_is_evidence_first_and_has_review_controls() -> None:
    html = (ROOT / "web/retrieval_ui/index.html").read_text(encoding="utf-8")
    assert 'id="qa-event-query"' in html
    assert 'id="qa-question"' in html
    assert 'id="qa-evidence"' in html
    assert 'id="qa-confirm-button"' in html
    assert "Answers are proposals only" in html


def test_qa_ui_calls_guarded_api_routes() -> None:
    app = (ROOT / "web/retrieval_ui/app.js").read_text(encoding="utf-8")
    assert "/api/qa/prepare" in app
    assert "/api/qa/draft-answer" in app
    assert "/api/qa/review" in app
    assert "/api/qa/export" in app
    assert "/api/qa/sessions" in app
    assert "qaConfirmButton.disabled" in app
    assert "selected_evidence_ids" in app
