from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_submission_workspace_exposes_guarded_kis_flow() -> None:
    html = (ROOT / "web/submission_ui/index.html").read_text(encoding="utf-8")
    app = (ROOT / "web/submission_ui/app.js").read_text(encoding="utf-8")
    for item in ('id="start-button"', 'id="query-id"', 'id="use-hybrid"', 'id="candidates"', 'id="confirm-button"', 'id="validate-button"', 'id="done-button"', 'id="download-link"'):
        assert item in html
    for route in ("/api/submission/session/start", "/api/submission/agent/run", "/api/submission/query/confirm", "/api/submission/query/remove", "/api/submission/session/validate", "/api/submission/session/done"):
        assert route in app
    assert "manual_official" in app
    assert "official-input" in app
    assert "/api/neighborhood?" in app
    assert "/api/submission/import/qa" in app
    assert "/api/submission/import/trake" in app
    assert 'value="QA"' in html and 'value="TRAKE"' in html
    assert 'id="workflow-session-id"' in html
    for item in ('id="agent-inspector"', 'id="gemini-state"', 'id="gemini-raw"', 'id="gemini-parsed"', 'id="validated-plan"'):
        assert item in html
    assert "agent_trace" in app
    assert "retriever_ranks" in app


def test_submission_workspace_has_responsive_candidate_and_validation_styles() -> None:
    css = (ROOT / "web/submission_ui/styles.css").read_text(encoding="utf-8")
    for selector in (".workspace", ".candidate", ".neighbors", ".validation-summary", "@media (max-width: 680px)"):
        assert selector in css
