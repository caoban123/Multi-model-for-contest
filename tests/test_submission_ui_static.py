from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_submission_workspace_exposes_guarded_kis_flow() -> None:
    html = (ROOT / "web/submission_ui/index.html").read_text(encoding="utf-8")
    app = (ROOT / "web/submission_ui/app.js").read_text(encoding="utf-8")
    styles = (ROOT / "web/submission_ui/styles.css").read_text(encoding="utf-8")
    for item in ('id="start-button"', 'id="query-id"', 'id="use-hybrid"', 'id="candidates"', 'id="confirm-button"', 'id="validate-button"', 'id="done-button"', 'id="download-link"'):
        assert item in html
    for route in ("/api/submission/session/start", "/api/submission/agent/run", "/api/submission/candidate/refine", "/api/submission/query/confirm", "/api/submission/query/remove", "/api/submission/session/validate", "/api/submission/session/done"):
        assert route in app
    assert "manual_official" in app
    assert "official-input" in app
    assert "/api/neighborhood?" in app
    assert "/api/submission/import/qa" in app
    assert "/api/submission/import/trake" in app
    assert 'value="QA"' in html and 'value="TRAKE"' in html
    for item in ('id="qa-question"', 'id="task-workflow"', 'id="workflow-content"', 'id="image-dialog"'):
        assert item in html
    for route in ("/api/qa/prepare", "/api/qa/draft-answer", "/api/qa/review", "/api/trake/plan", "/api/trake/search", "/api/trake/align", "/api/trake/review"):
        assert route in app
    assert "data-radius" in app
    assert "showModal" in app
    for item in ('id="agent-inspector"', 'id="gemini-state"', 'id="gemini-raw"', 'id="gemini-parsed"', 'id="validated-plan"'):
        assert item in html
    for item in ('id="csv-preview-list"', 'id="csv-file-count"'):
        assert item in html
    assert "agent_trace" in app
    assert "retriever_ranks" in app
    assert "CLIP visual queries" in html
    assert "visual_clip_queries_en" in app
    assert 'id="use-dense-frames"' in html
    assert "use_dense" in app
    assert "dense-frame-badge" in styles
    assert "Dense frame" in app
    assert "CLIP Q" in app
    assert 'id="clip-lanes"' in html
    assert "renderClipLanes" in app
    assert "variant_rescue" in app
    assert "temporal_sequence" in app
    assert "Sequence keeps events in time order" in app
    assert "SEQ ${Number(sequence.matched_events)}" in app
    assert "evidence-chip.is-sequence" in styles
    assert "/api/time-to-frame?" in app
    assert "timeFrameTool" in app
    assert "showSaveFilePicker" in app
    assert "csvEditorStates" in app
    assert "renderCsvPreviews" in app
    assert "focusCsvPreview" in app
    assert "/api/submission/query/csv" in app
    assert "saveCsvChanges" in app
    assert "textarea.readOnly = session.status !== 'ACTIVE'" in app
    assert "download.onclick = () => saveCsv(state.queryId)" in app
    for feature in ("activeQaVideoId", "renderQaCandidates", "Only evidence from this video", "updateTrakePlan", "addTrakeEvent", "moveTrakeEvent", "removeTrakeEvent", "replaceTrakeCandidate", "refineTrakeCandidate"):
        assert feature in app
    for feature in ("refineCandidate", "renderCandidateRefinement", "applyRefinedFrame", "data-refine-candidate", "data-use-refined-frame"):
        assert feature in app
    for route in ("/api/trake/update-plan", "/api/trake/replace", "/api/trake/refine"):
        assert route in app


def test_submission_workspace_has_responsive_candidate_and_validation_styles() -> None:
    css = (ROOT / "web/submission_ui/styles.css").read_text(encoding="utf-8")
    for selector in (".workspace", ".candidate", ".qa-candidate", ".candidate-refinement", ".refinement-frame", ".neighbors", ".task-workflow", ".image-dialog", ".trake-chain", ".trake-plan-editor", ".trake-alternative", ".clip-lanes", ".lane-button", ".time-frame-tool", ".preview-head", ".csv-preview-list", ".csv-file-item", ".preview-actions", ".validation-summary", "@media (max-width: 680px)"):
        assert selector in css
