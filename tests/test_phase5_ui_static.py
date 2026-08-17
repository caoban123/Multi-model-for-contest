from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_phase5_controls_are_opt_in_and_have_evidence_rendering():
    html=(ROOT/"web/retrieval_ui/index.html").read_text(encoding="utf-8")
    app=(ROOT/"web/retrieval_ui/app.js").read_text(encoding="utf-8")
    assert 'id="enable-ocr" type="checkbox"' in html
    assert 'id="enable-asr" type="checkbox"' in html
    assert "enable_ocr: ocrSearchAvailable && enableOcrInput.checked" in app
    assert "enable_asr: asrSearchAvailable && enableAsrInput.checked" in app
    assert "frame.phase5_evidence" in app

def test_phase5_health_capabilities_are_rendered():
    app=(ROOT/"web/retrieval_ui/app.js").read_text(encoding="utf-8")
    assert "payload.ocr_search_available" in app
    assert "payload.asr_search_available" in app
