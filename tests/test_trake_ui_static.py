from pathlib import Path


def test_trake_workspace_is_separate_and_exposes_required_controls() -> None:
    root=Path(__file__).parents[1]; html=(root/'web/trake_ui/index.html').read_text(encoding='utf-8'); js=(root/'web/trake_ui/app.js').read_text(encoding='utf-8')
    for text in ('TRAKE workspace','Event editor','Retrieved video (stage 1)','TRAKE answer (stage 2)','Manual correction & review','Confirm','Export internal record'):
        assert text in html
    for route in ("api('plan'","api('search'","api('align'","api('review'","api('export'"):
        assert route in js
    assert 'keyframeImage' in js
    assert '/keyframe?path=' in js
    assert 'chain-events' in js
    assert 'V2 debug & latency' in html
    assert "$('debug').textContent" in js
    assert 'Score breakdown' in js
    assert 'escapeHtml' in js
    assert 'Aligning timeline and running bounded dense refinement' in js
    assert 'Per-event candidates only' in js
    assert 'Dense temporal/scene-consistent candidate' in js
    assert 'semantic action review required' in js
    assert 'trake_answer?.answer_text' in js
    assert 'retrieved_video_id' in js
    assert 'semantic-seed quality and temporal order' in js
    assert 'Q&A workspace' not in html
