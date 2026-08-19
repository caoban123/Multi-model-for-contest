from pathlib import Path


def test_trake_workspace_is_separate_and_exposes_required_controls() -> None:
    root=Path(__file__).parents[1]; html=(root/'web/trake_ui/index.html').read_text(encoding='utf-8'); js=(root/'web/trake_ui/app.js').read_text(encoding='utf-8')
    for text in ('TRAKE workspace','Event editor','Candidate videos','Timeline & top chains','Manual correction & review','Confirm','Export internal record'):
        assert text in html
    for route in ("api('plan'","api('search'","api('align'","api('review'","api('export'"):
        assert route in js
    assert 'keyframeImage' in js
    assert '/keyframe?path=' in js
    assert 'chain-events' in js
    assert 'Q&A workspace' not in html
