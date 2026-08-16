from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_ui_contains_review_controls() -> None:
    html = (ROOT / "web" / "retrieval_ui" / "index.html").read_text(encoding="utf-8")

    assert 'id="export-button"' in html
    assert 'id="export-pins-button"' in html
    assert 'id="clear-history-button"' in html
    assert 'id="pins-list"' in html
    assert 'id="history-list"' in html
    assert 'id="visual-mode-button"' in html
    assert 'id="metadata-mode-button"' in html
    assert 'id="video-ranking-button"' in html
    assert 'id="frame-ranking-button"' in html
    assert 'class="pin-button secondary-button"' in html
    assert 'class="explore-button secondary-button"' in html
    assert 'class="open-video-button secondary-button"' in html
    assert 'id="video-dialog"' in html
    assert 'id="video-player"' in html
    assert 'class="explore-panel"' in html
    assert 'class="neighborhood-button secondary-button"' in html
    assert 'class="neighborhood-panel"' in html
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
        "is_pinned",
        "pinned_at",
        "video_url",
        "watch_url",
        "attribute_score",
        "attribute_rank",
        "matched_attributes",
        "manual_judgement",
        "manual_notes",
    ):
        assert f'"{field}"' in script
    assert "csvEscape" in script


def test_retrieval_ui_uses_local_storage_for_pins_and_history() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")

    assert "aic_retrieval_history_v1" in script
    assert "aic_retrieval_pins_v1" in script
    assert "buildPinnedCsvRows" in script
    assert "syncPinnedReview" in script
    assert "submission_video_id" in script
    assert "submission_pts_time" in script
    assert "Exported ${rows.length - 1} pinned rows." in script


def test_retrieval_ui_has_metadata_mode() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")

    assert "/api/metadata-search" in script
    assert 'activeMode = "visual"' in script
    assert 'setMode("metadata")' in script
    assert "normalizeResult" in script


def test_retrieval_ui_has_keyframe_neighborhood_viewer() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")

    assert "/api/neighborhood" in script
    assert "toggleNeighborhood" in script
    assert "renderNeighborhood" in script
    assert "neighborhood-button" in script


def test_retrieval_ui_defaults_to_video_ranking_and_keeps_frame_debug_mode() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")

    assert 'activeRankingMode = "video"' in script
    assert "payload.video_results" in script
    assert "payload.raw_results" in script
    assert 'setRankingMode("frame")' in script


def test_retrieval_ui_explores_matched_frames_and_reuses_neighborhood_api() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")

    assert "toggleExplore" in script
    assert "matchedFrameNode" in script
    assert "loadMatchedFrameTimeline" in script
    assert "Timeline neighbors" in script
    assert script.count("/api/neighborhood") >= 2

    styles = (ROOT / "web" / "retrieval_ui" / "styles.css").read_text(encoding="utf-8")
    assert ".explore-panel[hidden]" in styles
    assert ".neighborhood-panel[hidden]" in styles


def test_retrieval_ui_has_opt_in_structured_controls_and_evidence() -> None:
    html = (ROOT / "web" / "retrieval_ui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")
    for control_id in ("structured-filters","structured-enabled","debug-mode","enable-clip","enable-objects","enable-attributes","enable-metadata","object-label","object-min-count","object-position","object-min-confidence","attribute-color","attribute-filter-mode","metadata-author","metadata-date","metadata-title","fusion-method"):
        assert f'id="{control_id}"' in html
    assert "/api/structured-search" in script
    assert "Object evidence unavailable" in script
    assert "Attribute #" in script
    assert "renderEvidenceChips" in script
    assert "structured_config" in script
    assert "applyStructuredConfig" in script


def test_retrieval_ui_exports_structured_evidence_fields() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")
    for field in ("clip_score","clip_rank","object_score","object_rank","matched_objects","attribute_score","attribute_rank","matched_attributes","metadata_score","metadata_rank","matched_metadata_fields","fusion_score","fusion_rank","fusion_method"):
        assert f'"{field}"' in script


def test_retrieval_ui_has_raw_video_preview_flow() -> None:
    script = (ROOT / "web" / "retrieval_ui" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "web" / "retrieval_ui" / "styles.css").read_text(encoding="utf-8")

    assert "video_url" in script
    assert "openVideo" in script
    assert "openVideoResult" in script
    assert "closeVideo" in script
    assert "videoDialog.showModal" in script
    assert "open-video-button" in script
    assert "pin-actions" in script
    assert ".video-dialog" in styles
    assert ".open-video-button[hidden]" in styles
