from __future__ import annotations

from pathlib import Path

from aic_retrieval.trake_config import load_trake_config
from tools.phase8_ablation import build_variants, render_markdown, run_ablation


ROOT = Path(__file__).parents[1]


def test_ablation_ladder_is_cumulative_fingerprinted_and_has_guarded_optional_steps() -> None:
    variants = build_variants(load_trake_config(ROOT / "configs" / "phase8_trake_v2.json"))

    assert [variant["id"] for variant in variants] == list("ABCDEFGHIJKLMNO")
    assert len({variant["config_fingerprint"] for variant in variants}) == 15
    assert variants[0]["config"]["algorithm"] == "legacy"
    assert variants[1]["config"]["features"]["planner_v2"] is True
    assert variants[2]["config"]["features"]["query_variants"] is True
    assert variants[3]["availability"] == "SKIPPED_UNAVAILABLE"
    assert variants[9]["config"]["features"]["dante_coarse"] is True
    assert variants[10]["config"]["features"]["dense_refinement"] is True
    assert variants[11]["config"]["features"]["dante_fine"] is True
    assert variants[14]["availability"] == "SKIPPED_DISABLED"


def test_unlabelled_ablation_reports_json_and_markdown_nulls() -> None:
    result = run_ablation(
        ROOT / "benchmarks" / "trake_development_v1.json",
        ROOT / "artifacts" / "trake" / "phase8_trake.sqlite3",
        ROOT / "configs" / "phase8_trake_v2.json",
    )
    markdown = render_markdown(result)

    assert result["variant_count"] == 15 and result["labelled_query_count"] == 0
    assert all(all(value is None for value in variant["metrics"].values()) for variant in result["variants"])
    assert result["variants"][1]["status"] == "UNAVAILABLE_LABELS"
    assert "| A | Legacy Phase8 | UNAVAILABLE_LABELS |" in markdown
    assert "Quality claim: `null`" in markdown
