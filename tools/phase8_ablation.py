from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tools.phase8_benchmark import METRIC_KEYS, evaluate

from aic_retrieval.trake_config import FeatureSettings, TrakeRuntimeConfig, load_trake_config


ABLATION_VERSION = "phase8-trake-ablation-v1"
VARIANTS = (
    ("A", "Legacy Phase8"),
    ("B", "+ Planner V2"),
    ("C", "+ Query variants"),
    ("D", "+ Second visual encoder/RRF if available"),
    ("E", "+ Modality routing"),
    ("F", "+ Adaptive Top-K"),
    ("G", "+ Distinctiveness anchor"),
    ("H", "+ Temporal feasibility"),
    ("I", "+ Candidate windows"),
    ("J", "+ DANTE coarse"),
    ("K", "+ Automatic dense refinement"),
    ("L", "+ DANTE fine"),
    ("M", "+ Global context"),
    ("N", "+ Linkage"),
    ("O", "+ Conditional VLM if approved"),
)


def build_variants(base: TrakeRuntimeConfig) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    legacy = TrakeRuntimeConfig.legacy()
    output.append(_variant("A", VARIANTS[0][1], legacy, "AVAILABLE"))
    feature_order = [
        "planner_v2", "query_variants", "second_visual_encoder", "modality_routing",
        "adaptive_topk", "anchor", "temporal_feasibility", "candidate_windows",
        "dante_coarse", "dense_refinement", "dante_fine", "global_context",
        "linkage", "conditional_vlm",
    ]
    enabled = {name: False for name in FeatureSettings.__dataclass_fields__}
    for index, (letter, label) in enumerate(VARIANTS[1:], start=0):
        feature_name = feature_order[index]
        enabled[feature_name] = True
        features = FeatureSettings(**enabled)
        dante = features.dante_coarse
        config = replace(
            base,
            features=features,
            retrieval=replace(base.retrieval, adaptive_topk=features.adaptive_topk),
            temporal=replace(
                base.temporal,
                fast_feasibility_filter=features.temporal_feasibility,
                candidate_windows=features.candidate_windows,
                alignment="dante" if dante else "legacy_beam",
            ),
            refinement=replace(base.refinement, automatic=features.dense_refinement),
            reranking=replace(base.reranking, context_enabled=features.global_context, linkage_enabled=features.linkage),
        )
        availability = "AVAILABLE"
        reason = None
        if letter == "D" and not base.retrieval.siglip_enabled:
            availability = "SKIPPED_UNAVAILABLE"
            reason = "No verified second visual encoder/index is configured; no download was authorized."
        if letter == "O" and not base.vlm.enabled:
            availability = "SKIPPED_DISABLED"
            reason = "VLM is disabled and lacks explicit provider/decision-log approval."
        output.append(_variant(letter, label, config, availability, reason))
    return output


def run_ablation(queries_path: Path, store_path: Path, config_path: Path) -> dict[str, Any]:
    base = load_trake_config(config_path)
    query_payload = json.loads(queries_path.read_text(encoding="utf-8"))
    labelled = sum(bool(query.get("expected_video_id") or query.get("video_id")) for query in query_payload.get("queries", ()))
    stored_fingerprints = _stored_fingerprints(store_path)
    variants = build_variants(base)
    for variant in variants:
        metrics = {key: None for key in METRIC_KEYS}
        stage_latency = {}
        if variant["availability"] != "AVAILABLE":
            status = variant["availability"]
        elif labelled == 0:
            status = "UNAVAILABLE_LABELS"
        elif variant["config_fingerprint"] not in stored_fingerprints:
            status = "NOT_RUN"
            variant["reason"] = "No persisted session matches this exact config fingerprint."
        else:
            evaluated = evaluate(queries_path, store_path, config_fingerprint=variant["config_fingerprint"])
            status = "EVALUATED"
            metrics = evaluated["metrics"]
            stage_latency = evaluated["stage_latency_ms"]
        variant.update({"status": status, "metrics": metrics, "stage_latency_ms": stage_latency})
    return {
        "ablation_version": ABLATION_VERSION,
        "queries": str(queries_path),
        "store": str(store_path),
        "labelled_query_count": labelled,
        "variant_count": len(variants),
        "variants": variants,
        "quality_claim": None if labelled == 0 else "See per-variant measured metrics; fixture tests are excluded.",
        "notes": [
            "Variants are cumulative and identified by the exact typed-config fingerprint.",
            "Unavailable/not-run variants report null metrics, never fabricated zeroes.",
        ],
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Phase 8 TRAKE V2 ablation",
        "",
        f"Labelled queries: {result['labelled_query_count']}",
        "",
        "| Variant | Change | Status | Config fingerprint | Event R@100 | Video R@20 | Chain success | Order | Within 3s | p50 ms | p95 ms |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in result["variants"]:
        metrics = variant["metrics"]
        lines.append(
            f"| {variant['id']} | {variant['label']} | {variant['status']} | `{variant['config_fingerprint'][:12]}` | "
            f"{_display(metrics.get('event_recall_at_100'))} | {_display(metrics.get('video_recall_at_20'))} | "
            f"{_display(metrics.get('chain_success_rate'))} | {_display(metrics.get('temporal_order_accuracy'))} | "
            f"{_display(metrics.get('event_within_3s'))} | {_display(metrics.get('latency_ms_median'))} | {_display(metrics.get('latency_ms_p95'))} |"
        )
    lines.extend(["", "Quality claim: `null` when labelled development data is absent."])
    return "\n".join(lines) + "\n"


def _variant(identifier: str, label: str, config: TrakeRuntimeConfig, availability: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "availability": availability,
        "reason": reason,
        "config_fingerprint": config.fingerprint,
        "algorithm_version": config.algorithm_version,
        "config": config.to_dict(),
    }


def _stored_fingerprints(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT state_json FROM sessions").fetchall()
    return {str(json.loads(row[0]).get("config_fingerprint")) for row in rows if json.loads(row[0]).get("config_fingerprint")}


def _display(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fingerprint-isolated TRAKE ablation evaluation.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--trake-store", type=Path, required=True)
    parser.add_argument("--trake-config", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    result = run_ablation(args.queries, args.trake_store, args.trake_config)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
