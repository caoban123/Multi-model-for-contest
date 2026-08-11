from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryJudgementSummary:
    query_id: str
    query_text: str
    rows: int
    top1_judgement: str
    good: int
    partial: int
    bad: int
    blank: int
    precision_good_at_k: float
    precision_good_or_partial_at_k: float


def load_judgement_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_judgements(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("judgement CSV has no rows")

    totals = Counter(_norm(row.get("manual_judgement", "")) for row in rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("query_id", "")].append(row)

    query_summaries = []
    for query_id in sorted(grouped):
        group = sorted(grouped[query_id], key=lambda item: int(item.get("rank") or 0))
        counts = Counter(_norm(row.get("manual_judgement", "")) for row in group)
        rows_count = len(group)
        good = counts["good"]
        partial = counts["partial"]
        summary = QueryJudgementSummary(
            query_id=query_id,
            query_text=group[0].get("query_text", ""),
            rows=rows_count,
            top1_judgement=_norm(group[0].get("manual_judgement", "")),
            good=good,
            partial=partial,
            bad=counts["bad"],
            blank=counts[""],
            precision_good_at_k=good / rows_count if rows_count else 0.0,
            precision_good_or_partial_at_k=(good + partial) / rows_count if rows_count else 0.0,
        )
        query_summaries.append(summary)

    return {
        "rows": len(rows),
        "query_count": len(query_summaries),
        "totals": {
            "good": totals["good"],
            "partial": totals["partial"],
            "bad": totals["bad"],
            "blank": totals[""],
        },
        "top1": {
            "good": sum(1 for item in query_summaries if item.top1_judgement == "good"),
            "partial": sum(1 for item in query_summaries if item.top1_judgement == "partial"),
            "bad": sum(1 for item in query_summaries if item.top1_judgement == "bad"),
            "blank": sum(1 for item in query_summaries if item.top1_judgement == ""),
        },
        "queries_with_any_good": sum(1 for item in query_summaries if item.good > 0),
        "queries": [asdict(item) for item in query_summaries],
    }


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()
