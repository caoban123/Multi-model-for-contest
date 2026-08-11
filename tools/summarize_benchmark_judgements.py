from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.judgements import load_judgement_rows, summarize_judgements


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize manual judgement CSV for a text-query benchmark.")
    parser.add_argument("--input", default="artifacts/benchmarks/l21_text_benchmark_judged.csv")
    parser.add_argument("--output", default="artifacts/benchmarks/l21_text_judgement_summary.json")
    args = parser.parse_args()

    summary = summarize_judgements(load_judgement_rows(Path(args.input)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
