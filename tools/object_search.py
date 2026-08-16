from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from aic_retrieval.object_search import ObjectPredicate, ObjectSearchConfig, ObjectSearchService
from aic_retrieval.object_store import load_alias_dictionary


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the normalized Phase 4 object store.")
    parser.add_argument("--store", default="artifacts/structured/l21_objects.sqlite")
    parser.add_argument("--aliases", default="config/object_aliases_v1.json")
    parser.add_argument("--labels", default="phone", help="Comma-separated English or Vietnamese aliases")
    parser.add_argument("--count-operator", choices=["=", ">=", "<="], default=">=")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--horizontal", choices=["any", "left", "center", "right"], default="any")
    parser.add_argument("--vertical", choices=["any", "top", "middle", "bottom"], default="any")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--interaction", choices=["none", "phone_near_hand"], default="none")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    service = ObjectSearchService(Path(args.store), load_alias_dictionary(Path(args.aliases)))
    config = ObjectSearchConfig(args.min_confidence, args.nms_iou_threshold)
    if args.interaction == "phone_near_hand":
        payload = service.phone_near_hand(config)
    else:
        predicate = ObjectPredicate(tuple(item.strip() for item in args.labels.split(",") if item.strip()), args.count_operator, args.count, args.horizontal, args.vertical)
        payload = service.search(predicate, config)
    payload["results"] = payload["results"][: max(0, args.limit)]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__": raise SystemExit(main())
