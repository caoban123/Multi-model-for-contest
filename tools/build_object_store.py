from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.object_store import build_object_store


def _json_progress(processed: int, total: int, detections: int) -> None:
    print(
        json.dumps(
            {
                "stage": "object_store",
                "processed_frames": processed,
                "total_frames": total,
                "progress_percent": round(processed / total * 100, 2) if total else 100.0,
                "detections": detections,
            }
        ),
        flush=True,
    )


def _progress_reporter(
    mode: str,
) -> tuple[Callable[[int, int, int], None] | None, Callable[[], None]]:
    if mode == "none":
        return None, lambda: None
    if mode == "json":
        return _json_progress, lambda: None

    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("tqdm is unavailable; falling back to JSON progress.", file=sys.stderr, flush=True)
        return _json_progress, lambda: None

    progress_bar = None

    def report(processed: int, total: int, detections: int) -> None:
        nonlocal progress_bar
        if progress_bar is None:
            progress_bar = tqdm(
                total=total,
                desc="Object store",
                unit="frame",
                dynamic_ncols=True,
                smoothing=0.1,
            )
        progress_bar.update(max(0, processed - progress_bar.n))
        progress_bar.set_postfix(detections=f"{detections:,}", refresh=False)
        if processed == total:
            progress_bar.set_description("Object store: finalizing", refresh=False)
        progress_bar.refresh()

    def close() -> None:
        if progress_bar is not None:
            progress_bar.close()

    return report, close


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 4 normalized SQLite object store.")
    parser.add_argument("--refs", default="artifacts/indexes/l21_numpy/refs.json")
    parser.add_argument("--object-root", default="data/objects")
    parser.add_argument("--aliases", default="config/object_aliases_v1.json")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--output", default="artifacts/structured/l21_objects.sqlite")
    parser.add_argument("--manifest", default="artifacts/structured/l21_objects_manifest.json")
    parser.add_argument("--detection-threshold", type=float, default=None)
    parser.add_argument("--nms-threshold", type=float, default=None)
    parser.add_argument("--max-detections-per-frame", type=int, default=None)
    parser.add_argument(
        "--progress",
        choices=("tqdm", "json", "none"),
        default="tqdm",
        help="Progress display mode (default: tqdm; falls back to JSON if tqdm is unavailable).",
    )
    args = parser.parse_args()
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    report_progress, close_progress = _progress_reporter(args.progress)
    try:
        manifest = build_object_store(
            Path(args.refs),
            Path(args.object_root),
            Path(args.aliases),
            Path(args.output),
            groups,
            progress=report_progress,
            detection_threshold=args.detection_threshold,
            nms_threshold=args.nms_threshold,
            max_detections_per_frame=args.max_detections_per_frame,
        )
    finally:
        close_progress()
    manifest["store_path"] = args.output
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
