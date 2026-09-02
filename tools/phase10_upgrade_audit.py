from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def phase5_counts(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        output: dict[str, int] = {}
        for table, video_column in (("ocr", "video_id"), ("asr", "video_id")):
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if not exists:
                output[f"{table}_records"] = 0
                output[f"{table}_videos"] = 0
                continue
            output[f"{table}_records"] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            output[f"{table}_videos"] = int(connection.execute(f"SELECT COUNT(DISTINCT {video_column}) FROM {table}").fetchone()[0])
        return output
    finally:
        connection.close()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    corpus = read_json(args.corpus_manifest)
    bge = read_json(args.bge_manifest)
    bm25 = read_json(args.bm25_manifest)
    objects = read_json(args.object_manifest)
    phase5 = phase5_counts(args.phase5_store)
    checks: list[dict[str, Any]] = []

    def check(name: str, status: str, detail: Any) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    check("phase5_store", "READY" if phase5 else "MISSING", phase5 or str(args.phase5_store))
    corpus_counts = (corpus or {}).get("source_counts") or {}
    check("text_corpus", "READY" if corpus else "MISSING", {"path": str(args.corpus_manifest), "source_counts": corpus_counts})
    check("ocr_coverage", "READY" if int((phase5 or {}).get("ocr_videos", 0)) > 0 else "BLOCKED", (phase5 or {}).get("ocr_videos", 0))
    check("asr_coverage", "READY" if int((phase5 or {}).get("asr_videos", 0)) >= args.expected_videos else "PARTIAL", {"videos": int((phase5 or {}).get("asr_videos", 0)), "expected": args.expected_videos})

    corpus_sha = str((corpus or {}).get("corpus_sha256") or (corpus or {}).get("sha256") or "")
    bge_corpus = (bge or {}).get("corpus") or {}
    bge_sha = str(bge_corpus.get("sha256") or (bge or {}).get("corpus_sha256") or "")
    bge_current = bool(bge and corpus and corpus_sha and bge_sha == corpus_sha)
    check("bge_index", "READY" if bge_current else ("STALE" if bge else "MISSING"), {"path": str(args.bge_manifest), "corpus_sha256": bge_sha, "expected_sha256": corpus_sha})

    bm25_corpus = (bm25 or {}).get("corpus") or {}
    bm25_sha = str(bm25_corpus.get("sha256") or (bm25 or {}).get("corpus_sha256") or "")
    bm25_current = bool(bm25 and corpus and corpus_sha and bm25_sha == corpus_sha)
    check("bm25_index", "READY" if bm25_current else ("STALE" if bm25 else "MISSING"), {"path": str(args.bm25_manifest), "corpus_sha256": bm25_sha, "expected_sha256": corpus_sha})

    object_config = (objects or {}).get("build_configuration") or {}
    object_filtered = bool(objects and not object_config.get("all_valid_detections_preserved", True))
    frame_count = int((objects or {}).get("frame_count", 0))
    detection_count = int((objects or {}).get("detection_count", 0))
    check("object_store_v2", "READY" if object_filtered else ("BASELINE_NO_FILTER" if objects else "MISSING"), {
        "path": str(args.object_manifest),
        "detections_per_frame": round(detection_count / frame_count, 3) if frame_count else None,
        "build_configuration": object_config,
    })

    commands = {
        "bge_v2": (
            f"python tools\\build_bge_index.py --corpus {args.corpus_documents} "
            f"--corpus-manifest {args.corpus_manifest} --output-dir {args.bge_output} "
            "--model-path $env:AIC_BGE_MODEL_PATH --device cuda --batch-size 32 --progress tqdm"
        ),
        "object_v2": (
            f"python tools\\build_object_store.py --refs {args.refs} --object-root {args.object_root} "
            f"--aliases config\\object_aliases_v1.json --groups {args.groups} --output {args.object_output} "
            f"--manifest {args.object_manifest_v2} --detection-threshold 0.30 --nms-threshold 0.50 "
            "--max-detections-per-frame 30 --progress tqdm"
        ),
    }
    blocking = [item["name"] for item in checks if item["status"] in {"MISSING", "BLOCKED", "STALE"}]
    return {
        "schema_version": "phase10-upgrade-audit-v1",
        "status": "READY" if not blocking else "ACTION_REQUIRED",
        "checks": checks,
        "blocking": blocking,
        "commands": commands,
        "quality_claim": "NOT_EVALUATED_WITHOUT_LABELED_DEV_SET",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit Phase-10 L21-L30 retrieval upgrade artifacts.")
    value.add_argument("--corpus-manifest", type=Path, default=Path(r"E:\AIC2026\artifacts\corpora\l21_l30_text_v2\manifest.json"))
    value.add_argument("--corpus-documents", type=Path, default=Path(r"E:\AIC2026\artifacts\corpora\l21_l30_text_v2\documents.jsonl"))
    value.add_argument("--phase5-store", type=Path, default=Path(r"E:\AIC2026\artifacts\phase5\l21_l30_phase5.sqlite3"))
    value.add_argument("--bge-manifest", type=Path, default=Path(r"E:\AIC2026\artifacts\indexes\l21_l30_bge_v2\manifest.json"))
    value.add_argument("--bm25-manifest", type=Path, default=Path(r"E:\AIC2026\artifacts\indexes\l21_l30_bm25_v2\manifest.json"))
    value.add_argument("--object-manifest", type=Path, default=Path(r"E:\AIC2026\artifacts\structured\l21_l30_objects_manifest.json"))
    value.add_argument("--object-manifest-v2", type=Path, default=Path(r"E:\AIC2026\artifacts\structured\l21_l30_objects_v2_manifest.json"))
    value.add_argument("--refs", type=Path, default=Path(r"E:\AIC2026\artifacts\indexes\l21_l30_numpy\refs.json"))
    value.add_argument("--object-root", type=Path, default=Path(r"data\objects"))
    value.add_argument("--bge-output", type=Path, default=Path(r"E:\AIC2026\artifacts\indexes\l21_l30_bge_v2"))
    value.add_argument("--object-output", type=Path, default=Path(r"E:\AIC2026\artifacts\structured\l21_l30_objects_v2.sqlite"))
    value.add_argument("--groups", default="L21,L22,L23,L24,L25,L26,L27,L28,L29,L30")
    value.add_argument("--expected-videos", type=int, default=873)
    value.add_argument("--output", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    result = audit(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
