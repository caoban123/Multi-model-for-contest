from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.phase5_workflow import Phase5PreparationError, require_index_source_data, require_phase5_inputs, require_numpy_index


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run(command: list[str], cwd: Path = ROOT) -> None:
    print(f"> {command_text(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare(
    data_root: Path,
    registry: Path,
    validation: Path,
    groups: str,
    index_dir: Path,
    ocr_jsonl: Path | None,
    asr_jsonl: Path | None,
    output: Path,
    overwrite: bool,
    skip_store: bool,
    force_rebuild: bool,
) -> None:
    if not data_root.is_dir():
        raise Phase5PreparationError(f"Data root does not exist or is not a directory: {data_root}")
    require_index_source_data(data_root, groups)

    if force_rebuild or not registry.is_file():
        run([
            sys.executable, "tools/data_registry.py", "--data-root", str(data_root),
            "--output", str(registry), "--validation-output", str(validation),
        ])

    index_missing = bool([name for name in ("vectors.npy", "refs.json") if not (index_dir / name).is_file()])
    if force_rebuild or index_missing:
        run([
            sys.executable, "tools/build_numpy_index.py", "--registry", str(registry),
            "--groups", groups, "--output-dir", str(index_dir),
        ])
    require_numpy_index(index_dir, registry, groups)

    if skip_store:
        print("Phase 5 prerequisites are ready. Store build skipped by --skip-store.")
        return

    require_phase5_inputs(ocr_jsonl, asr_jsonl)
    command = [
        sys.executable, "tools/build_phase5_store.py", "--index-dir", str(index_dir),
        "--registry", str(registry), "--groups", groups, "--output", str(output),
    ]
    if ocr_jsonl is not None:
        command.extend(["--ocr-jsonl", str(ocr_jsonl)])
    if asr_jsonl is not None:
        command.extend(["--asr-jsonl", str(asr_jsonl)])
    if overwrite:
        command.append("--overwrite")
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild real Phase 5 prerequisites and store from data/ after artifacts/ is deleted.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry/data_registry.json"))
    parser.add_argument("--validation-output", type=Path, default=Path("artifacts/registry/validation_report.json"))
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes/l21_numpy"))
    parser.add_argument("--ocr-jsonl", type=Path)
    parser.add_argument("--asr-jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase5/manual/phase5.sqlite3"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-store", action="store_true", help="Build registry/index only; use when reviewed OCR/ASR JSONL is not available yet.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild registry/index even when generated files already exist.")
    args = parser.parse_args()
    try:
        prepare(args.data_root,args.registry,args.validation_output,args.groups,args.index_dir,args.ocr_jsonl,args.asr_jsonl,args.output,args.overwrite,args.skip_store,args.force_rebuild)
    except Phase5PreparationError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        parser.error(f"Preparation command failed with exit code {exc.returncode}: {command_text(exc.cmd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
