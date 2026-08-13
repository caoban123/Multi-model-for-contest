from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.retrieval_ui import RetrievalUiConfig, run_server
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local AIC L21 retrieval UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry" / "data_registry.json")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_numpy")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--clip-model-id", default=os.environ.get("AIC_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID))
    parser.add_argument("--clip-cache-dir", type=Path, default=Path(os.environ["AIC_CLIP_CACHE_DIR"]) if os.environ.get("AIC_CLIP_CACHE_DIR") else None)
    parser.add_argument("--clip-local-files-only", action="store_true")
    parser.add_argument("--allow-stale-index", action="store_true")
    args = parser.parse_args()

    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    config = RetrievalUiConfig(
        repo_root=ROOT,
        registry_path=args.registry,
        index_dir=args.index_dir,
        static_dir=ROOT / "web" / "retrieval_ui",
        groups=groups,
        clip_model_id=args.clip_model_id,
        clip_cache_dir=args.clip_cache_dir,
        clip_local_files_only=args.clip_local_files_only,
        allow_stale_index=args.allow_stale_index,
    )
    server = run_server(config, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"AIC retrieval UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
