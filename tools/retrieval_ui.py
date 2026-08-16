from __future__ import annotations

import argparse
import errno
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic_retrieval.retrieval_ui import RetrievalUiConfig, run_server
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID
from aic_retrieval.translation import (
    DEFAULT_GEMINI_API_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TRANSLATION_API_URL,
    DEFAULT_TRANSLATION_MODEL,
    TranslationConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local AIC L21 retrieval UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry" / "data_registry.json")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "artifacts" / "indexes" / "l21_numpy")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data" / "media-info")
    parser.add_argument("--object-store", type=Path, default=ROOT / "artifacts" / "structured" / "l21_objects.sqlite")
    parser.add_argument("--object-aliases", type=Path, default=ROOT / "config" / "object_aliases_v1.json")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--clip-model-id", default=os.environ.get("AIC_CLIP_MODEL_ID", DEFAULT_CLIP_MODEL_ID))
    parser.add_argument("--clip-cache-dir", type=Path, default=Path(os.environ["AIC_CLIP_CACHE_DIR"]) if os.environ.get("AIC_CLIP_CACHE_DIR") else None)
    parser.add_argument("--clip-local-files-only", action="store_true")
    parser.add_argument("--translation-provider", choices=["gemini", "openai-compatible"], default=os.environ.get("AIC_TRANSLATION_PROVIDER", "gemini"))
    parser.add_argument("--translation-api-url", default=os.environ.get("AIC_TRANSLATION_API_URL"))
    parser.add_argument("--translation-model", default=os.environ.get("AIC_TRANSLATION_MODEL"))
    parser.add_argument("--translation-timeout", type=float, default=float(os.environ.get("AIC_TRANSLATION_TIMEOUT", "30")))
    parser.add_argument("--allow-stale-index", action="store_true")
    args = parser.parse_args()

    if port_is_listening(args.host, args.port):
        print_port_in_use(args.host, args.port)
        return 2

    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    config = RetrievalUiConfig(
        repo_root=ROOT,
        registry_path=args.registry,
        index_dir=args.index_dir,
        metadata_dir=args.metadata_dir,
        static_dir=ROOT / "web" / "retrieval_ui",
        groups=groups,
        clip_model_id=args.clip_model_id,
        clip_cache_dir=args.clip_cache_dir,
        clip_local_files_only=args.clip_local_files_only,
        allow_stale_index=args.allow_stale_index,
        translation=TranslationConfig(
            api_key=os.environ.get("AIC_TRANSLATION_API_KEY"),
            api_url=args.translation_api_url or default_translation_api_url(args.translation_provider),
            model=args.translation_model or default_translation_model(args.translation_provider),
            provider=args.translation_provider,
            timeout_seconds=args.translation_timeout,
        ),
        object_store_path=args.object_store,
        object_aliases_path=args.object_aliases,
    )
    try:
        server = run_server(config, host=args.host, port=args.port)
    except OSError as exc:
        if exc.errno in {errno.EADDRINUSE, 10048}:
            print_port_in_use(args.host, args.port)
            return 2
        raise
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


def port_is_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def print_port_in_use(host: str, port: int) -> None:
    url = f"http://{host}:{port}/"
    print(f"Port {port} is already in use. The UI may already be running at {url}", file=sys.stderr)
    print(f"Check health: curl.exe -s http://{host}:{port}/api/health", file=sys.stderr)
    print(f"Find PID: netstat -ano | Select-String \":{port}\"", file=sys.stderr)
    print("Stop the existing server with: Stop-Process -Id <PID>", file=sys.stderr)


def default_translation_api_url(provider: str) -> str:
    return DEFAULT_GEMINI_API_URL if provider == "gemini" else DEFAULT_TRANSLATION_API_URL


def default_translation_model(provider: str) -> str:
    return DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_TRANSLATION_MODEL


if __name__ == "__main__":
    raise SystemExit(main())
