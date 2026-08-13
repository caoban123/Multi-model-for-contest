from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from aic_retrieval.search import (
    diversify_results_by_video,
    group_results_by_video,
    load_numpy_index,
    load_registry,
    search_numpy_index,
    validate_numpy_index,
)
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder


@dataclass(frozen=True)
class RetrievalUiConfig:
    repo_root: Path
    registry_path: Path
    index_dir: Path
    static_dir: Path
    groups: set[str]
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_cache_dir: Path | None = None
    clip_local_files_only: bool = False
    allow_stale_index: bool = False


class RetrievalUiService:
    def __init__(self, config: RetrievalUiConfig) -> None:
        self.config = config
        self.registry = load_registry(config.registry_path)
        self.index, self.refs, self.index_metadata = load_numpy_index(config.index_dir)
        validate_numpy_index(
            self.index,
            self.refs,
            self.index_metadata,
            requested_groups=config.groups,
            require_keyframes=False,
            registry_path=config.registry_path,
            repo_root=config.repo_root,
            allow_stale_index=config.allow_stale_index,
        )
        self.assets_by_video = {asset["video_id"]: asset for asset in self.registry.get("videos", [])}
        self.metadata_by_video: dict[str, dict[str, Any]] = {}
        self.encoder: ClipTextEncoder | None = None

    def search(self, query: str, top_k: int = 12, candidate_pool: int | None = None, max_frames_per_video: int = 1) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        top_k = max(1, min(top_k, 50))
        candidate_pool = max(candidate_pool or top_k, top_k)
        max_frames_per_video = max(1, min(max_frames_per_video, 10))

        started = time.perf_counter()
        query_vector = self._encoder().encode_text(query)
        raw_results = search_numpy_index(self.index, self.refs, query_vector, top_k=candidate_pool)
        results = diversify_results_by_video(raw_results, max_frames_per_video=max_frames_per_video)[:top_k]
        video_groups = group_results_by_video(raw_results, max_frames_per_video=max_frames_per_video)[:top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "query": query,
            "groups": sorted(self.config.groups),
            "top_k": top_k,
            "candidate_pool": candidate_pool,
            "max_frames_per_video": max_frames_per_video,
            "elapsed_ms": round(elapsed_ms, 3),
            "index_vectors": int(self.index.shape[0]),
            "index_dim": int(self.index.shape[1]),
            "results": [self.enrich_result(asdict(result)) for result in results],
            "video_groups": [asdict(group) for group in video_groups],
        }

    def enrich_result(self, result: dict[str, Any]) -> dict[str, Any]:
        video_id = result["video_id"]
        metadata = self.video_metadata(video_id)
        keyframe_path = result.get("keyframe_path")
        result["image_url"] = f"/keyframe?path={keyframe_path}" if keyframe_path else None
        result["metadata"] = {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "publish_date": metadata.get("publish_date", ""),
            "watch_url": metadata.get("watch_url", ""),
            "keywords": (metadata.get("keywords") or [])[:8],
        }
        return result

    def video_metadata(self, video_id: str) -> dict[str, Any]:
        if video_id in self.metadata_by_video:
            return self.metadata_by_video[video_id]
        asset = self.assets_by_video.get(video_id, {})
        metadata_path = asset.get("media_info_path")
        payload: dict[str, Any] = {}
        if metadata_path:
            path = self.config.repo_root / metadata_path
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
        self.metadata_by_video[video_id] = payload
        return payload

    def resolve_keyframe_path(self, value: str) -> Path:
        if not value:
            raise FileNotFoundError("empty keyframe path")
        path = (self.config.repo_root / value).resolve()
        repo_root = self.config.repo_root.resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise PermissionError("keyframe path is outside the repository") from exc
        if not path.is_file():
            raise FileNotFoundError(value)
        return path

    def _encoder(self) -> ClipTextEncoder:
        if self.encoder is None:
            self.encoder = ClipTextEncoder(
                model_id=self.config.clip_model_id,
                cache_dir=self.config.clip_cache_dir,
                local_files_only=self.config.clip_local_files_only,
            )
        return self.encoder


def run_server(config: RetrievalUiConfig, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    service = RetrievalUiService(config)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AICRetrievalUI/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._serve_static("index.html")
            elif parsed.path.startswith("/static/"):
                self._serve_static(parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/health":
                self._json(
                    {
                        "ok": True,
                        "groups": sorted(service.config.groups),
                        "index_vectors": int(service.index.shape[0]),
                        "index_dim": int(service.index.shape[1]),
                    }
                )
            elif parsed.path == "/api/search":
                self._handle_search(parsed.query)
            elif parsed.path == "/keyframe":
                self._serve_keyframe(parsed.query)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _handle_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.search(
                    first(params, "q"),
                    top_k=parse_int(first(params, "top_k", "12"), 12),
                    candidate_pool=parse_int(first(params, "candidate_pool", "30"), 30),
                    max_frames_per_video=parse_int(first(params, "max_frames_per_video", "1"), 1),
                )
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _serve_keyframe(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                path = service.resolve_keyframe_path(first(params, "path"))
            except PermissionError as exc:
                self._error(HTTPStatus.FORBIDDEN, str(exc))
                return
            except FileNotFoundError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

        def _serve_static(self, relative_path: str) -> None:
            path = (config.static_dir / relative_path).resolve()
            try:
                path.relative_to(config.static_dir.resolve())
            except ValueError:
                self._error(HTTPStatus.FORBIDDEN, "static path is outside static dir")
                return
            if not path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "static file not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

        def _json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, message: str) -> None:
            body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def first(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key)
    return values[0] if values else default


def parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default
