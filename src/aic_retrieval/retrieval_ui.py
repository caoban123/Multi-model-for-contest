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
    aggregate_results_by_video,
    diversify_results_by_video,
    load_numpy_index,
    load_registry,
    search_numpy_index,
    validate_numpy_index,
)
from aic_retrieval.metadata_search import load_metadata_documents, results_to_dict, search_metadata
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder
from aic_retrieval.translation import ExternalTranslator, TranslationConfig


DEFAULT_TOP_K_VIDEOS = 12
DEFAULT_CANDIDATE_POOL_SIZE = 40
DEFAULT_MAX_FRAMES_PER_VIDEO = 1
DEFAULT_MATCHED_FRAMES_PER_VIDEO = 5
DEFAULT_NEIGHBOR_RADIUS = 3
DEFAULT_AGGREGATION_METHOD = "max"
DEFAULT_MEAN_TOP_N = 3


@dataclass(frozen=True)
class RetrievalUiConfig:
    repo_root: Path
    registry_path: Path
    index_dir: Path
    metadata_dir: Path
    static_dir: Path
    groups: set[str]
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_cache_dir: Path | None = None
    clip_local_files_only: bool = False
    allow_stale_index: bool = False
    translation: TranslationConfig | None = None


class RetrievalUiService:
    def __init__(self, config: RetrievalUiConfig) -> None:
        self.config = config
        self.registry = load_registry(config.registry_path)
        self.metadata_docs = load_metadata_documents(config.metadata_dir, groups=config.groups)
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
        self.refs_by_video: dict[str, list[Any]] = {}
        for ref in self.refs:
            self.refs_by_video.setdefault(ref.video_id, []).append(ref)
        for video_refs in self.refs_by_video.values():
            video_refs.sort(key=lambda ref: (ref.pts_time, ref.keyframe_id))
        self.metadata_by_video: dict[str, dict[str, Any]] = {}
        self.encoder: ClipTextEncoder | None = None
        self.translator = ExternalTranslator(config.translation or TranslationConfig())

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_VIDEOS,
        candidate_pool: int | None = None,
        max_frames_per_video: int = DEFAULT_MAX_FRAMES_PER_VIDEO,
        matched_frames_per_video: int = DEFAULT_MATCHED_FRAMES_PER_VIDEO,
        aggregation_method: str = DEFAULT_AGGREGATION_METHOD,
        mean_top_n: int = DEFAULT_MEAN_TOP_N,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        _require_positive("top_k", top_k)
        candidate_pool = candidate_pool if candidate_pool is not None else DEFAULT_CANDIDATE_POOL_SIZE
        _require_positive("candidate_pool", candidate_pool)
        _require_positive("max_frames_per_video", max_frames_per_video)
        _require_positive("matched_frames_per_video", matched_frames_per_video)
        _require_positive("mean_top_n", mean_top_n)
        top_k = min(top_k, 50)
        candidate_pool = max(candidate_pool, top_k)
        max_frames_per_video = min(max_frames_per_video, 10)
        matched_frames_per_video = min(matched_frames_per_video, 50)

        started = time.perf_counter()
        encode_started = time.perf_counter()
        query_vector = self._encoder().encode_text(query)
        encode_ms = (time.perf_counter() - encode_started) * 1000
        retrieval_started = time.perf_counter()
        raw_results = search_numpy_index(self.index, self.refs, query_vector, top_k=candidate_pool)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        aggregation_started = time.perf_counter()
        results = diversify_results_by_video(raw_results, max_frames_per_video=max_frames_per_video)[:top_k]
        video_results = aggregate_results_by_video(
            raw_results,
            max_frames_per_video=matched_frames_per_video,
            aggregation_method=aggregation_method,
            mean_top_n=mean_top_n,
        )[:top_k]
        aggregation_ms = (time.perf_counter() - aggregation_started) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "mode": "visual",
            "query": query,
            "groups": sorted(self.config.groups),
            "top_k": top_k,
            "top_k_videos": top_k,
            "candidate_pool": candidate_pool,
            "candidate_pool_size": candidate_pool,
            "max_frames_per_video": max_frames_per_video,
            "matched_frames_per_video": matched_frames_per_video,
            "aggregation_method": aggregation_method,
            "mean_top_n": mean_top_n,
            "encode_ms": round(encode_ms, 3),
            "retrieval_ms": round(retrieval_ms, 3),
            "aggregation_ms": round(aggregation_ms, 3),
            "elapsed_ms": round(elapsed_ms, 3),
            "index_vectors": int(self.index.shape[0]),
            "index_dim": int(self.index.shape[1]),
            "raw_results": [self.enrich_result(asdict(result)) for result in raw_results],
            "results": [self.enrich_result(asdict(result)) for result in results],
            "video_results": [self.enrich_video_result(asdict(result)) for result in video_results],
            "video_groups": [self.enrich_video_result(asdict(result)) for result in video_results],
        }

    def metadata_search(self, query: str, top_k: int = 12, min_match: int = 1) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("metadata query must not be empty")
        top_k = max(1, min(top_k, 50))
        min_match = max(1, min(min_match, 10))
        started = time.perf_counter()
        results = search_metadata(self.metadata_docs, query, top_k=top_k, min_match=min_match)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "mode": "metadata",
            "query": query,
            "groups": sorted(self.config.groups),
            "top_k": top_k,
            "min_match": min_match,
            "elapsed_ms": round(elapsed_ms, 3),
            "total_documents": len(self.metadata_docs),
            "results": results_to_dict(results),
        }

    def keyframe_neighborhood(self, video_id: str, keyframe_id: int, radius: int = 3) -> dict[str, Any]:
        video_id = video_id.strip()
        if not video_id:
            raise ValueError("video_id must not be empty")
        _require_positive("radius", radius)
        radius = min(radius, 12)
        refs = self.refs_by_video.get(video_id)
        if not refs:
            raise ValueError(f"video_id not found in index: {video_id}")

        center_index = None
        for index, ref in enumerate(refs):
            if ref.keyframe_id == keyframe_id:
                center_index = index
                break
        if center_index is None:
            raise ValueError(f"{video_id} has no keyframe_id {keyframe_id}")

        start = max(0, center_index - radius)
        end = min(len(refs), center_index + radius + 1)
        frames = []
        for ref in refs[start:end]:
            item = asdict(ref)
            item["is_center"] = ref.keyframe_id == keyframe_id
            item["image_url"] = f"/keyframe?path={ref.keyframe_path}" if ref.keyframe_path else None
            frames.append(item)

        return {
            "video_id": video_id,
            "keyframe_id": keyframe_id,
            "radius": radius,
            "ordering": "pts_time_ascending",
            "total_frames": len(refs),
            "start_keyframe_id": frames[0]["keyframe_id"] if frames else None,
            "end_keyframe_id": frames[-1]["keyframe_id"] if frames else None,
            "frames": frames,
        }

    def translate(self, text: str) -> dict[str, Any]:
        source_text = text.strip()
        if not source_text:
            raise ValueError("translation query must not be empty")
        translated = self.translator.translate_vi_to_en(source_text)
        return {
            "source_text": source_text,
            "translated_text": translated,
            "provider": self.translator.config.provider,
            "model": self.translator.config.model,
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

    def enrich_video_result(self, result: dict[str, Any]) -> dict[str, Any]:
        video_id = result["video_id"]
        keyframe_path = result.get("best_keyframe_path")
        result["image_url"] = f"/keyframe?path={keyframe_path}" if keyframe_path else None
        result["metadata"] = self.enrich_result(
            {
                "video_id": video_id,
                "keyframe_path": keyframe_path,
            }
        )["metadata"]
        for frame in result.get("frames", []):
            frame_path = frame.get("keyframe_path")
            frame["image_url"] = f"/keyframe?path={frame_path}" if frame_path else None
            frame["is_representative"] = (
                frame.get("keyframe_id") == result.get("best_keyframe_id")
                and frame.get("frame_idx") == result.get("best_frame_idx")
            )
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
                        "metadata_documents": len(service.metadata_docs),
                        "translation_configured": service.translator.is_configured,
                        "translation_provider": service.translator.config.provider,
                        "translation_model": service.translator.config.model,
                        "default_ranking_mode": "video",
                        "default_aggregation_method": DEFAULT_AGGREGATION_METHOD,
                        "default_candidate_pool_size": DEFAULT_CANDIDATE_POOL_SIZE,
                    }
                )
            elif parsed.path == "/api/search":
                self._handle_search(parsed.query)
            elif parsed.path == "/api/metadata-search":
                self._handle_metadata_search(parsed.query)
            elif parsed.path == "/api/neighborhood":
                self._handle_neighborhood(parsed.query)
            elif parsed.path == "/api/translate":
                self._handle_translate(parsed.query)
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
                    top_k=parse_int(
                        first(params, "top_k", str(DEFAULT_TOP_K_VIDEOS)),
                        DEFAULT_TOP_K_VIDEOS,
                    ),
                    candidate_pool=parse_int(
                        first(params, "candidate_pool", str(DEFAULT_CANDIDATE_POOL_SIZE)),
                        DEFAULT_CANDIDATE_POOL_SIZE,
                    ),
                    max_frames_per_video=parse_int(
                        first(params, "max_frames_per_video", str(DEFAULT_MAX_FRAMES_PER_VIDEO)),
                        DEFAULT_MAX_FRAMES_PER_VIDEO,
                    ),
                    matched_frames_per_video=parse_int(
                        first(
                            params,
                            "matched_frames_per_video",
                            str(DEFAULT_MATCHED_FRAMES_PER_VIDEO),
                        ),
                        DEFAULT_MATCHED_FRAMES_PER_VIDEO,
                    ),
                    aggregation_method=first(
                        params, "aggregation_method", DEFAULT_AGGREGATION_METHOD
                    ),
                    mean_top_n=parse_int(
                        first(params, "mean_top_n", str(DEFAULT_MEAN_TOP_N)),
                        DEFAULT_MEAN_TOP_N,
                    ),
                )
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _handle_metadata_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.metadata_search(
                    first(params, "q"),
                    top_k=parse_int(first(params, "top_k", "12"), 12),
                    min_match=parse_int(first(params, "min_match", "1"), 1),
                )
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _handle_neighborhood(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.keyframe_neighborhood(
                    first(params, "video_id"),
                    keyframe_id=parse_int(first(params, "keyframe_id", "0"), 0),
                    radius=parse_int(first(params, "radius", "3"), 3),
                )
            except Exception as exc:  # UI boundary: return a readable error to the browser.
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(payload)

        def _handle_translate(self, query_string: str) -> None:
            params = parse_qs(query_string)
            try:
                payload = service.translate(first(params, "q"))
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


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
