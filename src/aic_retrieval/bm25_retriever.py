from __future__ import annotations

import json
import os
import platform
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from aic_retrieval.hybrid_audit import sha256_file
from aic_retrieval.phase5_schema import normalize_text
from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest, RetrieverHealth
from aic_retrieval.resilient_io import read_text_with_retry


BM25_INDEX_SCHEMA_VERSION = "bm25-sqlite-fts5-v1"
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
VIETNAMESE_DATE_PATTERN = re.compile(r"\bngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})\b")
NUMERIC_DATE_PATTERN = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")


def _iter_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            required = ("document_id", "source_type", "video_id", "text", "text_normalized", "text_folded")
            if not isinstance(payload, dict) or any(not str(payload.get(key, "")).strip() for key in required):
                raise ValueError(f"invalid corpus document at line {line_number}")
            yield payload


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _fts_query(text: str) -> str:
    normalized = normalize_text(text)
    folded = normalize_text(text, fold_accents=True)
    tokens: list[str] = []
    seen: set[str] = set()
    for candidate in (normalized, folded):
        for token in TOKEN_PATTERN.findall(candidate):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    for pattern, candidate in ((VIETNAMESE_DATE_PATTERN, folded), (NUMERIC_DATE_PATTERN, text)):
        for day, month, year in pattern.findall(candidate):
            compact_date = f"{int(day):02d}{int(month):02d}{year}"
            if compact_date not in seen:
                seen.add(compact_date)
                tokens.append(compact_date)
    if not tokens:
        raise ValueError("BM25 query must contain at least one searchable token")
    quoted = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
    return " OR ".join(f"text_normalized:{token} OR text_folded:{token}" for token in quoted)


def build_bm25_index(
    *,
    root: Path,
    corpus_path: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_manifest = json.loads(read_text_with_retry(corpus_manifest_path))
    corpus_checksum = sha256_file(corpus_path)
    if corpus_checksum != corpus_manifest.get("corpus_sha256"):
        raise ValueError("corpus checksum does not match its manifest")
    expected_count = int(corpus_manifest.get("document_count", -1))
    if expected_count < 1:
        raise ValueError("corpus manifest must contain a positive document count")

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "documents.sqlite3"
    manifest_path = output_dir / "manifest.json"
    building_path = database_path.with_suffix(database_path.suffix + ".building")
    manifest_building_path = manifest_path.with_suffix(manifest_path.suffix + ".building")
    building_path.unlink(missing_ok=True)
    manifest_building_path.unlink(missing_ok=True)

    source_counts: Counter[str] = Counter()
    source_videos: dict[str, set[str]] = {}
    seen_document_ids: set[str] = set()
    document_count = 0
    connection = sqlite3.connect(building_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE documents (
                rowid INTEGER PRIMARY KEY,
                document_id TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                video_id TEXT NOT NULL,
                text TEXT NOT NULL,
                text_normalized TEXT NOT NULL,
                text_folded TEXT NOT NULL,
                keyframe_id INTEGER,
                frame_idx INTEGER,
                pts_time REAL,
                source_field TEXT,
                provenance_json TEXT NOT NULL
            );
            CREATE INDEX idx_documents_source_type ON documents(source_type);
            CREATE INDEX idx_documents_video_id ON documents(video_id);
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                text_normalized,
                text_folded,
                content='documents',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        insert_sql = """
            INSERT INTO documents(
                rowid,document_id,source_type,video_id,text,text_normalized,text_folded,
                keyframe_id,frame_idx,pts_time,source_field,provenance_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        rows: list[tuple[Any, ...]] = []
        for position, document in enumerate(_iter_documents(corpus_path), 1):
            document_id = str(document["document_id"])
            if document_id in seen_document_ids:
                raise ValueError(f"duplicate corpus document ID: {document_id}")
            seen_document_ids.add(document_id)
            source_type = str(document["source_type"])
            video_id = str(document["video_id"])
            source_counts[source_type] += 1
            source_videos.setdefault(source_type, set()).add(video_id)
            rows.append(
                (
                    position,
                    document_id,
                    source_type,
                    video_id,
                    document["text"],
                    document["text_normalized"],
                    document["text_folded"],
                    document.get("keyframe_id"),
                    document.get("frame_idx"),
                    document.get("pts_time"),
                    document.get("source_field"),
                    json.dumps(document.get("provenance", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                )
            )
            if len(rows) == 1000:
                connection.executemany(insert_sql, rows)
                document_count += len(rows)
                rows.clear()
                if progress:
                    progress(document_count, expected_count)
        if rows:
            connection.executemany(insert_sql, rows)
            document_count += len(rows)
            if progress:
                progress(document_count, expected_count)
        if document_count != expected_count:
            raise ValueError("corpus document count does not match its manifest")
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        fts_count = int(connection.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0])
        if integrity != "ok" or fts_count != document_count:
            raise ValueError(f"BM25 index validation failed: integrity={integrity}, fts_count={fts_count}")
        connection.execute("VACUUM")
    finally:
        connection.close()

    database_checksum = sha256_file(building_path)
    database_size = building_path.stat().st_size
    expected_sources = ("metadata", "asr", "ocr", "object")
    manifest = {
        "schema_version": BM25_INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "group": corpus_manifest.get("group"),
        "groups": corpus_manifest.get("groups") or ([corpus_manifest["group"]] if corpus_manifest.get("group") else []),
        "engine": {
            "name": "SQLite FTS5 BM25",
            "sqlite_version": sqlite3.sqlite_version,
            "tokenizer": "unicode61 remove_diacritics 2",
            "query_policy": "OR over unique normalized/accent-folded Unicode tokens with deterministic DDMMYYYY date expansion",
            "score_policy": "negative sqlite bm25 so higher is better",
        },
        "corpus": {
            "path": _portable_path(corpus_path, root),
            "manifest_path": _portable_path(corpus_manifest_path, root),
            "sha256": corpus_checksum,
            "input_fingerprint": corpus_manifest.get("input_fingerprint"),
            "document_count": document_count,
            "source_counts": {source: source_counts[source] for source in expected_sources},
            "source_video_counts": {source: len(source_videos.get(source, set())) for source in expected_sources},
        },
        "database": {
            "path": _portable_path(database_path, root),
            "sha256": database_checksum,
            "size_bytes": database_size,
            "document_count": document_count,
            "fts_document_count": fts_count,
            "integrity_check": integrity,
        },
        "timing_ms": {"total": round((time.perf_counter() - started) * 1000, 3)},
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "status": "READY",
    }
    manifest_building_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(building_path, database_path)
    os.replace(manifest_building_path, manifest_path)
    return manifest


class Bm25Retriever:
    name = "bm25"

    def __init__(self, root: Path, index_dir: Path) -> None:
        self.root = root.resolve()
        self.index_dir = index_dir.resolve()
        self.manifest_path = self.index_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest = json.loads(read_text_with_retry(self.manifest_path))
        raw_database = Path(str(self.manifest["database"]["path"]))
        raw_corpus = Path(str(self.manifest["corpus"]["path"]))
        self.database_path = raw_database if raw_database.is_absolute() else self.root / raw_database
        self.corpus_path = raw_corpus if raw_corpus.is_absolute() else self.root / raw_corpus
        self._validate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{self.database_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _validate(self) -> None:
        if self.manifest.get("schema_version") != BM25_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported BM25 index schema")
        if not self.database_path.is_file() or sha256_file(self.database_path) != self.manifest["database"]["sha256"]:
            raise ValueError("BM25 database is missing or corrupted")
        if not self.corpus_path.is_file() or sha256_file(self.corpus_path) != self.manifest["corpus"]["sha256"]:
            raise ValueError("BM25 corpus is stale or corrupted")
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if count != int(self.manifest["database"]["document_count"]) or integrity != "ok":
            raise ValueError("BM25 database validation failed")

    def health(self) -> RetrieverHealth:
        source_counts = self.manifest["corpus"].get("source_counts", {})
        source_video_counts = self.manifest["corpus"].get("source_video_counts", {})
        group_video_count = int(source_video_counts.get("metadata", 0))
        warnings = tuple(
            warning
            for warning in (
                "OCR unavailable" if not source_counts.get("ocr") else None,
                "ASR coverage is partial" if int(source_video_counts.get("asr", 0)) < group_video_count else None,
            )
            if warning
        )
        return RetrieverHealth(
            self.name,
            "DEGRADED" if warnings else "READY",
            self.manifest.get("schema_version"),
            ("lexical_text", "unicode", "accent_insensitive"),
            warnings,
            {"documents": int(self.manifest["database"]["document_count"])},
        )

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        index_groups = {str(item) for item in self.manifest.get("groups", ()) if str(item)}
        if not index_groups and self.manifest.get("group"):
            index_groups = {str(self.manifest["group"])}
        active_groups = index_groups.intersection(request.groups)
        if not active_groups:
            return []
        source_filter = tuple(sorted({str(item) for item in request.filters.get("source_types", ())}))
        video_filter = tuple(sorted({str(item) for item in request.filters.get("video_ids", ())}))
        conditions = ["documents_fts MATCH ?"]
        parameters: list[Any] = [_fts_query(request.query_text)]
        if active_groups != index_groups:
            conditions.append(f"({' OR '.join('d.video_id LIKE ?' for _ in active_groups)})")
            parameters.extend(f"{group}_%" for group in sorted(active_groups))
        if source_filter:
            conditions.append(f"d.source_type IN ({','.join('?' for _ in source_filter)})")
            parameters.extend(source_filter)
        if video_filter:
            conditions.append(f"d.video_id IN ({','.join('?' for _ in video_filter)})")
            parameters.extend(video_filter)
        parameters.append(request.top_k)
        sql = f"""
            SELECT d.*, bm25(documents_fts, 1.0, 1.0) AS bm25_score
            FROM documents_fts
            JOIN documents d ON d.rowid=documents_fts.rowid
            WHERE {' AND '.join(conditions)}
            ORDER BY bm25_score ASC, d.rowid ASC
            LIMIT ?
        """
        with self._connect() as connection:
            rows = list(connection.execute(sql, parameters))
        return [
            RetrievalHit(
                retriever=self.name,
                rank=rank,
                raw_score=-float(row["bm25_score"]),
                video_id=str(row["video_id"]),
                source_type=str(row["source_type"]),
                document_id=str(row["document_id"]),
                keyframe_id=row["keyframe_id"],
                frame_idx=row["frame_idx"],
                pts_time=row["pts_time"],
                matched_text=str(row["text"]),
                provenance={
                    "source_field": row["source_field"],
                    **json.loads(str(row["provenance_json"])),
                },
            )
            for rank, row in enumerate(rows, 1)
        ]
