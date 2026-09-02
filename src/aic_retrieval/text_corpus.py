from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator

from aic_retrieval.hybrid_audit import sha256_file
from aic_retrieval.phase5_schema import normalize_text


CORPUS_SCHEMA_VERSION = "hybrid-text-corpus-v1"
METADATA_FIELDS = ("title", "author", "description", "keywords", "publish_date")


@dataclass(frozen=True)
class TextDocument:
    document_id: str
    source_type: str
    video_id: str
    text: str
    text_normalized: str
    text_folded: str
    content_checksum: str
    keyframe_id: int | None = None
    frame_idx: int | None = None
    pts_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    source_field: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id or not self.source_type or not self.video_id or not self.text.strip():
            raise ValueError("text document identity and text must not be empty")
        if not self.text_normalized or not self.text_folded:
            raise ValueError("text document must contain normalized and folded text")
        for name, value in (("keyframe_id", self.keyframe_id), ("frame_idx", self.frame_idx)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (("pts_time", self.pts_time), ("start_time", self.start_time), ("end_time", self.end_time)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.start_time is not None and self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusBuildConfig:
    group: str = "L21"
    object_min_confidence: float = 0.3
    object_max_labels_per_frame: int = 20
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.groups and not self.group.strip():
            raise ValueError("group must not be empty")
        if any(not group.strip() for group in self.groups):
            raise ValueError("groups must contain non-empty values")
        if not 0 <= self.object_min_confidence <= 1:
            raise ValueError("object_min_confidence must be between 0 and 1")
        if self.object_max_labels_per_frame < 1:
            raise ValueError("object_max_labels_per_frame must be positive")

    @property
    def selected_groups(self) -> tuple[str, ...]:
        values = self.groups or (self.group,)
        return tuple(sorted({group.strip() for group in values}))


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _document_id(source_type: str, identity: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return f"{source_type}:{digest[:24]}"


def _make_document(
    *,
    source_type: str,
    video_id: str,
    text: str,
    identity: dict[str, Any],
    keyframe_id: int | None = None,
    frame_idx: int | None = None,
    pts_time: float | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    source_field: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> TextDocument | None:
    raw = " ".join(str(text).split())
    normalized = normalize_text(raw)
    folded = normalize_text(raw, fold_accents=True)
    if not raw or not normalized:
        return None
    return TextDocument(
        document_id=_document_id(source_type, identity),
        source_type=source_type,
        video_id=video_id,
        text=raw,
        text_normalized=normalized,
        text_folded=folded,
        content_checksum=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        keyframe_id=keyframe_id,
        frame_idx=frame_idx,
        pts_time=pts_time,
        start_time=start_time,
        end_time=end_time,
        source_field=source_field,
        provenance=provenance or {},
    )


def _metadata_documents(data_root: Path, video_ids: Iterable[str]) -> Iterator[TextDocument]:
    for video_id in sorted(set(video_ids)):
        path = data_root / "media-info" / f"{video_id}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field_name in METADATA_FIELDS:
            value = payload.get(field_name)
            if isinstance(value, list):
                text = ", ".join(str(item).strip() for item in value if str(item).strip())
            else:
                text = str(value or "").strip()
            document = _make_document(
                source_type="metadata",
                video_id=video_id,
                text=text,
                identity={"video_id": video_id, "field": field_name},
                source_field=field_name,
                provenance={"path": path.relative_to(data_root.parent).as_posix(), "field": field_name},
            )
            if document:
                yield document


def _open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _phase5_documents(path: Path, group: str) -> Iterator[TextDocument]:
    if not path.is_file():
        return
    with _open_sqlite_readonly(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "asr" in tables:
            rows = connection.execute(
                "SELECT video_id,segment_id,start_time,end_time,text_raw,confidence,keyframe_id,frame_idx,pts_time,mapping_kind,distance_seconds,run_id "
                "FROM asr WHERE video_id LIKE ? ORDER BY video_id,segment_id",
                (f"{group}_%",),
            )
            for row in rows:
                document = _make_document(
                    source_type="asr",
                    video_id=str(row["video_id"]),
                    text=str(row["text_raw"]),
                    identity={"video_id": row["video_id"], "segment_id": row["segment_id"], "run_id": row["run_id"]},
                    keyframe_id=row["keyframe_id"],
                    frame_idx=row["frame_idx"],
                    pts_time=row["pts_time"],
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    source_field="transcript_segment",
                    provenance={
                        "segment_id": row["segment_id"],
                        "confidence": row["confidence"],
                        "mapping_kind": row["mapping_kind"],
                        "distance_seconds": row["distance_seconds"],
                        "run_id": row["run_id"],
                    },
                )
                if document:
                    yield document
        if "ocr" in tables:
            rows = connection.execute(
                "SELECT id,video_id,keyframe_id,frame_idx,pts_time,text_raw,confidence,bbox_json,run_id "
                "FROM ocr WHERE video_id LIKE ? ORDER BY video_id,keyframe_id,id",
                (f"{group}_%",),
            )
            identity = lambda row: (
                str(row["video_id"]), int(row["keyframe_id"]), int(row["frame_idx"]), float(row["pts_time"])
            )
            for (video_id, keyframe_id, frame_idx, pts_time), frame_rows in groupby(rows, key=identity):
                items = list(frame_rows)
                texts = list(dict.fromkeys(str(row["text_raw"]).strip() for row in items if str(row["text_raw"]).strip()))
                document = _make_document(
                    source_type="ocr",
                    video_id=video_id,
                    text=" ".join(texts),
                    identity={"video_id": video_id, "keyframe_id": keyframe_id, "source": "ocr_frame"},
                    keyframe_id=keyframe_id,
                    frame_idx=frame_idx,
                    pts_time=pts_time,
                    source_field="detected_frame_text",
                    provenance={
                        "detection_count": len(items),
                        "max_confidence": max(float(row["confidence"]) for row in items),
                        "run_ids": sorted({str(row["run_id"]) for row in items}),
                    },
                )
                if document:
                    yield document


def _object_documents(path: Path, config: CorpusBuildConfig) -> Iterator[TextDocument]:
    if not path.is_file():
        return
    query = """
        SELECT d.video_id,d.keyframe_id,f.frame_idx,f.pts_time,d.label_normalized,
               MAX(d.confidence) AS max_confidence
        FROM detections d
        JOIN frames f ON f.video_id=d.video_id AND f.keyframe_id=d.keyframe_id
        WHERE d.video_id LIKE ? AND d.confidence >= ? AND d.label_normalized <> ''
        GROUP BY d.video_id,d.keyframe_id,f.frame_idx,f.pts_time,d.label_normalized
        ORDER BY d.video_id,d.keyframe_id,max_confidence DESC,d.label_normalized
    """
    with _open_sqlite_readonly(path) as connection:
        for group in config.selected_groups:
            current_key: tuple[str, int, int, float] | None = None
            labels: list[dict[str, Any]] = []
            for row in connection.execute(query, (f"{group}_%", config.object_min_confidence)):
                key = (str(row["video_id"]), int(row["keyframe_id"]), int(row["frame_idx"]), float(row["pts_time"]))
                if current_key is not None and key != current_key:
                    document = _make_object_document(current_key, labels, config)
                    if document:
                        yield document
                    labels = []
                current_key = key
                if len(labels) < config.object_max_labels_per_frame:
                    labels.append({"label": str(row["label_normalized"]), "confidence": round(float(row["max_confidence"]), 6)})
            if current_key is not None:
                document = _make_object_document(current_key, labels, config)
                if document:
                    yield document


def _make_object_document(
    key: tuple[str, int, int, float],
    labels: list[dict[str, Any]],
    config: CorpusBuildConfig,
) -> TextDocument | None:
    video_id, keyframe_id, frame_idx, pts_time = key
    return _make_document(
        source_type="object",
        video_id=video_id,
        text=", ".join(item["label"] for item in labels),
        identity={"video_id": video_id, "keyframe_id": keyframe_id, "aggregation": "max_confidence_distinct_labels"},
        keyframe_id=keyframe_id,
        frame_idx=frame_idx,
        pts_time=pts_time,
        source_field="aggregated_labels",
        provenance={
            "labels": labels,
            "min_confidence": config.object_min_confidence,
            "max_labels": config.object_max_labels_per_frame,
        },
    )


def _fingerprint_paths(paths: Iterable[Path], root: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in sorted({item.resolve() for item in paths if item.is_file()}):
        checksum = sha256_file(path)
        try:
            name = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            name = str(path)
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(checksum))
        records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": checksum})
    return digest.hexdigest(), records


def build_text_corpus(
    root: Path,
    config: CorpusBuildConfig = CorpusBuildConfig(),
    registry_path: Path | None = None,
    phase5_store_path: Path | None = None,
    object_store_path: Path | None = None,
    audit_path: Path | None = None,
) -> tuple[list[TextDocument], dict[str, Any]]:
    root = root.resolve()
    data_root = root / "data"
    registry_path = registry_path or root / "artifacts" / "registry" / "data_registry.json"
    phase5_store_path = phase5_store_path or root / "artifacts" / "phase5" / "phase5_store.sqlite"
    object_store_path = object_store_path or root / "artifacts" / "structured" / "l21_objects.sqlite"
    audit_path = audit_path or root / "artifacts" / "audits" / "l21_hybrid_retrieval_audit.json"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected_groups = config.selected_groups
    videos = [item for item in registry.get("videos", []) if str(item.get("group")) in selected_groups]
    video_ids = sorted({str(item["video_id"]) for item in videos})
    warnings: list[str] = []
    if not video_ids:
        raise ValueError(f"registry has no videos for groups {selected_groups}")
    if not phase5_store_path.is_file():
        warnings.append("Phase 5 store is missing; OCR/ASR documents were skipped")
    if not object_store_path.is_file():
        warnings.append("Object store is missing; object documents were skipped")

    documents = list(_metadata_documents(data_root, video_ids))
    for group in selected_groups:
        documents.extend(_phase5_documents(phase5_store_path, group))
    documents.extend(_object_documents(object_store_path, config))
    documents.sort(
        key=lambda item: (
            item.source_type,
            item.video_id,
            item.keyframe_id if item.keyframe_id is not None else -1,
            item.start_time if item.start_time is not None else -1.0,
            item.document_id,
        )
    )
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate document IDs: {duplicates[:5]}")

    source_counts = Counter(document.source_type for document in documents)
    source_videos: dict[str, set[str]] = defaultdict(set)
    mapped_counts: Counter[str] = Counter()
    for document in documents:
        source_videos[document.source_type].add(document.video_id)
        if document.keyframe_id is not None:
            mapped_counts[document.source_type] += 1
    expected_sources = ("metadata", "asr", "ocr", "object")
    complete_source_counts = {source: source_counts[source] for source in expected_sources}
    complete_source_video_counts = {source: len(source_videos[source]) for source in expected_sources}
    complete_mapped_counts = {source: mapped_counts[source] for source in expected_sources}
    if phase5_store_path.is_file() and complete_source_counts["ocr"] == 0:
        warnings.append("OCR contributed 0 documents")
    if phase5_store_path.is_file() and complete_source_video_counts["asr"] < len(video_ids):
        warnings.append(f"ASR contributes documents for {complete_source_video_counts['asr']}/{len(video_ids)} videos")
    metadata_paths = [data_root / "media-info" / f"{video_id}.json" for video_id in video_ids]
    input_fingerprint, inputs = _fingerprint_paths([registry_path, phase5_store_path, object_store_path, *metadata_paths], root)
    audit_source_fingerprint = None
    if audit_path.is_file():
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_source_fingerprint = audit_payload.get("source_fingerprint")
    payload_lines = [_canonical_json(document.to_dict()) for document in documents]
    corpus_payload = "\n".join(payload_lines) + ("\n" if payload_lines else "")
    corpus_sha256 = hashlib.sha256(corpus_payload.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "group": selected_groups[0] if len(selected_groups) == 1 else None,
        "groups": list(selected_groups),
        "config": asdict(config),
        "input_fingerprint": input_fingerprint,
        "audit_source_fingerprint": audit_source_fingerprint,
        "inputs": inputs,
        "document_count": len(documents),
        "source_counts": complete_source_counts,
        "source_video_counts": complete_source_video_counts,
        "frame_mapped_counts": complete_mapped_counts,
        "text_characters": sum(len(document.text) for document in documents),
        "corpus_size_bytes": len(corpus_payload.encode("utf-8")),
        "corpus_sha256": corpus_sha256,
        "warnings": warnings,
        "status": "READY_WITH_WARNINGS" if warnings else "READY",
    }
    return documents, manifest


def write_text_corpus(documents: Iterable[TextDocument], manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    manifest_path = output_dir / "manifest.json"
    lines = [_canonical_json(document.to_dict()) for document in documents]
    with documents_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + ("\n" if lines else ""))
    actual_checksum = sha256_file(documents_path)
    if actual_checksum != manifest["corpus_sha256"]:
        raise ValueError("written corpus checksum does not match manifest")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"documents": str(documents_path), "manifest": str(manifest_path), "document_count": len(lines), "corpus_sha256": actual_checksum}
