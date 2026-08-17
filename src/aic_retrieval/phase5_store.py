from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from aic_retrieval.phase5_schema import AsrTranscript, OcrFrame, map_segment_to_frame, normalize_text

STORE_VERSION = "phase5-store-v1"


def create_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS ocr(
          id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, keyframe_id INTEGER NOT NULL,
          frame_idx INTEGER NOT NULL, pts_time REAL NOT NULL, text_raw TEXT NOT NULL,
          text_normalized TEXT NOT NULL, text_folded TEXT NOT NULL, confidence REAL NOT NULL,
          bbox_json TEXT NOT NULL, run_id TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(text_normalized, text_folded, content='ocr', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS ocr_ai AFTER INSERT ON ocr BEGIN
          INSERT INTO ocr_fts(rowid,text_normalized,text_folded) VALUES(new.id,new.text_normalized,new.text_folded);
        END;
        CREATE TABLE IF NOT EXISTS asr(
          id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, segment_id INTEGER NOT NULL,
          start_time REAL NOT NULL, end_time REAL NOT NULL, text_raw TEXT NOT NULL,
          text_normalized TEXT NOT NULL, text_folded TEXT NOT NULL, confidence REAL,
          keyframe_id INTEGER, frame_idx INTEGER, pts_time REAL, mapping_kind TEXT NOT NULL,
          distance_seconds REAL, run_id TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS asr_fts USING fts5(text_normalized, text_folded, content='asr', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS asr_ai AFTER INSERT ON asr BEGIN
          INSERT INTO asr_fts(rowid,text_normalized,text_folded) VALUES(new.id,new.text_normalized,new.text_folded);
        END;
        CREATE INDEX IF NOT EXISTS idx_ocr_frame ON ocr(video_id,keyframe_id);
        CREATE INDEX IF NOT EXISTS idx_asr_video_time ON asr(video_id,start_time,end_time);
    """)
    connection.execute("INSERT OR REPLACE INTO metadata VALUES('store_version',?)", (STORE_VERSION,))
    return connection


def ingest_ocr(connection: sqlite3.Connection, frames: Iterable[OcrFrame]) -> int:
    count = 0
    for frame in frames:
        for detection in frame.detections:
            connection.execute("INSERT INTO ocr(video_id,keyframe_id,frame_idx,pts_time,text_raw,text_normalized,text_folded,confidence,bbox_json,run_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (frame.video_id,frame.keyframe_id,frame.frame_idx,frame.pts_time,detection.text,detection.text_normalized,detection.text_folded,detection.confidence,json.dumps(detection.bbox),frame.run_id))
            count += 1
    connection.commit()
    return count


def ingest_asr(connection: sqlite3.Connection, transcripts: Iterable[AsrTranscript], refs: Iterable[Any], max_distance: float = 3.0) -> int:
    refs = list(refs); count = 0
    for transcript in transcripts:
        if transcript.status != "AVAILABLE": continue
        for segment in transcript.segments:
            mapping = map_segment_to_frame(transcript.video_id,segment.start_time,segment.end_time,refs,max_distance)
            connection.execute("INSERT INTO asr(video_id,segment_id,start_time,end_time,text_raw,text_normalized,text_folded,confidence,keyframe_id,frame_idx,pts_time,mapping_kind,distance_seconds,run_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (transcript.video_id,segment.segment_id,segment.start_time,segment.end_time,segment.text,segment.text_normalized,segment.text_folded,segment.confidence,mapping.keyframe_id,mapping.frame_idx,mapping.pts_time,mapping.mapping_kind,mapping.distance_seconds,transcript.run_id))
            count += 1
    connection.commit(); return count


class Phase5SearchService:
    def __init__(self, path: Path, refs: Iterable[Any] = ()) -> None:
        self.path = path; self.refs = list(refs)

    @property
    def available(self) -> bool: return self.path.is_file()

    def search_ocr(self, query: str, top_k: int = 100, min_confidence: float = 0.0) -> list[dict[str, Any]]:
        return self._search("ocr", query, top_k, min_confidence)

    def search_asr(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        return self._search("asr", query, top_k, None)

    def evidence_for_frame(self, video_id: str, keyframe_id: int) -> dict[str, list[dict[str, Any]]]:
        if not self.available: return {"ocr": [], "asr": []}
        with sqlite3.connect(self.path) as connection:
            connection.row_factory=sqlite3.Row
            ocr=[dict(row) for row in connection.execute("SELECT text_raw,confidence,bbox_json FROM ocr WHERE video_id=? AND keyframe_id=?",(video_id,keyframe_id))]
            asr=[dict(row) for row in connection.execute("SELECT segment_id,start_time,end_time,text_raw,mapping_kind,distance_seconds FROM asr WHERE video_id=? AND keyframe_id=?",(video_id,keyframe_id))]
        for item in ocr: item["bbox"] = json.loads(item.pop("bbox_json"))
        return {"ocr":ocr,"asr":asr}

    def _search(self, modality: str, query: str, top_k: int, min_confidence: float | None) -> list[dict[str, Any]]:
        if not self.available: raise FileNotFoundError(self.path)
        exact=normalize_text(query); folded=normalize_text(query,fold_accents=True)
        if not exact: raise ValueError("text query must not be empty")
        terms=" OR ".join(f'"{term}"' for term in sorted({exact,folded}) if term)
        where=" AND source.confidence >= ?" if min_confidence is not None else ""
        params:[Any]=[terms];
        if min_confidence is not None: params.append(min_confidence)
        params.append(max(1,min(top_k,500)))
        with sqlite3.connect(self.path) as connection:
            connection.row_factory=sqlite3.Row
            rows=connection.execute(f"SELECT source.*, bm25({modality}_fts) AS lexical_score FROM {modality}_fts JOIN {modality} source ON source.id={modality}_fts.rowid WHERE {modality}_fts MATCH ?{where} ORDER BY lexical_score, source.video_id, source.id LIMIT ?",params).fetchall()
        results=[]
        for rank,row in enumerate(rows,1):
            item=dict(row); item[f"{modality}_rank"]=rank; item[f"{modality}_score"]=-float(item.pop("lexical_score")); item["matched_text"]=item["text_raw"]
            if "bbox_json" in item: item["bbox"]=json.loads(item.pop("bbox_json"))
            results.append(item)
        return results
