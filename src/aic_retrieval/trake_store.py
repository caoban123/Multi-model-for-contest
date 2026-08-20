"""Separate SQLite audit store for TRAKE sessions and internal exports."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STORE_VERSION = "phase8-trake-store-v2"
READABLE_STORE_VERSIONS = {"phase8-trake-store-v1", STORE_VERSION}


class TrakeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            existing_row = connection.execute(
                "SELECT value FROM metadata WHERE key='store_version'"
            ).fetchone() if _table_exists(connection, "metadata") else None
            existing_version = str(existing_row[0]) if existing_row else None
            if existing_version is not None and existing_version not in READABLE_STORE_VERSIONS:
                raise ValueError(f"unsupported TRAKE store version: {existing_version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions(
                  session_id TEXT PRIMARY KEY, query_id TEXT NOT NULL, original_query TEXT NOT NULL,
                  group_name TEXT NOT NULL, plan_revision INTEGER NOT NULL, created_at REAL NOT NULL,
                  updated_at TEXT NOT NULL, state_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                  session_id TEXT NOT NULL, event_id TEXT NOT NULL, event_order INTEGER NOT NULL,
                  required INTEGER NOT NULL, payload_json TEXT NOT NULL,
                  PRIMARY KEY(session_id,event_id));
                CREATE TABLE IF NOT EXISTS candidates(
                  session_id TEXT NOT NULL, candidate_id TEXT NOT NULL, event_id TEXT NOT NULL,
                  video_id TEXT NOT NULL, pts_time REAL NOT NULL, payload_json TEXT NOT NULL,
                  PRIMARY KEY(session_id,candidate_id));
                CREATE TABLE IF NOT EXISTS video_candidates(
                  session_id TEXT NOT NULL, video_id TEXT NOT NULL, complete INTEGER NOT NULL,
                  payload_json TEXT NOT NULL, PRIMARY KEY(session_id,video_id));
                CREATE TABLE IF NOT EXISTS chains(
                  session_id TEXT NOT NULL, chain_id TEXT NOT NULL, video_id TEXT NOT NULL,
                  valid INTEGER NOT NULL, final_score REAL NOT NULL, selection TEXT NOT NULL,
                  payload_json TEXT NOT NULL, PRIMARY KEY(session_id,chain_id,selection));
                CREATE TABLE IF NOT EXISTS chain_events(
                  session_id TEXT NOT NULL, chain_id TEXT NOT NULL, event_id TEXT NOT NULL,
                  candidate_id TEXT, selection TEXT NOT NULL, payload_json TEXT NOT NULL,
                  PRIMARY KEY(session_id,chain_id,event_id,selection));
                CREATE TABLE IF NOT EXISTS reviews(
                  review_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, chain_id TEXT NOT NULL,
                  decision TEXT NOT NULL, reviewer TEXT NOT NULL, created_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS refinements(
                  refinement_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                  created_at TEXT NOT NULL, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS exports(
                  export_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, review_id TEXT NOT NULL,
                  created_at TEXT NOT NULL, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS schema_migrations(
                  migration_id TEXT PRIMARY KEY, from_version TEXT, to_version TEXT NOT NULL,
                  applied_at TEXT NOT NULL, notes TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_trake_reviews_session ON reviews(session_id,created_at);
                CREATE INDEX IF NOT EXISTS idx_trake_exports_session ON exports(session_id,created_at);
                """
            )
            if existing_version is not None and existing_version != STORE_VERSION:
                connection.execute(
                    "INSERT OR IGNORE INTO schema_migrations VALUES(?,?,?,?,?)",
                    (
                        f"{existing_version}-to-{STORE_VERSION}",
                        existing_version,
                        STORE_VERSION,
                        _now(),
                        "Non-destructive metadata/state JSON compatibility migration; existing tables and records retained.",
                    ),
                )
            connection.execute("INSERT OR REPLACE INTO metadata VALUES('store_version',?)", (STORE_VERSION,))

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save_state(self, payload: dict[str, Any]) -> None:
        state = payload["state"]
        request = state["request"]
        now = _now()
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?,?,?)",
                (state["session_id"], request["query_id"], request["original_query"], request["group"], state["plan_revision"], state["created_at_epoch"], now, _dump(state)),
            )
            for table in ("events", "candidates", "video_candidates", "chains", "chain_events"):
                connection.execute(f"DELETE FROM {table} WHERE session_id=?", (state["session_id"],))
            for event in request["events"]:
                connection.execute("INSERT INTO events VALUES(?,?,?,?,?)", (state["session_id"], event["event_id"], event["order"], int(event["required"]), _dump(event)))
            seen: set[str] = set()
            for pool in state.get("pools", []):
                for candidate in pool["candidates"]:
                    if candidate["candidate_id"] in seen: continue
                    seen.add(candidate["candidate_id"])
                    connection.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?)", (state["session_id"], candidate["candidate_id"], candidate["event_id"], candidate["video_id"], candidate["pts_time"], _dump(candidate)))
            for video in state.get("videos", []):
                connection.execute("INSERT INTO video_candidates VALUES(?,?,?,?)", (state["session_id"], video["video_id"], int(video["complete"]), _dump(video)))
            chains = [chain for result in state.get("alignments", []) for chain in result.get("chains", [])]
            if state.get("manual_chain"):
                chains.append(state["manual_chain"])
            for chain in chains:
                selection = "manual" if chain.get("manual", False) else "algorithm"
                persisted_score = (chain.get("score_components") or {}).get("final_rerank_score", chain["score"]["final_score"])
                connection.execute("INSERT OR REPLACE INTO chains VALUES(?,?,?,?,?,?,?)", (state["session_id"], chain["chain_id"], chain["video_id"], int(chain["valid"]), persisted_score, selection, _dump(chain)))
                for event in chain["events"]:
                    candidate = event.get("candidate")
                    connection.execute("INSERT OR REPLACE INTO chain_events VALUES(?,?,?,?,?,?)", (state["session_id"], chain["chain_id"], event["event_id"], candidate.get("candidate_id") if candidate else None, event.get("selection", selection), _dump(event)))

    def load_state(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT state_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["state_json"]) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT session_id,query_id,original_query,plan_revision,updated_at FROM sessions ORDER BY updated_at DESC,session_id").fetchall()
        return [dict(row) for row in rows]

    def append_review(self, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO reviews VALUES(?,?,?,?,?,?,?)", (payload["review_id"], payload["session_id"], payload["chain_id"], payload["decision"], payload["reviewer"], payload["created_at"], _dump(payload)))

    def reviews(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload_json FROM reviews WHERE session_id=? ORDER BY created_at,review_id", (session_id,)).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def append_export(self, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO exports VALUES(?,?,?,?,?)", (payload["export_id"], payload["session_id"], payload["review_id"], payload["created_at"], _dump(payload)))

    def append_refinement(self, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO refinements VALUES(?,?,?,?,?)", (payload["refinement_id"], payload["session_id"], payload["event_id"], payload["created_at"], _dump(payload)))

    def refinements(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows=connection.execute("SELECT payload_json FROM refinements WHERE session_id=? ORDER BY created_at,refinement_id",(session_id,)).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None
