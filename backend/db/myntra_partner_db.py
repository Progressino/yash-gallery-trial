"""Myntra Partner API — outbound webhook event persistence."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any


def _db_path() -> str:
    return os.environ.get("MYNTRA_PARTNER_DB_PATH", "./myntra_partner_dev.db")


def _conn():
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def init_db() -> None:
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS myntra_webhook_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                event_type      TEXT NOT NULL DEFAULT '',
                seller_order_id TEXT NOT NULL DEFAULT '',
                packet_id       TEXT NOT NULL DEFAULT '',
                order_line_id   TEXT NOT NULL DEFAULT '',
                payload_json    TEXT NOT NULL,
                headers_json    TEXT NOT NULL DEFAULT '{}',
                received_at     TEXT DEFAULT (datetime('now')),
                duplicate       INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_myntra_webhook_received
                ON myntra_webhook_events(received_at DESC);
            CREATE INDEX IF NOT EXISTS idx_myntra_webhook_seller_order
                ON myntra_webhook_events(seller_order_id);
            """
        )


def insert_webhook_event(
    *,
    idempotency_key: str,
    event_type: str,
    seller_order_id: str = "",
    packet_id: str = "",
    order_line_id: str = "",
    payload: dict | list | Any,
    headers: dict | None = None,
) -> tuple[int, bool]:
    """
    Insert event. Returns (row_id, is_duplicate).
    Duplicate idempotency_key returns existing row id with is_duplicate=True.
    """
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    headers_json = json.dumps(headers or {}, ensure_ascii=False, default=str)
    conn = _conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO myntra_webhook_events(
                idempotency_key, event_type, seller_order_id, packet_id,
                order_line_id, payload_json, headers_json, duplicate
            ) VALUES (?,?,?,?,?,?,?,0)
            """,
            (
                idempotency_key,
                event_type,
                seller_order_id or "",
                packet_id or "",
                order_line_id or "",
                payload_json,
                headers_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid), False
    except sqlite3.IntegrityError:
        row = conn.execute(
            "SELECT id FROM myntra_webhook_events WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        conn.close()
        return int(row["id"]) if row else 0, True
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_recent_events(*, limit: int = 50) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        """
        SELECT id, idempotency_key, event_type, seller_order_id, packet_id,
               order_line_id, received_at, duplicate
        FROM myntra_webhook_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_events() -> int:
    conn = _conn()
    n = conn.execute("SELECT COUNT(*) FROM myntra_webhook_events").fetchone()[0]
    conn.close()
    return int(n or 0)
