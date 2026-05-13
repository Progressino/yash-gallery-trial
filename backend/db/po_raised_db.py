"""
SQLite persistence for the daily "Raised PO" ledger.

When the ops team clicks "Export & Confirm PO" in the PO Engine, the same
payload is also recorded here. Tomorrow's PO calculation reads this ledger
and adds the recorded qty to the SKU's PO_Pipeline_Total, so the same SKU
isn't recommended again until the raised PO is either:
  - older than its lead time (auto-expired), or
  - the user explicitly clears the ledger, or
  - the user uploads a fresh "Existing PO" sheet that already includes it
    (operator can clear the internal ledger to avoid double-counting).

DB path: ``PO_RAISED_DB_PATH`` env var, default ``/data/po_raised.db`` with
a local-dev fallback to ``./po_raised_dev.db``.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

DB_PATH = os.environ.get("PO_RAISED_DB_PATH", "/data/po_raised.db")


def _connect() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:
        conn = sqlite3.connect("./po_raised_dev.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the ledger table on app startup."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS po_raised (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                oms_sku     TEXT    NOT NULL,
                qty         REAL    NOT NULL,
                raised_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                raised_date TEXT    NOT NULL DEFAULT (date('now')),
                lead_time   INTEGER,
                note        TEXT    DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_po_raised_sku_date "
            "ON po_raised(oms_sku, raised_date)"
        )
        conn.commit()
    finally:
        conn.close()


def record_raises(items: Iterable[dict]) -> int:
    """Persist a batch of raised POs. Each item must have ``oms_sku`` and
    ``qty``; ``lead_time`` (days) and ``note`` are optional. Returns count.
    """
    rows: list[tuple[str, float, Optional[int], str]] = []
    for it in items:
        sku = str(it.get("oms_sku") or it.get("OMS_SKU") or "").strip()
        try:
            qty = float(it.get("qty") or it.get("Final_PO_Qty") or 0)
        except Exception:
            qty = 0.0
        if not sku or qty <= 0:
            continue
        try:
            lt = it.get("lead_time")
            lt_int: Optional[int] = int(lt) if lt is not None else None
        except Exception:
            lt_int = None
        note = str(it.get("note") or "")[:512]
        rows.append((sku, qty, lt_int, note))

    if not rows:
        return 0

    conn = _connect()
    try:
        conn.executemany(
            "INSERT INTO po_raised (oms_sku, qty, lead_time, note) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def list_raises(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sku: Optional[str] = None,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """Return raised PO rows ordered newest-first."""
    where: list[str] = []
    args: list[Any] = []
    if start_date:
        where.append("raised_date >= ?")
        args.append(start_date)
    if end_date:
        where.append("raised_date <= ?")
        args.append(end_date)
    if sku:
        where.append("oms_sku = ?")
        args.append(sku)
    sql = (
        "SELECT id, oms_sku, qty, raised_at, raised_date, lead_time, note "
        "FROM po_raised"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY raised_at DESC LIMIT ?"
    args.append(int(max(1, min(50_000, limit))))

    conn = _connect()
    try:
        cur = conn.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def summary_by_sku(
    only_active: bool = True,
    default_lead_time: int = 60,
) -> list[dict[str, Any]]:
    """Aggregate active raised qty per SKU.

    ``only_active`` filters out rows whose age exceeds the row's
    ``lead_time`` (or ``default_lead_time`` if none stored). These are
    treated as already received and no longer part of pipeline.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT oms_sku, qty, raised_date, lead_time FROM po_raised"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    today = datetime.utcnow().date()
    tally: dict[str, dict[str, Any]] = {}
    for r in rows:
        sku = str(r["oms_sku"])
        qty = float(r["qty"] or 0)
        if qty <= 0:
            continue
        raised_date_str = str(r["raised_date"])
        try:
            rd = datetime.fromisoformat(raised_date_str).date()
        except Exception:
            rd = today
        lt = int(r["lead_time"]) if r["lead_time"] is not None else int(default_lead_time)
        active = (today - rd).days <= max(1, lt)
        if only_active and not active:
            continue
        slot = tally.setdefault(
            sku,
            {
                "oms_sku": sku,
                "qty": 0.0,
                "last_raised_date": raised_date_str,
                "first_raised_date": raised_date_str,
            },
        )
        slot["qty"] += qty
        if raised_date_str > slot["last_raised_date"]:
            slot["last_raised_date"] = raised_date_str
        if raised_date_str < slot["first_raised_date"]:
            slot["first_raised_date"] = raised_date_str

    return sorted(tally.values(), key=lambda x: x["last_raised_date"], reverse=True)


def clear_all() -> int:
    """Wipe the ledger. Returns number of rows deleted."""
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM po_raised")
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def clear_older_than_days(days: int) -> int:
    """Delete entries older than ``days``. Returns rowcount."""
    cutoff = (datetime.utcnow().date() - timedelta(days=max(0, int(days)))).isoformat()
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM po_raised WHERE raised_date < ?", (cutoff,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def delete_by_ids(ids: Iterable[int]) -> int:
    id_list = [int(i) for i in ids if i is not None]
    if not id_list:
        return 0
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(id_list))
        cur = conn.execute(
            f"DELETE FROM po_raised WHERE id IN ({placeholders})", id_list
        )
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()
