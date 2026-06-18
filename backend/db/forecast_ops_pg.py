"""
Durable PostgreSQL store for shared operational data (all users).

- ``forecast_shared_snapshot``: one row — warm-cache parquets + sku_mapping (survives deploy/restart)
- ``forecast_daily_uploads``: Tier-3 daily sales blobs (PostgreSQL backend for daily_store)

Enable with FORECAST_SESSION_DATABASE_URL (same DB as forecast_app_sessions).
Set DAILY_SALES_BACKEND=postgres to read Tier-3 from PG (dual-write always when PG is up).
"""
from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from ..services.helpers import _coerce_df_for_parquet

_log = logging.getLogger(__name__)

_BUNDLE_VERSION = 1
_SNAPSHOT_ID = "shared"

# Parquet + JSON keys mirrored from warm cache / disk manifest.
_SNAPSHOT_PARQUET_KEYS = (
    "sales_df",
    "mtr_df",
    "meesho_df",
    "myntra_df",
    "flipkart_df",
    "snapdeal_df",
    "inventory_df_variant",
    "inventory_df_parent",
    "existing_po_df",
    "po_raise_ledger_df",
    "po_return_overlay_df",
    "sku_status_lead_df",
    "daily_inventory_history_df",
    "manual_intransit_overlay_df",
)

_SNAPSHOT_JSON_KEYS = (
    "sku_mapping",
    "inventory_session_meta",
    "existing_po_meta",
    "return_overlay_meta",
)

_table_ready = False


def _connection_url() -> Optional[str]:
    u = (os.environ.get("FORECAST_SESSION_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    return u or None


def ops_pg_enabled() -> bool:
    if os.environ.get("FORECAST_OPS_PG", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool(_connection_url()) and _table_ready


def daily_uploads_pg_read() -> bool:
    raw = (os.environ.get("DAILY_SALES_BACKEND") or "postgres").strip().lower()
    return raw in ("postgres", "pg", "postgresql") and ops_pg_enabled()


def init_db() -> None:
    global _table_ready
    url = _connection_url()
    if not url:
        _table_ready = False
        return
    try:
        import psycopg
    except ImportError:
        _log.warning("psycopg not installed — forecast ops PostgreSQL disabled")
        _table_ready = False
        return
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_shared_snapshot (
                    snapshot_id TEXT PRIMARY KEY,
                    bundle      BYTEA NOT NULL,
                    manifest    JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_daily_uploads (
                    id           BIGSERIAL PRIMARY KEY,
                    platform     TEXT NOT NULL,
                    file_date    DATE NOT NULL,
                    filename     TEXT NOT NULL,
                    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    rows         INTEGER NOT NULL DEFAULT 0,
                    date_from    DATE,
                    date_to      DATE,
                    data_parquet BYTEA NOT NULL,
                    UNIQUE (platform, filename)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecast_daily_uploads_plat_date "
                "ON forecast_daily_uploads (platform, file_date DESC)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_platform_blocked_dates (
                    platform TEXT NOT NULL,
                    blocked_date DATE NOT NULL,
                    PRIMARY KEY (platform, blocked_date)
                )
            """)
        _table_ready = True
        _log.info("forecast ops PostgreSQL tables ready (shared snapshot + daily_uploads)")
    except Exception:
        _log.exception("forecast ops PostgreSQL init failed")
        _table_ready = False


def _require_conn():
    import psycopg

    url = _connection_url()
    if not url or not _table_ready:
        return None
    return psycopg.connect(url, autocommit=True)


def warm_cache_dict_to_bundle(cache_dict: dict) -> tuple[bytes, dict]:
    """Serialize warm-cache dict to zip bytes + manifest metadata."""
    buf = io.BytesIO()
    keys_saved: list[str] = []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in _SNAPSHOT_JSON_KEYS:
            val = cache_dict.get(key)
            if isinstance(val, dict) and val:
                zf.writestr(f"{key}.json", json.dumps(val, default=str).encode("utf-8"))
                keys_saved.append(key)
            elif key == "sku_mapping":
                sm = cache_dict.get("sku_mapping") or {}
                if isinstance(sm, dict) and sm:
                    zf.writestr("sku_mapping.json", json.dumps(sm, default=str).encode("utf-8"))
                    keys_saved.append("sku_mapping")
        for key in _SNAPSHOT_PARQUET_KEYS:
            df = cache_dict.get(key)
            if df is None or not hasattr(df, "to_parquet"):
                continue
            if getattr(df, "empty", True):
                continue
            bio = io.BytesIO()
            _coerce_df_for_parquet(df).to_parquet(bio, index=False, engine="pyarrow")
            zf.writestr(f"{key}.parquet", bio.getvalue())
            keys_saved.append(key)
    manifest = {
        "bundle_version": _BUNDLE_VERSION,
        "keys": sorted(set(keys_saved)),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    return buf.getvalue(), manifest


def warm_cache_dict_from_bundle(data: bytes) -> dict | None:
    """Deserialize zip bytes to warm-cache dict."""
    try:
        out: dict = {}
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            if "sku_mapping.json" in zf.namelist():
                out["sku_mapping"] = json.loads(zf.read("sku_mapping.json").decode("utf-8"))
            for key in _SNAPSHOT_JSON_KEYS:
                fn = f"{key}.json"
                if fn in zf.namelist():
                    out[key] = json.loads(zf.read(fn).decode("utf-8"))
            for key in _SNAPSHOT_PARQUET_KEYS:
                fn = f"{key}.parquet"
                if fn not in zf.namelist():
                    continue
                raw = zf.read(fn)
                if not raw:
                    continue
                out[key] = pd.read_parquet(io.BytesIO(raw))
        return out if out else None
    except Exception:
        _log.exception("warm_cache_dict_from_bundle failed")
        return None


def _snapshot_has_operational_data(data: dict) -> bool:
    if not data:
        return False
    sm = data.get("sku_mapping")
    if isinstance(sm, dict) and sm:
        return True
    for key in ("mtr_df", "sales_df", "myntra_df", "inventory_df_variant"):
        df = data.get(key)
        if df is not None and hasattr(df, "__len__") and len(df) > 0:
            return True
    return False


def persist_shared_snapshot(cache_dict: dict) -> bool:
    """Upsert shared warm-cache snapshot — all users hydrate from this after deploy."""
    if not ops_pg_enabled() or not cache_dict:
        return False
    if not _snapshot_has_operational_data(cache_dict):
        _log.warning("skip shared snapshot persist: no operational rows in cache_dict")
        return False
    conn = _require_conn()
    if conn is None:
        return False
    try:
        blob, manifest = warm_cache_dict_to_bundle(cache_dict)
        max_mb = int(os.environ.get("SHARED_SNAPSHOT_MAX_MB", "512"))
        if len(blob) > max_mb * 1024 * 1024:
            _log.warning(
                "skip shared snapshot persist: bundle %d MB exceeds SHARED_SNAPSHOT_MAX_MB=%d",
                len(blob) // (1024 * 1024),
                max_mb,
            )
            return False
        with conn:
            conn.execute(
                """
                INSERT INTO forecast_shared_snapshot (snapshot_id, bundle, manifest, updated_at)
                VALUES (%s, %s, %s::jsonb, NOW())
                ON CONFLICT (snapshot_id) DO UPDATE
                SET bundle = EXCLUDED.bundle,
                    manifest = EXCLUDED.manifest,
                    updated_at = NOW()
                """,
                (_SNAPSHOT_ID, blob, json.dumps(manifest)),
            )
        _log.info(
            "Shared operational snapshot saved to PostgreSQL (%d keys, %d MB)",
            len(manifest.get("keys") or []),
            len(blob) // (1024 * 1024),
        )
        return True
    except Exception:
        _log.exception("persist_shared_snapshot failed")
        return False


def load_shared_snapshot() -> dict | None:
    """Load shared warm-cache dict from PostgreSQL."""
    if not ops_pg_enabled():
        return None
    conn = _require_conn()
    if conn is None:
        return None
    try:
        with conn:
            row = conn.execute(
                "SELECT bundle, manifest FROM forecast_shared_snapshot WHERE snapshot_id = %s",
                (_SNAPSHOT_ID,),
            ).fetchone()
        if row is None or not row[0]:
            return None
        data = warm_cache_dict_from_bundle(bytes(row[0]))
        if data and _snapshot_has_operational_data(data):
            return data
        return None
    except Exception:
        _log.exception("load_shared_snapshot failed")
        return None


def shared_snapshot_status() -> dict[str, Any]:
    if not ops_pg_enabled():
        return {"enabled": False}
    conn = _require_conn()
    if conn is None:
        return {"enabled": False}
    try:
        with conn:
            row = conn.execute(
                "SELECT manifest, updated_at FROM forecast_shared_snapshot WHERE snapshot_id = %s",
                (_SNAPSHOT_ID,),
            ).fetchone()
            upload_count = conn.execute("SELECT COUNT(*) FROM forecast_daily_uploads").fetchone()
        if row is None:
            return {"enabled": True, "present": False, "daily_uploads": int(upload_count[0] or 0)}
        manifest = row[0] if isinstance(row[0], dict) else {}
        return {
            "enabled": True,
            "present": True,
            "updated_at": row[1].isoformat() if row[1] else None,
            "keys": manifest.get("keys") or [],
            "daily_uploads": int(upload_count[0] or 0),
        }
    except Exception:
        _log.exception("shared_snapshot_status failed")
        return {"enabled": True, "present": False, "error": True}


# ── Tier-3 daily uploads (PostgreSQL) ─────────────────────────────────────────

def pg_save_daily_file(
    platform: str,
    filename: str,
    file_date: str,
    date_from: str | None,
    date_to: str | None,
    parquet_bytes: bytes,
    row_count: int,
    *,
    max_files_per_platform: int = 200,
) -> None:
    if not ops_pg_enabled():
        return
    conn = _require_conn()
    if conn is None:
        return
    try:
        with conn:
            conn.execute(
                "DELETE FROM forecast_daily_uploads WHERE platform = %s AND filename = %s",
                (platform, filename),
            )
            conn.execute(
                """
                INSERT INTO forecast_daily_uploads
                    (platform, file_date, filename, rows, date_from, date_to, data_parquet)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    platform,
                    file_date,
                    filename,
                    row_count,
                    date_from or None,
                    date_to or None,
                    parquet_bytes,
                ),
            )
            conn.execute(
                """
                DELETE FROM forecast_daily_uploads
                WHERE platform = %s AND id NOT IN (
                    SELECT id FROM forecast_daily_uploads
                    WHERE platform = %s
                    ORDER BY file_date DESC, id DESC
                    LIMIT %s
                )
                """,
                (platform, platform, max_files_per_platform),
            )
    except Exception:
        _log.exception("pg_save_daily_file failed platform=%s file=%s", platform, filename[:60])


def pg_load_platform_rows(
    platform: str,
    months: int | None = None,
    max_files: int | None = None,
) -> list[tuple[str, bytes]]:
    if not ops_pg_enabled():
        return []
    conn = _require_conn()
    if conn is None:
        return []
    try:
        with conn:
            if months is None:
                rows = conn.execute(
                    """
                    SELECT filename, data_parquet FROM forecast_daily_uploads
                    WHERE platform = %s ORDER BY file_date ASC
                    """,
                    (platform,),
                ).fetchall()
            else:
                cutoff = (date.today() - timedelta(days=months * 30)).isoformat()
                rows = conn.execute(
                    """
                    SELECT filename, data_parquet FROM forecast_daily_uploads
                    WHERE platform = %s AND file_date >= %s::date
                    ORDER BY file_date ASC
                    """,
                    (platform, cutoff),
                ).fetchall()
        out = [(str(r[0]), bytes(r[1])) for r in rows if r[1]]
        if max_files is not None and len(out) > max_files:
            out = out[-max_files:]
        return out
    except Exception:
        _log.exception("pg_load_platform_rows failed platform=%s", platform)
        return []


def pg_get_summary() -> dict:
    if not ops_pg_enabled():
        return {}
    conn = _require_conn()
    if conn is None:
        return {}
    try:
        with conn:
            rows = conn.execute(
                """
                SELECT platform,
                       COUNT(*) AS file_count,
                       COALESCE(SUM(rows), 0) AS total_rows,
                       MAX(file_date) AS max_date,
                       MIN(file_date) AS min_date
                FROM forecast_daily_uploads
                GROUP BY platform
                """
            ).fetchall()
        return {
            str(r[0]): {
                "file_count": int(r[1] or 0),
                "total_rows": int(r[2] or 0),
                "max_date": str(r[3]) if r[3] else "",
                "min_date": str(r[4]) if r[4] else "",
            }
            for r in rows
        }
    except Exception:
        _log.exception("pg_get_summary failed")
        return {}


def migrate_sqlite_daily_uploads_to_pg(sqlite_path: str | None = None) -> int:
    """One-shot copy SQLite Tier-3 blobs into PostgreSQL."""
    if not ops_pg_enabled():
        return 0
    import sqlite3
    from pathlib import Path

    from ..services.daily_store import _resolve_db_path

    path = Path(sqlite_path) if sqlite_path else _resolve_db_path()
    if not path.is_file():
        return 0
    conn_sql = sqlite3.connect(str(path))
    rows = conn_sql.execute(
        "SELECT platform, file_date, filename, rows, date_from, date_to, data_parquet "
        "FROM daily_uploads ORDER BY id"
    ).fetchall()
    conn_sql.close()
    n = 0
    for platform, file_date, filename, row_count, date_from, date_to, blob in rows:
        pg_save_daily_file(
            str(platform),
            str(filename),
            str(file_date),
            str(date_from) if date_from else None,
            str(date_to) if date_to else None,
            bytes(blob),
            int(row_count or 0),
        )
        n += 1
    _log.info("Migrated %d daily_uploads rows from SQLite to PostgreSQL", n)
    return n
