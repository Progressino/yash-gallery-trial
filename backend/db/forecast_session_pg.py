"""
Durable server-side storage for forecast / upload AppSession state.

Without this, sessions live only in process memory: any deploy, restart, or
second app instance means the browser cookie's session_id no longer matches
RAM and the user appears to "lose" ~1M rows until warm/GitHub recovers — or
worse, if auto-restore is paused or GitHub failed.

Set FORECAST_SESSION_DATABASE_URL (PostgreSQL URL, e.g. postgresql://user:pass@host:5432/db)
to enable. Optional alias: DATABASE_URL if FORECAST_SESSION_DATABASE_URL is unset.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import zipfile
from typing import Any, Optional

import pandas as pd

from ..services.helpers import _coerce_df_for_parquet

_log = logging.getLogger(__name__)

_BUNDLE_VERSION = 1

# All DataFrame fields on AppSession that we persist (parquet in zip).
_PARQUET_KEYS = (
    "sales_df",
    "mtr_df",
    "meesho_df",
    "myntra_df",
    "flipkart_df",
    "snapdeal_df",
    "inventory_df_variant",
    "inventory_df_parent",
    "daily_orders_df",
    "existing_po_df",
    "sku_status_lead_df",
    "transfer_df",
    "cogs_df",
)

_META_JSON_FIELDS = (
    "amazon_date_basis",
    "include_replacements",
    "daily_sales_sources",
    "daily_sales_rows",
    "load_warnings",
    "snapdeal_parse_info",
    "inventory_debug",
    "daily_restored",
    "pause_auto_data_restore",
)

_table_ready = False
_pending_persist_handles: dict[str, asyncio.TimerHandle] = {}


def pg_session_persist_enabled() -> bool:
    return bool(_connection_url())


def _connection_url() -> Optional[str]:
    u = (os.environ.get("FORECAST_SESSION_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    return u or None


def init_db() -> None:
    global _table_ready
    url = _connection_url()
    if not url:
        _table_ready = False
        return
    try:
        import psycopg
    except ImportError:
        _log.warning("psycopg not installed — forecast session DB disabled")
        _table_ready = False
        return
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_app_sessions (
                    session_id TEXT PRIMARY KEY,
                    bundle     BYTEA NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecast_app_sessions_updated "
                "ON forecast_app_sessions (updated_at DESC)"
            )
        _table_ready = True
        _log.info("forecast_app_sessions table ready (PostgreSQL)")
    except Exception:
        _log.exception("forecast session PostgreSQL init failed — sessions stay in-memory only")
        _table_ready = False


def _require_conn():
    import psycopg

    url = _connection_url()
    if not url or not _table_ready:
        return None
    return psycopg.connect(url, autocommit=True)


def session_bundle_bytes(sess) -> bytes:
    """Serialize AppSession to a zip blob (parquet + JSON)."""
    buf = io.BytesIO()
    meta: dict[str, Any] = {}
    for name in _META_JSON_FIELDS:
        v = getattr(sess, name, None)
        if name == "daily_sales_sources" and v is not None:
            meta[name] = list(v)
        elif name == "load_warnings" and v is not None:
            meta[name] = list(v)
        else:
            meta[name] = v

    manifest = {"bundle_version": _BUNDLE_VERSION, "meta": meta}

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, default=str).encode("utf-8"))
        sm = getattr(sess, "sku_mapping", None) or {}
        zf.writestr("sku_mapping.json", json.dumps(sm, default=str).encode("utf-8"))
        for key in _PARQUET_KEYS:
            df = getattr(sess, key, None)
            if df is None or not hasattr(df, "to_parquet"):
                df = pd.DataFrame()
            if df.empty:
                zf.writestr(f"{key}.parquet", b"")
                continue
            bio = io.BytesIO()
            _coerce_df_for_parquet(df).to_parquet(bio, index=False, engine="pyarrow")
            zf.writestr(f"{key}.parquet", bio.getvalue())
    return buf.getvalue()


def _hydrate_session_from_bundle(data: bytes):
    """Build AppSession from zip bytes. Returns None on failure."""
    from ..session import AppSession

    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            names = set(zf.namelist())
            raw_man = zf.read("manifest.json")
            manifest = json.loads(raw_man.decode("utf-8"))
            if manifest.get("bundle_version") != _BUNDLE_VERSION:
                _log.warning("Unknown session bundle version %s", manifest.get("bundle_version"))
            meta = manifest.get("meta") or {}

            sess = AppSession()
            for name in _META_JSON_FIELDS:
                if name not in meta:
                    continue
                val = meta[name]
                if name in ("daily_restored", "pause_auto_data_restore", "include_replacements"):
                    setattr(sess, name, bool(val))
                elif name == "daily_sales_rows":
                    try:
                        setattr(sess, name, int(val) if val is not None else 0)
                    except (TypeError, ValueError):
                        setattr(sess, name, 0)
                elif name in ("daily_sales_sources", "load_warnings"):
                    setattr(sess, name, list(val or []))
                else:
                    setattr(sess, name, val)

            if "sku_mapping.json" in names:
                sess.sku_mapping = json.loads(zf.read("sku_mapping.json").decode("utf-8"))
            else:
                sess.sku_mapping = {}

            for key in _PARQUET_KEYS:
                fn = f"{key}.parquet"
                if fn not in names:
                    setattr(sess, key, pd.DataFrame())
                    continue
                raw = zf.read(fn)
                if not raw:
                    setattr(sess, key, pd.DataFrame())
                    continue
                setattr(sess, key, pd.read_parquet(io.BytesIO(raw)))
            sess._quarterly_cache.clear()
            return sess
    except Exception:
        _log.exception("Failed to hydrate session from PostgreSQL bundle")
        return None


def load_session_from_pg(session_id: str):
    """Return AppSession if a row exists; otherwise None."""
    if not session_id or not _table_ready:
        return None
    conn = _require_conn()
    if conn is None:
        return None
    try:
        with conn:
            row = conn.execute(
                "SELECT bundle FROM forecast_app_sessions WHERE session_id = %s",
                (session_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return _hydrate_session_from_bundle(bytes(row[0]))
    except Exception:
        _log.exception(
            "load_session_from_pg failed for session_id=%s",
            (session_id[:8] + "…") if len(session_id) > 8 else session_id,
        )
        return None


def persist_session_bundle(session_id: str, sess) -> bool:
    """Upsert full session state. Returns False if disabled or error."""
    if not session_id or not _table_ready:
        return False
    conn = _require_conn()
    if conn is None:
        return False
    try:
        blob = session_bundle_bytes(sess)
        with conn:
            conn.execute(
                """
                INSERT INTO forecast_app_sessions (session_id, bundle, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (session_id) DO UPDATE
                SET bundle = EXCLUDED.bundle, updated_at = NOW()
                """,
                (session_id, blob),
            )
        setattr(sess, "_last_pg_save_ts", time.monotonic())
        return True
    except Exception:
        _log.exception(
            "persist_session_bundle failed for session_id=%s",
            (session_id[:8] + "…") if len(session_id) > 8 else session_id,
        )
        return False


def delete_session_bundle(session_id: str) -> None:
    if not session_id or not _table_ready:
        return
    conn = _require_conn()
    if conn is None:
        return
    try:
        with conn:
            conn.execute("DELETE FROM forecast_app_sessions WHERE session_id = %s", (session_id,))
    except Exception:
        _log.exception("delete_session_bundle failed")


def debounced_persist_session(session_id: str, sess, delay: float = 8.0) -> None:
    """Coalesce many writes into one PostgreSQL upsert after `delay` seconds."""

    if not pg_session_persist_enabled() or not session_id:
        return

    def _make_runner(sid: str, s):
        def _run():
            _pending_persist_handles.pop(sid, None)
            try:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, lambda: persist_session_bundle(sid, s))
            except Exception:
                persist_session_bundle(sid, s)

        return _run

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        persist_session_bundle(session_id, sess)
        return

    h = _pending_persist_handles.get(session_id)
    if h is not None:
        try:
            h.cancel()
        except Exception:
            pass
    _pending_persist_handles[session_id] = loop.call_later(delay, _make_runner(session_id, sess))


def persist_session_bundle_thread_safe(session_id: str, sess) -> None:
    """For BackgroundTasks / thread pool — synchronous upsert."""
    persist_session_bundle(session_id, sess)
