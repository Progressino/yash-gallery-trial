"""
Indexed PostgreSQL tables for inventory lines and sales transactions.

Replaces parquet blobs in ``forecast_shared_snapshot`` when
``FORECAST_OPS_NORMALIZED=1`` (default on when ops PG is enabled).

Tables:
  forecast_inventory_snapshots + forecast_inventory_lines
  forecast_sales_transactions (platform-partitioned logically by ``platform`` column)
  forecast_sku_mapping
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

_PLATFORM_BULK_KEYS = {
    "mtr_df": "amazon",
    "myntra_df": "myntra",
    "meesho_df": "meesho",
    "flipkart_df": "flipkart",
    "snapdeal_df": "snapdeal",
    "sales_df": "unified",
}

_INVENTORY_KNOWN = {
    "OMS_SKU": "oms_sku",
    "OMS_Inventory": "oms_inventory",
    "Buffer_Stock": "buffer_stock",
    "Amazon_Inventory": "amazon_inventory",
    "Flipkart_Inventory": "flipkart_inventory",
    "Myntra_Other_Inventory": "myntra_other_inventory",
    "Meesho_Inventory": "meesho_inventory",
    "Manual_InTransit": "manual_intransit",
    "Not_In_Inventory_Qty": "not_in_inventory_qty",
    "FBA_InTransit": "fba_intransit",
    "Marketplace_Total": "marketplace_total",
    "Total_Inventory": "total_inventory",
}

_SALES_DB_COLUMNS = (
    "platform",
    "sku",
    "txn_date",
    "quantity",
    "transaction_type",
    "order_id",
    "line_key",
    "dsr_segment",
    "source_file",
    "units_effective",
    "extra",
)


def normalized_tables_enabled() -> bool:
    from .forecast_ops_pg import ops_pg_enabled

    if not ops_pg_enabled():
        return False
    raw = (os.environ.get("FORECAST_OPS_NORMALIZED") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _require_conn():
    from .forecast_ops_pg import _require_conn as _rc

    return _rc()


def ensure_tables(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_inventory_snapshots (
            id              BIGSERIAL PRIMARY KEY,
            snapshot_date   DATE,
            snapshot_label  TEXT,
            uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            debug           JSONB NOT NULL DEFAULT '{}'::jsonb,
            is_current      BOOLEAN NOT NULL DEFAULT TRUE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_inventory_lines (
            snapshot_id             BIGINT NOT NULL
                REFERENCES forecast_inventory_snapshots(id) ON DELETE CASCADE,
            oms_sku                 TEXT NOT NULL,
            oms_inventory           DOUBLE PRECISION NOT NULL DEFAULT 0,
            buffer_stock            DOUBLE PRECISION NOT NULL DEFAULT 0,
            amazon_inventory        DOUBLE PRECISION NOT NULL DEFAULT 0,
            flipkart_inventory      DOUBLE PRECISION NOT NULL DEFAULT 0,
            myntra_other_inventory  DOUBLE PRECISION NOT NULL DEFAULT 0,
            meesho_inventory        DOUBLE PRECISION NOT NULL DEFAULT 0,
            manual_intransit        DOUBLE PRECISION NOT NULL DEFAULT 0,
            not_in_inventory_qty    DOUBLE PRECISION NOT NULL DEFAULT 0,
            fba_intransit           DOUBLE PRECISION NOT NULL DEFAULT 0,
            marketplace_total       DOUBLE PRECISION NOT NULL DEFAULT 0,
            total_inventory         DOUBLE PRECISION NOT NULL DEFAULT 0,
            extra                   JSONB NOT NULL DEFAULT '{}'::jsonb,
            PRIMARY KEY (snapshot_id, oms_sku)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecast_inv_lines_sku "
        "ON forecast_inventory_lines (oms_sku)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecast_inv_lines_total "
        "ON forecast_inventory_lines (snapshot_id, total_inventory DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecast_inv_snapshots_current "
        "ON forecast_inventory_snapshots (uploaded_at DESC) "
        "WHERE is_current = TRUE"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_sales_transactions (
            id                BIGSERIAL PRIMARY KEY,
            platform          TEXT NOT NULL,
            sku               TEXT NOT NULL,
            txn_date          TIMESTAMPTZ NOT NULL,
            quantity          DOUBLE PRECISION NOT NULL DEFAULT 0,
            transaction_type  TEXT NOT NULL DEFAULT '',
            order_id          TEXT,
            line_key          TEXT,
            dsr_segment       TEXT,
            source_file       TEXT,
            units_effective   DOUBLE PRECISION,
            extra             JSONB NOT NULL DEFAULT '{}'::jsonb
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fst_platform_txn_date "
        "ON forecast_sales_transactions (platform, txn_date DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fst_platform_txn_date_asc "
        "ON forecast_sales_transactions (platform, txn_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fst_sku_txn_date "
        "ON forecast_sales_transactions (sku, txn_date DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fst_platform_line_key "
        "ON forecast_sales_transactions (platform, line_key) "
        "WHERE line_key IS NOT NULL AND line_key <> ''"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_sku_mapping (
            seller_key TEXT PRIMARY KEY,
            oms_sku    TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecast_sku_mapping_oms "
        "ON forecast_sku_mapping (oms_sku)"
    )


def _num(val) -> float:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _inventory_row_from_series(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {v: 0.0 for v in _INVENTORY_KNOWN.values() if v != "oms_sku"}
    extra: dict[str, Any] = {}
    for col, val in row.items():
        if col in _INVENTORY_KNOWN:
            if col == "OMS_SKU":
                out["oms_sku"] = str(val or "").strip().upper()
            else:
                out[_INVENTORY_KNOWN[col]] = _num(val)
        elif col not in ("Parent_SKU",):
            try:
                extra[str(col)] = _num(val) if pd.api.types.is_numeric_dtype(type(val)) else str(val)
            except Exception:
                extra[str(col)] = str(val)
    if not out.get("oms_sku"):
        out["oms_sku"] = str(row.get("OMS_SKU", "") or "").strip().upper()
    out["extra"] = json.dumps(extra)
    return out


def persist_inventory_dataframe(
    df: pd.DataFrame,
    *,
    snapshot_date: str | None = None,
    snapshot_label: str | None = None,
    debug: dict | None = None,
) -> int | None:
    if not normalized_tables_enabled() or df is None or df.empty:
        return None
    conn = _require_conn()
    if conn is None:
        return None
    rows = [_inventory_row_from_series(r) for _, r in df.iterrows() if str(r.get("OMS_SKU", "")).strip()]
    if not rows:
        return None
    try:
        with conn:
            conn.execute(
                "UPDATE forecast_inventory_snapshots SET is_current = FALSE WHERE is_current = TRUE"
            )
            snap = conn.execute(
                """
                INSERT INTO forecast_inventory_snapshots
                    (snapshot_date, snapshot_label, debug, is_current)
                VALUES (%s, %s, %s::jsonb, TRUE)
                RETURNING id
                """,
                (
                    snapshot_date or None,
                    snapshot_label or None,
                    json.dumps(debug or {}),
                ),
            ).fetchone()
            sid = int(snap[0])
            _copy_inventory_lines(conn, sid, rows)
        _log.info("Persisted inventory snapshot id=%s (%d SKUs) to PostgreSQL", sid, len(rows))
        return sid
    except Exception:
        _log.exception("persist_inventory_dataframe failed")
        return None


def _copy_inventory_lines(conn, snapshot_id: int, rows: list[dict]) -> None:
    with conn.cursor() as cur:
        with cur.copy(
            "COPY forecast_inventory_lines "
            "(snapshot_id, oms_sku, oms_inventory, buffer_stock, amazon_inventory, "
            "flipkart_inventory, myntra_other_inventory, meesho_inventory, "
            "manual_intransit, not_in_inventory_qty, fba_intransit, "
            "marketplace_total, total_inventory, extra) FROM STDIN"
        ) as copy:
            for r in rows:
                copy.write_row(
                    (
                        snapshot_id,
                        r["oms_sku"],
                        r.get("oms_inventory", 0),
                        r.get("buffer_stock", 0),
                        r.get("amazon_inventory", 0),
                        r.get("flipkart_inventory", 0),
                        r.get("myntra_other_inventory", 0),
                        r.get("meesho_inventory", 0),
                        r.get("manual_intransit", 0),
                        r.get("not_in_inventory_qty", 0),
                        r.get("fba_intransit", 0),
                        r.get("marketplace_total", 0),
                        r.get("total_inventory", 0),
                        r.get("extra", "{}"),
                    )
                )


def load_inventory_dataframe() -> pd.DataFrame | None:
    if not normalized_tables_enabled():
        return None
    conn = _require_conn()
    if conn is None:
        return None
    try:
        with conn:
            snap = conn.execute(
                """
                SELECT id FROM forecast_inventory_snapshots
                WHERE is_current = TRUE
                ORDER BY uploaded_at DESC
                LIMIT 1
                """
            ).fetchone()
            if not snap:
                return None
            sid = int(snap[0])
            cur = conn.execute(
                """
                SELECT oms_sku, oms_inventory, buffer_stock, amazon_inventory,
                       flipkart_inventory, myntra_other_inventory, meesho_inventory,
                       manual_intransit, not_in_inventory_qty, fba_intransit,
                       marketplace_total, total_inventory, extra
                FROM forecast_inventory_lines
                WHERE snapshot_id = %s
                """,
                (sid,),
            )
            rows = cur.fetchall()
        if not rows:
            return None
        records: list[dict] = []
        inv_rev = {v: k for k, v in _INVENTORY_KNOWN.items() if k != "OMS_SKU"}
        for r in rows:
            rec = {"OMS_SKU": r[0]}
            for i, col in enumerate(
                (
                    "oms_inventory",
                    "buffer_stock",
                    "amazon_inventory",
                    "flipkart_inventory",
                    "myntra_other_inventory",
                    "meesho_inventory",
                    "manual_intransit",
                    "not_in_inventory_qty",
                    "fba_intransit",
                    "marketplace_total",
                    "total_inventory",
                ),
                start=1,
            ):
                if r[i]:
                    rec[inv_rev[col]] = r[i]
            extra = r[12]
            if extra:
                if isinstance(extra, str):
                    extra = json.loads(extra)
                if isinstance(extra, dict):
                    rec.update(extra)
            records.append(rec)
        return pd.DataFrame(records)
    except Exception:
        _log.exception("load_inventory_dataframe failed")
        return None


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return str(lower[n.lower()])
    return None


def _sales_rows_from_dataframe(
    df: pd.DataFrame,
    platform: str,
    *,
    source_file: str | None = None,
) -> Iterator[tuple]:
    if df.empty:
        return
    sku_c = _first_col(df, ("Sku", "SKU", "sku"))
    date_c = _first_col(df, ("TxnDate", "Date", "Order Date", "order_date"))
    qty_c = _first_col(df, ("Quantity", "quantity", "Units_Effective", "units_effective"))
    tt_c = _first_col(df, ("Transaction Type", "Transaction_Type", "transaction_type"))
    oid_c = _first_col(df, ("OrderId", "Order_Id", "Order ID", "order_id"))
    lk_c = _first_col(df, ("LineKey", "line_key", "Line_Key"))
    dsr_c = _first_col(df, ("DSR_Segment", "dsr_segment"))
    src_c = _first_col(df, ("Source", "source"))
    ue_c = _first_col(df, ("Units_Effective", "units_effective"))
    known = {sku_c, date_c, qty_c, tt_c, oid_c, lk_c, dsr_c, src_c, ue_c} - {None}

    for _, row in df.iterrows():
        sku = str(row.get(sku_c, "") if sku_c else "").strip().upper()
        if not sku:
            continue
        txn_raw = row.get(date_c) if date_c else None
        try:
            txn = pd.Timestamp(txn_raw)
            if pd.isna(txn):
                continue
            if txn.tzinfo is None:
                txn = txn.tz_localize("UTC")
            else:
                txn = txn.tz_convert("UTC")
        except Exception:
            continue
        qty = _num(row.get(qty_c) if qty_c else 0)
        tt = str(row.get(tt_c, "") if tt_c else "").strip()
        oid = str(row.get(oid_c, "") if oid_c else "").strip() or None
        lk = str(row.get(lk_c, "") if lk_c else "").strip() or None
        dsr = str(row.get(dsr_c, "") if dsr_c else "").strip() or None
        ue = _num(row.get(ue_c)) if ue_c and row.get(ue_c) is not None else None
        extra = {}
        for c in df.columns:
            if c in known:
                continue
            try:
                v = row[c]
                if pd.isna(v):
                    continue
                extra[str(c)] = v.item() if hasattr(v, "item") else v
            except Exception:
                extra[str(c)] = str(row[c])
        yield (
            platform,
            sku,
            txn.to_pydatetime(),
            qty,
            tt,
            oid,
            lk,
            dsr,
            source_file,
            ue,
            json.dumps(extra, default=str),
        )


def persist_platform_sales_dataframe(
    platform: str,
    df: pd.DataFrame,
    *,
    source_file: str | None = None,
) -> int:
    if not normalized_tables_enabled() or df is None or df.empty:
        return 0
    conn = _require_conn()
    if conn is None:
        return 0
    batch: list[tuple] = []
    batch_size = int(os.environ.get("FORECAST_OPS_COPY_BATCH", "5000"))
    total = 0
    try:
        with conn:
            conn.execute(
                "DELETE FROM forecast_sales_transactions WHERE platform = %s",
                (platform,),
            )
            with conn.cursor() as cur:
                with cur.copy(
                    "COPY forecast_sales_transactions "
                    "(platform, sku, txn_date, quantity, transaction_type, order_id, "
                    "line_key, dsr_segment, source_file, units_effective, extra) FROM STDIN"
                ) as copy:
                    for rec in _sales_rows_from_dataframe(df, platform, source_file=source_file):
                        copy.write_row(rec)
                        total += 1
                        if total % batch_size == 0:
                            pass
        _log.info("Persisted %d %s sales rows to PostgreSQL", total, platform)
        return total
    except Exception:
        _log.exception("persist_platform_sales_dataframe failed platform=%s", platform)
        return 0


def load_platform_sales_dataframe(
    platform: str,
    *,
    months: int | None = None,
) -> pd.DataFrame | None:
    if not normalized_tables_enabled():
        return None
    conn = _require_conn()
    if conn is None:
        return None
    try:
        with conn:
            if months is None:
                rows = conn.execute(
                    """
                    SELECT sku, txn_date, quantity, transaction_type, order_id,
                           line_key, dsr_segment, source_file, units_effective, extra
                    FROM forecast_sales_transactions
                    WHERE platform = %s
                    ORDER BY txn_date
                    """,
                    (platform,),
                ).fetchall()
            else:
                cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)
                rows = conn.execute(
                    """
                    SELECT sku, txn_date, quantity, transaction_type, order_id,
                           line_key, dsr_segment, source_file, units_effective, extra
                    FROM forecast_sales_transactions
                    WHERE platform = %s AND txn_date >= %s
                    ORDER BY txn_date
                    """,
                    (platform, cutoff),
                ).fetchall()
        if not rows:
            return None
        records: list[dict] = []
        for r in rows:
            rec = {
                "Sku": r[0],
                "TxnDate": pd.Timestamp(r[1]).tz_localize(None),
                "Quantity": r[2],
                "Transaction Type": r[3] or "",
            }
            if r[4]:
                rec["OrderId"] = r[4]
            if r[5]:
                rec["LineKey"] = r[5]
            if r[6]:
                rec["DSR_Segment"] = r[6]
            if r[7]:
                rec["Source"] = r[7] if platform == "unified" else platform.title()
            if r[8] is not None:
                rec["Units_Effective"] = r[8]
            extra = r[9]
            if extra:
                if isinstance(extra, str):
                    extra = json.loads(extra)
                if isinstance(extra, dict):
                    rec.update(extra)
            if platform != "unified" and "Source" not in rec:
                rec["Source"] = {
                    "amazon": "Amazon",
                    "myntra": "Myntra",
                    "meesho": "Meesho",
                    "flipkart": "Flipkart",
                    "snapdeal": "Snapdeal",
                }.get(platform, platform.title())
            records.append(rec)
        return pd.DataFrame(records)
    except Exception:
        _log.exception("load_platform_sales_dataframe failed platform=%s", platform)
        return None


def persist_sku_mapping(mapping: dict[str, str]) -> int:
    if not normalized_tables_enabled() or not mapping:
        return 0
    conn = _require_conn()
    if conn is None:
        return 0
    try:
        with conn:
            conn.execute("DELETE FROM forecast_sku_mapping")
            with conn.cursor() as cur:
                with cur.copy(
                    "COPY forecast_sku_mapping (seller_key, oms_sku) FROM STDIN"
                ) as copy:
                    for k, v in mapping.items():
                        sk = str(k or "").strip()
                        ov = str(v or "").strip()
                        if sk and ov:
                            copy.write_row((sk, ov))
        return len(mapping)
    except Exception:
        _log.exception("persist_sku_mapping failed")
        return 0


def load_sku_mapping() -> dict[str, str]:
    if not normalized_tables_enabled():
        return {}
    conn = _require_conn()
    if conn is None:
        return {}
    try:
        with conn:
            rows = conn.execute(
                "SELECT seller_key, oms_sku FROM forecast_sku_mapping"
            ).fetchall()
        return {str(r[0]): str(r[1]) for r in rows}
    except Exception:
        _log.exception("load_sku_mapping failed")
        return {}


def persist_warm_cache_tables(cache_dict: dict) -> dict[str, int]:
    """Write inventory + platform sales + sku mapping from warm-cache dict."""
    stats: dict[str, int] = {}
    if not normalized_tables_enabled():
        return stats
    inv = cache_dict.get("inventory_df_variant")
    if inv is not None and hasattr(inv, "empty") and not inv.empty:
        meta = cache_dict.get("inventory_session_meta") or {}
        sid = persist_inventory_dataframe(
            inv,
            snapshot_date=str(meta.get("snapshot_date") or "")[:10] or None,
            snapshot_label=str(meta.get("snapshot_date_label") or "") or None,
            debug=cache_dict.get("inventory_debug") if isinstance(cache_dict.get("inventory_debug"), dict) else {},
        )
        if sid:
            stats["inventory_lines"] = len(inv)
    sm = cache_dict.get("sku_mapping")
    if isinstance(sm, dict) and sm:
        stats["sku_mapping"] = persist_sku_mapping(sm)
    for key, plat in _PLATFORM_BULK_KEYS.items():
        df = cache_dict.get(key)
        if df is not None and hasattr(df, "empty") and not df.empty:
            stats[f"sales_{plat}"] = persist_platform_sales_dataframe(plat, df)
    return stats


def load_warm_cache_tables() -> dict | None:
    """Rebuild warm-cache dict from indexed PostgreSQL tables."""
    if not normalized_tables_enabled():
        return None
    out: dict = {}
    sm = load_sku_mapping()
    if sm:
        out["sku_mapping"] = sm
    inv = load_inventory_dataframe()
    if inv is not None and not inv.empty:
        out["inventory_df_variant"] = inv
    loaded_any = False
    for key, plat in _PLATFORM_BULK_KEYS.items():
        df = load_platform_sales_dataframe(plat)
        if df is not None and not df.empty:
            if key == "mtr_df":
                df = df.rename(
                    columns={
                        "Sku": "SKU",
                        "TxnDate": "Date",
                        "Transaction Type": "Transaction_Type",
                    },
                    errors="ignore",
                )
            out[key] = df
            loaded_any = True
    if not sm and (inv is None or inv.empty) and not loaded_any:
        return None
    return out


def tables_status() -> dict[str, Any]:
    if not normalized_tables_enabled():
        return {"enabled": False}
    conn = _require_conn()
    if conn is None:
        return {"enabled": False}
    try:
        with conn:
            inv_snap = conn.execute(
                "SELECT COUNT(*) FROM forecast_inventory_snapshots WHERE is_current = TRUE"
            ).fetchone()
            inv_lines = conn.execute(
                """
                SELECT COUNT(*) FROM forecast_inventory_lines l
                JOIN forecast_inventory_snapshots s ON s.id = l.snapshot_id
                WHERE s.is_current = TRUE
                """
            ).fetchone()
            sku_n = conn.execute("SELECT COUNT(*) FROM forecast_sku_mapping").fetchone()
            sales = conn.execute(
                """
                SELECT platform, COUNT(*) FROM forecast_sales_transactions
                GROUP BY platform ORDER BY platform
                """
            ).fetchall()
        return {
            "enabled": True,
            "inventory_snapshots": int(inv_snap[0] or 0),
            "inventory_lines": int(inv_lines[0] or 0),
            "sku_mapping": int(sku_n[0] or 0),
            "sales_by_platform": {str(r[0]): int(r[1]) for r in sales},
        }
    except Exception:
        _log.exception("tables_status failed")
        return {"enabled": True, "error": True}
