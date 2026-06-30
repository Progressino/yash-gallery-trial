"""
Materialized SKU sales inputs for fast PO / forecast (30d / 90d / 180d rollups + daily grain).

PO calculate reads ``forecast_sku_sales_daily`` (~SKU×day rows) instead of scanning 1M+
line-level transactions. Rollup tables (``forecast_sku_sales_30d`` etc.) are refreshed
from the daily table for dashboards and ``SELECT * FROM sku_sales_90d`` style queries.

Refresh: after upload / warm-cache persist, and hourly on the server.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

ROLLUP_WINDOWS = (30, 90, 180)
_DAILY_LOOKBACK_DAYS = 400
_MEM_LOCK = threading.Lock()
_MEM: dict[str, Any] = {}  # daily_df, sales_through, refreshed_at, row_counts

_ROLLUP_DDL = """
    oms_sku              TEXT PRIMARY KEY,
    sold_units           DOUBLE PRECISION NOT NULL DEFAULT 0,
    return_units         DOUBLE PRECISION NOT NULL DEFAULT 0,
    net_units            DOUBLE PRECISION NOT NULL DEFAULT 0,
    ship_units_150d      DOUBLE PRECISION NOT NULL DEFAULT 0,
    first_active         DATE,
    last_active          DATE,
    distinct_active_days INTEGER NOT NULL DEFAULT 0,
    ly_sold_units        DOUBLE PRECISION NOT NULL DEFAULT 0,
    ly_net_units         DOUBLE PRECISION NOT NULL DEFAULT 0,
    sales_through        DATE NOT NULL,
    refreshed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
"""


def materializations_enabled() -> bool:
    raw = (os.environ.get("FORECAST_SKU_ROLLUPS") or "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    try:
        from .forecast_ops_tables import normalized_tables_enabled

        if normalized_tables_enabled():
            return True
    except Exception:
        pass
    # Local / in-memory path when unified sales exist but PG normalized tables are off.
    return raw in ("1", "true", "yes", "on", "memory")


def ensure_materialization_tables(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_sku_sales_daily (
            oms_sku       TEXT NOT NULL,
            txn_date      DATE NOT NULL,
            sold_units    DOUBLE PRECISION NOT NULL DEFAULT 0,
            return_units  DOUBLE PRECISION NOT NULL DEFAULT 0,
            net_units     DOUBLE PRECISION NOT NULL DEFAULT 0,
            PRIMARY KEY (oms_sku, txn_date)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fssd_txn_date "
        "ON forecast_sku_sales_daily (txn_date DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fssd_sku_date "
        "ON forecast_sku_sales_daily (oms_sku, txn_date DESC)"
    )
    for days in ROLLUP_WINDOWS:
        conn.execute(f"CREATE TABLE IF NOT EXISTS forecast_sku_sales_{days}d ({_ROLLUP_DDL})")


def _require_conn():
    from .forecast_ops_pg import _require_conn as _rc

    return _rc()


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return str(lower[n.lower()])
    return None


def sales_df_to_daily(sales_df: pd.DataFrame) -> tuple[pd.DataFrame, date | None]:
    """Aggregate line-level unified sales to SKU × calendar day."""
    if sales_df is None or sales_df.empty:
        return pd.DataFrame(), None
    sku_c = _first_col(sales_df, ("Sku", "SKU", "sku", "OMS_SKU"))
    date_c = _first_col(sales_df, ("TxnDate", "Date", "Order Date"))
    qty_c = _first_col(sales_df, ("Quantity", "quantity"))
    tt_c = _first_col(sales_df, ("Transaction Type", "Transaction_Type", "transaction_type"))
    ue_c = _first_col(sales_df, ("Units_Effective", "units_effective"))
    if not sku_c or not date_c:
        return pd.DataFrame(), None

    df = sales_df[[sku_c, date_c] + [c for c in (qty_c, tt_c, ue_c) if c]].copy()
    df = df.rename(columns={sku_c: "oms_sku", date_c: "txn_date"})
    df["oms_sku"] = df["oms_sku"].astype(str).str.strip().str.upper()
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce").dt.normalize()
    df = df[df["oms_sku"].str.len() > 0].dropna(subset=["txn_date"])
    if df.empty:
        return pd.DataFrame(), None

    qty = pd.to_numeric(df[qty_c], errors="coerce").fillna(0) if qty_c else pd.Series(0.0, index=df.index)
    tt = df[tt_c].astype(str).str.strip().str.lower() if tt_c else pd.Series("shipment", index=df.index)
    is_refund = tt.isin(("refund", "cancel"))
    if ue_c:
        net = pd.to_numeric(df[ue_c], errors="coerce").fillna(0.0)
    else:
        net = np.where(is_refund, -qty.abs(), qty)

    df["_sold"] = np.where(~is_refund, qty, 0.0)
    df["_ret"] = np.where(is_refund, qty.abs(), 0.0)
    df["_net"] = net

    daily = (
        df.groupby(["oms_sku", "txn_date"], as_index=False)
        .agg(
            sold_units=("_sold", "sum"),
            return_units=("_ret", "sum"),
            net_units=("_net", "sum"),
        )
    )
    sales_through = pd.Timestamp(daily["txn_date"].max()).date()
    return daily, sales_through


def daily_to_engine_sales_df(daily: pd.DataFrame) -> pd.DataFrame:
    """Expand SKU-day aggregates into minimal line-level frames for ``calculate_po_base``."""
    if daily is None or daily.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    ship = daily[pd.to_numeric(daily["sold_units"], errors="coerce").fillna(0) > 0].copy()
    if not ship.empty:
        parts.append(
            pd.DataFrame(
                {
                    "Sku": ship["oms_sku"].astype(str),
                    "TxnDate": pd.to_datetime(ship["txn_date"]),
                    "Transaction Type": "Shipment",
                    "Quantity": pd.to_numeric(ship["sold_units"], errors="coerce").fillna(0),
                    "Units_Effective": pd.to_numeric(ship["sold_units"], errors="coerce").fillna(0),
                }
            )
        )
    ret = daily[pd.to_numeric(daily["return_units"], errors="coerce").fillna(0) > 0].copy()
    if not ret.empty:
        rq = pd.to_numeric(ret["return_units"], errors="coerce").fillna(0)
        parts.append(
            pd.DataFrame(
                {
                    "Sku": ret["oms_sku"].astype(str),
                    "TxnDate": pd.to_datetime(ret["txn_date"]),
                    "Transaction Type": "Refund",
                    "Quantity": rq,
                    "Units_Effective": -rq,
                }
            )
        )
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["Sku", "TxnDate"]).reset_index(drop=True)


def _rollup_from_daily(
    daily: pd.DataFrame,
    *,
    window_days: int,
    sales_through: date,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    end = pd.Timestamp(sales_through).normalize()
    start = end - pd.Timedelta(days=int(window_days) - 1)
    ly_end = end - pd.Timedelta(days=365)
    ly_start = ly_end - pd.Timedelta(days=int(window_days) - 1)
    ship150_start = end - pd.Timedelta(days=149)

    d = daily.copy()
    d["txn_date"] = pd.to_datetime(d["txn_date"]).dt.normalize()
    win = d[(d["txn_date"] >= start) & (d["txn_date"] <= end)]
    ly = d[(d["txn_date"] >= ly_start) & (d["txn_date"] <= ly_end)]
    ship150 = d[(d["txn_date"] >= ship150_start) & (d["txn_date"] <= end)]

    base = (
        win.groupby("oms_sku", as_index=False)
        .agg(
            sold_units=("sold_units", "sum"),
            return_units=("return_units", "sum"),
            net_units=("net_units", "sum"),
        )
    )
    active = win[pd.to_numeric(win["sold_units"], errors="coerce").fillna(0) > 0]
    if not active.empty:
        span = (
            active.groupby("oms_sku", as_index=False)
            .agg(
                first_active=("txn_date", "min"),
                last_active=("txn_date", "max"),
                distinct_active_days=("txn_date", "nunique"),
            )
        )
        base = base.merge(span, on="oms_sku", how="left")
    else:
        base["first_active"] = pd.NaT
        base["last_active"] = pd.NaT
        base["distinct_active_days"] = 0

    if not ship150.empty:
        s150 = (
            ship150.groupby("oms_sku", as_index=False)["sold_units"]
            .sum()
            .rename(columns={"sold_units": "ship_units_150d"})
        )
        base = base.merge(s150, on="oms_sku", how="left")
    else:
        base["ship_units_150d"] = 0.0

    if not ly.empty:
        ly_agg = (
            ly.groupby("oms_sku", as_index=False)
            .agg(ly_sold_units=("sold_units", "sum"), ly_net_units=("net_units", "sum"))
        )
        base = base.merge(ly_agg, on="oms_sku", how="left")
    else:
        base["ly_sold_units"] = 0.0
        base["ly_net_units"] = 0.0

    base["sales_through"] = sales_through
    base["refreshed_at"] = datetime.now(timezone.utc)
    for c in ("sold_units", "return_units", "net_units", "ship_units_150d", "ly_sold_units", "ly_net_units"):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    if "distinct_active_days" in base.columns:
        base["distinct_active_days"] = pd.to_numeric(base["distinct_active_days"], errors="coerce").fillna(0).astype(int)
    return base


def _store_memory(daily: pd.DataFrame, sales_through: date | None, rollups: dict[int, pd.DataFrame]) -> None:
    with _MEM_LOCK:
        _MEM.clear()
        _MEM["daily_df"] = daily.copy()
        _MEM["sales_through"] = sales_through
        _MEM["refreshed_at"] = datetime.now(timezone.utc)
        _MEM["rollups"] = {k: v.copy() for k, v in rollups.items()}
        _MEM["daily_rows"] = len(daily)
        for k, v in rollups.items():
            _MEM[f"rollup_{k}"] = len(v)


def _persist_daily_and_rollups(conn, daily: pd.DataFrame, sales_through: date) -> dict[str, int]:
    stats: dict[str, int] = {}
    with conn:
        conn.execute("DELETE FROM forecast_sku_sales_daily")
        if not daily.empty:
            with conn.cursor() as cur:
                with cur.copy(
                    "COPY forecast_sku_sales_daily "
                    "(oms_sku, txn_date, sold_units, return_units, net_units) FROM STDIN"
                ) as copy:
                    for row in daily.itertuples(index=False):
                        copy.write_row(
                            (
                                str(row.oms_sku),
                                pd.Timestamp(row.txn_date).date(),
                                float(row.sold_units),
                                float(row.return_units),
                                float(row.net_units),
                            )
                        )
        stats["daily_rows"] = len(daily)
        for window in ROLLUP_WINDOWS:
            roll = _rollup_from_daily(daily, window_days=window, sales_through=sales_through)
            conn.execute(f"DELETE FROM forecast_sku_sales_{window}d")
            stats[f"sku_sales_{window}d"] = len(roll)
            if roll.empty:
                continue
            cols = (
                "oms_sku, sold_units, return_units, net_units, ship_units_150d, "
                "first_active, last_active, distinct_active_days, "
                "ly_sold_units, ly_net_units, sales_through, refreshed_at"
            )
            with conn.cursor() as cur:
                with cur.copy(
                    f"COPY forecast_sku_sales_{window}d ({cols}) FROM STDIN"
                ) as copy:
                    for row in roll.itertuples(index=False):
                        copy.write_row(
                            (
                                str(row.oms_sku),
                                float(row.sold_units),
                                float(row.return_units),
                                float(row.net_units),
                                float(row.ship_units_150d),
                                pd.Timestamp(row.first_active).date() if pd.notna(getattr(row, "first_active", None)) else None,
                                pd.Timestamp(row.last_active).date() if pd.notna(getattr(row, "last_active", None)) else None,
                                int(row.distinct_active_days),
                                float(row.ly_sold_units),
                                float(row.ly_net_units),
                                sales_through,
                                datetime.now(timezone.utc),
                            )
                        )
    return stats


def refresh_from_sales_df(sales_df: pd.DataFrame) -> dict[str, int]:
    """Rebuild daily + rollup tables from unified ``sales_df``."""
    if not materializations_enabled():
        return {}
    daily, sales_through = sales_df_to_daily(sales_df)
    if daily.empty or sales_through is None:
        return {}
    rollups = {
        w: _rollup_from_daily(daily, window_days=w, sales_through=sales_through) for w in ROLLUP_WINDOWS
    }
    _store_memory(daily, sales_through, rollups)
    with _MEM_LOCK:
        _MEM["sales_revision"] = None
    conn = _require_conn()
    if conn is None:
        _log.info(
            "SKU sales materialized in memory only (%d daily rows, through %s)",
            len(daily),
            sales_through,
        )
        return {"daily_rows": len(daily), **{f"sku_sales_{w}d": len(rollups[w]) for w in ROLLUP_WINDOWS}}
    try:
        ensure_materialization_tables(conn)
        stats = _persist_daily_and_rollups(conn, daily, sales_through)
        _log.info("SKU sales materializations refreshed: %s (through %s)", stats, sales_through)
        return stats
    except Exception:
        _log.exception("refresh_from_sales_df failed")
        return {}


def refresh_from_pg_transactions() -> dict[str, int]:
    """Rebuild materializations from ``forecast_sales_transactions`` (prefer unified platform)."""
    if not materializations_enabled():
        return {}
    conn = _require_conn()
    if conn is None:
        return {}
    try:
        ensure_materialization_tables(conn)
        with conn:
            unified_n = conn.execute(
                "SELECT COUNT(*) FROM forecast_sales_transactions WHERE platform = %s",
                ("unified",),
            ).fetchone()[0]
            if int(unified_n or 0) > 0:
                where = "WHERE platform = 'unified'"
            else:
                where = "WHERE platform <> 'unified'"
            rows = conn.execute(
                f"""
                SELECT
                    UPPER(TRIM(sku)) AS oms_sku,
                    (txn_date AT TIME ZONE 'UTC')::date AS txn_date,
                    SUM(CASE WHEN LOWER(TRIM(transaction_type)) IN ('refund', 'cancel')
                        THEN 0 ELSE quantity END) AS sold_units,
                    SUM(CASE WHEN LOWER(TRIM(transaction_type)) IN ('refund', 'cancel')
                        THEN ABS(quantity) ELSE 0 END) AS return_units,
                    SUM(COALESCE(
                        units_effective,
                        CASE WHEN LOWER(TRIM(transaction_type)) IN ('refund', 'cancel')
                            THEN -ABS(quantity) ELSE quantity END
                    )) AS net_units
                FROM forecast_sales_transactions
                {where}
                  AND txn_date >= NOW() - INTERVAL '{int(_DAILY_LOOKBACK_DAYS)} days'
                GROUP BY 1, 2
                """
            ).fetchall()
        if not rows:
            return {}
        daily = pd.DataFrame(
            rows,
            columns=["oms_sku", "txn_date", "sold_units", "return_units", "net_units"],
        )
        sales_through = pd.Timestamp(daily["txn_date"].max()).date()
        rollups = {
            w: _rollup_from_daily(daily, window_days=w, sales_through=sales_through) for w in ROLLUP_WINDOWS
        }
        _store_memory(daily, sales_through, rollups)
        return _persist_daily_and_rollups(conn, daily, sales_through)
    except Exception:
        _log.exception("refresh_from_pg_transactions failed")
        return {}


def _load_daily_from_pg(*, lookback_days: int, planning_date: str | None) -> pd.DataFrame | None:
    conn = _require_conn()
    if conn is None:
        return None
    try:
        with conn:
            meta = conn.execute(
                "SELECT MAX(sales_through) FROM forecast_sku_sales_90d"
            ).fetchone()
            if not meta or meta[0] is None:
                return None
            sales_through = meta[0]
            end = pd.Timestamp(planning_date).normalize() if planning_date else pd.Timestamp(sales_through)
            start = end - pd.Timedelta(days=int(lookback_days))
            rows = conn.execute(
                """
                SELECT oms_sku, txn_date, sold_units, return_units, net_units
                FROM forecast_sku_sales_daily
                WHERE txn_date >= %s AND txn_date <= %s
                ORDER BY txn_date, oms_sku
                """,
                (start.date(), end.date()),
            ).fetchall()
        if not rows:
            return None
        return pd.DataFrame(
            rows,
            columns=["oms_sku", "txn_date", "sold_units", "return_units", "net_units"],
        )
    except Exception:
        _log.exception("_load_daily_from_pg failed")
        return None


def _load_daily_from_memory(*, lookback_days: int, planning_date: str | None) -> pd.DataFrame | None:
    with _MEM_LOCK:
        daily = _MEM.get("daily_df")
        sales_through = _MEM.get("sales_through")
    if daily is None or daily.empty or sales_through is None:
        return None
    end = pd.Timestamp(planning_date).normalize() if planning_date else pd.Timestamp(sales_through)
    start = end - pd.Timedelta(days=int(lookback_days))
    d = daily.copy()
    d["txn_date"] = pd.to_datetime(d["txn_date"]).dt.normalize()
    out = d[(d["txn_date"] >= start) & (d["txn_date"] <= end)]
    return out if not out.empty else None


def load_rollup_table(window_days: int) -> pd.DataFrame | None:
    """Load ``forecast_sku_sales_{window}d`` (or in-memory equivalent)."""
    if window_days not in ROLLUP_WINDOWS:
        return None
    conn = _require_conn()
    if conn is not None:
        try:
            with conn:
                rows = conn.execute(f"SELECT * FROM forecast_sku_sales_{window_days}d").fetchall()
            if rows:
                cols = [
                    "oms_sku", "sold_units", "return_units", "net_units", "ship_units_150d",
                    "first_active", "last_active", "distinct_active_days",
                    "ly_sold_units", "ly_net_units", "sales_through", "refreshed_at",
                ]
                return pd.DataFrame(rows, columns=cols)
        except Exception:
            _log.exception("load_rollup_table pg failed window=%s", window_days)
    with _MEM_LOCK:
        roll = (_MEM.get("rollups") or {}).get(window_days)
    return roll.copy() if roll is not None and not roll.empty else None


def load_po_sales_df(
    sess,
    *,
    period_days: int,
    planning_date: str | None,
    use_seasonality: bool,
    use_ly_fallback: bool,
) -> pd.DataFrame | None:
    """
    Fast PO path: daily materialization → engine-compatible ``sales_df``.
    Returns None when stale / disabled so caller falls back to raw history.
    """
    if not materializations_enabled():
        return None

    sales_df = getattr(sess, "sales_df", None)
    rev = int(getattr(sess, "sales_data_revision", 0) or 0)
    with _MEM_LOCK:
        mem_rev = _MEM.get("sales_revision")
        mem_through = _MEM.get("sales_through")
    if sales_df is not None and not sales_df.empty:
        try:
            cur_through = pd.to_datetime(sales_df["TxnDate"], errors="coerce").max()
            cur_through = pd.Timestamp(cur_through).date() if pd.notna(cur_through) else None
        except Exception:
            cur_through = None
        stale = mem_rev != rev or (cur_through and mem_through and cur_through > mem_through)
        if stale:
            refresh_from_sales_df(sales_df)
            with _MEM_LOCK:
                _MEM["sales_revision"] = rev
        elif mem_rev is None:
            with _MEM_LOCK:
                _MEM["sales_revision"] = rev

    lookback = int(max(30, period_days) + 365 + period_days + 7)
    if use_seasonality or use_ly_fallback:
        from ..services.po_ads_horizon import po_ads_history_horizon_days

        lookback = max(
            lookback,
            po_ads_history_horizon_days(
                period_days,
                use_seasonality=use_seasonality,
                use_ly_fallback=use_ly_fallback,
            ),
        )

    daily = _load_daily_from_pg(lookback_days=lookback, planning_date=planning_date)
    if daily is None or daily.empty:
        daily = _load_daily_from_memory(lookback_days=lookback, planning_date=planning_date)
    if daily is None or daily.empty:
        return None

    engine_df = daily_to_engine_sales_df(daily)
    if engine_df.empty:
        return None
    _log.info(
        "PO sales from materialized daily grain: %d sku-days → %d engine rows (lookback=%dd)",
        len(daily),
        len(engine_df),
        lookback,
    )
    return engine_df


def materialization_status() -> dict[str, Any]:
    out: dict[str, Any] = {"enabled": materializations_enabled()}
    with _MEM_LOCK:
        out["memory"] = {
            "daily_rows": _MEM.get("daily_rows", 0),
            "sales_through": str(_MEM.get("sales_through") or ""),
            "refreshed_at": (_MEM.get("refreshed_at") or "").isoformat()
            if _MEM.get("refreshed_at")
            else "",
        }
        for w in ROLLUP_WINDOWS:
            out["memory"][f"sku_sales_{w}d"] = _MEM.get(f"rollup_{w}", 0)
    conn = _require_conn()
    if conn is None:
        return out
    try:
        with conn:
            d = conn.execute("SELECT COUNT(*), MAX(txn_date) FROM forecast_sku_sales_daily").fetchone()
            out["postgres"] = {"daily_rows": int(d[0] or 0), "max_txn_date": str(d[1] or "")}
            for w in ROLLUP_WINDOWS:
                r = conn.execute(
                    f"SELECT COUNT(*), MAX(sales_through) FROM forecast_sku_sales_{w}d"
                ).fetchone()
                out["postgres"][f"sku_sales_{w}d"] = int(r[0] or 0)
                out["postgres"][f"sales_through_{w}d"] = str(r[1] or "")
    except Exception:
        _log.exception("materialization_status failed")
    return out


def refresh_hourly_from_server() -> dict[str, int]:
    """Hourly job: PG transactions first, else warm-cache unified sales."""
    stats = refresh_from_pg_transactions()
    if stats:
        return stats
    try:
        import backend.main as _main

        sales = (_main._warm_cache or {}).get("sales_df")
        if sales is not None and not sales.empty:
            return refresh_from_sales_df(sales)
    except Exception:
        _log.exception("refresh_hourly_from_server warm cache failed")
    return {}
