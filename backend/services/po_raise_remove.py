"""Remove rows from the PO raise ledger (session + durable SQLite)."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _norm_day(day: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(str(day).strip()[:10]).normalize())


def remove_raise_ledger_day(sess, raised_date: str) -> dict:
    """Drop all ledger lines for one calendar day from session and DB."""
    day = str(raised_date).strip()[:10]
    if not day:
        return {"ok": False, "message": "raised_date is required (YYYY-MM-DD).", "removed": 0}

    from ..db.po_raised_db import delete_raises_for_date

    removed_db = delete_raises_for_date(day)
    removed_sess = _remove_day_from_session_df(sess, day)
    sess._quarterly_cache.clear()
    return {
        "ok": True,
        "message": (
            f"Removed raise ledger for {day} "
            f"({max(removed_sess, removed_db):,} line(s) cleared)."
        ),
        "removed": max(removed_sess, removed_db),
        "raised_date": day,
    }


def remove_raise_ledger_skus(sess, raised_date: str, oms_skus: Iterable[str]) -> dict:
    """Drop specific SKU lines for one day (admin correction)."""
    day = str(raised_date).strip()[:10]
    skus = [str(s).strip() for s in oms_skus if str(s).strip()]
    if not day:
        return {"ok": False, "message": "raised_date is required.", "removed": 0}
    if not skus:
        return {"ok": False, "message": "No SKUs provided.", "removed": 0}

    from ..db.po_raised_db import delete_raises_for_date_skus

    removed_db = delete_raises_for_date_skus(day, skus)
    removed_sess = _remove_skus_from_session_df(sess, day, skus)
    sess._quarterly_cache.clear()
    return {
        "ok": True,
        "message": f"Removed {len(skus)} SKU line(s) for {day}.",
        "removed": max(removed_sess, removed_db),
        "raised_date": day,
    }


def clear_raise_ledger_all(sess) -> dict:
    """Wipe session ledger and durable SQLite store."""
    import pandas as pd

    from ..db.po_raised_db import clear_all

    n_sess = int(len(getattr(sess, "po_raise_ledger_df", pd.DataFrame())))
    n_db = clear_all()
    sess.po_raise_ledger_df = pd.DataFrame()
    sess._quarterly_cache.clear()
    return {
        "ok": True,
        "message": f"PO raise ledger cleared ({max(n_sess, n_db):,} row(s) removed).",
        "removed": max(n_sess, n_db),
    }


def _remove_day_from_session_df(sess, day: str) -> int:
    df = getattr(sess, "po_raise_ledger_df", None)
    if df is None or df.empty or "Raised_Date" not in df.columns:
        return 0
    target = _norm_day(day)
    dates = pd.to_datetime(df["Raised_Date"], errors="coerce").dt.normalize()
    mask = dates == target
    removed = int(mask.sum())
    if removed:
        sess.po_raise_ledger_df = df.loc[~mask].reset_index(drop=True)
    return removed


def _remove_skus_from_session_df(sess, day: str, oms_skus: list[str]) -> int:
    df = getattr(sess, "po_raise_ledger_df", None)
    if df is None or df.empty:
        return 0
    need = {"OMS_SKU", "Raised_Date"}
    if not need.issubset({str(c).strip() for c in df.columns}):
        return 0
    target = _norm_day(day)
    sku_set = {s.upper() for s in oms_skus}
    work = df.copy()
    work["OMS_SKU"] = work["OMS_SKU"].astype(str).str.strip()
    work["Raised_Date"] = pd.to_datetime(work["Raised_Date"], errors="coerce").dt.normalize()
    mask = (work["Raised_Date"] == target) & work["OMS_SKU"].str.upper().isin(sku_set)
    removed = int(mask.sum())
    if removed:
        sess.po_raise_ledger_df = df.loc[~mask].reset_index(drop=True)
    return removed
