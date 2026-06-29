"""Remove rows from the PO raise ledger (session + durable SQLite)."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _norm_day(day: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(str(day).strip()[:10]).normalize())


def invalidate_po_calculate_result(sess) -> None:
    """Drop cached PO table — pipeline columns are stale after ledger edits."""
    import pandas as pd

    if getattr(sess, "po_calculate_status", "idle") == "running":
        return
    sess.po_calculate_status = "idle"
    sess.po_calculate_message = ""
    sess.po_calculate_progress = 0
    sess.po_calculate_result = {}
    sess.po_calculate_result_df = pd.DataFrame()
    sess.po_calculate_existing_po_generation = -1
    sess._quarterly_cache.clear()
    try:
        from .po_calculate_jobs import clear_po_job

        sid = getattr(sess, "_persist_sid", None)
        if sid:
            clear_po_job(sid)
            try:
                from .po_result_spill import clear_spill

                clear_spill(sid)
            except Exception:
                pass
    except Exception:
        pass


_RECALC_HINT = " Click Calculate PO to refresh PO Qty and pipeline columns."


def remove_raise_ledger_day(
    sess,
    raised_date: str,
    *,
    session_id: str | None = None,
) -> dict:
    """Drop all ledger lines for one calendar day from session, DB, and archives."""
    day = str(raised_date).strip()[:10]
    if not day:
        return {"ok": False, "message": "raised_date is required (YYYY-MM-DD).", "removed": 0}

    from ..db.po_raised_db import delete_raise_day_meta, delete_raises_for_date, suppress_raise_date
    from ..services.po_raise_archive import delete_archives_for_date

    removed_db = delete_raises_for_date(day)
    delete_raise_day_meta(day)
    removed_sess = _remove_day_from_session_df(sess, day)
    suppress_raise_date(day, note="removed via PO dashboard")
    archive_removed = delete_archives_for_date(day, session_id)
    invalidate_po_calculate_result(sess)
    try:
        import backend.main as _main

        _main.merge_po_optional_sheets_into_warm_cache(sess)
    except Exception:
        pass
    parts = [
        f"Removed raise ledger for {day} ({max(removed_sess, removed_db):,} line(s) cleared)."
    ]
    if archive_removed:
        parts.append("Archived export for that day was deleted — Calculate PO will not auto-import it again.")
    else:
        parts.append("Auto-import from archive is blocked for that day.")
    parts.append(_RECALC_HINT.strip())
    return {
        "ok": True,
        "message": " ".join(parts),
        "removed": max(removed_sess, removed_db),
        "raised_date": day,
        "archive_removed": archive_removed,
        "suppressed": True,
        "recalculate_required": True,
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
    invalidate_po_calculate_result(sess)
    return {
        "ok": True,
        "message": f"Removed {len(skus)} SKU line(s) for {day}.{_RECALC_HINT}",
        "removed": max(removed_sess, removed_db),
        "raised_date": day,
        "recalculate_required": True,
    }


def clear_raise_ledger_all(sess) -> dict:
    """Wipe session ledger, durable SQLite store, archives, and block auto-import."""
    import pandas as pd

    from ..db.po_raised_db import clear_all, list_raises, suppress_raise_date
    from ..services.po_raise_archive import _archive_csv_dates, delete_all_archives

    n_sess = int(len(getattr(sess, "po_raise_ledger_df", pd.DataFrame())))
    ledger = getattr(sess, "po_raise_ledger_df", pd.DataFrame())
    if ledger is not None and not ledger.empty and "Raised_Date" in ledger.columns:
        rd = pd.to_datetime(ledger["Raised_Date"], errors="coerce").dt.normalize()
        for d in rd.dropna().unique():
            suppress_raise_date(str(pd.Timestamp(d).date()), note="cleared all via PO dashboard")
    for row in list_raises(limit=50_000):
        day = str(row.get("raised_date") or "")[:10]
        if day:
            suppress_raise_date(day, note="cleared all via PO dashboard")
    for day in sorted(_archive_csv_dates()):
        suppress_raise_date(day, note="cleared all via PO dashboard")
    n_db = clear_all()
    archives_removed = delete_all_archives()
    sess.po_raise_ledger_df = pd.DataFrame()
    invalidate_po_calculate_result(sess)
    try:
        import backend.main as _main

        _main.merge_po_optional_sheets_into_warm_cache(sess)
    except Exception:
        pass
    msg = f"PO raise ledger cleared ({max(n_sess, n_db):,} row(s) removed)."
    if archives_removed:
        msg += f" {archives_removed} archive file(s) deleted."
    msg += " Auto-import is blocked for all prior raise dates." + _RECALC_HINT
    return {
        "ok": True,
        "message": msg,
        "removed": max(n_sess, n_db),
        "archives_removed": archives_removed,
        "recalculate_required": True,
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
