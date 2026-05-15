"""
PO Engine router.
POST /api/po/calculate  → run PO calculation, return table
GET  /api/po/quarterly  → quarterly history pivot
POST /api/po/sku-status-lead → upload SKU status & lead time (Excel/CSV) for PO rules
"""
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel

router = APIRouter()

_IST = ZoneInfo("Asia/Kolkata")


def _sync_po_sidecars_to_durable_storage(request: Request, sess) -> None:
    """Mirror PO-only uploads into warm cache + PostgreSQL session bundle.

    Without this, daily inventory / SKU status lived only in the current process
    until a full cache save — new logins or ``/api/data/coverage`` looked empty
    even though sales roll-forward was already implemented server-side.
    """
    try:
        import backend.main as _main

        _main.merge_po_optional_sheets_into_warm_cache(sess)
    except Exception:
        logging.getLogger(__name__).exception("merge_po_optional_sheets_into_warm_cache failed")
    sid = getattr(request.state, "session_id", None) or getattr(sess, "_persist_sid", None)
    if not sid:
        return
    try:
        from ..db.forecast_session_pg import persist_session_bundle

        persist_session_bundle(sid, sess)
    except Exception:
        logging.getLogger(__name__).exception("PostgreSQL persist after PO sidecar upload failed")


class PORequest(BaseModel):
    period_days:      int   = 90
    lead_time:        int   = 30
    target_days:      int   = 210
    demand_basis:     str   = "Sold"       # "Sold" or "Net"
    use_seasonality:  bool  = False
    seasonal_weight:  float = 0.5
    group_by_parent:  bool  = False
    min_denominator:  int   = 7
    grace_days:       int   = 0
    safety_pct:       float = 0.0
    enforce_two_size_minimum: bool = False
    # When True (default): for sheet-resolved lead rows only, block PO while
    # (Tot inv + eff. pipeline) / ADS > Lead_Time_Days.
    enforce_lead_time_release_gate: bool = True
    # Calendar day for PO raise-ledger "yesterday / today" columns (YYYY-MM-DD).
    # Defaults to server date if omitted; browser should send local date for daily PO.
    planning_date: Optional[str] = None
    # How many calendar days of confirmed raises (ending at planning_date) add to effective pipeline.
    raise_ledger_lookback_days: int = 14
    # When True (default), before PO math import yesterday's server-archived export if the
    # ledger has no rows for that day (see POST /raise-ledger/archive-export).
    auto_import_yesterday_ledger: bool = True


class RaiseConfirmItem(BaseModel):
    oms_sku: str
    qty: int


class RaiseConfirmBody(BaseModel):
    rows: List[RaiseConfirmItem]
    raised_date: Optional[str] = None
    group_by_parent: bool = False


@router.post("/sku-status-lead")
async def po_upload_sku_status_lead(request: Request, file: UploadFile = File(...)):
    """Upload optional SKU status & per-SKU lead overrides (Excel/CSV): SKU + Lead time columns required; Status optional."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.sku_status_lead import parse_sku_status_lead_upload

    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}
    try:
        df = parse_sku_status_lead_upload(
            BytesIO(raw),
            file.filename or "sku_status.xlsx",
            sku_mapping=sess.sku_mapping or None,
        )
    except Exception as e:
        return {"ok": False, "message": f"Parse error: {e}"}
    if df.empty:
        return {"ok": False, "message": "No valid SKU rows found (need SKU and Lead time columns; Status is optional)."}
    sess.sku_status_lead_df = df
    sess._quarterly_cache.clear()
    _sync_po_sidecars_to_durable_storage(request, sess)
    return {"ok": True, "rows": int(len(df)), "message": f"Loaded {len(df)} SKU rows (status + lead time) for PO."}


@router.get("/sku-status-lead")
def po_get_sku_status_lead(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}
    df = sess.sku_status_lead_df
    if df is None or df.empty:
        return {"ok": True, "loaded": False, "rows": [], "columns": []}
    return {
        "ok": True,
        "loaded": True,
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict("records"),
    }


@router.post("/daily-inventory-history")
async def po_upload_daily_inventory_history(request: Request, file: UploadFile = File(...)):
    """Upload Daily Inventory History (wide-format Excel: SKU rows × date columns)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.daily_inventory_history import parse_daily_inventory_history_upload

    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}
    try:
        df = parse_daily_inventory_history_upload(
            BytesIO(raw),
            file.filename or "daily_inventory_history.xlsx",
            sku_mapping=sess.sku_mapping or None,
        )
    except Exception as e:
        return {"ok": False, "message": f"Parse error: {e}"}
    if df.empty:
        return {
            "ok": False,
            "message": "No usable rows. Need a wide-format sheet: column 1 = SKU, "
            "column 2 = parent (optional), then daily snapshot columns whose first row is the date.",
        }
    sess.daily_inventory_history_df = df
    sess._quarterly_cache.clear()
    _sync_po_sidecars_to_durable_storage(request, sess)
    skus = int(df["OMS_SKU"].nunique())
    days = int(pd.to_datetime(df["Date"], errors="coerce").dt.normalize().nunique())
    return {
        "ok": True,
        "rows": int(len(df)),
        "skus": skus,
        "days": days,
        "message": f"Loaded {len(df):,} SKU-day rows ({skus:,} SKUs × {days:,} days) for effective-days math.",
    }


@router.get("/daily-inventory-history")
def po_get_daily_inventory_history(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}
    df = sess.daily_inventory_history_df
    if df is None or df.empty:
        return {"ok": True, "loaded": False, "rows": 0, "skus": 0, "days": 0}
    return {
        "ok": True,
        "loaded": True,
        "rows": int(len(df)),
        "skus": int(df["OMS_SKU"].nunique()),
        "days": int(pd.to_datetime(df["Date"], errors="coerce").dt.normalize().nunique()),
        "min_date": str(pd.to_datetime(df["Date"], errors="coerce").min().date())
        if len(df)
        else "",
        "max_date": str(pd.to_datetime(df["Date"], errors="coerce").max().date())
        if len(df)
        else "",
    }


@router.get("/daily-inventory-history/sku")
def po_get_daily_inventory_history_for_sku(
    request: Request,
    sku: str,
    window_days: int = 30,
    end_date: Optional[str] = None,
):
    """Return the day-by-day on-hand timeline for a single SKU.

    UI uses this to let the user verify "Eff_Days" — i.e. how many days the
    item was actually in stock within the ADS window. Without a window, the
    last ``window_days`` days from the latest record are returned.
    """
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    df = sess.daily_inventory_history_df
    if df is None or df.empty:
        return {"ok": True, "loaded": False, "sku": sku, "rows": []}
    from ..services.po_engine import canonical_oms_key
    from ..services.helpers import get_parent_sku
    from ..services.daily_inventory_history import (
        IN_STOCK_MIN_QTY,
        extend_history_with_sales,
    )

    sku_map = sess.sku_mapping or None
    canon = lambda v: canonical_oms_key(v, sku_map)  # noqa: E731
    target = canon(sku)
    work = df.copy()
    work["OMS_SKU"] = work["OMS_SKU"].astype(str).map(canon)
    # Auto-extend with sales-derived snapshots so the drawer shows the same
    # data the engine used to compute Eff_Days (including days after baseline).
    sales_for_ext = sess.sales_df if hasattr(sess, "sales_df") else None
    cap_now = pd.Timestamp.now().normalize()
    try:
        work_ext = extend_history_with_sales(work, sales_df=sales_for_ext, cap_date=cap_now)
        if work_ext is not None and not work_ext.empty:
            work = work_ext
    except Exception:
        pass  # fall back to the raw upload
    sub = work[work["OMS_SKU"] == target].copy()
    parent_used = False
    if sub.empty:
        parent_key = get_parent_sku(target)
        if parent_key and parent_key != target:
            sub = work[work["OMS_SKU"].map(get_parent_sku) == parent_key].copy()
            if not sub.empty:
                parent_used = True

    if sub.empty:
        return {"ok": True, "loaded": True, "sku": sku, "rows": [], "in_stock_days": 0,
                "window_days": int(window_days), "parent_used": False}

    sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
    sub = sub.dropna(subset=["Date"])
    sub["Qty"] = pd.to_numeric(sub["Qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "Source" not in sub.columns:
        sub["Source"] = "uploaded"
    # Collapse duplicates (parent rollup can dup days); prefer uploaded source
    # over derived when both exist on the same date so the user sees the
    # actual baseline snapshot when available.
    sub["_src_rank"] = sub["Source"].map(lambda x: 0 if str(x) == "uploaded" else 1)
    sub = (
        sub.sort_values(["Date", "_src_rank"])
        .groupby("Date", as_index=False)
        .agg({"Qty": "max", "Source": "first"})
    )

    sub = sub.sort_values("Date")
    # Anchor at today so the drawer reflects the engine's "last N days" intent.
    # Caller can override with end_date if they want a specific reference.
    today_norm = pd.Timestamp.now().normalize()
    if end_date:
        try:
            end_ts = pd.Timestamp(end_date).normalize()
        except Exception:
            end_ts = today_norm
    else:
        end_ts = today_norm
    start_ts = end_ts - pd.Timedelta(days=max(0, int(window_days) - 1))
    win = sub[(sub["Date"] >= start_ts) & (sub["Date"] <= end_ts)].copy()
    in_stock_days = int((win["Qty"] >= IN_STOCK_MIN_QTY).sum())

    rows = [
        {
            "date": str(r["Date"].date()),
            "qty": float(r["Qty"]),
            "in_stock": bool(r["Qty"] >= IN_STOCK_MIN_QTY),
            "source": str(r.get("Source", "uploaded") or "uploaded"),
        }
        for _, r in win.iterrows()
    ]
    derived_days = sum(1 for r in rows if r["source"] == "derived")
    uploaded_days = len(rows) - derived_days
    return {
        "ok": True,
        "loaded": True,
        "sku": sku,
        "canonical_sku": target,
        "parent_used": parent_used,
        "window_days": int(window_days),
        "window_start": str(start_ts.date()),
        "window_end": str(end_ts.date()),
        "covered_days": int(len(rows)),
        "uploaded_days": uploaded_days,
        "derived_days": derived_days,
        "in_stock_days": in_stock_days,
        "out_of_stock_days": int(len(rows) - in_stock_days),
        "in_stock_min_qty": float(IN_STOCK_MIN_QTY),
        "rows": rows,
    }


@router.delete("/daily-inventory-history")
def po_clear_daily_inventory_history(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    import pandas as _pd

    sess.daily_inventory_history_df = _pd.DataFrame()
    sess._quarterly_cache.clear()
    _sync_po_sidecars_to_durable_storage(request, sess)
    return {"ok": True, "message": "Daily inventory history cleared."}


@router.post("/raise-confirm")
def po_raise_confirm(request: Request, body: RaiseConfirmBody):
    """Record SKUs/qty confirmed via PO Engine (Export & Confirm) into the session ledger.

    Next PO runs treat these units as extra pipeline (within ``raise_ledger_lookback_days``)
    so the same SKU is not re-recommended at full quantity day after day.
    """
    import logging

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    if not body.rows:
        return {"ok": False, "message": "No rows to record."}

    from ..services.po_raise_ledger import append_raise_confirm_rows

    try:
        as_dt = (
            pd.Timestamp(pd.to_datetime(body.raised_date).normalize())
            if body.raised_date
            else pd.Timestamp.now().normalize()
        )
    except Exception:
        as_dt = pd.Timestamp.now().normalize()

    tuples = [(r.oms_sku, int(r.qty)) for r in body.rows]
    sess.po_raise_ledger_df = append_raise_confirm_rows(
        getattr(sess, "po_raise_ledger_df", pd.DataFrame()),
        tuples,
        as_dt,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=bool(body.group_by_parent),
    )
    sess._quarterly_cache.clear()

    _sync_po_sidecars_to_durable_storage(request, sess)

    n = int(len(sess.po_raise_ledger_df))
    return {
        "ok": True,
        "ledger_rows": n,
        "message": f"Recorded {len(body.rows)} SKU line(s); ledger now has {n} SKU-day row(s).",
    }


@router.post("/raise-ledger/archive-export")
async def po_raise_ledger_archive_export(
    request: Request,
    file: UploadFile = File(...),
    raised_date: str = Form(""),
):
    """Store a PO CSV export on the server (per session + date) for next-day auto-import."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    sid = getattr(request.state, "session_id", None)
    if not sid:
        return {"ok": False, "message": "No session id."}
    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}
    try:
        if raised_date and str(raised_date).strip():
            as_dt = pd.Timestamp(pd.to_datetime(str(raised_date).strip()).normalize())
        else:
            from ..services.po_raise_archive import ist_today

            as_dt = ist_today()
    except Exception:
        return {"ok": False, "message": "Invalid raised_date; use YYYY-MM-DD."}

    from ..services.po_raise_archive import save_archive

    try:
        path = save_archive(sid, as_dt, raw)
    except ValueError as e:
        return {"ok": False, "message": str(e)}
    return {
        "ok": True,
        "raised_date": str(as_dt.date()),
        "path": str(path),
        "message": (
            f"Archived export for {as_dt.date()}. Tomorrow's Calculate PO can auto-import it "
            f"into the raise ledger (no Downloads folder needed)."
        ),
    }


@router.post("/raise-ledger/import-file")
async def po_raise_ledger_import_file(
    request: Request,
    file: UploadFile = File(...),
    raised_date: str = Form(""),
    group_by_parent: str = Form("false"),
    replace_day: str = Form("true"),
):
    """Same as import-csv but accepts Excel (.xlsx) PO recommendation exports."""
    return await po_raise_ledger_import_csv(
        request, file, raised_date, group_by_parent, replace_day
    )


@router.post("/raise-ledger/import-csv")
async def po_raise_ledger_import_csv(
    request: Request,
    file: UploadFile = File(...),
    raised_date: str = Form(""),
    group_by_parent: str = Form("false"),
    replace_day: str = Form("true"),
):
    """Record quantities from a ``po_recommendation*.csv`` export into the raise ledger.

    Use this when operators saved a PO CSV **without** clicking **Export & Confirm** in
    the Raise PO modal (plain **Export CSV** does not write the ledger). After import,
    run **Calculate PO** (or open the PO Dashboard tab) so ``PO_Raised_*`` / effective
    pipeline columns refresh.
    """
    from ..services.po_raise_import import apply_ledger_import, parse_ledger_upload_bytes

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}
    accum, err = parse_ledger_upload_bytes(raw, file.filename or "")
    if err:
        return {"ok": False, "message": err}

    try:
        if raised_date and str(raised_date).strip():
            as_dt = pd.Timestamp(pd.to_datetime(str(raised_date).strip()).normalize())
        else:
            as_dt = pd.Timestamp((datetime.now(_IST) - timedelta(days=1)).date())
    except Exception:
        return {"ok": False, "message": "Invalid raised_date; use YYYY-MM-DD."}

    rep = str(replace_day).strip().lower() in ("1", "true", "yes", "on")
    gbp = str(group_by_parent).strip().lower() in ("1", "true", "yes", "on")

    out = apply_ledger_import(
        sess, accum, as_dt, group_by_parent=gbp, replace_day=rep
    )
    _sync_po_sidecars_to_durable_storage(request, sess)
    out["message"] = (
        f"{out['message']} Run Calculate PO to refresh columns."
    )
    return out


@router.get("/raise-ledger")
def po_get_raise_ledger(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}
    df = getattr(sess, "po_raise_ledger_df", pd.DataFrame())
    if df is None or df.empty:
        return {"ok": True, "loaded": False, "rows": [], "columns": []}
    return {
        "ok": True,
        "loaded": True,
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict("records"),
    }


@router.delete("/raise-ledger")
def po_clear_raise_ledger(request: Request):
    import logging
    import pandas as _pd

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    sess.po_raise_ledger_df = _pd.DataFrame()
    sess._quarterly_cache.clear()
    _sync_po_sidecars_to_durable_storage(request, sess)
    return {"ok": True, "message": "PO raise ledger cleared."}


class PODashboardRequest(PORequest):
    """Same knobs as PO calculate, plus short-horizon sales windows for the dashboard."""

    recent_days: int = 7
    prev_days: int = 7
    spike_ratio: float = 1.35
    min_recent_units: int = 5
    low_run_days: float = 40.0
    max_rows_per_section: int = 80


@router.post("/dashboard")
def po_dashboard(request: Request, body: PODashboardRequest):
    """One-shot PO dashboard: runs the same engine as ``/calculate`` and adds
    sections for pipeline, open recommendations, demand spikes, and tight cover.
    Uses the session-based raise ledger (``sess.po_raise_ledger_df``)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    if sess.sales_df.empty:
        return {"ok": False, "message": "Build Sales first (upload platforms, then POST /api/upload/build-sales)."}
    if sess.inventory_df_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    from ..services.po_dashboard import build_dashboard_payload
    from ..services.po_engine import calculate_po_base

    inv_df = sess.inventory_df_parent if body.group_by_parent else sess.inventory_df_variant
    _ledger = getattr(sess, "po_raise_ledger_df", None)

    try:
        po_df = calculate_po_base(
            sales_df=sess.sales_df,
            inv_df=inv_df,
            period_days=body.period_days,
            lead_time=body.lead_time,
            target_days=body.target_days,
            demand_basis=body.demand_basis,
            min_denominator=body.min_denominator,
            grace_days=body.grace_days,
            safety_pct=body.safety_pct,
            use_seasonality=body.use_seasonality,
            seasonal_weight=body.seasonal_weight,
            sku_mapping=sess.sku_mapping or None,
            group_by_parent=body.group_by_parent,
            existing_po_df=sess.existing_po_df if not sess.existing_po_df.empty else None,
            sku_status_df=sess.sku_status_lead_df if not sess.sku_status_lead_df.empty else None,
            enforce_two_size_minimum=body.enforce_two_size_minimum,
            enforce_lead_time_release_gate=body.enforce_lead_time_release_gate,
            inventory_history_df=(
                sess.daily_inventory_history_df
                if not sess.daily_inventory_history_df.empty
                else None
            ),
            po_raise_ledger_df=(_ledger if _ledger is not None and not _ledger.empty else None),
            planning_date=body.planning_date,
            raise_ledger_lookback_days=body.raise_ledger_lookback_days,
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}

    if po_df is None or po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    payload = build_dashboard_payload(
        po_df,
        sess.sales_df,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=body.group_by_parent,
        recent_days=body.recent_days,
        prev_days=body.prev_days,
        spike_ratio=body.spike_ratio,
        min_recent_units=body.min_recent_units,
        low_run_days=body.low_run_days,
        max_rows_per_section=body.max_rows_per_section,
        lead_time_default=body.lead_time,
    )

    # Active raise-ledger summary (same source as the engine's "Raised_Recently_*" merge).
    raised_active: list = []
    raised_skus = 0
    raised_units = 0
    if _ledger is not None and not _ledger.empty:
        try:
            df = _ledger.copy()
            df["OMS_SKU"] = df.get("OMS_SKU", "").astype(str)
            df["qty"] = pd.to_numeric(df.get("qty", 0), errors="coerce").fillna(0).astype(int)
            df = df[df["qty"] > 0]
            if "raised_date" in df.columns:
                df["raised_date"] = df["raised_date"].astype(str)
                grp = (
                    df.groupby("OMS_SKU", as_index=False)
                      .agg(qty=("qty", "sum"), last_raised_date=("raised_date", "max"))
                )
            else:
                grp = df.groupby("OMS_SKU", as_index=False).agg(qty=("qty", "sum"))
                grp["last_raised_date"] = ""
            grp = grp.sort_values("qty", ascending=False).head(int(max(10, min(500, body.max_rows_per_section))))
            raised_active = grp.to_dict("records")
            raised_skus = int(grp["OMS_SKU"].nunique())
            raised_units = int(grp["qty"].sum())
        except Exception:
            raised_active, raised_skus, raised_units = [], 0, 0

    payload["raised_ledger_active"] = raised_active
    payload["raised_ledger_skus"] = raised_skus
    payload["raised_ledger_units"] = raised_units
    return payload


@router.post("/calculate")
def po_calculate(request: Request, body: PORequest):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    if sess.sales_df.empty:
        return {"ok": False, "message": "Build Sales first (upload platforms, then POST /api/upload/build-sales)."}
    if sess.inventory_df_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    from ..services.po_engine import calculate_po_base

    inv_df = sess.inventory_df_parent if body.group_by_parent else sess.inventory_df_variant

    ledger_auto_import = None
    sid = getattr(request.state, "session_id", None)
    if body.auto_import_yesterday_ledger and sid:
        from ..services.po_raise_archive import try_auto_import_yesterday_ledger

        ledger_auto_import = try_auto_import_yesterday_ledger(
            sess,
            sid,
            body.planning_date,
            group_by_parent=body.group_by_parent,
        )
        if ledger_auto_import and ledger_auto_import.get("ok"):
            _sync_po_sidecars_to_durable_storage(request, sess)

    _ledger = getattr(sess, "po_raise_ledger_df", None)
    try:
        po_df = calculate_po_base(
            sales_df=sess.sales_df,
            inv_df=inv_df,
            period_days=body.period_days,
            lead_time=body.lead_time,
            target_days=body.target_days,
            demand_basis=body.demand_basis,
            min_denominator=body.min_denominator,
            grace_days=body.grace_days,
            safety_pct=body.safety_pct,
            use_seasonality=body.use_seasonality,
            seasonal_weight=body.seasonal_weight,
            sku_mapping=sess.sku_mapping or None,
            group_by_parent=body.group_by_parent,
            existing_po_df=sess.existing_po_df if not sess.existing_po_df.empty else None,
            sku_status_df=sess.sku_status_lead_df if not sess.sku_status_lead_df.empty else None,
            enforce_two_size_minimum=body.enforce_two_size_minimum,
            enforce_lead_time_release_gate=body.enforce_lead_time_release_gate,
            inventory_history_df=(
                sess.daily_inventory_history_df
                if not sess.daily_inventory_history_df.empty
                else None
            ),
            po_raise_ledger_df=(_ledger if _ledger is not None and not _ledger.empty else None),
            planning_date=body.planning_date,
            raise_ledger_lookback_days=body.raise_ledger_lookback_days,
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}

    if po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    import numpy as np
    import pandas as pd

    def _dedupe_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Pandas allows duplicate column labels; ``to_dict('records')`` keeps only one key
        per name which shuffles values into wrong headers (e.g. ``fds``, ``sdaf``) in CSV/UI."""
        seen: dict[str, int] = {}
        out: list[str] = []
        for c in df.columns:
            base = str(c)
            if base not in seen:
                seen[base] = 0
                out.append(base)
            else:
                seen[base] += 1
                out.append(f"{base}__dup{seen[base]}")
        df = df.copy()
        df.columns = out
        return df

    # Serialize — keep string hints; round numerics only
    po_df = po_df.copy()
    po_df = _dedupe_column_names(po_df)
    for c in ["Suggest_Close_SKU", "PO_Block_Reason", "SKU_Sheet_Status"]:
        if c in po_df.columns:
            po_df[c] = po_df[c].fillna("").astype(str)
    num_cols = po_df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        po_df[num_cols] = (
            po_df[num_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .round(3)
        )
    # Days columns should always render with an explicit value in UI/export.
    for c in ["Days_Left", "Projected_Running_Days"]:
        if c in po_df.columns:
            s = pd.to_numeric(po_df[c], errors="coerce")
            po_df[c] = s.where(np.isfinite(s), 999.0).fillna(999.0).round(1)
    rows = po_df.to_dict("records")
    sales_through = None
    try:
        st = pd.to_datetime(sess.sales_df["TxnDate"], errors="coerce").max()
        if pd.notna(st):
            sales_through = str(pd.Timestamp(st).date())
    except Exception:
        pass
    planning_out = None
    if body.planning_date and str(body.planning_date).strip():
        try:
            planning_out = str(pd.Timestamp(pd.to_datetime(body.planning_date).normalize()).date())
        except Exception:
            planning_out = str(body.planning_date).strip()
    _ledger_after = getattr(sess, "po_raise_ledger_df", None)
    ledger_n = (
        int(len(_ledger_after))
        if _ledger_after is not None and not getattr(_ledger_after, "empty", True)
        else 0
    )
    auto_msg = None
    if ledger_auto_import and ledger_auto_import.get("ok"):
        auto_msg = ledger_auto_import.get("message")
    return {
        "ok":      True,
        "rows":    rows,
        "columns": list(po_df.columns),
        "sales_through": sales_through,
        "planning_date": planning_out,
        "raise_ledger_rows": ledger_n,
        "ledger_auto_import": auto_msg,
    }


@router.get("/quarterly-debug")
def po_quarterly_debug(request: Request):
    """Diagnostic endpoint — clears quarterly cache, recomputes, returns sample."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "reason": "No session"}

    # Also report & clear the live cache so next Calculate PO gets fresh data
    cache_key = (False, 8)
    cached = sess._quarterly_cache.get(cache_key)
    cached_rows = len(cached.get("rows", [])) if cached else 0
    cached_sample_sku = cached["rows"][0].get("OMS_SKU") if cached and cached.get("rows") else None
    sess._quarterly_cache.clear()  # force fresh on next po/quarterly call

    from ..services.po_engine import calculate_quarterly_history
    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=None,
        myntra_df=None,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=False,
        n_quarters=8,
    )
    sales_skus  = sorted(str(x) for x in sess.sales_df["Sku"].unique()[:10]) if not sess.sales_df.empty and "Sku" in sess.sales_df.columns else []
    inv_skus    = sorted(str(x) for x in sess.inventory_df_variant["OMS_SKU"].unique()[:10]) if not sess.inventory_df_variant.empty else []
    q_skus      = sorted(str(x) for x in pivot["OMS_SKU"].unique()[:10]) if not pivot.empty else []
    q_cols      = [str(c) for c in pivot.columns] if not pivot.empty else []
    # Count how many quarterly SKUs actually exist in the inventory
    if not pivot.empty and not sess.inventory_df_variant.empty:
        inv_set = set(sess.inventory_df_variant["OMS_SKU"].astype(str))
        matched = int(pivot["OMS_SKU"].astype(str).isin(inv_set).sum())
    else:
        matched = 0
    sample_row  = {str(k): (None if str(v) in ("nan", "NaN") else float(v) if hasattr(v, '__float__') else str(v))
                   for k, v in (pivot.fillna(0).iloc[0].to_dict().items() if not pivot.empty else {}.items())}
    return {
        "ok": True,
        "cache_had_rows": cached_rows,
        "cache_sample_sku": cached_sample_sku,
        "cache_cleared": True,
        "sales_rows": int(len(sess.sales_df)),
        "inv_rows": int(len(sess.inventory_df_variant)),
        "quarterly_rows": int(len(pivot)) if not pivot.empty else 0,
        "quarterly_skus_matching_inventory": matched,
        "quarterly_columns": q_cols,
        "sales_sku_sample": sales_skus,
        "inv_sku_sample": inv_skus,
        "quarterly_sku_sample": q_skus,
        "sample_row": sample_row,
    }


@router.get("/quarterly")
def po_quarterly(request: Request, group_by_parent: bool = False, n_quarters: int = 8):
    sess = request.state.session
    if sess is None:
        return {"loaded": False}

    from ..routers.data import _restore_daily_if_needed
    _restore_daily_if_needed(sess)

    cache_key = (group_by_parent, n_quarters)
    if cache_key in sess._quarterly_cache:
        return sess._quarterly_cache[cache_key]

    from ..services.po_engine import calculate_quarterly_history

    _boot = sess.sales_df.empty or "Sku" not in sess.sales_df.columns
    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=sess.mtr_df if _boot and not sess.mtr_df.empty else None,
        myntra_df=sess.myntra_df if _boot and not sess.myntra_df.empty else None,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
    )
    if pivot.empty:
        result = {"loaded": False, "rows": []}
    else:
        result = {
            "loaded":   True,
            "columns":  list(pivot.columns),
            "rows":     pivot.fillna(0).to_dict("records"),
        }
    sess._quarterly_cache[cache_key] = result
    return result
