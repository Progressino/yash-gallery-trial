"""
PO Engine router.
POST /api/po/calculate  → run PO calculation, return table
GET  /api/po/quarterly  → quarterly history pivot
POST /api/po/sku-status-lead → upload SKU status & lead time (Excel/CSV) for PO rules
"""
import asyncio
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from pydantic import BaseModel

router = APIRouter()

_IST = ZoneInfo("Asia/Kolkata")


def _inventory_history_df_for_read(sess) -> pd.DataFrame:
    """Load daily inventory matrix from warm cache / disk — no full session PG restore."""
    try:
        from ..services.po_session_hydrate import ensure_inventory_history_authoritative_for_read

        return ensure_inventory_history_authoritative_for_read(sess)
    except Exception:
        pass
    try:
        import backend.main as _main

        _main.bootstrap_warm_cache_if_empty()
        _main._top_up_po_sidecars_from_loose_disk()
        from ..services.po_session_hydrate import ensure_po_sidecars_hydrated

        _main.restore_po_sidecars_from_warm(sess)
        ensure_po_sidecars_hydrated(sess)
    except Exception:
        pass
    df = getattr(sess, "daily_inventory_history_df", None)
    if df is not None and not df.empty:
        return df
    try:
        import backend.main as _main

        wc = (_main._warm_cache or {}).get("daily_inventory_history_df")
        if wc is not None and not wc.empty:
            return wc
    except Exception:
        pass
    return pd.DataFrame()


def _inventory_matrix_payload(
    sess,
    *,
    q: str = "",
    limit: int = 150,
    offset: int = 0,
    days: int = 30,
    end_date: Optional[str] = None,
) -> dict:
    from ..services.daily_inventory_history import inventory_history_wide_matrix

    try:
        df = _inventory_history_df_for_read(sess)
        out = inventory_history_wide_matrix(
            df,
            q=q,
            limit=min(max(1, int(limit)), 15000),
            offset=max(0, int(offset)),
            days=min(max(1, int(days)), 120),
            end_date=end_date,
        )
        out["ok"] = True
        return out
    except Exception as e:
        logging.getLogger(__name__).exception("inventory history matrix failed")
        return {
            "ok": False,
            "loaded": False,
            "message": str(e),
            "dates": [],
            "rows": [],
            "total": 0,
            "limit": int(limit),
            "offset": int(offset),
        }


@router.get("/readiness")
def po_readiness(request: Request):
    """Lightweight PO page gate — data ready for calculate, not all background jobs finished."""
    from ..models.schemas import PoReadinessResponse
    from ..routers.data import _build_coverage_response
    from ..services.po_readiness import build_po_readiness

    sess = request.state.session
    if sess is None:
        return PoReadinessResponse(po_ready=False, hydration="none")
    sid = getattr(request.state, "session_id", None) or ""
    cov = _build_coverage_response(sess, light=True)
    return PoReadinessResponse(**build_po_readiness(sess, cov, session_id=sid))


def _return_overlay_for_po_calc(sess):
    from ..services.po_return_import import aggregate_return_overlay_for_use

    ov = aggregate_return_overlay_for_use(getattr(sess, "po_return_overlay_df", None))
    if ov is None or getattr(ov, "empty", True):
        return None
    return ov


def _sync_po_sidecars_to_durable_storage(
    request: Request,
    sess,
    background_tasks: Optional[BackgroundTasks] = None,
) -> None:
    """Mirror PO-only uploads into warm cache + PostgreSQL session bundle.

    Without this, daily inventory / SKU status lived only in the current process
    until a full cache save — new logins or ``/api/data/coverage`` looked empty
    even though sales roll-forward was already implemented server-side.

    Full session bundles can be large (sales + inventory). Persist asynchronously
    so uploads return quickly and reverse proxies do not 502.
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
        from ..db.forecast_session_pg import (
            debounced_persist_session,
            persist_session_bundle_thread_safe,
            pg_session_persist_enabled,
        )

        if not pg_session_persist_enabled():
            return
        if background_tasks is not None:
            background_tasks.add_task(persist_session_bundle_thread_safe, sid, sess)
        else:
            debounced_persist_session(sid, sess, delay=3.0)
    except Exception:
        logging.getLogger(__name__).exception("PostgreSQL persist after PO sidecar upload failed")


class PORequest(BaseModel):
    period_days:      int   = 90
    lead_time:        int   = 30
    target_days:      int   = 135
    demand_basis:     str   = "Sold"       # "Sold" or "Net"
    use_seasonality:  bool  = False
    seasonal_weight:  float = 0.5
    group_by_parent:  bool  = False
    min_denominator:  int   = 7
    grace_days:       int   = 0
    safety_pct:       float = 0.0
    enforce_two_size_minimum: bool = False
    enforce_lead_time_release_gate: bool = True
    # Calendar day for PO raise-ledger "yesterday / today" columns (YYYY-MM-DD).
    # Defaults to server date if omitted; browser should send local date for daily PO.
    planning_date: Optional[str] = None
    # Raise-date picker in PO UI — fills ``PO_Raised_On_View_Date`` (YYYY-MM-DD).
    raise_view_date: Optional[str] = None
    # How many calendar days of confirmed raises (ending at planning_date) add to effective pipeline.
    raise_ledger_lookback_days: int = 14
    # When True (default), before PO math import yesterday's server-archived export if the
    # ledger has no rows for that day (see POST /raise-ledger/archive-export).
    auto_import_yesterday_ledger: bool = True
    # When any size of a parent SKU has Projected_Running_Days below this threshold,
    # automatically include ALL sibling sizes in the PO output so the operator
    # can review and raise for every size.  Set to 0 to disable.
    urgent_all_sizes_days: int = 45
    # When False, ADS uses only the current-period sales (no Last Year fallback).
    # Set True to blend LY data for seasonal planning, False for current-demand-only PO.
    use_ly_fallback: bool = True
    # When True (default), reuse another session's PO result on this server if planning
    # date, settings, and data snapshot match (see ``po_shared_cache``).
    use_shared_cache: bool = True


class RaiseConfirmItem(BaseModel):
    oms_sku: str
    qty: int


class RaiseLedgerDeleteSkusBody(BaseModel):
    raised_date: str
    oms_skus: List[str]


class RaiseConfirmBody(BaseModel):
    rows: List[RaiseConfirmItem]
    raised_date: Optional[str] = None
    group_by_parent: bool = False


@router.post("/sku-status-lead")
async def po_upload_sku_status_lead(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload optional SKU status & per-SKU lead overrides (Excel/CSV): SKU + Lead time columns required; Status optional."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.sku_status_lead import parse_sku_status_lead_upload

    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}
    try:
        from ..services.upload_file_sniff import check_upload_target

        wrong = check_upload_target("sku_status_lead", raw, file.filename or "")
        if wrong:
            return {"ok": False, "wrong_upload_target": True, "message": wrong}
    except Exception:
        pass
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
    try:
        from ..services.po_raise_remove import invalidate_po_calculate_result
        from ..services.po_shared_cache import invalidate_all_shared_caches

        invalidate_po_calculate_result(sess)
        invalidate_all_shared_caches()
    except Exception:
        pass
    _sync_po_sidecars_to_durable_storage(request, sess, background_tasks)
    return {"ok": True, "rows": int(len(df)), "message": f"Loaded {len(df)} SKU rows (status + lead time) for PO."}


@router.get("/sku-status-lead")
def po_get_sku_status_lead(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}
    try:
        from ..services.po_session_hydrate import ensure_po_sidecars_hydrated

        ensure_po_sidecars_hydrated(sess)
        import backend.main as _main

        _main.restore_po_sidecars_from_warm(sess)
    except Exception:
        pass
    df = sess.sku_status_lead_df
    if df is None or df.empty:
        return {"ok": True, "loaded": False, "rows": [], "columns": []}
    return {
        "ok": True,
        "loaded": True,
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict("records"),
    }


@router.post("/manual-intransit-sheet")
async def po_upload_manual_intransit_sheet(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Admin: upload Intrasit + Not In Inventory workbook (replaces prior upload — no duplicate rows)."""
    from ..services.manual_intransit_sheet import (
        apply_manual_intransit_import,
        parse_manual_intransit_workbook,
    )

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}

    intransit_df, not_in_df, report = parse_manual_intransit_workbook(
        raw,
        file.filename or "manual_intransit.xlsx",
        sku_mapping=sess.sku_mapping or None,
    )
    if report.get("error") and intransit_df.empty and not_in_df.empty:
        return {
            "ok": False,
            "message": str(report["error"]),
            "parse_report": report,
        }

    out = apply_manual_intransit_import(
        sess,
        intransit_df,
        not_in_df,
        report,
        filename=file.filename or "manual_intransit.xlsx",
    )
    if not out.get("ok"):
        return out

    sess._quarterly_cache.clear()
    _sync_po_sidecars_to_durable_storage(request, sess, background_tasks)
    try:
        import backend.main as _main

        _main.merge_po_optional_sheets_into_warm_cache(sess)
        _main.merge_inventory_into_warm_cache(sess)
    except Exception:
        logging.getLogger(__name__).exception("warm cache after manual intransit upload failed")
    return out


@router.get("/manual-intransit-sheet")
def po_get_manual_intransit_sheet(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}
    try:
        from ..services.po_session_hydrate import ensure_po_sidecars_hydrated

        ensure_po_sidecars_hydrated(sess)
        import backend.main as _main

        _main.restore_po_sidecars_from_warm(sess)
    except Exception:
        pass
    df = getattr(sess, "manual_intransit_overlay_df", None)
    report = getattr(sess, "manual_intransit_parse_report", None) or {}
    if df is None or df.empty:
        return {
            "ok": True,
            "loaded": False,
            "parse_report": report,
            "filename": getattr(sess, "manual_intransit_filename", "") or "",
            "uploaded_at": getattr(sess, "manual_intransit_uploaded_at", "") or "",
        }
    return {
        "ok": True,
        "loaded": True,
        "filename": getattr(sess, "manual_intransit_filename", "") or "",
        "uploaded_at": getattr(sess, "manual_intransit_uploaded_at", "") or "",
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict("records"),
        "skus": int(len(df)),
        "intransit_units": int(pd.to_numeric(df.get("Manual_InTransit"), errors="coerce").fillna(0).sum()),
        "not_in_inventory_units": int(
            pd.to_numeric(df.get("Not_In_Inventory_Qty"), errors="coerce").fillna(0).sum()
        ),
        "parse_report": report,
    }


@router.post("/daily-inventory-history")
async def po_upload_daily_inventory_history(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload Daily Inventory History (wide-format Excel: SKU rows × date columns).

    Parsing runs in the background so nginx does not 502 on large workbooks.
    """
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    sid = getattr(request.state, "session_id", None)
    if not sid:
        return {"ok": False, "message": "No session id."}

    if getattr(sess, "daily_inventory_upload_status", "idle") == "running":
        return {
            "ok": True,
            "status": "running",
            "message": getattr(sess, "daily_inventory_upload_message", "")
            or "Daily inventory upload already in progress…",
        }

    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}

    try:
        from ..services.upload_file_sniff import check_file_for_daily_inventory_history

        wrong = check_file_for_daily_inventory_history(raw, file.filename or "")
        if wrong:
            return {
                "ok": False,
                "wrong_upload_target": True,
                "suggested_section": "snapshot_inventory",
                "message": wrong.replace("**", ""),
            }
    except Exception:
        logging.getLogger(__name__).exception("daily inventory history upload sniff failed")

    from ..concurrency import INVENTORY_EXECUTOR
    from ..services.daily_inventory_upload_run import background_daily_inventory_upload

    sess.daily_inventory_upload_status = "running"
    sess.daily_inventory_upload_message = "Upload received — queued for parse…"
    sess.daily_inventory_upload_result = {}
    INVENTORY_EXECUTOR.submit(
        background_daily_inventory_upload,
        sid,
        raw,
        file.filename or "daily_inventory_history.xlsx",
    )
    return {
        "ok": True,
        "status": "running",
        "message": "Parsing daily inventory in background — this may take 1–3 minutes for large files.",
    }


@router.get("/daily-inventory-history/upload-status")
def po_daily_inventory_upload_status(request: Request):
    """Poll after POST /daily-inventory-history until status is ``done`` or ``error``."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "status": "error", "message": "No session"}
    st = getattr(sess, "daily_inventory_upload_status", "idle") or "idle"
    msg = getattr(sess, "daily_inventory_upload_message", "") or ""
    out: dict = {"status": st, "message": msg}
    result = getattr(sess, "daily_inventory_upload_result", None) or {}
    if st == "done":
        out.update(result)
        out["ok"] = True
    elif st == "error":
        out["ok"] = False
        if result.get("message"):
            out["message"] = result["message"]
    else:
        out["ok"] = True
    return out


@router.get("/daily-inventory-history")
async def po_get_daily_inventory_history(request: Request, days: int = 30, end_date: Optional[str] = None):
    from ..concurrency import run_read_api

    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}

    def _work() -> dict:
        from ..services.daily_inventory_history import inventory_history_summary

        df = _inventory_history_df_for_read(sess)
        summary = inventory_history_summary(
            df,
            days=min(max(1, int(days)), 120),
            end_date=end_date,
        )
        out = {"ok": True, **summary}
        out["uploaded_at"] = str(getattr(sess, "daily_inventory_history_uploaded_at", "") or "") or None
        out["filename"] = str(getattr(sess, "daily_inventory_history_filename", "") or "") or None
        return out

    return await run_read_api(_work)


@router.get("/daily-inventory-history/dates")
def po_list_daily_inventory_history_dates(request: Request, limit: int = 120):
    """List snapshot dates in the wide inventory matrix (newest first)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "dates": []}
    try:
        from ..services.po_session_hydrate import ensure_po_sidecars_hydrated

        ensure_po_sidecars_hydrated(sess)
    except Exception:
        pass
    df = getattr(sess, "daily_inventory_history_df", None)
    from ..services.daily_inventory_history import list_inventory_history_dates

    dates = list_inventory_history_dates(df if df is not None else pd.DataFrame(), limit=limit)
    return {"ok": True, "dates": dates, "count": len(dates)}


@router.get("/daily-inventory-history/by-date")
def po_daily_inventory_history_by_date(
    request: Request,
    date: str,
    q: str = "",
    limit: int = 500,
    offset: int = 0,
):
    """All SKU on-hand quantities for one snapshot date (admin verification)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    try:
        from ..services.po_session_hydrate import ensure_po_sidecars_hydrated

        ensure_po_sidecars_hydrated(sess)
    except Exception:
        pass
    df = getattr(sess, "daily_inventory_history_df", None)
    from ..services.daily_inventory_history import inventory_rows_for_date

    out = inventory_rows_for_date(
        df if df is not None else pd.DataFrame(),
        date,
        q=q,
        limit=min(max(1, int(limit)), 2000),
        offset=max(0, int(offset)),
    )
    out["ok"] = True
    return out


@router.get("/daily-inventory-history/matrix")
async def po_daily_inventory_history_matrix(
    request: Request,
    q: str = "",
    limit: int = 150,
    offset: int = 0,
    days: int = 30,
    end_date: Optional[str] = None,
):
    """Excel-style wide matrix: SKU rows × date columns (paginated)."""
    from ..concurrency import run_read_api

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    return await run_read_api(
        _inventory_matrix_payload,
        sess,
        q=q,
        limit=limit,
        offset=offset,
        days=days,
        end_date=end_date,
    )


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
    if "Source" not in work.columns:
        work["Source"] = "uploaded"
    uploaded_snap = work.copy()
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
    # Uploaded snapshots always win over derived for the same calendar day.
    if not uploaded_snap.empty:
        uploaded_snap["Source"] = "uploaded"
        work = pd.concat([work, uploaded_snap], ignore_index=True)
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
    uploaded_only = sub[sub["Source"].astype(str) == "uploaded"] if "Source" in sub.columns else sub
    uploaded_max = (
        pd.Timestamp(uploaded_only["Date"].max()).normalize()
        if not uploaded_only.empty and pd.notna(uploaded_only["Date"].max())
        else None
    )
    # Anchor at latest uploaded snapshot (or today) so today's upload is visible
    # even when sales / derived roll-forward only run through yesterday.
    today_norm = pd.Timestamp.now().normalize()
    if end_date:
        try:
            end_ts = pd.Timestamp(end_date).normalize()
        except Exception:
            end_ts = today_norm
    else:
        end_ts = today_norm
    if uploaded_max is not None:
        end_ts = max(end_ts, uploaded_max)
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


@router.get("/sku-audit")
def po_sku_audit(request: Request, sku: str):
    """
  Cross-check one PO row (shared cache / last session run) vs Tier-3 sales in the ADS window.
  Pass the same query params as ``GET /po/calculate/shared-cache`` (planning_date, period_days, …).
    """
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.po_sku_audit import build_po_sku_audit

    qp = {k: request.query_params.get(k) for k in request.query_params.keys()}
    return build_po_sku_audit(sess, sku, qp)


@router.delete("/daily-inventory-history")
def po_clear_daily_inventory_history(request: Request):
    from ..services.upload_policy import _DELETE_DENIED_MSG, may_delete_upload_data

    auth = getattr(request.state, "auth", None) or {}
    if not may_delete_upload_data(str(auth.get("role") or ""), str(auth.get("sub") or "")):
        return {"ok": False, "message": _DELETE_DENIED_MSG}
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

    from ..services.po_raise_batch import allocate_po_number

    po_number = allocate_po_number(str(as_dt.date()))
    total_qty = int(sum(int(r.qty) for r in body.rows))
    tuples = [(r.oms_sku, int(r.qty)) for r in body.rows]
    sess.po_raise_ledger_df = append_raise_confirm_rows(
        getattr(sess, "po_raise_ledger_df", pd.DataFrame()),
        tuples,
        as_dt,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=bool(body.group_by_parent),
    )
    sess._quarterly_cache.clear()

    from ..services.po_raise_import import sync_ledger_to_durable_db

    sync_ledger_to_durable_db(sess, as_dt)
    _sync_po_sidecars_to_durable_storage(request, sess)

    n = int(len(sess.po_raise_ledger_df))
    return {
        "ok": True,
        "po_number": po_number,
        "raised_date": str(as_dt.date()),
        "sku_count": len(body.rows),
        "total_qty": total_qty,
        "ledger_rows": n,
        "message": (
            f"{po_number}: recorded {len(body.rows)} SKU line(s) "
            f"({total_qty:,} units); ledger now has {n} SKU-day row(s)."
        ),
    }


@router.post("/raise-ledger/archive-export")
async def po_raise_ledger_archive_export(
    request: Request,
    background_tasks: BackgroundTasks,
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

    from ..services.po_raise_import import apply_ledger_import, parse_ledger_upload_bytes

    try:
        path = save_archive(sid, as_dt, raw)
    except ValueError as e:
        return {"ok": False, "message": str(e)}

    ledger_msg = ""
    accum, err = parse_ledger_upload_bytes(raw, file.filename or "")
    if not err and accum:
        imp = apply_ledger_import(
            sess,
            accum,
            as_dt,
            group_by_parent=False,
            replace_day=True,
        )
        if imp.get("ok"):
            _sync_po_sidecars_to_durable_storage(request, sess, background_tasks)
            ledger_msg = (
                f" Recorded {imp.get('imported_skus', 0):,} SKU(s) / "
                f"{imp.get('total_units', 0):,} units in the raise ledger."
            )
    return {
        "ok": True,
        "raised_date": str(as_dt.date()),
        "path": str(path),
        "message": (
            f"Archived export for {as_dt.date()}.{ledger_msg} "
            f"Future Calculate PO runs will not repeat these quantities."
        ),
    }


@router.post("/raise-ledger/import-file")
async def po_raise_ledger_import_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    raised_date: str = Form(""),
    group_by_parent: str = Form("false"),
    replace_day: str = Form("true"),
):
    """Same as import-csv but accepts Excel (.xlsx) PO recommendation exports."""
    return await po_raise_ledger_import_csv(
        request, background_tasks, file, raised_date, group_by_parent, replace_day
    )


@router.post("/raise-ledger/import-csv")
async def po_raise_ledger_import_csv(
    request: Request,
    background_tasks: BackgroundTasks,
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

    from ..services.po_raise_import import parse_raise_date_from_filename

    try:
        if raised_date and str(raised_date).strip():
            as_dt = pd.Timestamp(pd.to_datetime(str(raised_date).strip()).normalize())
        else:
            guessed = parse_raise_date_from_filename(file.filename or "")
            if guessed is not None:
                as_dt = guessed
            else:
                as_dt = pd.Timestamp((datetime.now(_IST) - timedelta(days=1)).date())
    except Exception:
        return {"ok": False, "message": "Invalid raised_date; use YYYY-MM-DD."}

    rep = str(replace_day).strip().lower() in ("1", "true", "yes", "on")
    gbp = str(group_by_parent).strip().lower() in ("1", "true", "yes", "on")

    from ..services.po_raise_archive import save_archive

    out = apply_ledger_import(
        sess, accum, as_dt, group_by_parent=gbp, replace_day=rep
    )
    sid = getattr(request.state, "session_id", None)
    if sid and out.get("ok"):
        try:
            save_archive(sid, as_dt, raw)
        except Exception:
            logging.getLogger(__name__).exception("save_archive on import failed")
    _sync_po_sidecars_to_durable_storage(request, sess, background_tasks)
    out["message"] = (
        f"{out['message']} Run Calculate PO to refresh columns."
    )
    return out


@router.get("/raise-ledger/summary")
def po_raise_ledger_summary(
    request: Request,
    lookback_days: int = 30,
    planning_date: str = "",
    max_skus_per_day: int = 500,
):
    """Daily raise totals + per-day SKU lines (lightweight — no full PO engine run)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.po_raise_ledger import summarize_raise_ledger_for_dashboard

    _ledger = getattr(sess, "po_raise_ledger_df", None)
    summary = summarize_raise_ledger_for_dashboard(
        _ledger,
        lookback_days=lookback_days,
        planning_date=planning_date or None,
        max_skus_per_day=max_skus_per_day,
    )
    return {"ok": True, **summary}


@router.get("/raise-ledger/dates")
def po_raise_ledger_dates(request: Request):
    from ..services.po_raise_ledger import list_raise_ledger_dates

    sess = request.state.session
    if sess is None:
        return {"ok": False, "dates": []}
    df = getattr(sess, "po_raise_ledger_df", pd.DataFrame())
    return {"ok": True, "dates": list_raise_ledger_dates(df)}


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


def _run_returns_import_worker(
    session_id: str,
    raw: bytes,
    filename: str,
    *,
    group_by_parent: bool,
    replace: bool,
    sku_mapping: dict | None,
) -> None:
    """Parse return archive, persist overlay to disk, rebuild net sales — off the HTTP thread."""
    from ..session import store

    sess = store.get(session_id)
    if sess is None:
        return
    try:
        from ..services.po_return_import import apply_return_overlay_import, parse_return_upload_bytes

        sess.returns_import_progress = 20
        sess.returns_import_message = "Extracting and parsing return files…"
        overlay, err, import_warnings = parse_return_upload_bytes(
            raw,
            filename,
            sku_mapping=sku_mapping or None,
            group_by_parent=group_by_parent,
            meesho_df=getattr(sess, "meesho_df", None),
        )
        sess.returns_import_warnings = list(import_warnings or [])
        if err:
            sess.returns_import_status = "error"
            warn_tail = ""
            if import_warnings:
                warn_tail = " " + "; ".join(import_warnings[:3])
            sess.returns_import_message = f"{err}{warn_tail}"
            sess.returns_import_progress = 0
            return
        sess.returns_import_progress = 55
        sess.returns_import_message = "Saving return data…"
        out = apply_return_overlay_import(sess, overlay, replace=replace, filename=filename)
        try:
            import backend.main as _main

            _main.merge_po_optional_sheets_into_warm_cache(sess)
        except Exception:
            logging.getLogger(__name__).exception("merge return overlay into warm cache failed")
        done_msg = str(out.get("message") or "Return sheet imported.")
        if import_warnings:
            preview = "; ".join(import_warnings[:4])
            extra = len(import_warnings) - 4
            if extra > 0:
                preview += f" (+{extra} more)"
            done_msg = f"{done_msg} ⚠ {preview}"
        sess.returns_import_status = "done"
        sess.returns_import_message = done_msg
        sess.returns_import_progress = 100
        from ..routers.upload import _run_returns_import_followup

        _run_returns_import_followup(session_id)
    except Exception as e:
        logging.getLogger(__name__).exception("return import worker failed")
        sess.returns_import_status = "error"
        sess.returns_import_message = str(e)
        sess.returns_import_progress = 0
    finally:
        sess.returns_import_started = 0.0


@router.post("/returns/import-file")
async def po_returns_import_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    group_by_parent: str = Form("false"),
    replace: str = Form("false"),
):
    """Import return units by SKU (CSV / Excel / RAR / ZIP). Reduces PO qty and feeds Net demand."""
    from ..concurrency import RETURNS_IMPORT_EXECUTOR

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}
    try:
        from ..services.upload_file_sniff import check_upload_target

        wrong = check_upload_target("returns", raw, file.filename or "")
        if wrong:
            return {
                "ok": False,
                "wrong_upload_target": True,
                "suggested_section": "daily_sales",
                "message": wrong,
            }
    except Exception:
        logging.getLogger(__name__).exception("returns upload sniff failed")
    gbp = str(group_by_parent).strip().lower() in ("1", "true", "yes", "on")
    rep = str(replace).strip().lower() in ("1", "true", "yes", "on")
    session_id = getattr(request.state, "session_id", None) or getattr(sess, "_persist_sid", None)
    if not session_id:
        return {"ok": False, "message": "No session id — refresh and try again."}
    if getattr(sess, "returns_import_status", "idle") == "running":
        return {
            "ok": False,
            "message": "A return import is already running on this session. Wait for it to finish.",
            "returns_import": "running",
        }

    import time

    sess.returns_import_status = "running"
    sess.returns_import_message = "Queued return import…"
    sess.returns_import_progress = 5
    sess.returns_import_started = time.monotonic()
    sku_mapping = dict(sess.sku_mapping or {})
    RETURNS_IMPORT_EXECUTOR.submit(
        _run_returns_import_worker,
        session_id,
        raw,
        file.filename or "",
        group_by_parent=gbp,
        replace=rep,
        sku_mapping=sku_mapping,
    )
    return {
        "ok": True,
        "status": "running",
        "returns_import": "running",
        "sales_rebuilt": False,
        "sales_rebuild": "pending",
        "message": (
            "Return file accepted — extracting and saving in the background (large RAR archives "
            "can take a few minutes). This page will update when finished; run Calculate PO "
            "afterward to refresh PO qty."
        ),
    }


@router.delete("/returns/overlay")
def po_clear_returns_overlay(request: Request):
    from ..services.upload_policy import _DELETE_DENIED_MSG, may_delete_upload_data

    auth = getattr(request.state, "auth", None) or {}
    if not may_delete_upload_data(str(auth.get("role") or ""), str(auth.get("sub") or "")):
        return {"ok": False, "message": _DELETE_DENIED_MSG}
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.po_return_import import clear_return_overlay_meta

    sess.po_return_overlay_df = pd.DataFrame()
    sess.return_overlay_as_of = None
    sess.return_overlay_sources = []
    clear_return_overlay_meta(sess)
    sess._quarterly_cache.clear()
    sales_note = ""
    try:
        with sess._daily_restore_lock:
            from ..routers import upload as _upload_router

            ok_rb, msg_rb = _upload_router._rebuild_sales_sync(sess)
            sales_note = msg_rb if ok_rb else f"Sales rebuild warning: {msg_rb}"
    except Exception:
        logging.getLogger(__name__).exception("sales rebuild after clear return overlay")
        sales_note = "Sales rebuild failed (see server logs)."
    try:
        import backend.main as _main

        _main.publish_warm_cache_from_session(sess)
    except Exception:
        logging.getLogger(__name__).exception("publish_warm_cache_from_session after clear returns")
    _sync_po_sidecars_to_durable_storage(request, sess)
    return {"ok": True, "message": f"Return sheet cleared. {sales_note}".strip()}


def _request_role(request: Request) -> str:
    auth = getattr(request.state, "auth", None) or {}
    return str(auth.get("role") or "Admin")


@router.delete("/raise-ledger/day")
def po_delete_raise_ledger_day(request: Request, raised_date: str = ""):
    """Remove all raises recorded for one calendar day (session + durable DB)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.po_raise_remove import remove_raise_ledger_day

    sid = getattr(request.state, "session_id", None)
    out = remove_raise_ledger_day(sess, raised_date, session_id=sid)
    if out.get("ok"):
        _sync_po_sidecars_to_durable_storage(request, sess)
    return out


@router.post("/raise-ledger/delete-skus")
def po_delete_raise_ledger_skus(request: Request, body: RaiseLedgerDeleteSkusBody):
    """Admin-only: remove specific SKU lines for one raise date (mistaken PO)."""
    from ..services.upload_policy import may_admin_po_session_edits

    role = _request_role(request)
    auth = getattr(request.state, "auth", None) or {}
    username = str(auth.get("sub") or "")
    if not may_admin_po_session_edits(role, username):
        return {
            "ok": False,
            "message": "Removing individual raised PO lines is Admin-only.",
        }
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.po_raise_remove import remove_raise_ledger_skus

    out = remove_raise_ledger_skus(sess, body.raised_date, body.oms_skus)
    if out.get("ok"):
        _sync_po_sidecars_to_durable_storage(request, sess)
    return out


@router.delete("/raise-ledger")
def po_clear_raise_ledger(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    from ..services.po_raise_remove import clear_raise_ledger_all

    out = clear_raise_ledger_all(sess)
    _sync_po_sidecars_to_durable_storage(request, sess)
    return out


class PODashboardRequest(PORequest):
    """Same knobs as PO calculate, plus short-horizon sales windows for the dashboard."""

    recent_days: int = 7
    prev_days: int = 7
    spike_ratio: float = 1.35
    min_recent_units: int = 5
    low_run_days: float = 45.0
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

    from ..services.po_calculate_run import _build_platform_sales_df
    from ..services.existing_po import ensure_existing_po_hydrated

    ensure_existing_po_hydrated(sess)
    inv_df = sess.inventory_df_parent if body.group_by_parent else sess.inventory_df_variant
    _ledger = getattr(sess, "po_raise_ledger_df", None)
    _platform_sales = _build_platform_sales_df(sess)
    _ads_source = _platform_sales if not _platform_sales.empty else sess.sales_df
    _existing_po = (
        sess.existing_po_df if getattr(sess, "existing_po_df", None) is not None and not sess.existing_po_df.empty else None
    )

    try:
        po_df = calculate_po_base(
            sales_df=_ads_source,
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
            existing_po_df=_existing_po,
            sku_status_df=sess.sku_status_lead_df if not sess.sku_status_lead_df.empty else None,
            enforce_two_size_minimum=body.enforce_two_size_minimum,
            enforce_lead_time_release_gate=False,
            inventory_history_df=(
                sess.daily_inventory_history_df
                if not sess.daily_inventory_history_df.empty
                else None
            ),
            po_raise_ledger_df=(_ledger if _ledger is not None and not _ledger.empty else None),
            planning_date=body.planning_date,
            raise_ledger_lookback_days=body.raise_ledger_lookback_days,
            raise_view_date=body.raise_view_date,
            po_return_overlay_df=_return_overlay_for_po_calc(sess),
            urgent_all_sizes_days=body.urgent_all_sizes_days,
            use_ly_fallback=body.use_ly_fallback,
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

    from ..services.po_raise_ledger import summarize_raise_ledger_for_dashboard

    ledger_summary = summarize_raise_ledger_for_dashboard(
        _ledger,
        lookback_days=max(body.raise_ledger_lookback_days, 30),
        planning_date=body.planning_date,
        max_skus_per_day=body.max_rows_per_section,
    )
    payload["raised_ledger_summary"] = ledger_summary
    payload["raised_ledger_active"] = ledger_summary.get("active_by_sku") or []
    payload["raised_ledger_skus"] = ledger_summary.get("total_skus", 0)
    payload["raised_ledger_units"] = ledger_summary.get("total_units", 0)
    payload["raised_ledger_daily"] = ledger_summary.get("daily_totals") or []
    payload["raised_ledger_by_day"] = ledger_summary.get("by_day") or {}
    return payload


def _po_calculate_status_payload(job: dict, *, sid: str = "") -> dict:
    """Build status JSON from job store row (or session fallback fields)."""
    st = str(job.get("status") or "idle")
    msg = str(job.get("message") or "")
    progress = max(0, min(100, int(job.get("progress") or 0)))
    out: dict = {"status": st, "message": msg, "progress": progress}
    if job.get("job_id"):
        out["job_id"] = str(job["job_id"])
    if job.get("from_shared_cache"):
        out["from_shared_cache"] = True
    if st == "error":
        out["ok"] = bool(job.get("ok", False))
    elif st == "done":
        out["ok"] = bool(job.get("ok", True))
        if job.get("total_rows") is not None:
            out["row_count"] = int(job["total_rows"])
        if job.get("columns"):
            out["columns"] = list(job["columns"])
    else:
        out["ok"] = True
    if sid and st in ("running", "done", "error"):
        try:
            from ..services.po_calculate_jobs import get_latest_job_id, set_po_job

            job_id = str(job.get("job_id") or get_latest_job_id(sid) or "")
            if job_id:
                set_po_job(job_id, **{k: v for k, v in job.items() if k != "updated_at"})
        except Exception:
            pass
    return out


def _require_po_job(request: Request, job_id: str) -> dict:
    from fastapi import HTTPException

    from ..services.po_calculate_jobs import get_po_job_by_id

    sid = getattr(request.state, "session_id", None) or ""
    job = get_po_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="PO job not found.")
    if str(job.get("session_id") or "") != sid:
        raise HTTPException(status_code=403, detail="PO job not accessible.")
    return job


def _fail_stale_po_job_if_needed(sess, sid: str) -> dict | None:
    """Mark long-idle running jobs as failed so the UI can retry."""
    if not sid:
        return None
    from ..services.po_calculate_jobs import get_latest_job_id, get_po_job_by_id, po_job_is_stale, set_po_job

    job_id = get_latest_job_id(sid)
    if not job_id or not po_job_is_stale(job_id):
        return None
    msg = (
        "PO calculation stopped responding (server may have restarted or run out of memory). "
        "Refresh the page and click Calculate PO again."
    )
    if sess is not None:
        sess.po_calculate_status = "error"
        sess.po_calculate_progress = 0
        sess.po_calculate_message = msg
        sess.po_calculate_result = {"ok": False, "message": msg}
    set_po_job(job_id, status="error", ok=False, progress=0, message=msg)
    return _po_calculate_status_payload(get_po_job_by_id(job_id), sid=sid)


@router.get("/calculate/status/{job_id}")
def po_calculate_status_by_job(request: Request, job_id: str):
    """Poll a specific PO calculate job."""
    from ..services.po_calculate_jobs import get_po_job_by_id

    sess = getattr(request.state, "session", None)
    sid = getattr(request.state, "session_id", None) or ""
    _require_po_job(request, job_id)
    stale = _fail_stale_po_job_if_needed(sess, sid)
    if stale and stale.get("job_id") == job_id:
        return stale
    job = get_po_job_by_id(job_id)
    return _po_calculate_status_payload(job, sid=sid)


@router.get("/calculate/status")
def po_calculate_status(request: Request):
    """Lightweight poll — in-memory job store; session fallback if job row was lost."""
    from ..services.po_calculate_jobs import get_po_job

    sid = getattr(request.state, "session_id", None) or ""
    sess = getattr(request.state, "session", None)
    stale = _fail_stale_po_job_if_needed(sess, sid)
    if stale:
        return stale
    job = get_po_job(sid)
    if job:
        return _po_calculate_status_payload(job, sid=sid)

    if sess is not None:
        st = getattr(sess, "po_calculate_status", "idle") or "idle"
        if st and st != "idle":
            meta = getattr(sess, "po_calculate_result", None) or {}
            fallback = {
                "status": st,
                "message": getattr(sess, "po_calculate_message", "") or "",
                "progress": int(getattr(sess, "po_calculate_progress", 0) or 0),
                "ok": st != "error",
            }
            if st == "done":
                if meta.get("total_rows") is not None:
                    fallback["total_rows"] = int(meta["total_rows"])
                if meta.get("columns"):
                    fallback["columns"] = list(meta["columns"])
            elif st == "error":
                fallback["ok"] = False
            return _po_calculate_status_payload(fallback, sid=sid)

    return {"status": "idle", "message": "", "progress": 0, "ok": True}


def _po_calculate_result_response(
    request: Request,
    *,
    offset: int,
    limit: int,
    compact: int,
):
    from ..services.po_calculate_result_api import build_result_page, default_page_size

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    st = getattr(sess, "po_calculate_status", "idle") or "idle"
    if st == "running":
        return {"ok": False, "status": "running", "message": "PO calculation still running."}
    if st == "error":
        result = getattr(sess, "po_calculate_result", None) or {}
        return {"ok": False, "status": "error", "message": result.get("message") or "PO calculation failed."}
    if st != "done":
        return {"ok": False, "status": st, "message": "No PO result yet — run Calculate PO first."}
    from ..services.existing_po import existing_po_needs_recalc
    from ..services.po_shared_cache import PO_MERGE_LOGIC_VERSION, po_merge_result_is_stale

    if existing_po_needs_recalc(sess):
        return {
            "ok": False,
            "status": "stale",
            "message": "Existing PO sheet was updated. Click Calculate PO to refresh per-size pipeline rows.",
        }
    meta = getattr(sess, "po_calculate_result", None) or {}
    if po_merge_result_is_stale(meta):
        return {
            "ok": False,
            "status": "stale",
            "message": (
                f"PO engine updated to v{PO_MERGE_LOGIC_VERSION}. "
                "Click Calculate PO to refresh recommendations."
            ),
            "po_merge_version": PO_MERGE_LOGIC_VERSION,
        }
    if not meta:
        return {"ok": False, "message": "PO result missing."}
    from ..services.tier3_session_merge import refresh_po_sales_through_meta

    meta = refresh_po_sales_through_meta(sess, meta)
    sid = getattr(request.state, "session_id", None)
    po_df = getattr(sess, "po_calculate_result_df", None)
    lim = int(limit) if int(limit or 0) > 0 else default_page_size()
    use_compact = str(compact).strip().lower() not in ("0", "false", "no", "off")
    return build_result_page(
        session_id=sid,
        po_df=po_df,
        meta=meta,
        offset=offset,
        limit=lim,
        compact=use_compact,
    )


@router.get("/calculate/result/{job_id}")
def po_calculate_result_by_job(
    request: Request,
    job_id: str,
    offset: int = 0,
    limit: int = 0,
    compact: int = 1,
):
    """PO table page for a finished job (paginated)."""
    job = _require_po_job(request, job_id)
    st = str(job.get("status") or "idle")
    if st == "running":
        return {"ok": False, "status": "running", "message": "PO calculation still running.", "job_id": job_id}
    if st == "error":
        return {
            "ok": False,
            "status": "error",
            "message": job.get("message") or "PO calculation failed.",
            "job_id": job_id,
        }
    if st != "done":
        return {"ok": False, "status": st, "message": "No PO result yet.", "job_id": job_id}
    return _po_calculate_result_response(request, offset=offset, limit=limit, compact=compact)


@router.get("/calculate/result")
def po_calculate_result(
    request: Request,
    offset: int = 0,
    limit: int = 0,
    compact: int = 1,
):
    """PO table page after ``status`` is ``done`` (paginated; compact matrix JSON by default)."""
    return _po_calculate_result_response(request, offset=offset, limit=limit, compact=compact)


@router.get("/calculate/shared-cache")
def po_shared_cache_info(request: Request):
    """Whether a shared PO run exists for this planning date and settings (no copy yet)."""
    sess = request.state.session
    if sess is None:
        return {"available": False}
    from ..services.po_shared_cache import _CALC_PARAM_KEYS, shared_cache_availability

    qp = request.query_params
    body: dict = {}
    for k in _CALC_PARAM_KEYS:
        if k not in qp:
            continue
        raw = qp.get(k)
        if raw is None:
            continue
        if k in (
            "use_seasonality",
            "group_by_parent",
            "enforce_two_size_minimum",
            "enforce_lead_time_release_gate",
            "auto_import_yesterday_ledger",
            "use_ly_fallback",
        ):
            body[k] = str(raw).strip().lower() in ("1", "true", "yes", "on")
        elif k in ("period_days", "lead_time", "target_days", "min_denominator", "grace_days", "urgent_all_sizes_days", "raise_ledger_lookback_days"):
            try:
                body[k] = int(raw)
            except (TypeError, ValueError):
                pass
        elif k in ("seasonal_weight", "safety_pct"):
            try:
                body[k] = float(raw)
            except (TypeError, ValueError):
                pass
        else:
            body[k] = raw
    if "planning_date" in qp:
        body["planning_date"] = qp.get("planning_date")
    return shared_cache_availability(sess, body)


@router.post("/calculate")
async def po_calculate(request: Request, body: PORequest, background_tasks: BackgroundTasks):
    """Enqueue PO calculation — returns immediately with a job_id (no gateway timeout)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}

    sid = getattr(request.state, "session_id", None)
    if not sid:
        return {"ok": False, "message": "No session id."}

    stale = _fail_stale_po_job_if_needed(sess, sid)
    if not stale:
        from ..services.po_calculate_jobs import get_latest_job_id, get_po_job_by_id

        latest_id = get_latest_job_id(sid)
        if latest_id:
            latest = get_po_job_by_id(latest_id)
            if str(latest.get("status") or "") == "running":
                return {"ok": True, "job_id": latest_id}

    body_dict = body.model_dump()
    sess.po_calculate_status = "running"
    sess.po_calculate_progress = 0
    sess.po_calculate_message = "Queued…"
    sess.po_calculate_result = {}
    sess.po_calculate_result_df = pd.DataFrame()

    import asyncio

    from ..concurrency import PO_CALC_EXECUTOR
    from ..services.po_calculate_jobs import create_po_job
    from ..services.po_calculate_run import background_po_calculate

    job_id = create_po_job(
        sid,
        status="queued",
        ok=True,
        progress=0,
        message="Queued…",
    )

    if body_dict.get("use_shared_cache", True):
        from ..services.po_shared_cache import apply_shared_cache_to_session

        cached = apply_shared_cache_to_session(sess, sid, body_dict, job_id=job_id)
        if cached:
            return {"ok": True, "job_id": job_id, "from_shared_cache": True}

    asyncio.get_running_loop().run_in_executor(
        PO_CALC_EXECUTOR,
        background_po_calculate,
        job_id,
        sid,
        body_dict,
    )
    return {"ok": True, "job_id": job_id}


@router.get("/quarterly-debug")
def po_quarterly_debug(request: Request):
    """Diagnostic endpoint — clears quarterly cache, recomputes, returns sample."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "reason": "No session"}

    # Also report & clear the live cache so next Calculate PO gets fresh data
    from ..services.po_quarterly_warmup import quarterly_cache_key

    cache_key = quarterly_cache_key(False, 8)
    cached = sess._quarterly_cache.get(cache_key)
    cached_rows = len(cached.get("rows", [])) if cached else 0
    cached_sample_sku = cached["rows"][0].get("OMS_SKU") if cached and cached.get("rows") else None
    sess._quarterly_cache.clear()  # force fresh on next po/quarterly call

    from ..services.po_quarterly_warmup import build_quarterly_payload

    payload = build_quarterly_payload(sess, group_by_parent=False, n_quarters=8)
    import pandas as pd

    pivot = (
        pd.DataFrame(payload.get("rows") or [])
        if payload.get("loaded")
        else pd.DataFrame()
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

    from ..services.po_quarterly_cache import (
        get_shared_quarterly,
        quarterly_build_status,
    )
    from ..services.po_quarterly_jobs import (
        get_quarterly_job,
        start_quarterly_background,
    )
    from ..services.po_quarterly_warmup import (
        quarterly_cache_key,
        try_build_quarterly_payload_sync,
        normalize_quarterly_payload,
    )

    sid = getattr(request.state, "session_id", None) or ""
    cache_key = quarterly_cache_key(group_by_parent, n_quarters)

    shared = get_shared_quarterly(cache_key)
    if shared and shared.get("loaded") and shared.get("rows"):
        shared = normalize_quarterly_payload(shared, n_quarters=n_quarters)
        sess._quarterly_cache[cache_key] = shared
        return shared

    cached = sess._quarterly_cache.get(cache_key)
    if cached and cached.get("loaded") and cached.get("rows"):
        cached = normalize_quarterly_payload(cached, n_quarters=n_quarters)
        return cached

    build_st = quarterly_build_status()
    job = get_quarterly_job(sid)
    if build_st.get("building") or job.get("status") == "running":
        pct = int(build_st.get("progress") or job.get("progress") or 10)
        msg = str(
            build_st.get("message")
            or job.get("message")
            or "Loading quarterly history…"
        )
        return {
            "loaded": False,
            "status": "warming",
            "progress": pct,
            "message": msg,
        }
    if job.get("status") == "ready":
        ready = job.get("result")
        if isinstance(ready, dict) and ready.get("loaded"):
            ready = normalize_quarterly_payload(ready, n_quarters=n_quarters)
            sess._quarterly_cache[cache_key] = ready
            return ready
    if job.get("status") == "error":
        return {
            "loaded": False,
            "status": "error",
            "message": job.get("message") or "Quarterly build failed",
        }

    result = try_build_quarterly_payload_sync(
        sess,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
    )
    if result and result.get("loaded") and result.get("rows"):
        result = normalize_quarterly_payload(result, n_quarters=n_quarters)
        sess._quarterly_cache[cache_key] = result
        return result

    if sid:
        start_quarterly_background(
            sid,
            group_by_parent=group_by_parent,
            n_quarters=n_quarters,
        )
    return {
        "loaded": False,
        "status": "warming",
        "progress": 8,
        "message": "Building quarterly history (first load may take 1–3 minutes)…",
    }
