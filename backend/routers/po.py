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
    # When True: for sheet-resolved lead rows only, block PO while
    # (Tot inv + eff. pipeline) / ADS > Lead_Time_Days.
    # Default False so Post_PO_Cover_Days always reaches target_days.
    enforce_lead_time_release_gate: bool = False
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
    _sync_po_sidecars_to_durable_storage(request, sess, background_tasks)
    return {"ok": True, "rows": int(len(df)), "message": f"Loaded {len(df)} SKU rows (status + lead time) for PO."}


@router.get("/sku-status-lead")
def po_get_sku_status_lead(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}
    try:
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

    from ..concurrency import HEAVY_EXECUTOR
    from ..services.daily_inventory_upload_run import background_daily_inventory_upload

    sess.daily_inventory_upload_status = "running"
    sess.daily_inventory_upload_message = "Upload received. Parsing sheet…"
    sess.daily_inventory_upload_result = {}
    asyncio.get_running_loop().run_in_executor(
        HEAVY_EXECUTOR,
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
def po_get_daily_inventory_history(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "loaded": False}
    try:
        import backend.main as _main

        _main.restore_po_sidecars_from_warm(sess)
    except Exception:
        pass
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

    from ..services.po_raise_import import sync_ledger_to_durable_db

    sync_ledger_to_durable_db(sess, as_dt)
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


@router.post("/returns/import-file")
async def po_returns_import_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    group_by_parent: str = Form("false"),
    replace: str = Form("true"),
):
    """Import return units by SKU (CSV / Excel / RAR / ZIP). Reduces PO qty and feeds Net demand."""
    from ..services.po_return_import import apply_return_overlay_import, parse_return_upload_bytes

    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    raw = await file.read()
    if not raw:
        return {"ok": False, "message": "Empty file."}
    gbp = str(group_by_parent).strip().lower() in ("1", "true", "yes", "on")
    rep = str(replace).strip().lower() in ("1", "true", "yes", "on")
    overlay, err = parse_return_upload_bytes(
        raw,
        file.filename or "",
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=gbp,
    )
    if err:
        return {"ok": False, "message": err}
    out = apply_return_overlay_import(sess, overlay, replace=rep)
    _sync_po_sidecars_to_durable_storage(request, sess, background_tasks)

    session_id = getattr(request.state, "session_id", None) or getattr(sess, "_persist_sid", None)
    base_msg = str(out.get("message") or "").strip()
    try:
        from ..concurrency import DAILY_UPLOAD_EXECUTOR
        from ..routers.upload import _run_returns_import_followup

        if session_id:
            DAILY_UPLOAD_EXECUTOR.submit(_run_returns_import_followup, session_id)
    except Exception:
        logging.getLogger(__name__).exception("queue return import follow-up")

    out["sales_rebuilt"] = False
    out["sales_rebuild"] = "pending"
    out["message"] = (
        f"{base_msg} Dashboard net sales are updating in the background (usually under a minute). "
        "Run Calculate PO on PO Engine to refresh PO qty columns."
    ).strip()
    return out


@router.delete("/returns/overlay")
def po_clear_returns_overlay(request: Request):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    sess.po_return_overlay_df = pd.DataFrame()
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
    from ..services.upload_policy import may_reset_shared_data

    role = _request_role(request)
    if not may_reset_shared_data(role):
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
            raise_view_date=body.raise_view_date,
            po_return_overlay_df=(
                getattr(sess, "po_return_overlay_df", None)
                if getattr(sess, "po_return_overlay_df", None) is not None
                and not getattr(sess, "po_return_overlay_df", pd.DataFrame()).empty
                else None
            ),
            urgent_all_sizes_days=body.urgent_all_sizes_days,
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


@router.get("/calculate/status")
def po_calculate_status(request: Request):
    """Lightweight poll — in-memory job store only (must stay fast during heavy PO math)."""
    from ..services.po_calculate_jobs import get_po_job

    sid = getattr(request.state, "session_id", None) or ""
    job = get_po_job(sid)
    if not job:
        return {"status": "idle", "message": "", "progress": 0, "ok": True}

    st = str(job.get("status") or "idle")
    msg = str(job.get("message") or "")
    progress = max(0, min(100, int(job.get("progress") or 0)))
    out: dict = {"status": st, "message": msg, "progress": progress}
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
    return out


@router.get("/calculate/result")
def po_calculate_result(
    request: Request,
    offset: int = 0,
    limit: int = 0,
    compact: int = 1,
):
    """PO table page after ``status`` is ``done`` (paginated; compact matrix JSON by default)."""
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
    meta = getattr(sess, "po_calculate_result", None) or {}
    if not meta:
        return {"ok": False, "message": "PO result missing."}
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


@router.get("/calculate/shared-cache")
def po_shared_cache_info(
    request: Request,
    planning_date: Optional[str] = None,
    period_days: int = 90,
    lead_time: int = 30,
    target_days: int = 135,
    demand_basis: str = "Sold",
    group_by_parent: bool = False,
    raise_ledger_lookback_days: int = 14,
):
    """Whether a shared PO run exists for this planning date and settings (no copy yet)."""
    sess = request.state.session
    if sess is None:
        return {"available": False}
    from ..services.po_shared_cache import shared_cache_availability

    return shared_cache_availability(
        sess,
        {
            "planning_date": planning_date,
            "period_days": period_days,
            "lead_time": lead_time,
            "target_days": target_days,
            "demand_basis": demand_basis,
            "group_by_parent": group_by_parent,
            "raise_ledger_lookback_days": raise_ledger_lookback_days,
        },
    )


@router.post("/calculate")
async def po_calculate(request: Request, body: PORequest, background_tasks: BackgroundTasks):
    """Start PO calculation in the background (avoids 502 on large catalogs)."""
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    if sess.sales_df.empty:
        return {"ok": False, "message": "Build Sales first (upload platforms, then POST /api/upload/build-sales)."}
    if sess.inventory_df_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    sid = getattr(request.state, "session_id", None)
    if not sid:
        return {"ok": False, "message": "No session id."}

    body_dict = body.model_dump()
    if body.use_shared_cache:
        from ..services.po_shared_cache import apply_shared_cache_to_session

        cached = apply_shared_cache_to_session(sess, sid, body_dict)
        if cached:
            return cached

    if getattr(sess, "po_calculate_status", "idle") == "running":
        return {
            "ok": True,
            "status": "running",
            "message": getattr(sess, "po_calculate_message", "") or "PO calculation already in progress…",
        }

    sess.po_calculate_status = "running"
    sess.po_calculate_progress = 2
    try:
        from ..services.po_result_spill import clear_spill

        clear_spill(sid)
    except Exception:
        pass
    _inv_n = int(len(getattr(sess, "daily_inventory_history_df", pd.DataFrame())))
    if _inv_n > 500_000:
        sess.po_calculate_message = (
            f"Calculating PO (using last {int(body.period_days)} days of "
            f"{_inv_n:,}-row inventory history)…"
        )
    else:
        sess.po_calculate_message = "Calculating PO recommendations…"
    sess.po_calculate_result = {}
    sess.po_calculate_result_df = pd.DataFrame()

    import asyncio

    from ..concurrency import PO_CALC_EXECUTOR
    from ..services.po_calculate_jobs import set_po_job
    from ..services.po_calculate_run import background_po_calculate

    set_po_job(
        sid,
        status="running",
        ok=True,
        progress=2,
        message=sess.po_calculate_message,
    )
    # Fire-and-forget — must not await (large catalogs run several minutes).
    asyncio.get_running_loop().run_in_executor(
        PO_CALC_EXECUTOR,
        background_po_calculate,
        sid,
        body.model_dump(),
    )
    return {
        "ok": True,
        "status": "running",
        "message": "PO calculation started. This may take 1–3 minutes on large catalogs.",
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

    cache_key = (group_by_parent, n_quarters)
    if cache_key in sess._quarterly_cache:
        cached = sess._quarterly_cache[cache_key]
        if cached.get("loaded") and cached.get("rows"):
            return cached

    from ..services.po_quarterly_warmup import warmup_quarterly_cache

    result, _ = warmup_quarterly_cache(
        sess, group_by_parent=group_by_parent, n_quarters=n_quarters
    )
    return result
