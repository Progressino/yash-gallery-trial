"""
PO Engine router.
POST /api/po/calculate  → run PO calculation, return table
GET  /api/po/quarterly  → quarterly history pivot
POST /api/po/sku-status-lead → upload SKU status & lead time (Excel/CSV) for PO rules
"""
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, Request, UploadFile
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


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

    sku_map = sess.sku_mapping or None
    canon = lambda v: canonical_oms_key(v, sku_map)  # noqa: E731
    target = canon(sku)
    work = df.copy()
    work["OMS_SKU"] = work["OMS_SKU"].astype(str).map(canon)
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
    if parent_used:
        sub = sub.groupby("Date", as_index=False)["Qty"].max()
    else:
        sub = sub.groupby("Date", as_index=False)["Qty"].max()

    sub = sub.sort_values("Date")
    if end_date:
        try:
            end_ts = pd.Timestamp(end_date).normalize()
        except Exception:
            end_ts = sub["Date"].max().normalize()
    else:
        end_ts = sub["Date"].max().normalize()
    start_ts = end_ts - pd.Timedelta(days=max(0, int(window_days) - 1))
    win = sub[(sub["Date"] >= start_ts) & (sub["Date"] <= end_ts)].copy()
    in_stock_days = int((win["Qty"] >= 1.0).sum())

    rows = [
        {"date": str(r["Date"].date()), "qty": float(r["Qty"]), "in_stock": bool(r["Qty"] >= 1.0)}
        for _, r in win.iterrows()
    ]
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
        "in_stock_days": in_stock_days,
        "out_of_stock_days": int(len(rows) - in_stock_days),
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
    return {"ok": True, "message": "Daily inventory history cleared."}


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
            inventory_history_df=(
                sess.daily_inventory_history_df
                if not sess.daily_inventory_history_df.empty
                else None
            ),
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}

    if po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    # Serialize — keep string hints; round numerics only
    import numpy as np
    import pandas as pd

    po_df = po_df.copy()
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
    return {
        "ok":      True,
        "rows":    rows,
        "columns": list(po_df.columns),
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
