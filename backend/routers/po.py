"""
PO Engine router.
POST /api/po/calculate  → run PO calculation, return table
GET  /api/po/quarterly  → quarterly history pivot
POST /api/po/sku-status-lead → upload SKU status & lead time (Excel/CSV) for PO rules
"""
from io import BytesIO

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
    """Upload 'Sku Status & Lead Time' style sheet: SKU, Status, Lead time (days)."""
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
        return {"ok": False, "message": "No valid SKU rows found (need SKU, Status, and Lead time columns)."}
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
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}

    if po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    # Serialize — keep string hints; round numerics only
    import numpy as np

    po_df = po_df.copy()
    for c in ["Suggest_Close_SKU", "PO_Block_Reason", "SKU_Sheet_Status"]:
        if c in po_df.columns:
            po_df[c] = po_df[c].fillna("").astype(str)
    num_cols = po_df.select_dtypes(include=[np.number]).columns
    po_df[num_cols] = po_df[num_cols].fillna(0).round(3)
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
