"""
PO Engine router.
POST /api/po/calculate  → run PO calculation, return table
GET  /api/po/quarterly  → quarterly history pivot
"""
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class PORequest(BaseModel):
    period_days:      int   = 90
    lead_time:        int   = 30
    target_days:      int   = 60
    demand_basis:     str   = "Sold"       # "Sold" or "Net"
    use_seasonality:  bool  = False
    seasonal_weight:  float = 0.5
    group_by_parent:  bool  = False
    min_denominator:  int   = 7
    grace_days:       int   = 7
    safety_pct:       float = 20.0


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
            mtr_df=sess.mtr_df if not sess.mtr_df.empty else None,
            myntra_df=sess.myntra_df if not sess.myntra_df.empty else None,
            sku_mapping=sess.sku_mapping or None,
            group_by_parent=body.group_by_parent,
            existing_po_df=sess.existing_po_df if not sess.existing_po_df.empty else None,
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}

    if po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    # Serialize — convert any period types etc.
    rows = po_df.fillna(0).round(3).to_dict("records")
    return {
        "ok":      True,
        "rows":    rows,
        "columns": list(po_df.columns),
    }


@router.get("/quarterly")
def po_quarterly(request: Request, group_by_parent: bool = False, n_quarters: int = 8):
    sess = request.state.session
    if sess is None:
        return {"loaded": False}

    from ..services.po_engine import calculate_quarterly_history

    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=sess.mtr_df if not sess.mtr_df.empty else None,
        myntra_df=sess.myntra_df if not sess.myntra_df.empty else None,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
    )
    if pivot.empty:
        return {"loaded": False, "rows": []}

    return {
        "loaded":   True,
        "columns":  list(pivot.columns),
        "rows":     pivot.fillna(0).to_dict("records"),
    }
