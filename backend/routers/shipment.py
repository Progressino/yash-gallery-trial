"""
Shipment engine router.
POST /api/shipment/calculate -> recommend shipment qty marketplace-wise.
"""
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class ShipmentRequest(BaseModel):
    marketplace: str = "amazon"
    period_days: int = 30
    lead_time: int = 7
    target_days: int = 14
    demand_basis: str = "Sold"  # Sold or Net
    min_denominator: int = 7
    safety_pct: float = 10.0
    round_to: int = 5
    cap_to_oms_inventory: bool = True


@router.post("/calculate")
def shipment_calculate(request: Request, body: ShipmentRequest):
    sess = request.state.session
    if sess is None:
        return {"ok": False, "message": "No session"}
    if sess.sales_df.empty:
        return {"ok": False, "message": "Build Sales first."}
    if sess.inventory_df_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    from ..services.shipment_engine import calculate_shipment_plan

    try:
        ship_df = calculate_shipment_plan(
            sales_df=sess.sales_df,
            inv_df=sess.inventory_df_variant,
            marketplace=body.marketplace,
            period_days=body.period_days,
            lead_time=body.lead_time,
            target_days=body.target_days,
            min_denominator=body.min_denominator,
            safety_pct=body.safety_pct,
            round_to=body.round_to,
            demand_basis=body.demand_basis,
            cap_to_oms_inventory=body.cap_to_oms_inventory,
        )
    except Exception as e:
        return {"ok": False, "message": f"Shipment calculation error: {e}"}

    if ship_df.empty:
        return {"ok": False, "message": "Shipment result is empty."}

    return {"ok": True, "rows": ship_df.fillna(0).to_dict("records"), "columns": list(ship_df.columns)}
