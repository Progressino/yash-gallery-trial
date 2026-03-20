"""Sales Orders & Demand Management router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from ..db.sales_db import (
    list_demands, create_demand, update_demand_status,
    list_orders, create_order, update_order, update_so_line, get_open_orders
)

router = APIRouter()

class DemandLineIn(BaseModel):
    sku: str
    sku_name: Optional[str] = ''
    demand_qty: int = 0

class DemandIn(BaseModel):
    demand_date: Optional[str] = None
    demand_source: Optional[str] = 'Sales Team'
    buyer: Optional[str] = ''
    priority: Optional[str] = 'Normal'
    status: Optional[str] = 'Draft'
    notes: Optional[str] = ''
    lines: List[DemandLineIn] = []

class SOLineIn(BaseModel):
    sku: str
    sku_name: Optional[str] = ''
    qty: int = 0
    unit: Optional[str] = 'PCS'

class SOIn(BaseModel):
    so_date: Optional[str] = None
    buyer: Optional[str] = ''
    warehouse: Optional[str] = ''
    sales_team: Optional[str] = ''
    source_type: Optional[str] = 'Sales Team Demand'
    ref_demand: Optional[str] = ''
    delivery_date: Optional[str] = ''
    payment_terms: Optional[str] = ''
    status: Optional[str] = 'Draft'
    notes: Optional[str] = ''
    lines: List[SOLineIn] = []

class StatusUpdate(BaseModel):
    status: str

class SOLineUpdate(BaseModel):
    produced_qty: Optional[int] = None
    dispatch_qty: Optional[int] = None
    received_qty: Optional[int] = None

# ── Demands ───────────────────────────────────────────────────────────────────
@router.get("/demands")
def get_demands(status: Optional[str] = None):
    return list_demands(status)

@router.post("/demands")
def post_demand(body: DemandIn):
    num = create_demand(body.model_dump())
    return {"demand_number": num}

@router.patch("/demands/{did}/status")
def patch_demand_status(did: int, body: StatusUpdate):
    update_demand_status(did, body.status)
    return {"ok": True}

# ── Sales Orders ──────────────────────────────────────────────────────────────
@router.get("/orders")
def get_orders(status: Optional[str] = None):
    return list_orders(status)

@router.get("/orders/open")
def get_open():
    return get_open_orders()

@router.post("/orders")
def post_order(body: SOIn):
    num = create_order(body.model_dump())
    return {"so_number": num}

@router.patch("/orders/{soid}")
def patch_order(soid: int, body: dict):
    update_order(soid, body)
    return {"ok": True}

@router.patch("/orders/lines/{lid}")
def patch_so_line(lid: int, body: SOLineUpdate):
    update_so_line(lid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}
