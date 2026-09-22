"""Sales Orders & Demand Management router"""
import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
from ..db.sales_db import (
    list_demands, create_demand, update_demand_status, get_demand_by_number,
    list_orders, create_order, update_order, update_so_line, get_open_orders
)
from ..services.sales_import import (
    demand_import_template_csv,
    so_import_template_csv,
    parse_demand_import_rows,
    parse_so_import_rows,
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
    rate: Optional[float] = 0
    delivery_date: Optional[str] = ''
    remarks: Optional[str] = ''
    hsn_code: Optional[str] = ''
    gst_pct: Optional[float] = 0
    merchant_code: Optional[str] = ''
    priority: Optional[str] = 'Normal'
    line_delivery_date: Optional[str] = ''
    procurement_override: Optional[str] = ''

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
    dispatch_date: Optional[str] = ''
    ref_number: Optional[str] = ''
    ref_date: Optional[str] = ''
    production_mode: Optional[str] = 'inhouse'
    lines: List[SOLineIn] = []

class StatusUpdate(BaseModel):
    status: str

class SOLineUpdate(BaseModel):
    produced_qty:  Optional[int]   = None
    dispatch_qty:  Optional[int]   = None
    received_qty:  Optional[int]   = None
    qty:           Optional[int]   = None
    rate:          Optional[float] = None
    delivery_date: Optional[str]   = None
    remarks:       Optional[str]   = None
    procurement_override: Optional[str] = None


def _read_upload_rows(file: UploadFile, raw: bytes) -> list[dict]:
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}") from e
    if df.empty:
        raise HTTPException(400, "Import file is empty")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df.fillna("").to_dict(orient="records")


# ── Demands ───────────────────────────────────────────────────────────────────
@router.get("/demands")
def get_demands(status: Optional[str] = None):
    return list_demands(status)

@router.post("/demands")
def post_demand(body: DemandIn):
    num = create_demand(body.model_dump())
    return {"demand_number": num}

@router.get("/demands/by-number/{demand_number}")
def get_demand_detail(demand_number: str):
    d = get_demand_by_number(demand_number)
    if not d:
        raise HTTPException(status_code=404, detail="Demand not found")
    return d

@router.patch("/demands/{did}/status")
def patch_demand_status(did: int, body: StatusUpdate):
    update_demand_status(did, body.status)
    return {"ok": True}


@router.get("/demands/import-template")
def download_demand_import_template():
    return Response(
        content=demand_import_template_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="demand_lines_import_template.csv"'},
    )


@router.post("/demands/import-lines")
async def import_demand_lines(file: UploadFile = File(...)):
    """Parse Demand SKU lines from CSV/XLSX (does not create the demand — fills the form)."""
    raw = await file.read()
    rows = _read_upload_rows(file, raw)
    lines, errors = parse_demand_import_rows(rows)
    if not lines and errors:
        raise HTTPException(400, "; ".join(errors[:8]))
    return {
        "ok": True,
        "lines": lines,
        "imported": len(lines),
        "errors": errors,
        "message": f"Parsed {len(lines)} SKU line(s)" + (f"; {len(errors)} row warning(s)" if errors else ""),
    }


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


@router.get("/orders/import-template")
def download_so_import_template():
    return Response(
        content=so_import_template_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="sales_order_lines_import_template.csv"'},
    )


@router.post("/orders/import-lines")
async def import_so_lines(file: UploadFile = File(...)):
    """Parse Sales Order SKU lines from CSV/XLSX (does not create the SO — fills the form)."""
    raw = await file.read()
    rows = _read_upload_rows(file, raw)
    lines, errors = parse_so_import_rows(rows)
    if not lines and errors:
        raise HTTPException(400, "; ".join(errors[:8]))
    return {
        "ok": True,
        "lines": lines,
        "imported": len(lines),
        "errors": errors,
        "message": f"Parsed {len(lines)} SKU line(s)" + (f"; {len(errors)} row warning(s)" if errors else ""),
    }
