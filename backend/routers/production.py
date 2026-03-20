"""Production Module + MRP router"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from ..db.production_db import (
    list_jos, create_jo, update_jo,
    list_reservations, create_reservation, release_reservation, get_reserved_qty,
    get_production_stats
)
from ..db.sales_db import get_open_orders

router = APIRouter()

class JOLineIn(BaseModel):
    sku: Optional[str] = ''
    sku_name: Optional[str] = ''
    planned_qty: int = 0
    input_material: Optional[str] = ''
    input_qty: float = 0
    input_unit: Optional[str] = 'PCS'
    remarks: Optional[str] = ''

class JOIn(BaseModel):
    jo_date: Optional[str] = None
    so_number: Optional[str] = ''
    sku: Optional[str] = ''
    sku_name: Optional[str] = ''
    process: Optional[str] = 'Cutting'
    exec_type: Optional[str] = 'Inhouse'
    vendor_name: Optional[str] = ''
    so_qty: Optional[int] = 0
    planned_qty: Optional[int] = 0
    expected_completion: Optional[str] = ''
    issued_to: Optional[str] = ''
    remarks: Optional[str] = ''
    lines: List[JOLineIn] = []

class JOUpdate(BaseModel):
    status: Optional[str] = None
    output_qty: Optional[int] = None
    completed_date: Optional[str] = None
    remarks: Optional[str] = None
    issued_to: Optional[str] = None
    exec_type: Optional[str] = None
    vendor_name: Optional[str] = None

class ReservationIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    reserved_qty: float = 0
    unit: Optional[str] = 'PCS'
    against_so: Optional[str] = ''
    remarks: Optional[str] = ''

# ── Stats ─────────────────────────────────────────────────────────────────────
@router.get("/stats")
def get_stats():
    return get_production_stats()

# ── MRP ───────────────────────────────────────────────────────────────────────
@router.get("/mrp")
def run_mrp(so_number: Optional[str] = None):
    """Run MRP: explode open SO lines through BOMs to get material requirements."""
    open_lines = get_open_orders()
    if so_number:
        open_lines = [l for l in open_lines if l['so_number'] == so_number]

    material_req: dict = {}
    for line in open_lines:
        sku = line.get('sku', '')
        qty_needed = (line.get('qty', 0) or 0) - (line.get('produced_qty', 0) or 0)
        if qty_needed <= 0 or not sku:
            continue
        # Try to get BOM for the item and explode requirements
        bom_found = False
        try:
            import sqlite3, os as _os
            item_db_path = _os.environ.get("ITEM_DB_PATH", "/data/items.db")
            if not _os.path.exists(item_db_path):
                item_db_path = _os.path.join(_os.path.dirname(__file__), '..', 'items_dev.db')
            if os.path.exists(item_db_path):
                conn = sqlite3.connect(item_db_path)
                conn.row_factory = sqlite3.Row
                # Find item by code
                item_row = conn.execute("SELECT id FROM items WHERE item_code=?", (sku,)).fetchone()
                if item_row:
                    bom = conn.execute(
                        "SELECT id FROM bom_headers WHERE item_id=? AND is_default=1 LIMIT 1",
                        (item_row['id'],)
                    ).fetchone()
                    if not bom:
                        bom = conn.execute(
                            "SELECT id FROM bom_headers WHERE item_id=? LIMIT 1",
                            (item_row['id'],)
                        ).fetchone()
                    if bom:
                        lines_rows = conn.execute(
                            "SELECT * FROM bom_lines WHERE bom_id=?", (bom['id'],)
                        ).fetchall()
                        for bline in lines_rows:
                            bline = dict(bline)
                            net_qty = bline['quantity'] * (1 + bline.get('shrinkage_pct', 0) / 100) * (1 + bline.get('wastage_pct', 0) / 100)
                            mat_code = bline.get('component_name') or str(bline.get('component_item_id', 'UNKNOWN'))
                            if mat_code not in material_req:
                                material_req[mat_code] = {
                                    'material_code': mat_code,
                                    'material_name': bline.get('component_name', ''),
                                    'unit': bline.get('unit', 'PCS'),
                                    'required_qty': 0,
                                    'so_refs': [],
                                }
                            material_req[mat_code]['required_qty'] += net_qty * qty_needed
                            so_ref = line.get('so_number', '')
                            if so_ref not in material_req[mat_code]['so_refs']:
                                material_req[mat_code]['so_refs'].append(so_ref)
                            bom_found = True
                conn.close()
        except Exception:
            pass
        # If no BOM found, list the SKU itself as a requirement
        if not bom_found:
            if sku not in material_req:
                material_req[sku] = {
                    'material_code': sku,
                    'material_name': line.get('sku_name', sku),
                    'unit': 'PCS',
                    'required_qty': 0,
                    'so_refs': [],
                }
            material_req[sku]['required_qty'] += qty_needed
            so_ref = line.get('so_number', '')
            if so_ref not in material_req[sku]['so_refs']:
                material_req[sku]['so_refs'].append(so_ref)

    # Add reserved qty info
    result = []
    for mat in material_req.values():
        mat['reserved_qty'] = get_reserved_qty(mat['material_code'])
        mat['net_requirement'] = max(0, mat['required_qty'] - mat['reserved_qty'])
        result.append(mat)

    return sorted(result, key=lambda x: -x['net_requirement'])

# ── Job Orders ────────────────────────────────────────────────────────────────
@router.get("/orders")
def get_jos(status: Optional[str] = None, so_number: Optional[str] = None):
    return list_jos(status, so_number)

@router.post("/orders")
def post_jo(body: JOIn):
    num = create_jo(body.model_dump())
    return {"jo_number": num}

@router.patch("/orders/{joid}")
def patch_jo(joid: int, body: JOUpdate):
    update_jo(joid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}

# ── Soft Reservations ─────────────────────────────────────────────────────────
@router.get("/reservations")
def get_reservations(status: str = 'Active'):
    return list_reservations(status)

@router.post("/reservations")
def post_reservation(body: ReservationIn):
    create_reservation(body.model_dump())
    return {"ok": True}

@router.delete("/reservations/{rid}")
def delete_reservation(rid: int):
    release_reservation(rid)
    return {"ok": True}
