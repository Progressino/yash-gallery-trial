"""Production Module + MRP router"""
import os
import sqlite3
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from ..db.production_db import (
    list_jos, create_jo, update_jo,
    list_reservations, create_reservation, release_reservation, get_reserved_qty,
    get_production_stats,
    save_mrp_result, get_last_mrp_result,
    soft_reserve_all, release_so_reservations,
    list_soft_reservations_v2, get_soft_reserved_by_material,
)
from ..db.sales_db import get_open_orders, list_orders

router = APIRouter()

# ── Item DB helper ─────────────────────────────────────────────────────────────

_ITEM_DB_PATH = os.environ.get(
    "ITEM_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "items_dev.db")
)


def _item_connect():
    path = _ITEM_DB_PATH
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "items_dev.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _get_item_by_code(conn, code: str):
    row = conn.execute(
        "SELECT i.*, t.name AS item_type_name, t.code AS item_type_code FROM items i "
        "JOIN item_types t ON t.id = i.item_type_id WHERE i.item_code=?", (code,)
    ).fetchone()
    return dict(row) if row else None


def _get_item_by_id(conn, item_id: int):
    row = conn.execute(
        "SELECT i.*, t.name AS item_type_name, t.code AS item_type_code FROM items i "
        "JOIN item_types t ON t.id = i.item_type_id WHERE i.id=?", (item_id,)
    ).fetchone()
    return dict(row) if row else None


def _get_default_bom(conn, item_id: int):
    row = conn.execute(
        "SELECT * FROM bom_headers WHERE item_id=? AND is_default=1 LIMIT 1", (item_id,)
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT * FROM bom_headers WHERE item_id=? LIMIT 1", (item_id,)
        ).fetchone()
    return dict(row) if row else None


def _get_bom_lines(conn, bom_id: int):
    rows = conn.execute("SELECT * FROM bom_lines WHERE bom_id=?", (bom_id,)).fetchall()
    return [dict(r) for r in rows]


# ── Full MRP Engine ────────────────────────────────────────────────────────────

def calculate_mrp(so_numbers: List[str]) -> dict:
    """
    Returns dict: {material_code: {name, type, unit, total_req, stock, reserved,
    available, soft_reserved, net_available, net_req, net_req_with_soft,
    breakdown: [{so_no, sku, qty_req, source}], level}}
    """
    open_orders = get_open_orders()
    selected_lines = [l for l in open_orders if l['so_number'] in so_numbers]

    result = {}

    try:
        conn = _item_connect()
    except Exception:
        return result

    def explode_bom(item_code: str, qty: float, so_no: str, sku: str, depth: int = 0):
        if depth > 10 or qty <= 0:
            return

        item_row = _get_item_by_code(conn, item_code)
        if not item_row:
            return

        bom = _get_default_bom(conn, item_row['id'])
        if not bom:
            return

        lines = _get_bom_lines(conn, bom['id'])
        for line in lines:
            comp_code = line.get('component_name') or str(line.get('component_item_id', ''))
            if not comp_code:
                continue

            line_qty = float(line.get('quantity') or 0)
            shrinkage = float(line.get('shrinkage_pct') or 0) / 100
            wastage = float(line.get('wastage_pct') or 0) / 100
            adj_qty = line_qty * (1 + shrinkage + wastage)
            total_qty = round(adj_qty * qty, 3)
            unit = line.get('unit', 'PCS') or 'PCS'
            comp_type = line.get('component_type', 'RM') or 'RM'

            # Get stock from items table if component_item_id is set
            comp_item = None
            if line.get('component_item_id'):
                comp_item = _get_item_by_id(conn, line['component_item_id'])

            stock = float(comp_item.get('stock') or 0) if comp_item else 0

            if comp_code not in result:
                result[comp_code] = {
                    'name': comp_item.get('item_name', comp_code) if comp_item else comp_code,
                    'type': comp_type,
                    'unit': unit,
                    'total_req': 0.0,
                    'stock': stock,
                    'reserved': 0.0,
                    'breakdown': [],
                    'level': depth,
                }
            else:
                # Update level to shallowest
                if depth < result[comp_code]['level']:
                    result[comp_code]['level'] = depth

            result[comp_code]['total_req'] = round(result[comp_code]['total_req'] + total_qty, 3)
            result[comp_code]['breakdown'].append({
                'so_no': so_no,
                'sku': sku,
                'qty_req': total_qty,
                'source': f'BOM: {item_code} → Level {depth}',
            })

            # Recurse if component also has a BOM (multi-level)
            if comp_item:
                sub_bom = _get_default_bom(conn, comp_item['id'])
                if sub_bom:
                    sub_lines = _get_bom_lines(conn, sub_bom['id'])
                    if sub_lines:
                        explode_bom(comp_code, total_qty, so_no, sku, depth + 1)

    for line in selected_lines:
        sku = line.get('sku', '')
        so_no = line.get('so_number', '')
        qty = (line.get('qty') or 0) - (line.get('produced_qty') or 0)
        if qty <= 0 or not sku:
            continue

        # If item is a size variant, use parent item's BOM
        item = _get_item_by_code(conn, sku)
        bom_item_code = sku
        if item and item.get('parent_id'):
            parent = _get_item_by_id(conn, item['parent_id'])
            if parent:
                bom_item_code = parent['item_code']

        explode_bom(bom_item_code, qty, so_no, sku)

    conn.close()

    # Calculate net requirements with soft reservations
    for code, mat in result.items():
        hard_reserved = mat['reserved']
        soft_reserved = get_soft_reserved_by_material(code)
        available = max(0.0, mat['stock'] - hard_reserved)
        mat['available'] = available
        mat['soft_reserved'] = soft_reserved
        mat['net_available'] = max(0.0, available - soft_reserved)
        mat['net_req'] = max(0.0, round(mat['total_req'] - available, 3))
        mat['net_req_with_soft'] = max(0.0, round(mat['total_req'] - mat['net_available'], 3))

    return result


# ── Pydantic Models ────────────────────────────────────────────────────────────

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

class MRPRunBody(BaseModel):
    so_numbers: List[str]
    filter_buyer: Optional[str] = None


# ── Stats ─────────────────────────────────────────────────────────────────────
@router.get("/stats")
def get_stats():
    return get_production_stats()


# ── MRP Endpoints ─────────────────────────────────────────────────────────────

@router.get("/mrp/open-sos")
def get_open_sos():
    """Return open SOs with buyer, delivery_date, lines summary for the MRP UI."""
    all_orders = list_orders()
    open_orders = [o for o in all_orders if o.get('status') not in ('Closed', 'Cancelled')]
    result = []
    for so in open_orders:
        lines = so.get('lines', [])
        total_qty = sum(l.get('qty', 0) or 0 for l in lines)
        pending_qty = sum(
            max(0, (l.get('qty') or 0) - (l.get('produced_qty') or 0))
            for l in lines
        )
        result.append({
            'so_number': so['so_number'],
            'so_date': so.get('so_date', ''),
            'buyer': so.get('buyer', ''),
            'delivery_date': so.get('delivery_date', ''),
            'status': so.get('status', ''),
            'total_qty': total_qty,
            'pending_qty': pending_qty,
            'line_count': len(lines),
            'skus': list({l.get('sku', '') for l in lines if l.get('sku')}),
        })
    return result


@router.post("/mrp/run")
def run_mrp_full(body: MRPRunBody):
    """Full MRP engine: explode BOMs, calculate net requirements, save result."""
    so_numbers = body.so_numbers
    if not so_numbers:
        return {'run_time': datetime.now().isoformat(), 'so_numbers': [], 'result': {}}

    result = calculate_mrp(so_numbers)
    save_mrp_result(so_numbers, result)
    return {
        'run_time': datetime.now().isoformat(),
        'so_numbers': so_numbers,
        'result': result,
    }


@router.get("/mrp/last")
def get_last_mrp():
    """Returns {run_time, so_numbers, result} from saved DB or 404."""
    data = get_last_mrp_result()
    if not data:
        return {'run_time': None, 'so_numbers': [], 'result': {}}
    return data


@router.post("/mrp/soft-reserve-all")
def mrp_soft_reserve_all():
    """Use last MRP result to create soft reservations for all materials/SOs."""
    data = get_last_mrp_result()
    if not data:
        return {'ok': False, 'message': 'No MRP result found. Run MRP first.'}

    result = data.get('result', {})
    reservations = []
    for mat_code, mat in result.items():
        for bd in mat.get('breakdown', []):
            reservations.append({
                'material_code': mat_code,
                'material_name': mat.get('name', mat_code),
                'unit': mat.get('unit', 'PCS'),
                'so_no': bd['so_no'],
                'sku': bd.get('sku', ''),
                'qty': bd.get('qty_req', 0),
            })

    soft_reserve_all(reservations)
    return {'ok': True, 'reserved': len(reservations)}


@router.delete("/mrp/soft-reservations/{so_no}")
def release_so_mrp_reservations(so_no: str):
    """Release all MRP soft reservations for a given SO."""
    release_so_reservations(so_no)
    return {'ok': True}


@router.get("/mrp/soft-reservations")
def get_mrp_soft_reservations():
    """List all active MRP soft reservations."""
    return list_soft_reservations_v2()


@router.get("/mrp")
def run_mrp_legacy(so_number: Optional[str] = None):
    """Legacy MRP endpoint - basic BOM explosion."""
    open_lines = get_open_orders()
    if so_number:
        open_lines = [l for l in open_lines if l['so_number'] == so_number]

    material_req: dict = {}
    for line in open_lines:
        sku = line.get('sku', '')
        qty_needed = (line.get('qty', 0) or 0) - (line.get('produced_qty', 0) or 0)
        if qty_needed <= 0 or not sku:
            continue
        bom_found = False
        try:
            if os.path.exists(_ITEM_DB_PATH):
                conn = sqlite3.connect(_ITEM_DB_PATH)
                conn.row_factory = sqlite3.Row
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

# ── Soft Reservations (legacy) ─────────────────────────────────────────────────
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
