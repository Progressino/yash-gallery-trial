"""Production Module — Dynamic Routing, Multi-process JO"""
import os, sqlite3
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from ..db.production_db import (
    list_jos, get_jo, create_jo, update_jo,
    issue_fabric, return_fabric, issue_pieces, receive_pieces, add_cost,
    create_next_process_jo, validate_jo_creation,
    get_process_stock, get_all_process_stocks, get_ready_to_process,
    get_item_routing, get_next_process, get_all_routing_steps,
    get_process_report, get_production_stats,
    save_mrp_result, get_last_mrp_result,
    soft_reserve_all, release_so_reservations,
    list_soft_reservations_v2, get_soft_reserved_by_material,
    list_reservations, create_reservation, release_reservation, get_reserved_qty,
)
from ..db.sales_db import get_open_orders, list_orders

router = APIRouter()

_ITEM_DB_PATH = os.environ.get("ITEM_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "items_dev.db"))

def _item_connect():
    path = _ITEM_DB_PATH
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "items_dev.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def _get_item_by_code(conn, code):
    row = conn.execute(
        "SELECT i.*, t.name AS item_type_name FROM items i JOIN item_types t ON t.id=i.item_type_id WHERE i.item_code=?", (code,)
    ).fetchone()
    return dict(row) if row else None

def _get_item_by_id(conn, iid):
    row = conn.execute(
        "SELECT i.*, t.name AS item_type_name FROM items i JOIN item_types t ON t.id=i.item_type_id WHERE i.id=?", (iid,)
    ).fetchone()
    return dict(row) if row else None

def _get_default_bom(conn, item_id):
    row = conn.execute("SELECT * FROM bom_headers WHERE item_id=? AND is_default=1 LIMIT 1", (item_id,)).fetchone()
    if not row:
        row = conn.execute("SELECT * FROM bom_headers WHERE item_id=? LIMIT 1", (item_id,)).fetchone()
    return dict(row) if row else None

def _get_bom_lines(conn, bom_id):
    return [dict(r) for r in conn.execute("SELECT * FROM bom_lines WHERE bom_id=?", (bom_id,)).fetchall()]


# ── MRP Engine ─────────────────────────────────────────────────────────────────

def calculate_mrp(so_numbers):
    open_orders = get_open_orders()
    selected = [l for l in open_orders if l['so_number'] in so_numbers]
    result = {}
    try:
        conn = _item_connect()
    except:
        return result

    def explode(item_code, qty, so_no, sku, depth=0):
        if depth > 10 or qty <= 0: return
        item = _get_item_by_code(conn, item_code)
        if not item: return
        bom = _get_default_bom(conn, item['id'])
        if not bom: return
        for line in _get_bom_lines(conn, bom['id']):
            ctype = (line.get('component_type') or 'RM').upper()
            if ctype in ('SVC','SERVICE','PROCESS'): continue
            comp = _get_item_by_id(conn, line['component_item_id']) if line.get('component_item_id') else None
            if comp:
                code = comp['item_code']
            else:
                raw = line.get('component_name') or ''
                code = raw.split(' — ')[0].strip() if ' — ' in raw else raw
                comp = _get_item_by_code(conn, code) if code else None
            if not code: continue
            adj_qty = float(line.get('quantity') or 0) * (1 + float(line.get('shrinkage_pct') or 0)/100 + float(line.get('wastage_pct') or 0)/100)
            total = round(adj_qty * qty, 3)
            if code not in result:
                result[code] = {'name': comp.get('item_name', code) if comp else code, 'type': ctype,
                                'unit': line.get('unit','PCS'), 'total_req': 0., 'stock': float(comp.get('stock') or 0) if comp else 0.,
                                'reserved': 0., 'breakdown': [], 'level': depth}
            result[code]['total_req'] = round(result[code]['total_req'] + total, 3)
            result[code]['breakdown'].append({'so_no': so_no, 'sku': sku, 'qty_req': total})
            if comp:
                sub = _get_default_bom(conn, comp['id'])
                if sub and [l for l in _get_bom_lines(conn, sub['id']) if (l.get('component_type') or 'RM').upper() not in ('SVC','SERVICE','PROCESS')]:
                    explode(comp['item_code'], total, so_no, sku, depth+1)

    for line in selected:
        sku = line.get('sku','')
        qty = (line.get('qty') or 0) - (line.get('produced_qty') or 0)
        if qty <= 0 or not sku: continue
        item = _get_item_by_code(conn, sku)
        bom_code = sku
        if item and item.get('parent_id'):
            parent = _get_item_by_id(conn, item['parent_id'])
            if parent: bom_code = parent['item_code']
        explode(bom_code, qty, line.get('so_number',''), sku)

    conn.close()
    for code, mat in result.items():
        soft = get_soft_reserved_by_material(code)
        avail = max(0., mat['stock'] - mat['reserved'])
        mat['available'] = avail
        mat['soft_reserved'] = soft
        mat['net_available'] = max(0., avail - soft)
        mat['net_req'] = max(0., round(mat['total_req'] - avail, 3))
        mat['net_req_with_soft'] = max(0., round(mat['total_req'] - mat['net_available'], 3))
    return result


# ── Pydantic Models ────────────────────────────────────────────────────────────

class JOLineIn(BaseModel):
    so_number: Optional[str] = ''
    sku: Optional[str] = ''
    sku_name: Optional[str] = ''
    style: Optional[str] = ''
    planned_qty: int = 0
    vendor_rate: Optional[float] = 0
    remarks: Optional[str] = ''

class JOIn(BaseModel):
    jo_date: Optional[str] = None
    so_number: Optional[str] = ''
    sku: Optional[str] = ''
    sku_name: Optional[str] = ''
    process: Optional[str] = 'Cutting'
    exec_type: Optional[str] = 'Inhouse'
    vendor_name: Optional[str] = ''
    vendor_rate: Optional[float] = 0
    so_qty: Optional[int] = 0
    planned_qty: Optional[int] = 0
    expected_completion: Optional[str] = ''
    issued_to: Optional[str] = ''
    remarks: Optional[str] = ''
    fabric_code: Optional[str] = ''
    fabric_qty: Optional[float] = 0
    fabric_unit: Optional[str] = 'MTR'
    lines: List[JOLineIn] = []

class JOUpdate(BaseModel):
    status: Optional[str] = None
    output_qty: Optional[int] = None
    received_qty: Optional[int] = None
    rejected_qty: Optional[int] = None
    balance_qty: Optional[int] = None
    completed_date: Optional[str] = None
    remarks: Optional[str] = None
    issued_to: Optional[str] = None
    exec_type: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_rate: Optional[float] = None
    fabric_issued_qty: Optional[float] = None
    fabric_received_qty: Optional[float] = None
    fabric_consumption: Optional[float] = None
    process_cost: Optional[float] = None
    total_cost: Optional[float] = None

class FabricIssueIn(BaseModel):
    fabric_code: str
    fabric_name: Optional[str] = ''
    issued_qty: float = 0
    unit: Optional[str] = 'MTR'
    jo_line_id: Optional[int] = None
    issue_date: Optional[str] = None
    issued_by: Optional[str] = ''
    remarks: Optional[str] = ''

class FabricReturnIn(BaseModel):
    fabric_code: str
    returned_qty: float = 0
    unit: Optional[str] = 'MTR'
    return_date: Optional[str] = None
    returned_by: Optional[str] = ''
    remarks: Optional[str] = ''

class PieceIssueIn(BaseModel):
    issued_qty: int = 0
    from_process: Optional[str] = None
    to_process: Optional[str] = None
    sku: Optional[str] = ''
    jo_line_id: Optional[int] = None
    issue_date: Optional[str] = None
    issued_by: Optional[str] = ''
    remarks: Optional[str] = ''

class PieceReceiptIn(BaseModel):
    received_qty: int = 0
    rejected_qty: Optional[int] = 0
    process: Optional[str] = None
    sku: Optional[str] = ''
    jo_line_id: Optional[int] = None
    receipt_date: Optional[str] = None
    received_by: Optional[str] = ''
    remarks: Optional[str] = ''

class CostEntryIn(BaseModel):
    process: str = 'Cutting'
    cost_type: Optional[str] = 'Labour'
    amount: float = 0
    description: Optional[str] = ''
    cost_date: Optional[str] = None

class ReservationIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    reserved_qty: float = 0
    unit: Optional[str] = 'PCS'
    against_so: Optional[str] = ''
    remarks: Optional[str] = ''

class MRPRunBody(BaseModel):
    so_numbers: List[str]


# ── Stats & Meta ───────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats():
    return get_production_stats()

@router.get("/processes")
def get_processes():
    """All available routing steps."""
    return get_all_routing_steps()

@router.get("/item-routing/{sku}")
def get_routing(sku: str):
    """Get process routing for a specific SKU."""
    return {'sku': sku, 'routing': get_item_routing(sku)}


# ── Ready to Process ───────────────────────────────────────────────────────────

@router.get("/ready-to-process/{process}")
def ready_to_process(process: str):
    """Get lines ready to be processed at given stage."""
    return get_ready_to_process(process)


# ── Process Stock ──────────────────────────────────────────────────────────────

@router.get("/process-stock")
def process_stock_list(so_number: Optional[str] = None, sku: Optional[str] = None):
    if so_number and sku:
        return get_all_process_stocks(so_number, sku)
    return []


# ── Process Report ─────────────────────────────────────────────────────────────

@router.get("/process-report")
def process_report():
    return get_process_report()


# ── Job Orders ─────────────────────────────────────────────────────────────────

@router.get("/orders")
def get_jos(status: Optional[str] = None, so_number: Optional[str] = None, process: Optional[str] = None):
    return list_jos(status, so_number, process)

@router.get("/orders/validate")
def validate_jo(process: str, so_number: str, sku: str, planned_qty: int = 0):
    result = validate_jo_creation(process, so_number, sku, planned_qty)
    stocks = get_all_process_stocks(so_number, sku) if so_number and sku else {}
    return {**result, 'process_stocks': stocks}

@router.get("/orders/{joid}")
def get_jo_detail(joid: int):
    jo = get_jo(joid)
    if not jo:
        raise HTTPException(404, "Job order not found")
    return jo

@router.post("/orders")
def post_jo(body: JOIn):
    data = body.model_dump()
    process = data.get('process','Cutting')
    so_number = data.get('so_number','')
    sku = data.get('sku','')
    planned_qty = int(data.get('planned_qty') or 0)
    if process != 'Cutting' and so_number and sku and planned_qty:
        v = validate_jo_creation(process, so_number, sku, planned_qty)
        if not v['ok']:
            raise HTTPException(400, v['message'])
    num = create_jo(data)
    return {"jo_number": num, "ok": True}

@router.patch("/orders/{joid}")
def patch_jo(joid: int, body: JOUpdate):
    update_jo(joid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}

@router.post("/orders/{joid}/issue-fabric")
def post_issue_fabric(joid: int, body: FabricIssueIn):
    try:
        issue_fabric(joid, body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}

@router.post("/orders/{joid}/return-fabric")
def post_return_fabric(joid: int, body: FabricReturnIn):
    return_fabric(joid, body.model_dump())
    return {"ok": True}

@router.post("/orders/{joid}/issue-pieces")
def post_issue_pieces(joid: int, body: PieceIssueIn):
    try:
        issue_pieces(joid, body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}

@router.post("/orders/{joid}/receive-pieces")
def post_receive_pieces(joid: int, body: PieceReceiptIn):
    try:
        receive_pieces(joid, body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}

@router.post("/orders/{joid}/add-cost")
def post_add_cost(joid: int, body: CostEntryIn):
    add_cost(joid, body.model_dump())
    return {"ok": True}

@router.post("/orders/{joid}/next-process")
def post_next_process(joid: int):
    result = create_next_process_jo(joid)
    if not result.get('ok'):
        raise HTTPException(400, result.get('message','Cannot create next process JO'))
    return result


# ── BOM Inputs ────────────────────────────────────────────────────────────────

@router.get("/bom-inputs/{item_code}")
def get_bom_inputs(item_code: str, qty: float = 1.0):
    try:
        conn = _item_connect()
        item = _get_item_by_code(conn, item_code)
        if not item:
            conn.close()
            return {'inputs': []}
        bom = _get_default_bom(conn, item['id'])
        if not bom:
            conn.close()
            return {'inputs': []}
        inputs = []
        for ln in _get_bom_lines(conn, bom['id']):
            ctype = (ln.get('component_type') or 'RM').upper()
            if ctype in ('SVC','SERVICE','PROCESS'): continue
            comp = _get_item_by_id(conn, ln['component_item_id']) if ln.get('component_item_id') else None
            if not comp:
                raw = ln.get('component_name') or ''
                code = raw.split(' — ')[0].strip() if ' — ' in raw else raw
                comp = _get_item_by_code(conn, code) if code else None
            bom_qty = float(ln.get('quantity') or 0)
            adj = round(bom_qty * (1 + float(ln.get('shrinkage_pct') or 0)/100 + float(ln.get('wastage_pct') or 0)/100) * qty, 3)
            inputs.append({'material_code': comp['item_code'] if comp else '', 'material_name': comp['item_name'] if comp else '',
                           'material_type': ctype, 'bom_qty': bom_qty, 'adj_qty': adj, 'unit': ln.get('unit','MTR')})
        conn.close()
        return {'item_code': item_code, 'item_name': item['item_name'], 'inputs': inputs}
    except Exception as ex:
        return {'inputs': [], 'error': str(ex)}


# ── MRP ────────────────────────────────────────────────────────────────────────

@router.get("/mrp/open-sos")
def get_open_sos():
    all_orders = list_orders()
    open_orders = [o for o in all_orders if o.get('status') not in ('Closed','Cancelled')]
    result = []
    for so in open_orders:
        lines = so.get('lines', [])
        result.append({
            'so_number': so['so_number'], 'so_date': so.get('so_date',''),
            'buyer': so.get('buyer',''), 'delivery_date': so.get('delivery_date',''),
            'status': so.get('status',''),
            'total_qty': sum(l.get('qty',0) or 0 for l in lines),
            'pending_qty': sum(max(0,(l.get('qty') or 0)-(l.get('produced_qty') or 0)) for l in lines),
            'line_count': len(lines),
            'skus': list({l.get('sku','') for l in lines if l.get('sku')}),
        })
    return result

@router.post("/mrp/run")
def run_mrp_full(body: MRPRunBody):
    if not body.so_numbers:
        return {'run_time': datetime.now().isoformat(), 'so_numbers': [], 'result': {}}
    result = calculate_mrp(body.so_numbers)
    save_mrp_result(body.so_numbers, result)
    return {'run_time': datetime.now().isoformat(), 'so_numbers': body.so_numbers, 'result': result}

@router.get("/mrp/last")
def get_last_mrp():
    data = get_last_mrp_result()
    return data or {'run_time': None, 'so_numbers': [], 'result': {}}

@router.get("/mrp/lines-for-so")
def get_mrp_lines_for_so(so_number: str = ''):
    data = get_last_mrp_result()
    if not data:
        return {'purchase_items': [], 'sfg_items': [], 'error': 'No MRP result. Run MRP first.'}
    result = data.get('result', {})
    so_numbers = data.get('so_numbers', [])
    if so_number and so_number not in so_numbers:
        return {'purchase_items': [], 'sfg_items': [], 'warning': f'{so_number} not in last MRP run'}
    purchase_items, sfg_items = [], []
    for code, mat in result.items():
        so_qty = (sum(bd['qty_req'] for bd in mat.get('breakdown',[]) if bd.get('so_no')==so_number) if so_number else mat['total_req'])
        if so_qty <= 0: continue
        item_data = {'material_code': code, 'material_name': mat['name'], 'material_type': mat.get('type','RM'),
                     'required_qty': round(so_qty,3), 'unit': mat['unit'], 'net_req': mat.get('net_req_with_soft', so_qty)}
        if mat.get('type','').upper() == 'SFG':
            sfg_items.append(item_data)
        else:
            purchase_items.append(item_data)
    return {'so_number': so_number, 'purchase_items': sorted(purchase_items, key=lambda x: x['material_type']), 'sfg_items': sfg_items}

@router.post("/mrp/soft-reserve-all")
def mrp_soft_reserve_all():
    data = get_last_mrp_result()
    if not data:
        return {'ok': False, 'message': 'No MRP result. Run MRP first.'}
    reservations = []
    for mat_code, mat in data.get('result',{}).items():
        for bd in mat.get('breakdown',[]):
            reservations.append({'material_code': mat_code, 'material_name': mat.get('name',mat_code),
                                 'unit': mat.get('unit','PCS'), 'so_no': bd['so_no'],
                                 'sku': bd.get('sku',''), 'qty': bd.get('qty_req',0)})
    soft_reserve_all(reservations)
    return {'ok': True, 'reserved': len(reservations)}

@router.delete("/mrp/soft-reservations/{so_no}")
def release_so_mrp_reservations(so_no: str):
    release_so_reservations(so_no)
    return {'ok': True}

@router.get("/mrp/soft-reservations")
def get_mrp_soft_reservations():
    return list_soft_reservations_v2()

@router.get("/mrp")
def run_mrp_legacy(so_number: Optional[str] = None):
    open_lines = get_open_orders()
    if so_number:
        open_lines = [l for l in open_lines if l['so_number'] == so_number]
    material_req = {}
    for line in open_lines:
        sku = line.get('sku','')
        qty = (line.get('qty',0) or 0) - (line.get('produced_qty',0) or 0)
        if qty <= 0 or not sku: continue
        if sku not in material_req:
            material_req[sku] = {'material_code': sku, 'material_name': sku, 'unit': 'PCS', 'required_qty': 0, 'so_refs': []}
        material_req[sku]['required_qty'] += qty
        so_ref = line.get('so_number','')
        if so_ref not in material_req[sku]['so_refs']:
            material_req[sku]['so_refs'].append(so_ref)
    result = []
    for mat in material_req.values():
        mat['reserved_qty'] = get_reserved_qty(mat['material_code'])
        mat['net_requirement'] = max(0, mat['required_qty'] - mat['reserved_qty'])
        result.append(mat)
    return sorted(result, key=lambda x: -x['net_requirement'])

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
