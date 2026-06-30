"""
Item Master & BOM router.

GET  /api/items/meta                              — item types, size groups, routing steps
GET  /api/items                                   — list items
POST /api/items                                   — create item + size variants
GET  /api/items/{id}                              — item detail (with variants + routing)
PUT  /api/items/{id}                              — update item
DELETE /api/items/{id}                            — delete item + cascade

GET  /api/items/routing                           — list routing steps
POST /api/items/routing                           — create routing step
DELETE /api/items/routing/{step_id}               — delete routing step

GET  /api/items/{id}/boms                         — list BOMs for item
POST /api/items/{id}/boms                         — create BOM
GET  /api/items/{id}/boms/{bom_id}               — BOM with lines
PUT  /api/items/{id}/boms/{bom_id}               — update BOM header
DELETE /api/items/{id}/boms/{bom_id}             — delete BOM
POST /api/items/{id}/boms/{bom_id}/lines         — add BOM line
PUT  /api/items/{id}/boms/{bom_id}/lines/{lid}   — update BOM line
DELETE /api/items/{id}/boms/{bom_id}/lines/{lid} — delete BOM line
POST /api/items/{id}/boms/{bom_id}/copy          — copy BOM to another item

POST /api/items/import                            — bulk import from Excel/CSV
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field

from ..services.permissions import may_access_erp_admin
from ..db.item_db import (
    list_item_types, create_item_type,
    list_size_groups, create_size_group,
    list_routing_steps, create_routing_step, delete_routing_step,
    list_merchants, create_merchant, delete_merchant,
    list_buyers, create_buyer, delete_buyer,
    list_items, get_item, create_item, update_item, update_item_stock, delete_item,
    create_size_variants, set_item_routing,
    list_boms, get_bom_with_lines, create_bom, update_bom, delete_bom,
    certify_bom, uncertify_bom,
    add_bom_line, update_bom_line, delete_bom_line, copy_bom,
    list_item_packaging, get_buyer_packaging, add_packaging_line, delete_packaging_line,
    bulk_create_items, get_item_stats, list_all_boms,
    adjust_item_stock, list_stock_adjustments, book_stock_for_code, get_item_by_code,
)
from ..services.item_import import parse_item_import

router = APIRouter()


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ItemTypeCreate(BaseModel):
    name: str
    code: str

class SizeGroupCreate(BaseModel):
    name: str
    sizes: List[str]

class RoutingStepCreate(BaseModel):
    name: str
    description: str = ""
    sort_order: int  = 0

class ItemCreate(BaseModel):
    item_code:          str
    item_name:          str
    item_type_id:       int
    hsn_code:           str   = ""
    season:             str   = ""
    merchant_code:      str   = ""
    selling_price:      float = 0.0
    purchase_price:     float = 0.0
    launch_date:        str   = ""
    uom:                str   = "PCS"
    alias:              str   = ""
    gst_applicability:  str   = "Applicable"
    type_of_supply:     str   = "Goods"
    gst_rate:           float = 0.0
    procurement_type:   str   = ""           # e.g. Purchase, Make — Grey Fabric (GF) typically Purchase
    sizes:              List[str] = []       # triggers size variant generation
    routing_step_ids:   List[int] = []

class ItemUpdate(BaseModel):
    item_code:          Optional[str]   = None
    item_name:          Optional[str]   = None
    item_type_id:       Optional[int]   = None
    hsn_code:           Optional[str]   = None
    season:             Optional[str]   = None
    merchant_code:      Optional[str]   = None
    selling_price:      Optional[float] = None
    purchase_price:     Optional[float] = None
    launch_date:        Optional[str]   = None
    uom:                Optional[str]   = None
    alias:              Optional[str]   = None
    gst_applicability:  Optional[str]   = None
    type_of_supply:     Optional[str]   = None
    gst_rate:           Optional[float] = None
    procurement_type:   Optional[str]   = None
    routing_step_ids:   Optional[List[int]] = None
    add_sizes:          Optional[List[str]] = None   # new sizes to append (existing are skipped)

class BOMCreate(BaseModel):
    bom_name:   str = "Default"
    applies_to: str = "all"
    is_default: int = 0

class BOMUpdate(BaseModel):
    bom_name:   Optional[str]   = None
    applies_to: Optional[str]   = None
    is_default: Optional[int]   = None
    cmt_cost:   Optional[float] = None
    other_cost: Optional[float] = None

class BOMLineCreate(BaseModel):
    component_name:    str
    component_type:    str   = "RM"
    quantity:          float = 1.0
    unit:              str   = "PCS"
    rate:              float = 0.0
    component_item_id: Optional[int]   = None
    process_id:        Optional[int]   = None
    shrinkage_pct:     float = 0.0
    wastage_pct:       float = 0.0
    remarks:           str   = ""

class BOMLineUpdate(BaseModel):
    component_name:    Optional[str]   = None
    component_type:    Optional[str]   = None
    quantity:          Optional[float] = None
    unit:              Optional[str]   = None
    rate:              Optional[float] = None
    component_item_id: Optional[int]   = None
    process_id:        Optional[int]   = None
    shrinkage_pct:     Optional[float] = None
    wastage_pct:       Optional[float] = None
    remarks:           Optional[str]   = None

class BOMCopyRequest(BaseModel):
    target_item_id: int
    new_name:       str = "Copied BOM"

class MerchantCreate(BaseModel):
    merchant_code: str
    merchant_name: str

class BuyerCreate(BaseModel):
    buyer_code: str
    buyer_name: str

class PackagingLineCreate(BaseModel):
    packaging_item_id: int
    quantity:          float = 1.0
    unit:              str   = "PCS"
    remarks:           str   = ""

class StockUpdate(BaseModel):
    stock: float


class StockAdjustBody(BaseModel):
    qty: float = Field(gt=0)
    direction: str = Field(description="IN to add stock, OUT to remove")
    entry_date: Optional[str] = None
    reason: str = ""
    reference_no: str = ""
    unit: str = ""


# ── Meta ──────────────────────────────────────────────────────────────────────

@router.get("/meta")
def get_meta():
    return {
        "item_types":    list_item_types(),
        "size_groups":   list_size_groups(),
        "routing_steps": list_routing_steps(),
        "merchants":     list_merchants(),
        "buyers":        list_buyers(),
    }


@router.get("/stats")
def get_stats():
    return get_item_stats()


@router.get("/boms/all")
def get_all_boms():
    return list_all_boms()


# ── Merchants ──────────────────────────────────────────────────────────────────

@router.get("/merchants")
def get_merchants():
    return list_merchants()

@router.post("/merchants")
def add_merchant(body: MerchantCreate):
    new_id = create_merchant(body.merchant_code, body.merchant_name)
    return {"ok": True, "id": new_id}

@router.delete("/merchants/{merchant_id}")
def remove_merchant(merchant_id: int):
    if not delete_merchant(merchant_id):
        raise HTTPException(status_code=404, detail="Merchant not found")
    return {"ok": True}


# ── Buyers ────────────────────────────────────────────────────────────────────

@router.get("/buyers")
def get_buyers():
    return list_buyers()

@router.post("/buyers")
def add_buyer(body: BuyerCreate):
    new_id = create_buyer(body.buyer_code, body.buyer_name)
    return {"ok": True, "id": new_id}

@router.delete("/buyers/{buyer_id}")
def remove_buyer(buyer_id: int):
    if not delete_buyer(buyer_id):
        raise HTTPException(status_code=404, detail="Buyer not found")
    return {"ok": True}


# ── Item Types ────────────────────────────────────────────────────────────────

@router.post("/types")
def add_item_type(body: ItemTypeCreate):
    new_id = create_item_type(body.name, body.code)
    return {"ok": True, "id": new_id}


# ── Size Groups ───────────────────────────────────────────────────────────────

@router.post("/size-groups")
def add_size_group(body: SizeGroupCreate):
    new_id = create_size_group(body.name, body.sizes)
    return {"ok": True, "id": new_id}


# ── Routing ───────────────────────────────────────────────────────────────────

@router.get("/routing")
def get_routing():
    return list_routing_steps()


@router.post("/routing")
def add_routing_step(body: RoutingStepCreate):
    new_id = create_routing_step(body.name, body.description, body.sort_order)
    return {"ok": True, "id": new_id}


@router.delete("/routing/{step_id}")
def remove_routing_step(step_id: int):
    if not delete_routing_step(step_id):
        raise HTTPException(status_code=404, detail="Routing step not found")
    return {"ok": True}


# ── Items ─────────────────────────────────────────────────────────────────────

@router.get("")
def get_items(
    type_id:     Optional[int] = None,
    search:      Optional[str] = None,
    parent_only: bool          = False,
):
    return list_items(type_id=type_id, search=search, parent_only=parent_only)


@router.post("")
def add_item(body: ItemCreate):
    item_id = create_item(
        item_code         = body.item_code,
        item_name         = body.item_name,
        item_type_id      = body.item_type_id,
        hsn_code          = body.hsn_code,
        season            = body.season,
        merchant_code     = body.merchant_code,
        selling_price     = body.selling_price,
        purchase_price    = body.purchase_price,
        launch_date       = body.launch_date,
        uom               = body.uom,
        alias             = body.alias,
        gst_applicability = body.gst_applicability,
        type_of_supply    = body.type_of_supply,
        gst_rate          = body.gst_rate,
        procurement_type  = body.procurement_type,
    )
    variant_ids: list[int] = []
    if body.sizes:
        variant_ids = create_size_variants(item_id, body.sizes)
    if body.routing_step_ids:
        set_item_routing(item_id, body.routing_step_ids)
    return {"ok": True, "id": item_id, "variant_count": len(variant_ids)}


@router.get("/search")
def search_items(q: str = ""):
    """Lightweight search for BOM component lookup."""
    return list_items(search=q, parent_only=False)


@router.get("/{item_id}")
def get_item_detail(item_id: int):
    item = get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.put("/{item_id}")
def edit_item(item_id: int, body: ItemUpdate):
    fields = {k: v for k, v in body.model_dump().items()
              if v is not None and k not in ("routing_step_ids", "add_sizes")}
    if fields:
        update_item(item_id, **fields)
    if body.routing_step_ids is not None:
        set_item_routing(item_id, body.routing_step_ids)
    new_variant_ids: list[int] = []
    if body.add_sizes:
        new_variant_ids = create_size_variants(item_id, body.add_sizes)
    return {"ok": True, "added_variants": len(new_variant_ids)}


@router.delete("/{item_id}")
def remove_item(item_id: int):
    if not delete_item(item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}


@router.put("/{item_id}/stock")
def update_stock(item_id: int, body: StockUpdate, request: Request):
    role = str((getattr(request.state, "auth", None) or {}).get("role", "") or "")
    if not may_access_erp_admin(role):
        raise HTTPException(status_code=403, detail="Admin access required for stock changes.")
    if not update_item_stock(item_id, body.stock):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True, "stock": body.stock}


@router.post("/{item_id}/stock/adjust")
def adjust_stock(item_id: int, body: StockAdjustBody, request: Request):
    """Manual stock IN/OUT (+/-) — admin only. Use for opening/existing stock and corrections."""
    role = str((getattr(request.state, "auth", None) or {}).get("role", "") or "")
    if not may_access_erp_admin(role):
        raise HTTPException(status_code=403, detail="Admin access required for stock adjustment.")
    from datetime import date as _date

    entry_date = (body.entry_date or "").strip() or str(_date.today())
    created_by = str((getattr(request.state, "auth", None) or {}).get("sub", "") or "")
    try:
        out = adjust_item_stock(
            item_id,
            body.qty,
            body.direction,
            entry_date=entry_date,
            reason=body.reason,
            reference_no=body.reference_no,
            unit=body.unit,
            created_by=created_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, **out}


# ── BOM ───────────────────────────────────────────────────────────────────────

@router.get("/{item_id}/boms")
def get_boms(item_id: int):
    return list_boms(item_id)


@router.post("/{item_id}/boms")
def add_bom(item_id: int, body: BOMCreate):
    new_id = create_bom(item_id, body.bom_name, body.applies_to, body.is_default)
    return {"ok": True, "id": new_id}


@router.get("/{item_id}/boms/{bom_id}")
def get_bom(item_id: int, bom_id: int):
    bom = get_bom_with_lines(bom_id)
    if bom is None or bom["item_id"] != item_id:
        raise HTTPException(status_code=404, detail="BOM not found")
    return bom


@router.put("/{item_id}/boms/{bom_id}")
def edit_bom(item_id: int, bom_id: int, body: BOMUpdate):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    update_bom(bom_id, **fields)
    return {"ok": True}


@router.delete("/{item_id}/boms/{bom_id}")
def remove_bom(item_id: int, bom_id: int):
    try:
        if not delete_bom(bom_id):
            raise HTTPException(status_code=404, detail="BOM not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@router.post("/{item_id}/boms/{bom_id}/certify")
def certify_bom_endpoint(item_id: int, bom_id: int):
    if not certify_bom(bom_id):
        raise HTTPException(status_code=404, detail="BOM not found")
    return {"ok": True}


@router.delete("/{item_id}/boms/{bom_id}/certify")
def uncertify_bom_endpoint(item_id: int, bom_id: int):
    if not uncertify_bom(bom_id):
        raise HTTPException(status_code=404, detail="BOM not found")
    return {"ok": True}


@router.post("/{item_id}/boms/{bom_id}/lines")
def add_line(item_id: int, bom_id: int, body: BOMLineCreate):
    try:
        new_id = add_bom_line(
            bom_id            = bom_id,
            component_name    = body.component_name,
            component_type    = body.component_type,
            quantity          = body.quantity,
            unit              = body.unit,
            rate              = body.rate,
            component_item_id = body.component_item_id,
            process_id        = body.process_id,
            shrinkage_pct     = body.shrinkage_pct,
            wastage_pct       = body.wastage_pct,
            remarks           = body.remarks,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "id": new_id}


@router.put("/{item_id}/boms/{bom_id}/lines/{line_id}")
def edit_line(item_id: int, bom_id: int, line_id: int, body: BOMLineUpdate):
    try:
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        update_bom_line(line_id, **fields)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@router.delete("/{item_id}/boms/{bom_id}/lines/{line_id}")
def remove_line(item_id: int, bom_id: int, line_id: int):
    try:
        if not delete_bom_line(line_id):
            raise HTTPException(status_code=404, detail="BOM line not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@router.post("/{item_id}/boms/{bom_id}/copy")
def copy_bom_to_item(item_id: int, bom_id: int, body: BOMCopyRequest):
    try:
        new_id = copy_bom(bom_id, body.target_item_id, body.new_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "id": new_id}


# ── Buyer Packaging ───────────────────────────────────────────────────────────

@router.get("/{item_id}/packaging")
def get_item_packaging(item_id: int):
    """All packaging lines for an item across all buyers."""
    return list_item_packaging(item_id)

@router.get("/{item_id}/packaging/{buyer_id}")
def get_packaging_for_buyer(item_id: int, buyer_id: int):
    """Packaging lines for a specific item + buyer."""
    return get_buyer_packaging(item_id, buyer_id)

@router.post("/{item_id}/packaging/{buyer_id}/lines")
def add_packaging(item_id: int, buyer_id: int, body: PackagingLineCreate):
    new_id = add_packaging_line(
        item_id=item_id,
        buyer_id=buyer_id,
        packaging_item_id=body.packaging_item_id,
        quantity=body.quantity,
        unit=body.unit,
        remarks=body.remarks,
    )
    return {"ok": True, "id": new_id}

@router.delete("/{item_id}/packaging/{buyer_id}/lines/{line_id}")
def remove_packaging(item_id: int, buyer_id: int, line_id: int):
    if not delete_packaging_line(line_id):
        raise HTTPException(status_code=404, detail="Packaging line not found")
    return {"ok": True}


# ── Bulk Import ───────────────────────────────────────────────────────────────

@router.post("/import/preview")
async def import_preview(file: UploadFile = File(...)):
    """Parse file and return first 20 rows for user review."""
    file_bytes = await file.read()
    try:
        rows = parse_item_import(file_bytes, file.filename or "import.xlsx")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"rows": rows[:20], "total": len(rows)}


@router.post("/import/confirm")
async def import_confirm(file: UploadFile = File(...)):
    """Parse file and bulk-insert all items."""
    file_bytes = await file.read()
    try:
        rows = parse_item_import(file_bytes, file.filename or "import.xlsx")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = bulk_create_items(rows)
    return {"ok": True, **result}


# ── Item Stock Tracking ────────────────────────────────────────────────────────
#
# Aggregates inbound (GRN, production receipts) and outbound (MIN, JWO issues,
# fabric issues) movements across the purchase + production SQLite files for a
# single item code. Tolerant of missing DBs (returns whatever it could read).

import os
import re
import sqlite3

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _safe_date(value):
    """Accept only ISO ``YYYY-MM-DD`` strings; everything else is dropped."""
    if isinstance(value, str) and _DATE_RE.match(value):
        return value
    return None


def _purchase_db_path() -> str:
    return os.environ.get(
        "PURCHASE_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "purchase.db"),
    )


def _production_db_path() -> str:
    return os.environ.get(
        "PRODUCTION_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "production.db"),
    )


def _date_clause(column: str, frm, to):
    """Return ``(sql_fragment, params)`` for an optional ``column BETWEEN`` filter."""
    parts: List[str] = []
    bind: List[str] = []
    if frm:
        parts.append(f" AND {column} >= ?")
        bind.append(frm)
    if to:
        parts.append(f" AND {column} <= ?")
        bind.append(to)
    return "".join(parts), bind


@router.get("/{item_code_str}/tracking")
def get_item_tracking(
    item_code_str: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    frm = _safe_date(from_date)
    to = _safe_date(to_date)
    like = f"%{item_code_str}%"
    results: list[dict] = []

    purchase_db = _purchase_db_path()
    if os.path.exists(purchase_db):
        try:
            pc = sqlite3.connect(purchase_db)
            pc.row_factory = sqlite3.Row

            grn_filter, grn_bind = _date_clause("h.grn_date", frm, to)
            for r in pc.execute(
                "SELECT h.grn_number,h.grn_date,h.party_name,l.material_code,l.material_name,"
                "l.accepted_qty as qty,l.unit,l.rate,l.amount,h.challan_no,h.so_reference "
                "FROM grn_lines l JOIN grn_headers h ON h.id=l.grn_id "
                "WHERE l.material_code LIKE ? AND h.status!='Cancelled'" + grn_filter +
                " ORDER BY h.grn_date DESC",
                [like, *grn_bind],
            ).fetchall():
                results.append({'date': r['grn_date'], 'direction': 'IN', 'txn_type': 'GRN',
                                'doc_number': r['grn_number'], 'doc_ref': r['challan_no'] or '',
                                'party': r['party_name'], 'item_code': r['material_code'],
                                'item_name': r['material_name'], 'qty': float(r['qty'] or 0),
                                'unit': r['unit'], 'rate': float(r['rate'] or 0),
                                'amount': float(r['amount'] or 0), 'so_ref': r['so_reference'] or ''})

            jwo_filter, jwo_bind = _date_clause("h.jwo_date", frm, to)
            # JWO material issue is recorded on confirmed MIN — not at JWO creation.

            min_filter, min_bind = _date_clause("h.min_date", frm, to)
            for r in pc.execute(
                "SELECT h.min_number,h.min_date,h.to_vendor,h.status,l.material_code,l.material_name,"
                "l.issue_qty,l.unit,l.rate,l.amount "
                "FROM min_lines l JOIN material_issue_notes h ON h.id=l.min_id "
                "WHERE l.material_code LIKE ? AND h.status IN ('Confirmed','Issued')" + min_filter +
                " ORDER BY h.min_date DESC",
                [like, *min_bind],
            ).fetchall():
                results.append({'date': r['min_date'], 'direction': 'OUT', 'txn_type': 'MIN',
                                'doc_number': r['min_number'], 'doc_ref': '',
                                'party': r['to_vendor'] or '', 'item_code': r['material_code'],
                                'item_name': r['material_name'] or r['material_code'],
                                'qty': float(r['issue_qty'] or 0),
                                'unit': r['unit'] or 'MTR', 'rate': float(r['rate'] or 0),
                                'amount': float(r['amount'] or 0),
                                'so_ref': ''})
            pc.close()
        except Exception:
            pass

    production_db = _production_db_path()
    if os.path.exists(production_db):
        try:
            pr = sqlite3.connect(production_db)
            pr.row_factory = sqlite3.Row

            fi_filter, fi_bind = _date_clause("fi.issue_date", frm, to)
            for r in pr.execute(
                "SELECT fi.issue_date,fi.fabric_code,fi.fabric_name,fi.issued_qty as qty,"
                "fi.unit,j.jo_number,j.so_number,j.process "
                "FROM jo_fabric_issues fi JOIN job_orders j ON j.id=fi.jo_id "
                "WHERE fi.fabric_code LIKE ?" + fi_filter +
                " ORDER BY fi.issue_date DESC",
                [like, *fi_bind],
            ).fetchall():
                results.append({'date': r['issue_date'], 'direction': 'OUT',
                                'txn_type': 'Production Fabric Issue', 'doc_number': r['jo_number'],
                                'doc_ref': r['so_number'] or '', 'party': 'Process: ' + str(r['process']),
                                'item_code': r['fabric_code'], 'item_name': r['fabric_name'],
                                'qty': float(r['qty'] or 0), 'unit': r['unit'], 'rate': 0, 'amount': 0,
                                'so_ref': r['so_number'] or ''})

            rc_filter, rc_bind = _date_clause("pr.receipt_date", frm, to)
            for r in pr.execute(
                "SELECT pr.receipt_date,pr.sku,pr.received_qty as qty,pr.process,pr.so_number,"
                "j.jo_number "
                "FROM jo_piece_receipts pr JOIN job_orders j ON j.id=pr.jo_id "
                "WHERE pr.sku LIKE ?" + rc_filter +
                " ORDER BY pr.receipt_date DESC",
                [like, *rc_bind],
            ).fetchall():
                results.append({'date': r['receipt_date'], 'direction': 'IN',
                                'txn_type': 'Production Receipt (' + str(r['process']) + ')',
                                'doc_number': r['jo_number'], 'doc_ref': r['so_number'] or '',
                                'party': 'Process: ' + str(r['process']), 'item_code': r['sku'],
                                'item_name': r['sku'], 'qty': float(r['qty'] or 0), 'unit': 'PCS',
                                'rate': 0, 'amount': 0, 'so_ref': r['so_number'] or ''})
            pr.close()
        except Exception:
            pass

    for adj in list_stock_adjustments(item_code_str, frm, to):
        label = (adj.get("reason") or "Stock Adjustment").strip() or "Stock Adjustment"
        if adj.get("reference_no"):
            label = f"{label} ({adj['reference_no']})"
        results.append({
            'date': adj.get('entry_date') or '',
            'direction': adj.get('direction') or 'IN',
            'txn_type': 'Stock Adjustment',
            'doc_number': f"ADJ-{adj.get('id', '')}",
            'doc_ref': adj.get('reference_no') or '',
            'party': adj.get('created_by') or 'Admin',
            'item_code': adj.get('item_code') or item_code_str,
            'item_name': adj.get('item_code') or item_code_str,
            'qty': float(adj.get('qty') or 0),
            'unit': adj.get('unit') or 'PCS',
            'rate': 0,
            'amount': 0,
            'so_ref': label,
        })

    results.sort(key=lambda x: x.get('date') or '', reverse=True)
    bal = 0.0
    for r in reversed(results):
        if r['direction'] == 'IN':
            bal += r['qty']
        else:
            bal -= r['qty']
        r['balance'] = round(bal, 3)
    results.reverse()
    in_qty = sum(r['qty'] for r in results if r['direction'] == 'IN')
    out_qty = sum(r['qty'] for r in results if r['direction'] == 'OUT')
    ledger_stock = round(in_qty - out_qty, 3)
    book_stock = book_stock_for_code(item_code_str)
    return {
        'item_code': item_code_str,
        'total_in': round(in_qty, 3),
        'total_out': round(out_qty, 3),
        'current_stock': book_stock,
        'ledger_stock': ledger_stock,
        'book_stock': book_stock,
        'transactions': results,
    }
