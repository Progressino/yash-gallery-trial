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

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from ..db.item_db import (
    list_item_types, create_item_type,
    list_size_groups, create_size_group,
    list_routing_steps, create_routing_step, delete_routing_step,
    list_merchants, create_merchant, delete_merchant,
    list_buyers, create_buyer, delete_buyer,
    list_items, get_item, create_item, update_item, delete_item,
    create_size_variants, set_item_routing,
    list_boms, get_bom_with_lines, create_bom, update_bom, delete_bom,
    certify_bom, uncertify_bom,
    add_bom_line, update_bom_line, delete_bom_line, copy_bom,
    list_item_packaging, get_buyer_packaging, add_packaging_line, delete_packaging_line,
    bulk_create_items, get_item_stats, list_all_boms,
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
    item_code:      str
    item_name:      str
    item_type_id:   int
    hsn_code:       str   = ""
    season:         str   = ""
    merchant_code:  str   = ""
    selling_price:  float = 0.0
    purchase_price: float = 0.0
    launch_date:    str   = ""
    sizes:          List[str] = []       # triggers size variant generation
    routing_step_ids: List[int] = []

class ItemUpdate(BaseModel):
    item_code:      Optional[str]   = None
    item_name:      Optional[str]   = None
    item_type_id:   Optional[int]   = None
    hsn_code:       Optional[str]   = None
    season:         Optional[str]   = None
    merchant_code:  Optional[str]   = None
    selling_price:  Optional[float] = None
    purchase_price: Optional[float] = None
    launch_date:    Optional[str]   = None
    routing_step_ids: Optional[List[int]] = None

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
        item_code      = body.item_code,
        item_name      = body.item_name,
        item_type_id   = body.item_type_id,
        hsn_code       = body.hsn_code,
        season         = body.season,
        merchant_code  = body.merchant_code,
        selling_price  = body.selling_price,
        purchase_price = body.purchase_price,
        launch_date    = body.launch_date,
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
              if v is not None and k != "routing_step_ids"}
    if fields:
        update_item(item_id, **fields)
    if body.routing_step_ids is not None:
        set_item_routing(item_id, body.routing_step_ids)
    return {"ok": True}


@router.delete("/{item_id}")
def remove_item(item_id: int):
    if not delete_item(item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}


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
