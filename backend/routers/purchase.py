"""Purchase Module router — Suppliers, Processors, PR, PO, JWO, GRN,list_mins, create_min, update_min_status, get_min_by_number, create_gate_pass, list_gate_passes, get_gate_pass_by_number,"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from ..db.purchase_db import (
    list_suppliers, create_supplier, update_supplier,
    list_processors, create_processor,
    list_prs, create_pr, approve_pr, reject_pr, update_pr_status, create_pos_from_pr, mark_pr_lines_ordered,
    list_pos, create_po, update_po_status, update_po,
    list_jwos, create_jwo, update_jwo_status, update_jwo, get_po_by_number, get_jwo_by_number,
    list_grns, create_grn, update_grn_status,
    list_mins, create_min, update_min_status, get_min_by_number,
    list_gate_passes, create_gate_pass, get_gate_pass_by_number,
    get_purchase_stats
)

router = APIRouter()

class SupplierIn(BaseModel):
    supplier_code: Optional[str] = None
    supplier_name: str
    supplier_type: Optional[str] = 'Others'
    contact_person: Optional[str] = ''
    email: Optional[str] = ''
    phone: Optional[str] = ''
    address: Optional[str] = ''
    gst_number: Optional[str] = ''
    payment_terms: Optional[str] = 'Net 30'

class ProcessorIn(BaseModel):
    processor_code: Optional[str] = None
    processor_name: str
    processor_type: Optional[str] = 'Others'
    contact_person: Optional[str] = ''
    email: Optional[str] = ''
    phone: Optional[str] = ''
    address: Optional[str] = ''

class PRLineIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    material_type: Optional[str] = 'RM'
    required_qty: float = 0
    unit: Optional[str] = 'PCS'
    required_by_date: Optional[str] = ''
    purpose: Optional[str] = ''
    remarks: Optional[str] = ''

class PRIn(BaseModel):
    pr_date: Optional[str] = None
    requested_by: Optional[str] = ''
    department: Optional[str] = 'Production'
    priority: Optional[str] = 'Normal'
    so_reference: Optional[str] = ''
    pr_type: Optional[str] = 'Purchase'
    source: Optional[str] = 'Manual'
    required_by_date: Optional[str] = ''
    notes: Optional[str] = ''
    lines: List[PRLineIn] = []

class ApproveIn(BaseModel):
    approver: str
    remarks: Optional[str] = ''

class RejectIn(BaseModel):
    remarks: Optional[str] = ''

class POFromPRLineIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    material_type: Optional[str] = 'RM'
    unit: Optional[str] = 'PCS'
    qty: float = 0
    rate: float = 0
    gst_pct: Optional[int] = 0
    supplier_id: Optional[int] = None
    supplier_name: Optional[str] = ''

class POFromPRIn(BaseModel):
    pr_id: int
    delivery_date: Optional[str] = ''
    payment_terms: Optional[str] = 'Immediate'
    lines: List[POFromPRLineIn] = []

class POLineIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    material_type: Optional[str] = 'RM'
    po_qty: float = 0
    unit: Optional[str] = 'PCS'
    rate: float = 0
    gst_pct: Optional[int] = 0
    amount: Optional[float] = None
    remarks: Optional[str] = ''

class POIn(BaseModel):
    po_date: Optional[str] = None
    supplier_id: Optional[int] = None
    supplier_name: Optional[str] = ''
    currency: Optional[str] = 'INR'
    payment_terms: Optional[str] = ''
    delivery_location: Optional[str] = ''
    delivery_date: Optional[str] = ''
    pr_reference: Optional[str] = ''
    so_reference: Optional[str] = ''
    remarks: Optional[str] = ''
    lines: List[POLineIn] = []

class JWOLineIn(BaseModel):
    input_material: str
    input_qty: float = 0
    input_unit: Optional[str] = 'MTR'
    output_material: str
    output_qty: float = 0
    output_unit: Optional[str] = 'MTR'
    process_type: Optional[str] = 'Printing'
    rate: float = 0
    amount: Optional[float] = None
    remarks: Optional[str] = ''

class JWOIn(BaseModel):
    jwo_date: Optional[str] = None
    processor_id: Optional[int] = None
    processor_name: Optional[str] = ''
    pr_reference: Optional[str] = ''
    so_reference: Optional[str] = ''
    expected_return_date: Optional[str] = ''
    remarks: Optional[str] = ''
    issued_by: Optional[str] = ''
    lines: List[JWOLineIn] = []

# ── NEW: JWO Update model (same as JWOIn but all optional) ────────────────────
class JWOUpdateIn(BaseModel):
    processor_id: Optional[int] = None
    processor_name: Optional[str] = None
    pr_reference: Optional[str] = None
    so_reference: Optional[str] = None
    expected_return_date: Optional[str] = None
    remarks: Optional[str] = None
    issued_by: Optional[str] = None
    lines: Optional[List[JWOLineIn]] = None

class GRNLineIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    material_type: Optional[str] = 'RM'
    po_qty: Optional[float] = 0
    received_qty: float = 0
    accepted_qty: float = 0
    rejected_qty: Optional[float] = 0
    unit: Optional[str] = 'PCS'
    rate: float = 0
    amount: Optional[float] = None
    qc_status: Optional[str] = 'Pending'
    rejection_reason: Optional[str] = ''

class GRNIn(BaseModel):
    grn_date: Optional[str] = None
    grn_type: Optional[str] = 'PO Receipt'
    reference_number: Optional[str] = ''
    party_name: Optional[str] = ''
    challan_no: Optional[str] = ''
    invoice_no: Optional[str] = ''
    invoice_date: Optional[str] = ''
    vehicle_no: Optional[str] = ''
    transporter: Optional[str] = ''
    warehouse: Optional[str] = ''
    so_reference: Optional[str] = ''
    remarks: Optional[str] = ''
    lines: List[GRNLineIn] = []

class StatusUpdate(BaseModel):
    status: str

class GRNVerify(BaseModel):
    status: str
    qc_checked_by: Optional[str] = ''

# ── Stats ─────────────────────────────────────────────────────────────────────
@router.get("/stats")
def get_stats():
    return get_purchase_stats()

# ── Suppliers ─────────────────────────────────────────────────────────────────
@router.get("/suppliers")
def get_suppliers(active_only: bool = True):
    return list_suppliers(active_only)

@router.post("/suppliers")
def post_supplier(body: SupplierIn):
    create_supplier(body.model_dump())
    return {"ok": True}

@router.patch("/suppliers/{sid}")
def patch_supplier(sid: int, body: dict):
    update_supplier(sid, body)
    return {"ok": True}

# ── Processors ────────────────────────────────────────────────────────────────
@router.get("/processors")
def get_processors(active_only: bool = True):
    return list_processors(active_only)

@router.post("/processors")
def post_processor(body: ProcessorIn):
    create_processor(body.model_dump())
    return {"ok": True}

# ── Purchase Requisitions ─────────────────────────────────────────────────────
@router.get("/pr")
def get_prs(status: Optional[str] = None):
    return list_prs(status)

@router.post("/pr")
def post_pr(body: PRIn):
    num = create_pr(body.model_dump())
    return {"pr_number": num}

@router.post("/pr/{prid}/approve")
def post_approve_pr(prid: int, body: ApproveIn):
    approve_pr(prid, body.approver, body.remarks or '')
    return {"ok": True}

@router.post("/pr/{prid}/reject")
def post_reject_pr(prid: int, body: RejectIn):
    reject_pr(prid, body.remarks or '')
    return {"ok": True}

@router.patch("/pr/{prid}/status")
def patch_pr_status(prid: int, body: StatusUpdate):
    update_pr_status(prid, body.status)
    return {"ok": True}

class MarkOrderedIn(BaseModel):
    updates: List[dict] = []

@router.post("/pr/{prid}/mark-ordered")
def post_mark_pr_ordered(prid: int, body: MarkOrderedIn):
    mark_pr_lines_ordered(prid, body.updates)
    return {"ok": True}

@router.post("/po/from-pr")
def post_po_from_pr(body: POFromPRIn):
    po_numbers = create_pos_from_pr(
        body.pr_id,
        [l.model_dump() for l in body.lines],
        body.delivery_date or '',
        body.payment_terms or 'Immediate',
    )
    return {"po_numbers": po_numbers, "count": len(po_numbers)}

# ── Purchase Orders ───────────────────────────────────────────────────────────
@router.get("/po")
def get_pos(status: Optional[str] = None):
    return list_pos(status)

@router.post("/po")
def post_po(body: POIn):
    num = create_po(body.model_dump())
    return {"po_number": num}

@router.patch("/po/{poid}/status")
def patch_po_status(poid: int, body: StatusUpdate):
    update_po_status(poid, body.status)
    return {"ok": True}

@router.patch("/po/{poid}")
def patch_po(poid: int, body: dict):
    update_po(poid, body)
    return {"ok": True}

# ── Job Work Orders ───────────────────────────────────────────────────────────
@router.get("/jwo")
def get_jwos(status: Optional[str] = None):
    return list_jwos(status)

@router.post("/jwo")
def post_jwo(body: JWOIn):
    num = create_jwo(body.model_dump())
    return {"jwo_number": num}

@router.patch("/jwo/{jwoid}/status")
def patch_jwo_status(jwoid: int, body: StatusUpdate):
    update_jwo_status(jwoid, body.status)
    return {"ok": True}

# ── NEW: Update JWO (header + lines) — mirrors PATCH /po/{poid} ──────────────
@router.patch("/jwo/{jwoid}")
def patch_jwo(jwoid: int, body: JWOUpdateIn):
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    if "lines" in data:
        # Compute amount for each line before saving
        for ln in data["lines"]:
            if ln.get("amount") is None:
                ln["amount"] = ln.get("output_qty", 0) * ln.get("rate", 0)
    update_jwo(jwoid, data)
    return {"ok": True}
# ── Material Issue Notes (MIN) ────────────────────────────────────────────────
class MINLineIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    material_type: Optional[str] = 'GF'
    issue_qty: float = 0
    unit: Optional[str] = 'MTR'
    rate: Optional[float] = 0
    remarks: Optional[str] = ''

class MINIn(BaseModel):
    min_date: Optional[str] = None
    jwo_reference: Optional[str] = ''
    so_reference: Optional[str] = ''
    from_location: Optional[str] = 'Grey Warehouse'
    to_location: Optional[str] = ''
    to_vendor: Optional[str] = ''
    issued_by: Optional[str] = ''
    remarks: Optional[str] = ''
    lines: List[MINLineIn] = []

@router.get("/min")
def get_mins(status: Optional[str] = None):
    return list_mins(status)

@router.post("/min")
def post_min(body: MINIn):
    num = create_min(body.model_dump())
    return {"min_number": num}

@router.patch("/min/{minid}/status")
def patch_min_status(minid: int, body: StatusUpdate):
    update_min_status(minid, body.status)
    return {"ok": True}

@router.get("/min/by-number/{min_number}")
def get_min_by_num(min_number: str):
    doc = get_min_by_number(min_number)
    if not doc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="MIN not found")
    return doc
# ── GRN Auto-fill helpers ─────────────────────────────────────────────────────
@router.get("/po/by-number/{po_number}")
def get_po_by_num(po_number: str):
    doc = get_po_by_number(po_number)
    if not doc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="PO not found")
    return doc

@router.get("/jwo/by-number/{jwo_number}")
def get_jwo_by_num(jwo_number: str):
    doc = get_jwo_by_number(jwo_number)
    if not doc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="JWO not found")
    return doc
# ── GRN ───────────────────────────────────────────────────────────────────────
@router.get("/grn")
def get_grns(status: Optional[str] = None):
    return list_grns(status)

@router.post("/grn")
def post_grn(body: GRNIn):
    num = create_grn(body.model_dump())
    return {"grn_number": num}

@router.patch("/grn/{grnid}/verify")
def patch_grn_verify(grnid: int, body: GRNVerify):
    update_grn_status(grnid, body.status, body.qc_checked_by or '')
    return {"ok": True}
# ── Gate Pass ──────────────────────────────────────────────────────────────
class GPLineIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    qty: float = 0
    unit: Optional[str] = 'MTR'
    remarks: Optional[str] = ''

class GatePassIn(BaseModel):
    gp_date: Optional[str] = None
    min_reference: Optional[str] = ''
    jwo_reference: Optional[str] = ''
    from_location: Optional[str] = 'Factory'
    to_location: Optional[str] = ''
    party_name: Optional[str] = ''
    vehicle_no: Optional[str] = ''
    driver_name: Optional[str] = ''
    material_desc: Optional[str] = ''
    unit: Optional[str] = 'MTR'
    purpose: Optional[str] = 'Job Work'
    remarks: Optional[str] = ''
    lines: List[GPLineIn] = []

@router.get("/gate-pass")
def get_gate_passes():
    return list_gate_passes()

@router.post("/gate-pass")
def post_gate_pass(body: GatePassIn):
    num = create_gate_pass(body.model_dump())
    return {"gp_number": num}

@router.get("/gate-pass/by-number/{gp_number}")
def get_gp_by_number(gp_number: str):
    doc = get_gate_pass_by_number(gp_number)
    if not doc:
        raise HTTPException(status_code=404, detail="Gate pass not found")
    return doc