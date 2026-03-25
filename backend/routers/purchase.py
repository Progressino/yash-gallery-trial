"""Purchase Module router — Suppliers, Processors, PR, PO, JWO, GRN"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from ..db.purchase_db import (
    list_suppliers, create_supplier, update_supplier,
    list_processors, create_processor,
    list_prs, create_pr, approve_pr, reject_pr, update_pr_status, create_pos_from_pr,
    list_pos, create_po, update_po_status,
    list_jwos, create_jwo, update_jwo_status,
    list_grns, create_grn, update_grn_status,
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
