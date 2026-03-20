"""Grey Fabric Module router"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from ..db.grey_db import (
    list_grey, create_grey_entry, update_grey_status,
    list_ledger, list_hard_reservations, create_hard_reservation,
    release_hard_reservation, get_grey_stats
)

router = APIRouter()

class GreyEntryIn(BaseModel):
    po_number: Optional[str] = ''
    material_code: Optional[str] = ''
    material_name: Optional[str] = ''
    supplier: Optional[str] = ''
    so_reference: Optional[str] = ''
    ordered_qty: float = 0

class GreyUpdateIn(BaseModel):
    status: Optional[str] = None
    dispatched_qty: Optional[float] = None
    received_qty: Optional[float] = None
    transport_qty: Optional[float] = None
    factory_qty: Optional[float] = None
    printer_qty: Optional[float] = None
    checked_qty: Optional[float] = None
    rejected_qty: Optional[float] = None
    rework_qty: Optional[float] = None
    dispatch_date: Optional[str] = None
    bilty_no: Optional[str] = None
    vendor_invoice: Optional[str] = None
    vendor_challan: Optional[str] = None
    vehicle_no: Optional[str] = None
    transporter: Optional[str] = None
    expected_arrival: Optional[str] = None
    qc_status: Optional[str] = None
    qc_checked_by: Optional[str] = None
    qc_date: Optional[str] = None
    qc_remarks: Optional[str] = None

class HardReservationIn(BaseModel):
    fabric_code: str
    fabric_name: Optional[str] = ''
    so_number: Optional[str] = ''
    sku: Optional[str] = ''
    qty: float = 0
    unit: Optional[str] = 'MTR'
    remarks: Optional[str] = ''

@router.get("/stats")
def get_stats():
    return get_grey_stats()

@router.get("")
def get_grey(status: Optional[str] = None):
    return list_grey(status)

@router.post("")
def post_grey(body: GreyEntryIn):
    key = create_grey_entry(body.model_dump())
    if key is None:
        return {"error": "Entry already exists for this PO + material combination"}
    return {"tracker_key": key}

@router.patch("/{gid}")
def patch_grey(gid: int, body: GreyUpdateIn):
    update_grey_status(gid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}

@router.get("/ledger")
def get_ledger(material_code: Optional[str] = None):
    return list_ledger(material_code)

@router.get("/reservations")
def get_hard_reservations(status: str = 'Active'):
    return list_hard_reservations(status)

@router.post("/reservations")
def post_hard_reservation(body: HardReservationIn):
    create_hard_reservation(body.model_dump())
    return {"ok": True}

@router.delete("/reservations/{rid}")
def delete_hard_reservation(rid: int):
    release_hard_reservation(rid)
    return {"ok": True}
