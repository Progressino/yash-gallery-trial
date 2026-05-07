"""Grey Fabric Module router — lifecycle, MRP lines, job work, QC, reports."""
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db import grey_db as gdb

router = APIRouter()


class GreyEntryIn(BaseModel):
    po_number: str = ""
    material_code: str = ""
    material_name: str = ""
    supplier: str = ""
    so_reference: str = ""
    ordered_qty: float = 0
    rate: float = 0
    delivery_location: str = ""


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
    passed_qty: Optional[float] = None
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
    delivery_location: Optional[str] = None
    rate: Optional[float] = None
    in_transit_qty: Optional[float] = None
    return_to_vendor_qty: Optional[float] = None
    rework_issue_qty: Optional[float] = None
    rework_receive_qty: Optional[float] = None
    debit_note_no: Optional[str] = None
    return_challan_no: Optional[str] = None
    gate_pass_no: Optional[str] = None
    job_work_order_no: Optional[str] = None
    printed_fabric_qty: Optional[float] = None


class VendorDispatchIn(BaseModel):
    bilty_no: str
    transporter: str = ""
    dispatch_date: str = ""
    expected_arrival: str = ""
    dispatched_qty: float = 0
    vehicle_no: str = ""


class ArriveTransportIn(BaseModel):
    qty: Optional[float] = None


class TransferIn(BaseModel):
    to_location: str = Field(..., description="'factory' or 'printer'")
    qty: float = 0


class QCIn(BaseModel):
    received_qty: float = 0
    checked_qty: float = 0
    passed_qty: float = 0
    rejected_qty: float = 0
    rework_qty: float = 0
    outcome: str = ""
    qc_remarks: str = ""
    qc_by: str = ""
    qc_date: str = ""


class ReturnVendorIn(BaseModel):
    return_qty: float = 0
    debit_note_no: str = ""
    return_challan: str = ""
    return_date: str = ""
    remarks: str = ""


class PrinterIssueIn(BaseModel):
    tracker_id: int
    job_order_no: str = ""
    material_code: str = ""
    issue_qty: float = 0
    from_location: str = "Transport Location"
    to_vendor: str = ""
    issue_date: str = ""
    challan_no: str = ""
    gate_pass: str = ""
    remarks: str = ""


class PrintedReceiveIn(BaseModel):
    received_back_qty: float = 0
    grey_input_mtr: float = 0
    printed_item_code: str = ""
    printed_output_mtr: float = 0
    wastage_mtr: float = 0
    conversion_date: str = ""
    remarks: str = ""


class MRPRequirementIn(BaseModel):
    material_code: str
    material_name: str = ""
    so_number: str = ""
    sku: str = ""
    qty_required: float = 0
    run_label: str = ""
    notes: str = ""


class HardReservationIn(BaseModel):
    fabric_code: str
    fabric_name: str = ""
    so_number: str = ""
    sku: str = ""
    qty: float = 0
    unit: str = "MTR"
    remarks: str = ""


class FabricCheckIn(BaseModel):
    fabric_code: str
    fabric_name: Optional[str] = ''
    printer: Optional[str] = ''
    check_date: Optional[str] = ''
    checked_by: Optional[str] = ''
    checked_qty: float = 0
    passed_qty: float = 0
    rejected_qty: float = 0
    rework_qty: float = 0
    remarks: Optional[str] = ''


class ReserveFabricIn(BaseModel):
    fabric_code: str
    fabric_name: str = ''
    so_number: str = ''
    sku: str = ''
    qty: float = 0
    remarks: str = ''


class PrintedFabricQCIn(BaseModel):
    fabric_code: Optional[str] = ''
    fabric_name: Optional[str] = ''
    jwo_ref: Optional[str] = ''
    passed_qty: float = 0
    failed_qty: float = 0
    qc_by: Optional[str] = ''
    qc_date: Optional[str] = ''


class PrintedFabricReserveIn(BaseModel):
    fabric_code: str
    fabric_name: str = ''
    so_number: str = ''
    sku: str = ''
    qty: float = 0
    remarks: str = ''


# ── Basic routes ──────────────────────────────────────────────────────────────
@router.get("/meta")
def grey_meta():
    return {"statuses": gdb.GREY_STATUSES}


@router.get("/stats")
def get_stats():
    return gdb.get_grey_stats()


@router.get("/locations")
def location_summary():
    return gdb.get_location_summary()


@router.get("/ledger")
def get_ledger(material_code: Optional[str] = None):
    return gdb.list_ledger(material_code)


@router.get("/qc-events")
def list_qc_events(tracker_id: Optional[int] = None):
    return gdb.list_qc_events(tracker_id)


@router.get("/conversions/list")
def list_conversions(tracker_id: Optional[int] = None):
    return gdb.list_conversions(tracker_id)


# ── Grey Fabric Check (Grey fabric ki QC) ─────────────────────────────────────
@router.post("/fabric-check")
def post_fabric_check(body: FabricCheckIn):
    gdb.save_fabric_check(body.model_dump())
    return {"ok": True}


@router.get("/fabric-check/unchecked")
def get_unchecked():
    return gdb.list_unchecked_fabric()


@router.get("/fabric-check/checked")
def get_checked():
    return gdb.list_checked_fabric()


@router.post("/fabric-check/reserve")
def post_reserve_fabric(body: ReserveFabricIn):
    gdb.reserve_checked_fabric(body.model_dump())
    return {"ok": True}


@router.get("/fabric-check/ready-to-cut")
def get_ready_to_cut():
    return gdb.list_ready_to_cut()


# ── Printed Fabric (JWO GRN ke baad — alag warehouse) ─────────────────────────
@router.get("/printed-fabric/unchecked")
def get_printed_fabric_unchecked():
    return gdb.list_printed_fabric_unchecked()


@router.post("/printed-fabric/qc")
def post_printed_fabric_qc(body: PrintedFabricQCIn):
    gdb.do_printed_fabric_qc(body.model_dump())
    return {"ok": True}


@router.get("/printed-fabric/checked")
def get_printed_fabric_checked():
    return gdb.list_printed_fabric_checked()


@router.post("/printed-fabric/reserve")
def post_printed_fabric_reserve(body: PrintedFabricReserveIn):
    gdb.reserve_printed_fabric(body.model_dump())
    return {"ok": True}


@router.get("/printed-fabric/ready-to-cut")
def get_printed_fabric_ready_to_cut():
    return gdb.list_printed_fabric_ready_to_cut()


# ── Reservations ──────────────────────────────────────────────────────────────
@router.get("/reservations")
def get_hard_reservations(status: str = "Active"):
    return gdb.list_hard_reservations(status)


@router.post("/reservations")
def post_hard_reservation(body: HardReservationIn):
    gdb.create_hard_reservation(body.model_dump())
    return {"ok": True}


@router.delete("/reservations/{rid}")
def delete_hard_reservation(rid: int):
    gdb.release_hard_reservation(rid)
    return {"ok": True}


# ── MRP ───────────────────────────────────────────────────────────────────────
@router.get("/mrp/requirements")
def mrp_requirements(material_code: Optional[str] = None):
    return gdb.list_mrp_requirements(material_code)


@router.post("/mrp/requirements")
def mrp_requirements_post(body: MRPRequirementIn):
    iid = gdb.create_mrp_requirement(body.model_dump())
    return {"ok": True, "id": iid}


@router.get("/mrp/by-material/{material_code}")
def mrp_drilldown(material_code: str):
    return gdb.mrp_drilldown_by_material(material_code)


@router.get("/mrp/totals")
def mrp_totals():
    return gdb.mrp_totals_by_material()


@router.get("/mrp/availability/{fabric_code}")
def mrp_availability(fabric_code: str):
    return gdb.available_for_other_so(fabric_code)


@router.get("/mrp/stock-snapshot")
def mrp_stock_snapshot():
    return gdb.mrp_stock_snapshot()


# ── Printer Issues ────────────────────────────────────────────────────────────
@router.post("/printer-issue")
def post_printer_issue(body: PrinterIssueIn):
    iid = gdb.create_printer_issue(body.model_dump())
    return {"ok": True, "id": iid}


@router.get("/printer-issue/list")
def get_printer_issue_list(tracker_id: Optional[int] = None):
    return gdb.list_printer_issues(tracker_id)


@router.post("/printer-issue/{issue_id}/receive-printed")
def post_receive_printed(issue_id: int, body: PrintedReceiveIn):
    conv: Optional[Dict[str, Any]] = None
    if body.printed_item_code or body.printed_output_mtr or body.grey_input_mtr:
        conv = {
            "grey_input_mtr": body.grey_input_mtr,
            "printed_item_code": body.printed_item_code,
            "printed_output_mtr": body.printed_output_mtr,
            "wastage_mtr": body.wastage_mtr,
            "conversion_date": body.conversion_date,
            "remarks": body.remarks,
        }
    ok = gdb.receive_printed_fabric(issue_id, body.received_back_qty, conv)
    if not ok:
        raise HTTPException(404, "Printer issue not found")
    return {"ok": True}


# ── Reports ───────────────────────────────────────────────────────────────────
@router.get("/reports/transit")
def rep_transit():
    return gdb.report_transit()


@router.get("/reports/stock-locations")
def rep_stock_locations():
    return gdb.report_stock_by_location()


@router.get("/reports/qc")
def rep_qc():
    return gdb.report_qc()


@router.get("/reports/rejects-returns")
def rep_rejects():
    return gdb.report_rejects_returns()


@router.get("/reports/printer-issues")
def rep_printer():
    return gdb.report_printer_issues()


@router.get("/reports/consumption")
def rep_consumption():
    return gdb.report_grey_consumption()


# ── Grey Tracker CRUD — dynamic {gid} routes LAST ────────────────────────────
@router.get("")
def get_grey(status: Optional[str] = None):
    return gdb.list_grey(status)


@router.post("")
def post_grey(body: GreyEntryIn):
    key = gdb.create_grey_entry(body.model_dump())
    if key is None:
        return {"error": "Entry already exists for this PO + material combination"}
    return {"tracker_key": key}


@router.get("/{gid:int}")
def get_one(gid: int):
    row = gdb.get_grey(gid)
    if not row:
        raise HTTPException(404, "Not found")
    return row


@router.patch("/{gid}")
def patch_grey(gid: int, body: GreyUpdateIn):
    gdb.update_grey_status(gid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


@router.post("/{gid}/vendor-dispatch")
def post_vendor_dispatch(gid: int, body: VendorDispatchIn):
    ok = gdb.vendor_dispatch(gid, body.bilty_no, body.transporter, body.dispatch_date,
                              body.expected_arrival, body.dispatched_qty, body.vehicle_no)
    if not ok:
        raise HTTPException(404, "Tracker not found")
    return {"ok": True, "status": "In Transit"}


@router.post("/{gid}/arrive-transport")
def post_arrive_transport(gid: int, body: ArriveTransportIn):
    ok = gdb.arrive_at_transport(gid, body.qty)
    if not ok:
        raise HTTPException(400, "Tracker not found or invalid qty")
    return {"ok": True, "status": "At Transport Location"}


@router.post("/{gid}/transfer")
def post_transfer(gid: int, body: TransferIn):
    ok = gdb.transfer_qty(gid, body.to_location, body.qty)
    if not ok:
        raise HTTPException(400, "Tracker not found, invalid destination, or insufficient transport qty")
    return {"ok": True}


@router.post("/{gid}/qc")
def post_qc(gid: int, body: QCIn):
    ok = gdb.record_qc(gid, body.received_qty, body.checked_qty, body.passed_qty,
                        body.rejected_qty, body.rework_qty, body.outcome,
                        body.qc_remarks, body.qc_by, body.qc_date)
    if not ok:
        raise HTTPException(404, "Tracker not found")
    return {"ok": True}


@router.post("/{gid}/return-vendor")
def post_return_vendor(gid: int, body: ReturnVendorIn):
    ok = gdb.return_to_vendor(gid, body.return_qty, body.debit_note_no,
                               body.return_challan, body.return_date, body.remarks)
    if not ok:
        raise HTTPException(404, "Tracker not found")
    return {"ok": True}