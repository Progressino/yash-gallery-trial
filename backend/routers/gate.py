"""Gate Inward (GIN) — barcode scan + receive for PO/JWO/JO."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from ..db.purchase_db import (
    create_grn,
    create_gin_record,
    get_gin,
    get_gin_by_number,
    get_jwo_receive_balance,
    get_po_receive_balance,
    link_gin_grn,
    list_gins,
    update_gin_line_receipt,
)
from ..db.production_db import get_jo_by_number, receive_pieces
from ..services.document_barcode import barcode_bundle, make_payload, parse_payload

router = APIRouter()


class GinLineIn(BaseModel):
    line_key: str = ""
    material_code: str = ""
    material_name: str = ""
    sku: str = ""
    planned_qty: float = 0
    already_received_qty: float = 0
    pending_qty: float = 0
    received_qty: float
    unit: str = "PCS"
    jo_id: Optional[int] = None
    jo_line_id: Optional[int] = None


class GinCreateIn(BaseModel):
    source_type: str
    source_number: str
    party_name: str = ""
    stage: str = ""
    vehicle_no: str = ""
    challan_no: str = ""
    remarks: str = ""
    created_by: str = "Gate"
    lines: List[GinLineIn] = Field(default_factory=list)


def _scan_po(number: str) -> dict:
    bal = get_po_receive_balance(number)
    if not bal:
        raise HTTPException(404, f"PO {number} not found")
    if bal.get("grn_blocked"):
        raise HTTPException(400, f"PO {number} is {bal.get('status')} — receive blocked")
    from ..db.purchase_db import _connect

    conn = _connect()
    po = conn.execute("SELECT * FROM po_headers WHERE po_number=?", (number,)).fetchone()
    conn.close()
    party = (dict(po).get("supplier_name") if po else "") or ""
    lines = []
    for ln in bal.get("lines") or []:
        pending = float(ln.get("balance_qty") or 0)
        if pending <= 0:
            continue
        lines.append(
            {
                "line_key": f"PO:{number}:{ln.get('material_code')}",
                "material_code": ln.get("material_code") or "",
                "material_name": ln.get("material_name") or "",
                "sku": ln.get("material_code") or "",
                "planned_qty": float(ln.get("po_qty") or 0),
                "already_received_qty": float(ln.get("grn_accepted_qty") or 0),
                "pending_qty": pending,
                "unit": ln.get("unit") or "PCS",
                "received_qty": pending,
            }
        )
    return {
        "source_type": "PO",
        "source_number": number,
        "party_name": party,
        "stage": "Warehouse",
        "barcode_payload": make_payload("PO", number),
        "lines": lines,
        "blocked": False,
    }


def _scan_jwo(number: str) -> dict:
    bal = get_jwo_receive_balance(number)
    if not bal:
        raise HTTPException(404, f"JWO {number} not found")
    if bal.get("grn_blocked"):
        raise HTTPException(400, f"JWO {number} is {bal.get('status')} — receive blocked")
    from ..db.purchase_db import _connect

    conn = _connect()
    jwo = conn.execute("SELECT * FROM jwo_headers WHERE jwo_number=?", (number,)).fetchone()
    conn.close()
    party = (dict(jwo).get("processor_name") if jwo else "") or ""
    lines = []
    for ln in bal.get("lines") or []:
        pending = float(ln.get("balance_qty") or 0)
        if pending <= 0:
            continue
        code = ln.get("material_code") or ln.get("output_material") or ""
        lines.append(
            {
                "line_key": f"JWO:{number}:{code}",
                "material_code": code,
                "material_name": ln.get("material_name") or code,
                "sku": code,
                "planned_qty": float(ln.get("output_qty") or 0),
                "already_received_qty": float(ln.get("grn_accepted_qty") or 0),
                "pending_qty": pending,
                "unit": ln.get("unit") or "PCS",
                "received_qty": pending,
            }
        )
    return {
        "source_type": "JWO",
        "source_number": number,
        "party_name": party,
        "stage": "Job Work Return",
        "barcode_payload": make_payload("JWO", number),
        "lines": lines,
        "blocked": False,
    }


def _scan_jo(number: str) -> dict:
    jo = get_jo_by_number(number)
    if not jo:
        raise HTTPException(404, f"Job Order {number} not found")
    if (jo.get("status") or "") in ("Cancelled", "Closed"):
        raise HTTPException(400, f"JO {number} is {jo.get('status')} — receive blocked")
    lines = []
    for ln in jo.get("lines") or []:
        planned = int(ln.get("planned_qty") or 0)
        already = int(ln.get("received_qty") or 0)
        pending = max(0, planned - already)
        if pending <= 0:
            continue
        lines.append(
            {
                "line_key": f"JO:{number}:{ln.get('id')}",
                "material_code": ln.get("sku") or "",
                "material_name": ln.get("sku_name") or "",
                "sku": ln.get("sku") or "",
                "planned_qty": planned,
                "already_received_qty": already,
                "pending_qty": pending,
                "unit": "PCS",
                "received_qty": pending,
                "jo_id": jo["id"],
                "jo_line_id": ln.get("id"),
            }
        )
    if not lines:
        # Header-level JO without line balances
        planned = int(jo.get("planned_qty") or 0)
        already = int(jo.get("received_qty") or 0)
        pending = max(0, planned - already)
        if pending > 0:
            lines.append(
                {
                    "line_key": f"JO:{number}:HDR",
                    "material_code": jo.get("sku") or "",
                    "material_name": jo.get("sku_name") or "",
                    "sku": jo.get("sku") or "",
                    "planned_qty": planned,
                    "already_received_qty": already,
                    "pending_qty": pending,
                    "unit": "PCS",
                    "received_qty": pending,
                    "jo_id": jo["id"],
                    "jo_line_id": None,
                }
            )
    return {
        "source_type": "JO",
        "source_number": jo.get("jo_number") or number,
        "party_name": jo.get("vendor_name") or "",
        "stage": jo.get("process") or "",
        "jo_id": jo["id"],
        "barcode_payload": make_payload("JO", jo.get("jo_number") or number),
        "lines": lines,
        "blocked": False,
    }


@router.get("/scan")
def gate_scan(code: str = Query(..., min_length=1)):
    try:
        dtype, number = parse_payload(code)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    if dtype == "GIN":
        gin = get_gin_by_number(number)
        if not gin:
            raise HTTPException(404, f"GIN {number} not found")
        return {"source_type": "GIN", "source_number": number, "gin": gin, "lines": gin.get("lines") or []}
    if dtype == "PO":
        return _scan_po(number)
    if dtype == "JWO":
        return _scan_jwo(number)
    if dtype == "JO":
        return _scan_jo(number)
    if dtype == "GRN":
        raise HTTPException(400, "Scan the PO / JWO / JO document for gate inward (not GRN).")
    if dtype in ("MIN", "GP", "DC"):
        return {
            "source_type": dtype,
            "source_number": number,
            "party_name": "",
            "stage": "",
            "barcode_payload": make_payload(dtype, number),
            "lines": [],
            "message": f"{dtype} documents are for issue/outward lookup — use document search, not gate receive.",
            "lookup_only": True,
        }
    raise HTTPException(400, f"Unsupported document type: {dtype}")


@router.post("/gin")
def post_gin(body: GinCreateIn):
    st = (body.source_type or "").strip().upper()
    if st not in ("PO", "JWO", "JO"):
        raise HTTPException(400, "source_type must be PO, JWO, or JO")
    if not body.lines:
        raise HTTPException(400, "At least one line with received_qty is required")

    recv_lines = []
    for ln in body.lines:
        qty = float(ln.received_qty or 0)
        if qty <= 0:
            continue
        pending = float(ln.pending_qty or 0)
        if pending > 0 and qty > pending + 1e-6:
            raise HTTPException(
                400,
                f"Received {qty} exceeds pending {pending} for {ln.sku or ln.material_code}",
            )
        recv_lines.append(ln)
    if not recv_lines:
        raise HTTPException(400, "Enter received quantity greater than 0 on at least one line")

    # Re-validate live pending
    if st == "PO":
        live = _scan_po(body.source_number)
    elif st == "JWO":
        live = _scan_jwo(body.source_number)
    else:
        live = _scan_jo(body.source_number)
    live_pending = {
        (x.get("line_key") or x.get("sku") or x.get("material_code")): float(x.get("pending_qty") or 0)
        for x in live.get("lines") or []
    }
    for ln in recv_lines:
        key = ln.line_key or ln.sku or ln.material_code
        pend = live_pending.get(key)
        if pend is None:
            # fallback match by sku
            for k, v in live_pending.items():
                if ln.sku and ln.sku in k:
                    pend = v
                    break
        if pend is not None and float(ln.received_qty) > pend + 1e-6:
            raise HTTPException(400, f"Received qty exceeds current pending ({pend}) for {key}")

    header = {
        "source_type": st,
        "source_number": body.source_number.strip(),
        "party_name": body.party_name or live.get("party_name") or "",
        "stage": body.stage or live.get("stage") or "",
        "vehicle_no": body.vehicle_no or "",
        "challan_no": body.challan_no or "",
        "remarks": body.remarks or "",
        "created_by": body.created_by or "Gate",
        "barcode_payload": make_payload(st, body.source_number.strip()),
        "jo_id": live.get("jo_id"),
        "status": "Completed",
    }
    gin_lines = [ln.model_dump() for ln in recv_lines]
    gin = create_gin_record(header, gin_lines)
    gin_id = gin["id"]

    grn_number = None
    jo_receipt_ids: list[int] = []

    try:
        if st in ("PO", "JWO"):
            grn_lines = []
            for ln in recv_lines:
                qty = float(ln.received_qty)
                grn_lines.append(
                    {
                        "material_code": ln.material_code or ln.sku,
                        "material_name": ln.material_name or "",
                        "material_type": "RM" if st == "PO" else "SFG",
                        "po_qty": float(ln.planned_qty or 0),
                        "received_qty": qty,
                        "accepted_qty": qty,
                        "rejected_qty": 0,
                        "unit": ln.unit or "PCS",
                        "rate": 0,
                        "qc_status": "Pending",
                    }
                )
            grn_result = create_grn(
                {
                    "grn_type": "PO Receipt" if st == "PO" else "JWO Receipt",
                    "reference_number": body.source_number.strip(),
                    "party_name": header["party_name"],
                    "challan_no": body.challan_no or gin["gin_number"],
                    "vehicle_no": body.vehicle_no or "",
                    "remarks": f"Auto GRN from {gin['gin_number']}. {body.remarks or ''}".strip(),
                    "gin_id": gin_id,
                    "return_id": True,
                    "lines": grn_lines,
                    "grn_date": datetime.now().strftime("%Y-%m-%d"),
                }
            )
            grn_number = grn_result["grn_number"]
            link_gin_grn(gin_id, int(grn_result["grn_id"]))
        else:
            jo_id = live.get("jo_id")
            if not jo_id:
                raise HTTPException(400, "JO id missing for gate receive")
            for ln in recv_lines:
                result = receive_pieces(
                    int(jo_id),
                    {
                        "jo_line_id": ln.jo_line_id,
                        "sku": ln.sku or ln.material_code,
                        "received_qty": int(float(ln.received_qty)),
                        "rejected_qty": 0,
                        "process": header["stage"] or live.get("stage"),
                        "received_by": body.created_by or "Gate",
                        "remarks": f"GIN {gin['gin_number']}",
                        "gin_id": gin_id,
                        "split_components": True,
                    },
                )
                rid = result.get("receipt_id") if isinstance(result, dict) else None
                if rid:
                    jo_receipt_ids.append(int(rid))
                    # best-effort link on matching gin line
                    for gl in gin.get("lines") or []:
                        if gl.get("jo_line_id") == ln.jo_line_id or (
                            gl.get("sku") == ln.sku and not gl.get("jo_receipt_id")
                        ):
                            update_gin_line_receipt(int(gl["id"]), int(rid))
                            break
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"GIN saved but receive failed: {e}") from e

    detail = get_gin(gin_id)
    return {
        "ok": True,
        "gin": detail,
        "gin_number": detail["gin_number"] if detail else gin["gin_number"],
        "grn_number": grn_number,
        "jo_receipt_ids": jo_receipt_ids,
        "message": f"GIN {detail['gin_number'] if detail else gin['gin_number']} completed",
    }


@router.get("/gin")
def gin_list(status: Optional[str] = None, limit: int = 100):
    return list_gins(status=status, limit=limit)


@router.get("/gin/{gin_id}")
def gin_detail(gin_id: int):
    g = get_gin(gin_id)
    if not g:
        raise HTTPException(404, "GIN not found")
    return g


@router.get("/barcode")
def barcode_assets(type: str = Query(...), number: str = Query(...)):
    try:
        return barcode_bundle(type, number)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/barcode.svg")
def barcode_qr_svg(type: str = Query(...), number: str = Query(...)):
    """Return QR as SVG bytes for <img src> embeds in print windows."""
    try:
        bundle = barcode_bundle(type, number)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    # Decode data URL
    import base64

    data = bundle["qr_data_url"].split(",", 1)[1]
    return Response(content=base64.b64decode(data), media_type="image/svg+xml")


@router.get("/gin/{gin_id}/print", response_class=HTMLResponse)
def gin_print(gin_id: int):
    g = get_gin(gin_id)
    if not g:
        raise HTTPException(404, "GIN not found")
    try:
        bundle = barcode_bundle("GIN", g["gin_number"])
        qr = bundle["qr_data_url"]
    except Exception:
        qr = ""
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{ln.get('sku') or ln.get('material_code')}</td>"
        f"<td>{ln.get('material_name') or ''}</td>"
        f"<td style='text-align:right'>{ln.get('planned_qty')}</td>"
        f"<td style='text-align:right'>{ln.get('received_qty')}</td>"
        f"<td>{ln.get('unit') or ''}</td></tr>"
        for i, ln in enumerate(g.get("lines") or [])
    )
    html = f"""<!DOCTYPE html><html><head><title>{g['gin_number']}</title>
    <style>
      body{{font-family:Segoe UI,sans-serif;padding:24px;font-size:12px}}
      .hdr{{display:flex;justify-content:space-between;border-bottom:2px solid #002B5B;padding-bottom:12px}}
      .title{{font-size:18px;font-weight:700;color:#002B5B}}
      table{{width:100%;border-collapse:collapse;margin-top:16px}}
      th{{background:#002B5B;color:#fff;padding:6px;text-align:left}}
      td{{padding:6px;border-bottom:1px solid #e2e8f0}}
    </style></head><body>
    <div class="hdr">
      <div><div class="title">GATE INWARD NOTE</div>
        <div>{g['gin_number']} · {g.get('gin_date','')}</div>
        <div>{g.get('source_type')}: {g.get('source_number')} · {g.get('party_name') or ''}</div>
        <div>Stage: {g.get('stage') or '—'}</div>
      </div>
      <div style="text-align:right">
        {"<img src='"+qr+"' width='110' height='110' alt='QR'/>" if qr else ""}
        <div style="font-size:10px;margin-top:4px">{g.get('barcode_payload') or ''}</div>
      </div>
    </div>
    <table><thead><tr><th>#</th><th>SKU / Code</th><th>Name</th><th>Planned</th><th>Received</th><th>Unit</th></tr></thead>
    <tbody>{rows}</tbody></table>
    <script>window.onload=()=>window.print()</script>
    </body></html>"""
    return HTMLResponse(html)
