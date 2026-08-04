"""Production Module — Dynamic Routing, Multi-process JO"""
import os, sqlite3
from datetime import datetime
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
import io
import pandas as pd
from ..db.production_db import (
    list_jos, get_jo, create_jo, update_jo,
    issue_fabric, return_fabric, issue_pieces, receive_pieces, add_cost,
    create_next_process_jo, validate_jo_creation,
    get_process_stock, get_all_process_stocks, get_ready_to_process,
    get_item_routing, get_next_process, get_all_routing_steps,
    get_process_report, get_production_stats,
    save_mrp_result, get_last_mrp_result,
    sync_mrp_commitments_from_run, get_mrp_commitments_for_so, check_mrp_commitment,
    list_embroidery_stock_for_skus,
    soft_reserve_all, release_so_reservations,
    list_soft_reservations_v2, get_soft_reserved_by_material,
    list_reservations, create_reservation, release_reservation, get_reserved_qty,
    list_set_boms, get_set_bom, get_set_bom_for_sku, upsert_set_bom, delete_set_bom,
    preview_set_match, commit_set_match, list_set_split_events, list_set_match_events,
    get_component_routing, preview_bundle_ready, get_partial_wip_board, get_jo_panel_wip,
    list_embroidery_stock_for_sku,
    list_embroidery_stock_for_skus,
    import_ready_to_wip, ready_to_wip_template_rows, get_previous_process,
)
from ..db.sales_db import get_open_orders, list_orders
from ..services.helpers import get_parent_sku
from ..services import jo_issue_notes
from ..services.jo_import import (
    build_jo_payload_from_import_row,
    jo_import_template_csv,
    looks_like_ready_to_wip_columns,
)

router = APIRouter()

_ITEM_DB_PATH = os.environ.get("ITEM_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "items_dev.db"))

def _item_connect():
    """Open Item Master DB. Falls back to a local dev path only if it actually exists.

    Previously, ``sqlite3.connect`` on a missing path silently created an empty
    DB and MRP returned ``{}`` for every SO. Now we surface that misconfig.
    """
    candidates = [_ITEM_DB_PATH, os.path.join(os.path.dirname(__file__), "..", "items_dev.db")]
    for path in candidates:
        if path and os.path.exists(path):
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            return conn
    raise RuntimeError(
        f"Item Master DB not found at any of: {candidates}. "
        "Set ITEM_DB_PATH (e.g. /data/items.db) and ensure the file is mounted."
    )

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

def _parent_candidates(sku: str) -> list[str]:
    """Build progressively-broader parent-style candidates for an MRP fallback.

    Returns ordered, deduped list. Strategy:
      1. Iteratively strip one ``-suffix`` at a time (handles ``STYLE-1-XL`` → ``STYLE-1``
         where ``get_parent_sku`` would over-strip the numeric ``1``).
      2. Append ``get_parent_sku`` as the last resort.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(c: str) -> None:
        c = (c or "").strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    cur = (sku or "").strip()
    while "-" in cur:
        cur = cur.rsplit("-", 1)[0]
        _add(cur)
    try:
        _add(get_parent_sku(sku))
    except Exception:
        pass
    return out


def _resolve_bom_anchor(conn, sku: str):
    """Find an Item Master row whose default BOM should drive MRP for ``sku``.

    Order of attempts:
        1. Exact code match (with its ``parent_id`` if its own BOM is missing).
        2. Hyphen-suffix strip cascade (``STYLE-1-XL`` → ``STYLE-1`` → ``STYLE``).
        3. ``get_parent_sku`` heuristic (handles tokens without hyphens).

    Returns ``(bom_code, item_row, reason)`` — ``reason`` is empty on success.
    """
    if not sku:
        return None, None, "Empty SKU"

    item = _get_item_by_code(conn, sku)
    if item:
        if item.get("parent_id"):
            parent = _get_item_by_id(conn, item["parent_id"])
            if parent and _get_default_bom(conn, parent["id"]):
                return parent["item_code"], parent, ""
        if _get_default_bom(conn, item["id"]):
            return item["item_code"], item, ""

    for candidate in _parent_candidates(sku):
        if candidate == sku:
            continue
        parent = _get_item_by_code(conn, candidate)
        if parent and _get_default_bom(conn, parent["id"]):
            return parent["item_code"], parent, ""

    if item:
        return sku, item, f"No BOM defined for SKU '{sku}' (or its parent style)"
    return None, None, f"SKU '{sku}' is not in Item Master"


def _is_printed_component(ctype: str, code: str) -> bool:
    t = (ctype or "").upper()
    c = (code or "").strip().upper()
    if t in ("SFG", "PRINTED", "PRINTED FABRIC", "PF"):
        return True
    # Heuristic: P-codes are often P###… (printed fabric)
    return bool(c) and c.startswith("P") and any(ch.isdigit() for ch in c[:6])


def calculate_mrp(so_numbers):
    """
    Returns ``{"materials": {...}, "warnings": [...]}``.

    Warnings surface every SKU that couldn't be exploded so newly created SOs
    don't fail silently when their items / BOMs are missing on the server.

    Breakdown rows keep full hierarchy for traceability:
      FG SKU · P-Code · Required · Allocated · Status
    Grey planning still targets **P-Code**; FG stays visible for SKU status.
    """
    open_orders = get_open_orders()
    requested = set(so_numbers or [])
    selected = [l for l in open_orders if l.get('so_number') in requested]
    materials: dict = {}
    warnings: list[str] = []
    matched_sos: set[str] = set()

    missing_sos = sorted(requested - {l.get('so_number') for l in selected})
    for so in missing_sos:
        warnings.append(f"{so}: not found among open SOs (it may be Closed/Cancelled or have no lines).")

    try:
        conn = _item_connect()
    except Exception as e:
        warnings.append(f"Item Master DB unreachable: {e}. Verify ITEM_DB_PATH on the server.")
        return {"materials": materials, "warnings": warnings, "matched_sos": [], "missing_sos": missing_sos}

    stock_consumed: dict[str, float] = {}

    def explode(item_code, qty, so_no, sku, depth=0, parent_printed_code=""):
        if depth > 10 or qty <= 0:
            return
        item = _get_item_by_code(conn, item_code)
        if not item:
            return
        bom = _get_default_bom(conn, item['id'])
        if not bom:
            return
        for line in _get_bom_lines(conn, bom['id']):
            ctype = (line.get('component_type') or 'RM').upper()
            if ctype in ('SVC', 'SERVICE', 'PROCESS'):
                continue
            comp = _get_item_by_id(conn, line['component_item_id']) if line.get('component_item_id') else None
            if comp:
                code = comp['item_code']
            else:
                raw = line.get('component_name') or ''
                code = raw.split(' — ')[0].strip() if ' — ' in raw else raw
                comp = _get_item_by_code(conn, code) if code else None
            if not code:
                continue
            adj_qty = float(line.get('quantity') or 0) * (
                1 + float(line.get('shrinkage_pct') or 0) / 100 + float(line.get('wastage_pct') or 0) / 100
            )
            total = round(adj_qty * qty, 3)
            is_printed = _is_printed_component(ctype, code)
            p_code = code if is_printed else (parent_printed_code or "")
            if code not in materials:
                materials[code] = {
                    'name': comp.get('item_name', code) if comp else code,
                    'type': ctype,
                    'unit': line.get('unit', 'PCS'),
                    'total_req': 0.,
                    'stock': float(comp.get('stock') or 0) if comp else 0.,
                    'reserved': 0.,
                    'breakdown': [],
                    'level': depth,
                }
            materials[code]['total_req'] = round(materials[code]['total_req'] + total, 3)
            materials[code]['breakdown'].append({
                'so_no': so_no,
                'sku': sku,
                'fg_sku': sku,
                'p_code': p_code,
                'printed_code': p_code,
                'qty_req': total,
                'allocated_qty': 0.0,
                'status': 'Pending',
            })
            if comp:
                sub = _get_default_bom(conn, comp['id'])
                if sub and [
                    l for l in _get_bom_lines(conn, sub['id'])
                    if (l.get('component_type') or 'RM').upper() not in ('SVC', 'SERVICE', 'PROCESS')
                ]:
                    comp_stock = float(comp.get('stock') or 0)
                    already_used = stock_consumed.get(code, 0.)
                    remaining_stock = max(0., comp_stock - already_used)
                    net_for_sub = max(0., round(total - remaining_stock, 3))
                    stock_consumed[code] = already_used + min(remaining_stock, total)
                    next_printed = code if is_printed else parent_printed_code
                    explode(
                        comp['item_code'],
                        net_for_sub,
                        so_no,
                        sku,
                        depth + 1,
                        parent_printed_code=next_printed,
                    )

    for line in selected:
        so_no = line.get('so_number', '') or ''
        sku = line.get('sku', '') or ''
        qty = (line.get('qty') or 0) - (line.get('produced_qty') or 0)
        if not sku:
            warnings.append(f"{so_no}: line has no SKU — can't compute material requirements.")
            continue
        if qty <= 0:
            continue
        bom_code, anchor, reason = _resolve_bom_anchor(conn, sku)
        if reason:
            warnings.append(f"{so_no} · {sku}: {reason}.")
        if not bom_code or not anchor:
            continue
        before = len(materials)
        explode(bom_code, qty, so_no, sku)
        if len(materials) > before:
            matched_sos.add(so_no)

    conn.close()
    for code, mat in materials.items():
        soft = get_soft_reserved_by_material(code)
        avail = max(0., mat['stock'] - mat['reserved'])
        mat['available'] = avail
        mat['soft_reserved'] = soft
        mat['net_available'] = max(0., avail - soft)
        mat['net_req'] = max(0., round(mat['total_req'] - avail, 3))
        mat['net_req_with_soft'] = max(0., round(mat['total_req'] - mat['net_available'], 3))

    try:
        from ..services.fabric_allocation_engine import annotate_mrp_breakdown_with_allocations

        annotate_mrp_breakdown_with_allocations(materials)
    except Exception:
        pass

    so_skus = sorted({
        str(l.get("sku") or "").strip().upper()
        for l in selected
        if str(l.get("sku") or "").strip()
    })
    embroidery_stock = list_embroidery_stock_for_skus(so_skus)

    return {
        "materials": materials,
        "warnings": warnings,
        "matched_sos": sorted(matched_sos),
        "missing_sos": missing_sos,
        "embroidery_stock": embroidery_stock,
        "embroidery_stock_skus": so_skus,
    }


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
    # None = auto (component JOs when Set BOM exists); False = legacy single JO + receive split
    create_component_jos: Optional[bool] = None

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
    # Cutting receive: when Set BOM exists, explode into component SKUs (default on)
    split_components: Optional[bool] = True
    # Embroidery receive: unused border/yog/etc. returned to stock (measurement units)
    leftover_measurement: Optional[float] = 0


class SetBomMaterialIn(BaseModel):
    material_code: str
    material_name: Optional[str] = ''
    quantity: float = 0
    unit: Optional[str] = 'MTR'
    sort_order: Optional[int] = 0


class SetBomLineIn(BaseModel):
    component_code: str
    component_name: Optional[str] = ''
    qty_per_set: int = 1
    default_next_process: Optional[str] = ''
    routing: Optional[str] = ''  # e.g. Cutting>Embroidery>Cutting>Stitching
    requires_embroidery: Optional[bool] = False
    embroidery_before_cutting: Optional[bool] = False
    embroidery_type: Optional[str] = ''
    embroidery_qty_per_piece: Optional[float] = 0
    embroidery_unit: Optional[str] = ''
    # Blank = infer (FRONT/BACK → PANEL; TOP/BOTTOM → SET_COMPONENT)
    component_role: Optional[str] = ''
    parent_component_code: Optional[str] = ''
    creates_cutting_jo: Optional[bool] = None
    sort_order: Optional[int] = 0
    materials: Optional[List[SetBomMaterialIn]] = []


class SetBomIn(BaseModel):
    style_key: str
    style_name: Optional[str] = ''
    remarks: Optional[str] = ''
    active: Optional[bool] = True
    stitching_requires_complete_set: Optional[bool] = True
    bundle_gate_process: Optional[str] = 'Cutting'
    lines: List[SetBomLineIn]


class SetMatchIn(BaseModel):
    so_number: str
    main_sku: str
    match_qty: Optional[int] = None
    from_process: Optional[str] = 'Finishing'
    to_process: Optional[str] = 'Packing'
    matched_by: Optional[str] = ''
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
    """Get process routing for a specific SKU (honors component-level Set BOM routing)."""
    from ..services.set_components import parse_component_sku

    main, comp = parse_component_sku(sku)
    emb_before = False
    requires_emb = False
    emb_type = ""
    emb_per_piece = 0.0
    emb_unit = ""
    if comp:
        bom = get_set_bom_for_sku(main or sku)
        if bom:
            for ln in bom.get("lines") or []:
                if str(ln.get("component_code") or "").strip().upper() == str(comp).upper():
                    requires_emb = bool(int(ln.get("requires_embroidery") or 0))
                    emb_before = bool(int(ln.get("embroidery_before_cutting") or 0))
                    emb_type = str(ln.get("embroidery_type") or "")
                    emb_per_piece = float(ln.get("embroidery_qty_per_piece") or 0)
                    emb_unit = str(ln.get("embroidery_unit") or "")
                    break
    return {
        "sku": sku,
        "routing": get_component_routing(sku),
        "item_routing": get_item_routing(sku),
        "next_after_cutting": get_next_process(sku, "Cutting"),
        "requires_embroidery": requires_emb,
        "embroidery_before_cutting": emb_before,
        "embroidery_type": emb_type,
        "embroidery_qty_per_piece": emb_per_piece,
        "embroidery_unit": emb_unit,
        "embroidery_stock": list_embroidery_stock_for_sku(sku),
    }


@router.get("/embroidery-stock/{sku}")
def embroidery_stock_for_sku(sku: str):
    """Leftover embroidery material (border meters, Yog, etc.) for a SKU/style."""
    return {"sku": sku, "items": list_embroidery_stock_for_sku(sku)}


@router.get("/bundle-ready")
def get_bundle_ready(
    so_number: str,
    main_sku: str,
    parent_component_code: Optional[str] = None,
):
    """Stitching gate preview — component/panel bundle ready at the gate?

    Pass ``parent_component_code`` (e.g. TOP) to check only that component's
    panels. Omit it for full-style set-match readiness (TOP+PANT+DUPATTA).
    """
    if not so_number or not main_sku:
        raise HTTPException(400, "so_number and main_sku are required")
    return preview_bundle_ready(
        so_number,
        main_sku,
        parent_component_code=parent_component_code,
    )


@router.get("/wip-board")
def get_wip_board(so_number: str, main_sku: str):
    """Panel-level WIP locations and statuses for partial operation routing."""
    if not so_number or not main_sku:
        raise HTTPException(400, "so_number and main_sku are required")
    return get_partial_wip_board(so_number, main_sku)


# ── Ready to Process ───────────────────────────────────────────────────────────

@router.get("/ready-to-process/{process}")
def ready_to_process(
    process: str,
    q: Optional[str] = None,
    jo: Optional[str] = None,
    sku: Optional[str] = None,
    vendor: Optional[str] = None,
    min_qty: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """Get lines ready to be processed at given stage (with search/filters)."""
    return get_ready_to_process(
        process,
        q=q,
        jo=jo,
        sku=sku,
        vendor=vendor,
        min_qty=min_qty,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/ready-to-wip/import-template")
def ready_to_wip_template(stage: str = "Stitching"):
    """CSV template for Ready-To WIP migration."""
    rows = ready_to_wip_template_rows(stage)
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="ready_to_{stage.lower()}_wip_template.csv"'
        },
    )


@router.post("/ready-to-wip/import")
async def ready_to_wip_import(
    file: UploadFile = File(...),
    stage: Optional[str] = Form(None),
):
    """Import existing Ready-To WIP into process_stock (from_process feeder)."""
    raw = await file.read()
    name = (file.filename or "wip.csv").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            # utf-8-sig strips Excel BOM; sep=None with engine python can mis-detect —
            # try comma first then semicolon.
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", sep=";")
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}") from e
    if df is None or df.empty:
        raise HTTPException(400, "Empty import file")
    # Strip BOM / whitespace from headers so OMS_SKU etc. resolve after Excel exports.
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    rows = df.fillna("").to_dict(orient="records")
    try:
        result = import_ready_to_wip(rows, default_stage=stage)
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    return result


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


@router.get("/orders/import-template")
def download_jo_import_template():
    """CSV template for bulk JO import (main SKU + optional component_code; panels not separate rows)."""
    return Response(
        content=jo_import_template_csv(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="production_jo_import_template.csv"'
        },
    )


@router.post("/orders/import")
async def import_jos(
    file: UploadFile = File(...),
    process: str = Form("Cutting"),
):
    """
    Import job orders from CSV/XLSX.

    Cutting (recommended): one row per main size SKU → auto-creates TOP/PANT/DUPATTA
    Cutting JOs from Set BOM. FRONT/BACK are panels under TOP — do not import them
    as separate rows; they appear on the TOP JO after Receive.

    Optional ``component_code`` = TOP | PANT | DUPATTA creates only that Cutting JO.
    """
    raw = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}") from e
    if df.empty:
        raise HTTPException(400, "Import file is empty")
    # Preserve original headers for Ready-To WIP detection, then normalize for JO parse.
    raw_columns = [str(c) for c in df.columns]
    if looks_like_ready_to_wip_columns(raw_columns):
        # User often uploads ready_to_*_wip_template.csv via the JO import button.
        rows = df.fillna("").to_dict(orient="records")
        try:
            wip = import_ready_to_wip(rows, default_stage=process or "Stitching")
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        errs = wip.get("errors") or []
        imported = int(wip.get("imported") or 0)
        return {
            "ok": True,
            "kind": "ready_to_wip",
            "created": 0,
            "imported": imported,
            "import_batch": wip.get("import_batch"),
            "jo_numbers": [],
            "errors": errs,
            "message": (
                f"Detected Ready-To WIP template — credited {imported} stock row(s)"
                + (f"; {len(errs)} failed" if errs else "")
                + ". (Use ⬇ WIP template + Import Ready-To WIP for this format.)"
            ),
            "hint": (
                "This file stocks Ready-To for the stage (Cutting → Stitching feeder). "
                "To create Job Orders, download production_jo_import_template.csv instead."
            ),
        }
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    created: list[str] = []
    errors: list[str] = []
    for i, row in df.iterrows():
        try:
            data = build_jo_payload_from_import_row(
                row.to_dict(),
                default_process=process,
            )
            result = create_jo(data)
            if isinstance(result, list):
                created.extend(result)
            else:
                created.append(result)
        except Exception as e:
            sku_hint = str(row.get("sku") or row.get("oms_sku") or "").strip() or "?"
            errors.append(f"Row {int(i) + 2} ({sku_hint}): {e}")
    return {
        "ok": True,
        "kind": "job_order",
        "created": len(created),
        "jo_numbers": created,
        "errors": errors,
        "message": f"Imported {len(created)} job order(s)"
        + (f"; {len(errors)} failed" if errors else ""),
        "hint": (
            "FRONT/BACK are panels inside the TOP Cutting JO after Receive — "
            "not separate import rows. Use one main size SKU row, or component_code=TOP/PANT/DUPATTA. "
            "For existing Ready-To stock, use ready_to_*_wip_template.csv + Import Ready-To WIP."
        ),
    }


@router.get("/orders/{joid}")
def get_jo_detail(joid: int):
    jo = get_jo(joid)
    if not jo:
        raise HTTPException(404, "Job order not found")
    return jo


@router.get("/orders/{joid}/panel-wip")
def get_jo_panel_wip_detail(joid: int):
    """Child panel WIP for a Cutting JO (TOP → FRONT/BACK, etc.)."""
    return get_jo_panel_wip(joid)

@router.post("/orders")
def post_jo(body: JOIn):
    data = body.model_dump()
    exec_type = str(data.get("exec_type") or "Inhouse").strip()
    vendor_name = str(data.get("vendor_name") or "").strip()
    if exec_type.lower() == "outsource" and not vendor_name:
        raise HTTPException(400, "Vendor name is required when execution type is Outsource.")
    data["exec_type"] = exec_type
    data["vendor_name"] = vendor_name
    process = data.get('process','Cutting')
    so_number = data.get('so_number','')
    sku = data.get('sku','')
    planned_qty = int(data.get('planned_qty') or 0)
    if process != 'Cutting' and so_number and sku and planned_qty:
        v = validate_jo_creation(process, so_number, sku, planned_qty)
        if not v['ok']:
            raise HTTPException(400, v['message'])
    fabric_code = (data.get("fabric_code") or "").strip()
    fabric_qty = float(data.get("fabric_qty") or 0)
    if so_number and fabric_code and fabric_qty > 0:
        try:
            check_mrp_commitment(so_number, fabric_code, fabric_qty)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    try:
        result = create_jo(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if isinstance(result, list):
        orders = []
        for num in result:
            jo_row = next((j for j in list_jos() if j.get("jo_number") == num), None)
            issue_note = jo_issue_notes.get_issue_note_by_jo_id(jo_row["id"]) if jo_row else None
            orders.append(
                {
                    "jo_number": num,
                    "id": jo_row["id"] if jo_row else None,
                    "sku": jo_row.get("sku") if jo_row else None,
                    "component_code": jo_row.get("component_code") if jo_row else None,
                    "issue_note": issue_note,
                }
            )
        return {
            "ok": True,
            "component_jos": True,
            "jo_numbers": result,
            "orders": orders,
            "message": f"Created {len(result)} component Cutting JO(s)",
        }
    num = result
    jo_row = next((j for j in list_jos() if j.get("jo_number") == num), None)
    issue_note = jo_issue_notes.get_issue_note_by_jo_id(jo_row["id"]) if jo_row else None
    return {"jo_number": num, "ok": True, "issue_note": issue_note}


@router.patch("/orders/{joid}")
def patch_jo(joid: int, body: JOUpdate):
    jo = get_jo(joid)
    if not jo:
        raise HTTPException(404, "Job order not found")
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    if "exec_type" in data:
        data["exec_type"] = str(data["exec_type"] or "Inhouse").strip()
    if "vendor_name" in data:
        data["vendor_name"] = str(data["vendor_name"] or "").strip()
    eff_exec = str(data.get("exec_type") or jo.get("exec_type") or "Inhouse").strip()
    eff_vendor = str(
        data["vendor_name"] if "vendor_name" in data else jo.get("vendor_name") or ""
    ).strip()
    if ("exec_type" in data or "vendor_name" in data):
        if eff_exec.lower() == "outsource" and not eff_vendor:
            raise HTTPException(400, "Vendor name is required when execution type is Outsource.")
        if eff_exec.lower() == "inhouse":
            data["vendor_name"] = ""
    update_jo(joid, data)
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
        result = issue_pieces(joid, body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result if isinstance(result, dict) else {"ok": True}

@router.post("/orders/{joid}/receive-pieces")
def post_receive_pieces(joid: int, body: PieceReceiptIn):
    try:
        result = receive_pieces(joid, body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Receive failed: {e}")
    return result if isinstance(result, dict) else {"ok": True}


# ── Set BOM + Set Match ────────────────────────────────────────────────────────

@router.get("/set-bom")
def get_set_bom_list():
    return list_set_boms()


@router.get("/set-bom/{style_key}")
def get_set_bom_detail(style_key: str):
    bom = get_set_bom(style_key)
    if not bom:
        raise HTTPException(404, f"No Set BOM for '{style_key}'")
    return bom


@router.get("/set-bom-for-sku/{sku:path}")
def get_set_bom_by_sku(sku: str):
    from ..services.component_bom import (
        effective_set_bom_for_cutting,
        panel_lines,
        set_component_lines,
    )

    bom = effective_set_bom_for_cutting(sku) or get_set_bom_for_sku(sku)
    cutting = set_component_lines(bom)
    panels = panel_lines(bom)
    return {
        "sku": sku,
        "bom": bom,
        "has_set_bom": bool(cutting),
        "cutting_components": cutting,
        "panels": panels,
    }


@router.post("/set-bom")
def post_set_bom(body: SetBomIn):
    try:
        return upsert_set_bom(body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/set-bom/{style_key}")
def remove_set_bom(style_key: str):
    if not delete_set_bom(style_key):
        raise HTTPException(404, f"No Set BOM for '{style_key}'")
    return {"ok": True}


@router.get("/set-match")
def get_set_match_preview(
    so_number: str,
    main_sku: str,
    from_process: str = "Finishing",
):
    try:
        return preview_set_match(so_number, main_sku, from_process)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/set-match")
def post_set_match(body: SetMatchIn):
    try:
        return commit_set_match(body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/set-events/splits")
def get_set_splits(so_number: Optional[str] = None, main_sku: Optional[str] = None):
    return list_set_split_events(so_number or "", main_sku or "")


@router.get("/set-events/matches")
def get_set_matches(so_number: Optional[str] = None, main_sku: Optional[str] = None):
    return list_set_match_events(so_number or "", main_sku or "")

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


# ── Material issue notes (JO-linked, BOM-driven) ─────────────────────────────

@router.get("/issue-notes")
def get_production_issue_notes(jo_number: Optional[str] = None, status: Optional[str] = None):
    return jo_issue_notes.list_issue_notes(jo_number=jo_number, status=status)


@router.get("/orders/{joid}/issue-note")
def get_production_issue_note(joid: int):
    note = jo_issue_notes.get_issue_note_by_jo_id(joid)
    if not note:
        raise HTTPException(404, "Issue note not found for this job order")
    return note


@router.post("/orders/{joid}/regenerate-issue-note")
def regenerate_production_issue_note(joid: int):
    jo = get_jo(joid)
    if not jo:
        raise HTTPException(404, "Job order not found")
    note = jo_issue_notes.create_issue_note_for_jo(
        joid, jo["jo_number"], jo, jo.get("lines") or []
    )
    return {"ok": True, "issue_note": note}


# ── BOM Inputs ────────────────────────────────────────────────────────────────

@router.get("/bom-inputs/{item_code}")
def get_bom_inputs(item_code: str, qty: float = 1.0):
    try:
        conn = _item_connect()
        bom_code, item, reason = _resolve_bom_anchor(conn, item_code)
        if not item:
            conn.close()
            return {'inputs': [], 'error': reason or f"SKU '{item_code}' not found"}
        bom = _get_default_bom(conn, item['id'])
        if not bom:
            conn.close()
            return {'inputs': [], 'error': reason or f"No BOM for '{item_code}'"}
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
        return {
            'item_code': item_code,
            'bom_item_code': bom_code or item.get('item_code', item_code),
            'item_name': item['item_name'],
            'inputs': inputs,
        }
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

def _normalize_mrp_payload(payload):
    """Accept both new (``{materials, warnings, ...}``) and legacy (flat dict) shapes."""
    if isinstance(payload, dict) and "materials" in payload:
        return {
            "materials": payload.get("materials") or {},
            "warnings": payload.get("warnings") or [],
            "matched_sos": payload.get("matched_sos") or [],
            "missing_sos": payload.get("missing_sos") or [],
            "embroidery_stock": payload.get("embroidery_stock") or [],
            "embroidery_stock_skus": payload.get("embroidery_stock_skus") or [],
        }
    return {
        "materials": payload or {},
        "warnings": [],
        "matched_sos": [],
        "missing_sos": [],
        "embroidery_stock": [],
        "embroidery_stock_skus": [],
    }


@router.post("/mrp/run")
def run_mrp_full(body: MRPRunBody):
    if not body.so_numbers:
        return {
            'run_time': datetime.now().isoformat(),
            'so_numbers': [],
            'result': {},
            'warnings': ['Select at least one SO before running material requirement planning.'],
            'matched_sos': [],
            'missing_sos': [],
            'embroidery_stock': [],
        }
    payload = calculate_mrp(body.so_numbers)
    save_mrp_result(body.so_numbers, payload)
    norm = _normalize_mrp_payload(payload)
    try:
        from ..services.fabric_allocation_engine import annotate_mrp_breakdown_with_allocations

        annotate_mrp_breakdown_with_allocations(norm["materials"])
    except Exception:
        pass
    sync_mrp_commitments_from_run(body.so_numbers, norm['materials'])
    return {
        'run_time': datetime.now().isoformat(),
        'so_numbers': body.so_numbers,
        'result': norm['materials'],
        'warnings': norm['warnings'],
        'matched_sos': norm['matched_sos'],
        'missing_sos': norm['missing_sos'],
        'embroidery_stock': norm['embroidery_stock'],
    }

@router.get("/mrp/last")
def get_last_mrp():
    data = get_last_mrp_result()
    if not data:
        return {
            'run_time': None,
            'so_numbers': [],
            'result': {},
            'warnings': [],
            'matched_sos': [],
            'missing_sos': [],
            'embroidery_stock': [],
        }
    norm = _normalize_mrp_payload(data.get('result'))
    # Live allocation status so hierarchy columns stay current without re-run.
    try:
        from ..services.fabric_allocation_engine import annotate_mrp_breakdown_with_allocations

        annotate_mrp_breakdown_with_allocations(norm["materials"])
    except Exception:
        pass
    # Refresh leftover stock live so MRP UI always shows current balances
    # even when the last planning snapshot is older than a receive/leftover credit.
    so_skus: list[str] = []
    try:
        open_orders = get_open_orders()
        so_set = set(data.get('so_numbers') or [])
        so_skus = sorted({
            str(l.get("sku") or "").strip().upper()
            for l in open_orders
            if l.get("so_number") in so_set and str(l.get("sku") or "").strip()
        })
    except Exception:
        so_skus = list(norm.get("embroidery_stock_skus") or [])
    live_stock = list_embroidery_stock_for_skus(so_skus) if so_skus else list(norm.get("embroidery_stock") or [])
    return {
        'run_time': data.get('run_time'),
        'so_numbers': data.get('so_numbers', []),
        'result': norm['materials'],
        'warnings': norm['warnings'],
        'matched_sos': norm['matched_sos'],
        'missing_sos': norm['missing_sos'],
        'embroidery_stock': live_stock,
    }

@router.get("/mrp/embroidery-stock")
def mrp_embroidery_stock(so_numbers: str = ""):
    """Leftover Border / Yog / Boota stock for SKUs on the given SOs.

    Surfaces embroidery leftovers on the MRP / PO planning screen before the
    next purchase or job order is placed for those styles.
    """
    so_list = [s.strip() for s in str(so_numbers or "").split(",") if s.strip()]
    if not so_list:
        return {"so_numbers": [], "items": []}
    so_set = set(so_list)
    open_orders = get_open_orders()
    skus = sorted({
        str(l.get("sku") or "").strip().upper()
        for l in open_orders
        if l.get("so_number") in so_set and str(l.get("sku") or "").strip()
    })
    return {
        "so_numbers": so_list,
        "skus": skus,
        "items": list_embroidery_stock_for_skus(skus),
    }


@router.get("/mrp/lines-for-so")
def get_mrp_lines_for_so(so_number: str = ''):
    data = get_last_mrp_result()
    if not data:
        return {'purchase_items': [], 'sfg_items': [], 'error': 'No planning result. Run material requirement planning first.'}
    norm = _normalize_mrp_payload(data.get('result'))
    result = norm['materials']
    so_numbers = data.get('so_numbers', [])
    if so_number and so_number not in so_numbers:
        return {'purchase_items': [], 'sfg_items': [], 'warning': f'{so_number} not in last planning run'}
    commitments = {c['material_code']: c for c in get_mrp_commitments_for_so(so_number)} if so_number else {}
    purchase_items, sfg_items = [], []
    for code, mat in result.items():
        so_gross = (
            sum(bd['qty_req'] for bd in mat.get('breakdown', []) if bd.get('so_no') == so_number)
            if so_number
            else mat['total_req']
        )
        if so_gross <= 0:
            continue
        net_req = round(float(mat.get('net_req_with_soft', mat.get('net_req', 0)) or 0), 3)
        if net_req <= 1e-9:
            continue
        total_req = round(float(mat.get('total_req') or 0), 3)
        so_net_req = (
            round(net_req * (so_gross / total_req), 3)
            if so_number and total_req > 1e-9
            else net_req
        )
        commit = commitments.get(code, {})
        po_c = float(commit.get('po_committed_qty') or 0)
        jo_c = float(commit.get('jo_committed_qty') or 0)
        mrp_qty = float(commit.get('mrp_qty') or 0)
        if mrp_qty <= 0:
            mrp_qty = so_net_req
        remaining = round(max(0.0, mrp_qty - po_c - jo_c), 3)
        if remaining <= 1e-9:
            continue
        stock = round(float(mat.get('stock') or 0), 3)
        reserved = round(float(mat.get('reserved') or 0), 3)
        available = round(float(mat.get('available') or max(0.0, stock - reserved)), 3)
        item_data = {
            'material_code': code,
            'material_name': mat['name'],
            'material_type': mat.get('type', 'RM'),
            'required_qty': round(so_gross, 3),
            'total_req': total_req,
            'stock': stock,
            'available': available,
            'unit': mat['unit'],
            'net_req': so_net_req,
            'mrp_qty': mrp_qty,
            'po_committed_qty': round(po_c, 3),
            'jo_committed_qty': round(jo_c, 3),
            'remaining_qty': remaining,
            'can_create_po': bool(commit.get('can_create_po', remaining > 0)),
            'can_create_jo': bool(commit.get('can_create_jo', remaining > 0)),
            'commitment_status': commit.get('status', 'Open'),
        }
        if mat.get('type','').upper() == 'SFG':
            sfg_items.append(item_data)
        else:
            purchase_items.append(item_data)
    return {'so_number': so_number, 'purchase_items': sorted(purchase_items, key=lambda x: x['material_type']), 'sfg_items': sfg_items}


@router.get("/mrp/commitments")
def mrp_commitments(so_number: str = ''):
    if not so_number:
        raise HTTPException(400, 'so_number is required')
    return get_mrp_commitments_for_so(so_number)


@router.post("/mrp/resync-commitments")
def mrp_resync_commitments():
    """Rebuild per-SO commitment totals from the last MRP snapshot."""
    data = get_last_mrp_result()
    if not data:
        return {"ok": False, "message": "No MRP run on file."}
    norm = _normalize_mrp_payload(data.get("result"))
    sync_mrp_commitments_from_run(data.get("so_numbers") or [], norm.get("materials") or {})
    return {"ok": True, "so_numbers": data.get("so_numbers") or []}


@router.get("/mrp/audit-chain")
def mrp_audit_chain(so_number: str = ''):
    from ..services.document_chain_audit import get_document_chain_audit
    if not so_number.strip():
        raise HTTPException(400, 'so_number is required')
    return get_document_chain_audit(so_number.strip())

@router.post("/mrp/soft-reserve-all")
def mrp_soft_reserve_all():
    data = get_last_mrp_result()
    if not data:
        return {'ok': False, 'message': 'No planning result. Run material requirement planning first.'}
    norm = _normalize_mrp_payload(data.get('result'))
    reservations = []
    for mat_code, mat in norm['materials'].items():
        for bd in mat.get('breakdown', []):
            reservations.append({'material_code': mat_code, 'material_name': mat.get('name', mat_code),
                                 'unit': mat.get('unit', 'PCS'), 'so_no': bd['so_no'],
                                 'sku': bd.get('sku', ''), 'qty': bd.get('qty_req', 0)})
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
