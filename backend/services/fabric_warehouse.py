"""Fabric Warehouse — read-only dashboard + date-wise ledger for printed fabric (P-Codes).

This module is a *view* over records that already exist; it never posts stock and
has no parallel stock engine:

    GRN receipt (JWO)          grey.db   printed_fabric_stock           -> IN  Unchecked
    Printed QC pass / reject   grey.db   printed_fabric_stock           -> Unchecked -> Checked / OUT
    Opening / conversions      items.db  item_stock_adjustments         -> IN  Opening pool
                               grey.db   printed_fabric_opening_conversions -> Opening -> Checked
    Purchase MIN (confirmed)   purchase.db material_issue_notes/min_lines -> OUT Checked
    Cutting fabric issue       production.db jo_fabric_issues           -> OUT Checked
    Fabric return from JO      production.db jo_fabric_returns          -> IN  Checked
    Manual stock adjustments   items.db  item_stock_adjustments         -> IN/OUT Checked
    Reservations (SO/Advance)  grey.db   printed_fabric_reservations    -> allocation only
                               grey.db   fabric_allocation_history      (release / consume)

Every event carries per-warehouse deltas (``checked_delta`` / ``unchecked_delta`` /
``opening_delta``) and allocation deltas (``so_alloc_delta`` / ``adv_alloc_delta``).
Reservations set ``physical_move=False`` and only ever touch the allocation deltas, so
they reduce *Free* but never count as OUT.

Free = physical (checked + opening pool) - SO allocated - Advance reserved.

Stock Category (Phase 2) — ``stock_category`` = ``sfg`` (default) | ``grey`` | ``all``:

    SFG   printed / dyed fabric (everything above).  A P-Code is SFG when it appears in any
          ``printed_fabric_*`` table, or Item Master type is SFG, or it is not known as grey.
    Grey  grey (greige) fabric — Item Master type "Grey Fabric" (GF), or the code appears in the
          grey tracker / grey checked stock / grey hard reservations and is not printed.
          Grey events come from grey_tracker (receipt), fabric_check_log (QC pass / reject),
          hard_reservations (SO allocation), confirmed MINs and Item Master opening / adjustments.

The category is independent of the Checked / Unchecked warehouse filter.
"""
from __future__ import annotations

import csv
import io
import os
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

_EPS = 1e-6

WAREHOUSES = ("all", "checked", "unchecked")
DIRECTIONS = ("all", "in", "out", "alloc")
STOCK_CATEGORIES = ("sfg", "grey", "all")
DEFAULT_STOCK_CATEGORY = "sfg"

_GREY_TYPE_CODES = {"GF", "GREY", "GREIGE"}
_GREY_TYPE_NAMES = {"grey fabric", "greige", "greige fabric"}

# Ledger txn types (stable strings — the UI filters on them).
T_GRN = "GRN Receipt"
T_QC_PASS = "QC Pass (Unchecked→Checked)"
T_QC_REJECT = "QC Reject"
T_OPENING = "Opening Stock"
T_OPENING_CONV = "Opening → Checked"
T_MIN = "Purchase MIN Issue"
T_JO_ISSUE = "JO Fabric Issue"
T_JO_RETURN = "JO Fabric Return"
T_ADJ = "Adjustment"
T_RESERVE_SO = "SO Allocation"
T_RESERVE_ADV = "Advance Reservation"
T_RELEASE = "Reservation Released"
T_CONSUME = "Reservation Consumed (Cutting)"

ALL_TXN_TYPES = [
    T_GRN, T_QC_PASS, T_QC_REJECT, T_OPENING, T_OPENING_CONV, T_MIN,
    T_JO_ISSUE, T_JO_RETURN, T_ADJ, T_RESERVE_SO, T_RESERVE_ADV, T_RELEASE, T_CONSUME,
]


# ── DB path helpers (resolve at call time so tests can monkeypatch module paths) ──

def _grey_path() -> str:
    from ..db import grey_db

    return grey_db._DB


def _purchase_path() -> str:
    from ..db import purchase_db

    return purchase_db._DB


def _production_path() -> str:
    from ..db import production_db

    return production_db._DB


def _open(path: str) -> Optional[sqlite3.Connection]:
    if not path or not os.path.exists(path):
        return None
    try:
        conn = sqlite3.connect(path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _rows(conn: Optional[sqlite3.Connection], sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    if conn is None:
        return []
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def _items_rows(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    try:
        from ..db import item_db

        conn = item_db._connect()
    except Exception:
        return []
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _day(v: Any) -> str:
    s = str(v or "").strip()
    return s[:10] if len(s) >= 10 else s


def _norm(code: Any) -> str:
    return str(code or "").strip()


def norm_category(value: Any) -> str:
    v = str(value or DEFAULT_STOCK_CATEGORY).strip().lower()
    return v if v in STOCK_CATEGORIES else DEFAULT_STOCK_CATEGORY


def _grey_code_set(grey: Optional[sqlite3.Connection]) -> set:
    """Upper-cased P-Codes classified as Grey Fabric (everything else is SFG).

    Precedence: printed_fabric_* tables (SFG) > Item Master type SFG (SFG) >
    Item Master type Grey Fabric / grey tracker-checked-reservation tables (Grey) > SFG.
    """
    printed: set = set()
    for tbl in ("printed_fabric_stock", "printed_fabric_checked_stock",
                "printed_fabric_reservations", "printed_fabric_opening_conversions"):
        for r in _rows(grey, f"SELECT DISTINCT UPPER(TRIM(fabric_code)) AS c FROM {tbl}"):
            if r["c"]:
                printed.add(str(r["c"]))
    grey_codes: set = set()
    sfg_typed: set = set()
    for r in _items_rows(
        """SELECT UPPER(TRIM(i.item_code)) AS c, UPPER(COALESCE(t.code,'')) AS tcode,
                  LOWER(COALESCE(t.name,'')) AS tname
           FROM items i JOIN item_types t ON t.id = i.item_type_id"""
    ):
        if not r["c"]:
            continue
        if r["tcode"] in _GREY_TYPE_CODES or r["tname"] in _GREY_TYPE_NAMES:
            grey_codes.add(str(r["c"]))
        elif r["tcode"] == "SFG":
            sfg_typed.add(str(r["c"]))
    for tbl, col in (
        ("grey_tracker", "material_code"),
        ("fabric_checked_stock", "fabric_code"),
        ("fabric_check_log", "fabric_code"),
        ("hard_reservations", "fabric_code"),
    ):
        for r in _rows(grey, f"SELECT DISTINCT UPPER(TRIM({col})) AS c FROM {tbl}"):
            if r["c"] and str(r["c"]) not in sfg_typed:
                grey_codes.add(str(r["c"]))
    return grey_codes - printed


def category_of(code: Any, grey_codes: set) -> str:
    return "grey" if _norm(code).upper() in grey_codes else "sfg"


def _cat_ok(category: str, want: str) -> bool:
    return want == "all" or category == want


def _event(
    *,
    day: Any,
    ts: Any = "",
    pcode: str,
    name: str = "",
    txn_type: str,
    ref_type: str = "",
    ref_no: str = "",
    from_wh: str = "",
    to_wh: str = "",
    qty: float = 0.0,
    checked: float = 0.0,
    unchecked: float = 0.0,
    opening: float = 0.0,
    so_alloc: float = 0.0,
    adv_alloc: float = 0.0,
    physical_move: bool = True,
    so_ref: str = "",
    jo_ref: str = "",
    user: str = "",
    remarks: str = "",
    source: str = "",
) -> Dict[str, Any]:
    d = _day(day)
    return {
        "date": d,
        "ts": str(ts or d),
        "pcode": _norm(pcode),
        "name": name or "",
        "txn_type": txn_type,
        "ref_type": ref_type,
        "ref_no": ref_no or "",
        "from_wh": from_wh,
        "to_wh": to_wh,
        "qty": round(abs(qty), 3),
        "checked_delta": round(checked, 3),
        "unchecked_delta": round(unchecked, 3),
        "opening_delta": round(opening, 3),
        "so_alloc_delta": round(so_alloc, 3),
        "adv_alloc_delta": round(adv_alloc, 3),
        "physical_move": bool(physical_move),
        "so_ref": so_ref or "",
        "jo_ref": jo_ref or "",
        "user": user or "",
        "remarks": remarks or "",
        "source": source,
    }


# ── Source collectors ─────────────────────────────────────────────────────────

def _opening_reason(reason: str, reference_no: str) -> Optional[bool]:
    """True = opening-type adjustment, False = manual adjustment, None = skip (document auto-ADJ)."""
    from ..db.item_db import is_document_auto_adjustment

    if is_document_auto_adjustment(str(reason or ""), str(reference_no or "")):
        return None
    r = str(reason or "").lower()
    return ("opening" in r) or ("existing" in r)


def _collect_grn_and_qc(grey: Optional[sqlite3.Connection]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _rows(grey, "SELECT * FROM printed_fabric_stock"):
        code = _norm(r["fabric_code"])
        if not code:
            continue
        remaining = _f(r["qty"])
        passed = _f(r["passed_qty"]) if "passed_qty" in r.keys() else 0.0
        failed = _f(r["failed_qty"]) if "failed_qty" in r.keys() else 0.0
        received = remaining + passed + failed
        grn_ref = r["grn_ref"] or ""
        jwo = r["jwo_ref"] or ""
        recv_date = r["receive_date"] or _day(r["created_at"])
        name = r["fabric_name"] or ""
        printer = r["printer"] or ""
        if received > _EPS:
            out.append(_event(
                day=recv_date, ts=r["created_at"] or recv_date, pcode=code, name=name,
                txn_type=T_GRN, ref_type="GRN", ref_no=grn_ref or jwo,
                from_wh=printer or "Printer / Vendor", to_wh="Unchecked", qty=received,
                unchecked=received, jo_ref=jwo,
                remarks=f"JWO {jwo}".strip() if jwo else "", source="printed_fabric_stock",
            ))
        qc_day = (r["qc_date"] if "qc_date" in r.keys() else "") or recv_date
        qc_by = (r["qc_by"] if "qc_by" in r.keys() else "") or ""
        qc_ts = f"{_day(qc_day)} 99:{r['id']:08d}"  # QC sorts after same-day receipt
        if passed > _EPS:
            out.append(_event(
                day=qc_day, ts=qc_ts, pcode=code, name=name,
                txn_type=T_QC_PASS, ref_type="QC", ref_no=grn_ref or jwo,
                from_wh="Unchecked", to_wh="Checked", qty=passed,
                unchecked=-passed, checked=passed, jo_ref=jwo, user=qc_by,
                source="printed_fabric_stock",
            ))
        if failed > _EPS:
            out.append(_event(
                day=qc_day, ts=qc_ts, pcode=code, name=name,
                txn_type=T_QC_REJECT, ref_type="QC", ref_no=grn_ref or jwo,
                from_wh="Unchecked", to_wh="Rejected / Return", qty=failed,
                unchecked=-failed, jo_ref=jwo, user=qc_by,
                source="printed_fabric_stock",
            ))
    return out


def _collect_opening(grey: Optional[sqlite3.Connection]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # Item Master opening / existing stock adjustments
    adj_rows = _items_rows(
        """SELECT a.id, a.item_code, a.entry_date, a.direction, a.qty, a.reason,
                  a.reference_no, a.created_by, a.created_at, i.item_name
           FROM item_stock_adjustments a LEFT JOIN items i ON i.id = a.item_id
           ORDER BY a.entry_date, a.id"""
    )
    for a in adj_rows:
        code = _norm(a["item_code"])
        kind = _opening_reason(a["reason"], a["reference_no"])
        if kind is None or not code:
            continue
        qty = _f(a["qty"])
        sign = 1.0 if str(a["direction"] or "").upper() == "IN" else -1.0
        base = dict(
            day=a["entry_date"], ts=a["created_at"] or a["entry_date"], pcode=code,
            name=a["item_name"] or "", qty=qty, user=a["created_by"] or "",
            remarks=a["reason"] or "", source="item_stock_adjustments",
        )
        if kind:
            out.append(_event(
                txn_type=T_OPENING, ref_type="ADJ", ref_no=f"ADJ-{a['id']}",
                from_wh="" if sign > 0 else "Opening", to_wh="Opening" if sign > 0 else "",
                opening=sign * qty, **base,
            ))
        else:
            out.append(_event(
                txn_type=T_ADJ, ref_type="ADJ", ref_no=f"ADJ-{a['id']}",
                from_wh="" if sign > 0 else "Checked", to_wh="Checked" if sign > 0 else "",
                checked=sign * qty, **base,
            ))
    # Opening converted into checked stock (Opening pool -> Checked)
    for r in _rows(grey, "SELECT * FROM printed_fabric_opening_conversions ORDER BY id"):
        code = _norm(r["fabric_code"])
        qty = _f(r["qty"])
        if not code or qty <= _EPS:
            continue
        out.append(_event(
            day=r["created_at"], ts=r["created_at"], pcode=code,
            txn_type=T_OPENING_CONV, ref_type="OPENING", ref_no=f"OPC-{r['id']}",
            from_wh="Opening", to_wh="Checked", qty=qty, opening=-qty, checked=qty,
            user=r["user_name"] or "", remarks=r["reason"] or "",
            source="printed_fabric_opening_conversions",
        ))
    return out


def _collect_min(pur: Optional[sqlite3.Connection], codes: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not codes:
        return out
    for r in _rows(
        pur,
        """SELECT h.id AS min_id, h.min_number, h.min_date, h.jwo_reference, h.so_reference,
                  h.to_vendor, h.to_location, h.issued_by, h.created_at,
                  l.material_code, l.material_name, l.issue_qty
           FROM min_lines l JOIN material_issue_notes h ON h.id = l.min_id
           WHERE h.status IN ('Confirmed','Issued')""",
    ):
        code = _norm(r["material_code"])
        qty = _f(r["issue_qty"])
        if code not in codes or qty <= _EPS:
            continue
        party = r["to_vendor"] or r["to_location"] or "Vendor"
        out.append(_event(
            day=r["min_date"], ts=r["created_at"] or r["min_date"], pcode=code,
            name=r["material_name"] or "", txn_type=T_MIN, ref_type="MIN",
            ref_no=r["min_number"] or "", from_wh="Checked", to_wh=party, qty=qty,
            checked=-qty, so_ref=r["so_reference"] or "", jo_ref=r["jwo_reference"] or "",
            user=r["issued_by"] or "", source="material_issue_notes",
        ))
    return out


def _collect_jo(prod: Optional[sqlite3.Connection]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _rows(
        prod,
        """SELECT fi.id, fi.issue_date, fi.fabric_code, fi.fabric_name, fi.issued_qty,
                  fi.issued_by, fi.remarks, fi.created_at, j.jo_number, j.so_number, j.process
           FROM jo_fabric_issues fi LEFT JOIN job_orders j ON j.id = fi.jo_id""",
    ):
        code = _norm(r["fabric_code"])
        qty = _f(r["issued_qty"])
        if not code or qty <= _EPS:
            continue
        out.append(_event(
            day=r["issue_date"], ts=r["created_at"] or r["issue_date"], pcode=code,
            name=r["fabric_name"] or "", txn_type=T_JO_ISSUE, ref_type="JO",
            ref_no=r["jo_number"] or "", from_wh="Checked",
            to_wh=f"{r['process'] or 'Cutting'} (JO)", qty=qty, checked=-qty,
            so_ref=r["so_number"] or "", jo_ref=r["jo_number"] or "",
            user=r["issued_by"] or "", remarks=r["remarks"] or "", source="jo_fabric_issues",
        ))
    for r in _rows(
        prod,
        """SELECT fr.id, fr.return_date, fr.fabric_code, fr.returned_qty, fr.returned_by,
                  fr.remarks, fr.created_at, j.jo_number, j.so_number, j.process
           FROM jo_fabric_returns fr LEFT JOIN job_orders j ON j.id = fr.jo_id""",
    ):
        code = _norm(r["fabric_code"])
        qty = _f(r["returned_qty"])
        if not code or qty <= _EPS:
            continue
        out.append(_event(
            day=r["return_date"], ts=r["created_at"] or r["return_date"], pcode=code,
            txn_type=T_JO_RETURN, ref_type="JO", ref_no=r["jo_number"] or "",
            from_wh=f"{r['process'] or 'Cutting'} (JO)", to_wh="Checked", qty=qty, checked=qty,
            so_ref=r["so_number"] or "", jo_ref=r["jo_number"] or "",
            user=r["returned_by"] or "", remarks=r["remarks"] or "", source="jo_fabric_returns",
        ))
    return out


def _collect_reservations(grey: Optional[sqlite3.Connection]) -> List[Dict[str, Any]]:
    """Allocation rows — physical_move=False, only affect Free."""
    out: List[Dict[str, Any]] = []
    res_rows = {int(r["id"]): r for r in _rows(grey, "SELECT * FROM printed_fabric_reservations")}
    history = _rows(
        grey,
        """SELECT * FROM fabric_allocation_history
           WHERE event_type IN ('PF_RELEASE','PF_LOCK_CUTTING') AND entity_type='printed_reservation'
           ORDER BY id""",
    )
    released_by_id: Dict[int, Any] = {}
    for h in history:
        if h["entity_id"] is not None:
            released_by_id.setdefault(int(h["entity_id"]), []).append(h)  # type: ignore[arg-type]

    for rid, r in res_rows.items():
        code = _norm(r["fabric_code"])
        qty = _f(r["qty"])
        if not code:
            continue
        kind = str(r["reservation_kind"] if "reservation_kind" in r.keys() else "so").lower() or "so"
        is_adv = kind == "advance"
        status = str(r["status"] or "")
        # Cutting-issued reservation qty is stored on the row as qty; consumed qty is in cutting_issued_qty
        so_ref = r["so_number"] or ""
        remarks = " ".join(x for x in [r["sku"] or "", r["remarks"] or ""] if x).strip()
        base_ts = r["reserved_date"] or ""
        if qty > _EPS:
            out.append(_event(
                day=base_ts, ts=base_ts, pcode=code, name=r["fabric_name"] or "",
                txn_type=T_RESERVE_ADV if is_adv else T_RESERVE_SO, ref_type="RES",
                ref_no=f"RES-{rid}", from_wh="Free", to_wh="Advance" if is_adv else "SO Allocated",
                qty=qty, so_alloc=0.0 if is_adv else qty, adv_alloc=qty if is_adv else 0.0,
                physical_move=False, so_ref=so_ref,
                user=(r["reserved_by"] if "reserved_by" in r.keys() else "") or "",
                remarks=remarks, source="printed_fabric_reservations",
            ))
        for h in released_by_id.get(rid, []):
            hq = _f(h["qty"]) or qty
            consumed = h["event_type"] == "PF_LOCK_CUTTING"
            # A cutting lock row is one history row per lock event; the physical OUT is the
            # jo_fabric_issues row. Here we only free the allocation bucket.
            out.append(_event(
                day=h["created_at"], ts=h["created_at"], pcode=code, name=r["fabric_name"] or "",
                txn_type=T_CONSUME if consumed else T_RELEASE, ref_type="RES",
                ref_no=f"RES-{rid}", from_wh="Advance" if is_adv else "SO Allocated",
                to_wh="Cutting" if consumed else "Free", qty=hq,
                so_alloc=0.0 if is_adv else -hq, adv_alloc=-hq if is_adv else 0.0,
                physical_move=False, so_ref=so_ref, user=h["user_name"] or "",
                remarks=h["reason"] or "", source="fabric_allocation_history",
            ))
    return out


def _collect_grey(grey: Optional[sqlite3.Connection]) -> List[Dict[str, Any]]:
    """Grey fabric events (read-only): tracker receipt, grey QC (check log), SO hard reservations."""
    out: List[Dict[str, Any]] = []
    for r in _rows(grey, "SELECT * FROM grey_tracker"):
        code = _norm(r["material_code"])
        recv = _f(r["received_qty"])
        if not code or recv <= _EPS:
            continue
        day = (r["qc_date"] or "") or _day(r["created_at"])
        out.append(_event(
            day=day, ts=f"{_day(day)} 00:{int(r['id']):08d}", pcode=code,
            name=r["material_name"] or "", txn_type=T_GRN, ref_type="GREY",
            ref_no=r["po_number"] or r["tracker_key"] or "",
            from_wh=r["supplier"] or "Supplier", to_wh="Unchecked", qty=recv, unchecked=recv,
            so_ref=r["so_reference"] or "", remarks=f"Tracker {r['tracker_key'] or ''}".strip(),
            source="grey_tracker",
        ))
    for r in _rows(grey, "SELECT * FROM fabric_check_log"):
        code = _norm(r["fabric_code"])
        if not code:
            continue
        day = r["check_date"] or _day(r["created_at"])
        ts = f"{_day(day)} 99:{int(r['id']):08d}"
        passed, rejected = _f(r["passed_qty"]), _f(r["rejected_qty"])
        ref = f"GCHK-{r['id']}"
        if passed > _EPS:
            out.append(_event(
                day=day, ts=ts, pcode=code, name=r["fabric_name"] or "", txn_type=T_QC_PASS,
                ref_type="GREY", ref_no=ref, from_wh="Unchecked", to_wh="Checked", qty=passed,
                unchecked=-passed, checked=passed, user=r["checked_by"] or "",
                remarks=r["remarks"] or "", source="fabric_check_log",
            ))
        if rejected > _EPS:
            out.append(_event(
                day=day, ts=ts, pcode=code, name=r["fabric_name"] or "", txn_type=T_QC_REJECT,
                ref_type="GREY", ref_no=ref, from_wh="Unchecked", to_wh="Rejected / Return",
                qty=rejected, unchecked=-rejected, user=r["checked_by"] or "",
                remarks=r["remarks"] or "", source="fabric_check_log",
            ))
    for r in _rows(grey, "SELECT * FROM hard_reservations WHERE status='Active'"):
        code = _norm(r["fabric_code"])
        qty = _f(r["qty"])
        if not code or qty <= _EPS:
            continue
        out.append(_event(
            day=r["reserved_date"], ts=r["reserved_date"], pcode=code, name=r["fabric_name"] or "",
            txn_type=T_RESERVE_SO, ref_type="GREY", ref_no=f"HRES-{r['id']}", from_wh="Free",
            to_wh="SO Allocated", qty=qty, so_alloc=qty, physical_move=False,
            so_ref=r["so_number"] or "",
            remarks=" ".join(x for x in [r["sku"] or "", r["remarks"] or ""] if x).strip(),
            source="hard_reservations",
        ))
    return out


def _fabric_universe(grey: Optional[sqlite3.Connection], prod: Optional[sqlite3.Connection]) -> set:
    """P-Codes = every code seen in a printed/grey fabric table / cutting issue / opening adjustment."""
    codes: set = set()
    for tbl, col in (
        ("printed_fabric_stock", "fabric_code"),
        ("printed_fabric_checked_stock", "fabric_code"),
        ("printed_fabric_reservations", "fabric_code"),
        ("printed_fabric_opening_conversions", "fabric_code"),
    ):
        for r in _rows(grey, f"SELECT DISTINCT TRIM({col}) AS c FROM {tbl}"):
            if r["c"]:
                codes.add(str(r["c"]))
    for tbl, col in (
        ("grey_tracker", "material_code"),
        ("fabric_checked_stock", "fabric_code"),
        ("fabric_check_log", "fabric_code"),
        ("hard_reservations", "fabric_code"),
    ):
        for r in _rows(grey, f"SELECT DISTINCT TRIM({col}) AS c FROM {tbl}"):
            if r["c"]:
                codes.add(str(r["c"]))
    for tbl in ("jo_fabric_issues", "jo_fabric_returns"):
        for r in _rows(prod, f"SELECT DISTINCT TRIM(fabric_code) AS c FROM {tbl}"):
            if r["c"]:
                codes.add(str(r["c"]))
    # Opening-only fabrics (never QC'd) — keep SFG / GF / RM; drop finished goods & non-fabric.
    for r in _items_rows(
        """SELECT DISTINCT TRIM(a.item_code) AS c, t.code AS tcode
           FROM item_stock_adjustments a
           LEFT JOIN items i ON i.id = a.item_id
           LEFT JOIN item_types t ON t.id = i.item_type_id
           WHERE LOWER(COALESCE(a.reason,'')) LIKE '%opening%'
              OR LOWER(COALESCE(a.reason,'')) LIKE '%existing%'"""
    ):
        tcode = str(r["tcode"] or "").upper()
        if r["c"] and tcode not in ("FG", "ACC", "PKG", "FUEL", "SVC"):
            codes.add(str(r["c"]))
    return codes


def collect_events() -> List[Dict[str, Any]]:
    """All-time event stream across existing sources, oldest first (stable)."""
    grey = _open(_grey_path())
    pur = _open(_purchase_path())
    prod = _open(_production_path())
    try:
        universe = _fabric_universe(grey, prod)
        universe_u = {str(c).upper() for c in universe}
        events: List[Dict[str, Any]] = []
        events += _collect_grn_and_qc(grey)
        events += _collect_grey(grey)
        events += _collect_opening(grey)
        events += _collect_min(pur, universe)
        events += _collect_jo(prod)
        events += _collect_reservations(grey)
        # Opening-only adjustments are only P-Codes in the universe (item type filtered above).
        events = [
            e for e in events
            if str(e["pcode"]).upper() in universe_u
            or e["source"] not in ("item_stock_adjustments",)
        ]
        grey_codes = _grey_code_set(grey)
    finally:
        for c in (grey, pur, prod):
            if c is not None:
                c.close()
    # name backfill per code
    names: Dict[str, str] = {}
    for e in events:
        if e["name"] and e["pcode"] not in names:
            names[e["pcode"]] = e["name"]
    for e in events:
        if not e["name"]:
            e["name"] = names.get(e["pcode"], "")
    order = {t: i for i, t in enumerate(ALL_TXN_TYPES)}
    events.sort(key=lambda e: (e["date"], e["ts"], order.get(e["txn_type"], 99)))
    for i, e in enumerate(events):
        e["seq"] = i + 1
        e["category"] = category_of(e["pcode"], grey_codes)
    return events


# ── Scope maths ───────────────────────────────────────────────────────────────

def scope_delta(e: Dict[str, Any], warehouse: str) -> float:
    """Physical quantity effect of one event within a warehouse scope."""
    c, u, o = e["checked_delta"], e["unchecked_delta"], e["opening_delta"]
    if warehouse == "checked":
        return c
    if warehouse == "unchecked":
        return u
    return c + u + o


def _in_scope(e: Dict[str, Any], warehouse: str, direction: str) -> bool:
    is_alloc = not e["physical_move"]
    if warehouse != "all" and is_alloc:
        # Allocations hit Checked free only; hide in Unchecked view.
        if warehouse == "unchecked":
            return False
    d = scope_delta(e, warehouse)
    if direction == "alloc":
        return is_alloc
    if is_alloc:
        return direction == "all"
    if direction == "in":
        return d > _EPS
    if direction == "out":
        return d < -_EPS
    if warehouse == "all":
        return True
    # Warehouse view: only rows touching that warehouse.
    return abs(d) > _EPS


def _types(txn_type: str) -> set:
    """``txn_type`` may be a comma-separated list (used by KPI drill-downs)."""
    return {t.strip() for t in str(txn_type or "").split("|") if t.strip()}


def build_ledger(
    *,
    from_date: str = "",
    to_date: str = "",
    pcode: str = "",
    warehouse: str = "all",
    direction: str = "all",
    txn_type: str = "",
    q: str = "",
    stock_category: str = DEFAULT_STOCK_CATEGORY,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Date-wise register with per-P-Code running balance for the selected warehouse scope."""
    warehouse = warehouse if warehouse in WAREHOUSES else "all"
    direction = direction if direction in DIRECTIONS else "all"
    category = norm_category(stock_category)
    evs = events if events is not None else collect_events()
    evs = [e for e in evs if _cat_ok(e.get("category", "sfg"), category)]
    code_f = _norm(pcode).upper()
    q_f = (q or "").strip().lower()
    types_f = _types(txn_type)

    bal: Dict[str, float] = {}      # running physical balance per code (scope), all history
    pre: Dict[str, float] = {}      # balance per code just before from_date
    rows: List[Dict[str, Any]] = []
    range_net = 0.0                 # net physical effect of every in-range event (code filter only)
    for e in evs:
        code = e["pcode"]
        if code_f and code.upper() != code_f:
            continue
        d = scope_delta(e, warehouse) if e["physical_move"] else 0.0
        if from_date and e["date"] < from_date:
            bal[code] = bal.get(code, 0.0) + d
            pre[code] = bal[code]
            continue
        if to_date and e["date"] > to_date:
            continue
        bal[code] = bal.get(code, 0.0) + d
        range_net += d
        if not _in_scope(e, warehouse, direction):
            continue
        if types_f and e["txn_type"] not in types_f:
            continue
        if q_f:
            hay = " ".join(str(e.get(k) or "") for k in
                           ("pcode", "name", "ref_no", "so_ref", "jo_ref", "user", "remarks")).lower()
            if q_f not in hay:
                continue
        row = dict(e)
        row.update({
            "in_qty": round(max(d, 0.0), 3),
            "out_qty": round(max(-d, 0.0), 3),
            "balance": round(bal[code], 3),
            "is_transfer": bool(
                e["physical_move"] and abs(d) < _EPS
                and (abs(e["checked_delta"]) > _EPS or abs(e["unchecked_delta"]) > _EPS)
            ),
            "alloc_qty": round(e["so_alloc_delta"] + e["adv_alloc_delta"], 3),
        })
        row["link"] = doc_link(row)
        rows.append(row)

    opening = round(sum(pre.values()), 3)
    return {
        "rows": rows,
        "opening": opening,
        "total_in": round(sum(r["in_qty"] for r in rows), 3),
        "total_out": round(sum(r["out_qty"] for r in rows), 3),
        # Period closing for the code/warehouse scope (independent of type/search filters).
        "closing": round(opening + range_net, 3),
        "warehouse": warehouse,
        "stock_category": category,
    }


def doc_link(row: Dict[str, Any]) -> Dict[str, str]:
    """Deep link into the module that owns the source document."""
    ref = row.get("ref_no") or ""
    rt = row.get("ref_type")
    if rt == "GRN" and ref:
        return {"path": "/purchase", "query": f"tab=grn&doc={ref}", "label": f"Open GRN {ref}"}
    if rt == "MIN" and ref:
        return {"path": "/purchase", "query": f"tab=min&doc={ref}", "label": f"Open MIN {ref}"}
    if rt == "JO" and ref:
        return {"path": "/production", "query": f"jo={ref}", "label": f"Open JO {ref}"}
    if rt == "GREY":
        return {"path": "/grey", "query": "", "label": "Open Grey Fabric"}
    if rt in ("QC", "RES", "OPENING"):
        return {"path": "/grey", "query": "tab=printed-fabric", "label": "Open Grey › Printed Fabric"}
    if rt == "ADJ":
        return {"path": "/items", "query": f"q={row.get('pcode') or ''}", "label": "Open Item Master"}
    return {}


# ── Live stock (authoritative tables, not ledger) ────────────────────────────

def _grey_stock_rows(
    grey: Optional[sqlite3.Connection], needle: str, *, include_zero: bool = False,
) -> List[Dict[str, Any]]:
    """Grey checked stock (fabric_checked_stock) + tracker receipts still awaiting QC."""
    names: Dict[str, str] = {}
    checked: Dict[str, Dict[str, float]] = {}
    for r in _rows(grey, "SELECT * FROM fabric_checked_stock"):
        c = _norm(r["fabric_code"])
        if not c:
            continue
        names[c] = r["fabric_name"] or ""
        reserved = _f(r["reserved_qty"])
        avail = _f(r["available_qty"])
        checked[c] = {"checked": round(avail + reserved, 3), "reserved": round(reserved, 3),
                      "free": round(max(avail, 0.0), 3)}
    received: Dict[str, float] = {}
    for r in _rows(
        grey,
        """SELECT TRIM(material_code) AS c, MAX(material_name) AS n, SUM(COALESCE(received_qty,0)) AS q
           FROM grey_tracker GROUP BY TRIM(material_code)""",
    ):
        if r["c"]:
            received[str(r["c"])] = _f(r["q"])
            names.setdefault(str(r["c"]), r["n"] or "")
    logged: Dict[str, float] = {}
    for r in _rows(
        grey,
        """SELECT TRIM(fabric_code) AS c, SUM(COALESCE(passed_qty,0)+COALESCE(rejected_qty,0)) AS q
           FROM fabric_check_log GROUP BY TRIM(fabric_code)""",
    ):
        if r["c"]:
            logged[str(r["c"])] = _f(r["q"])
    out: List[Dict[str, Any]] = []
    for c in sorted(set(checked) | set(received)):
        if needle and needle not in c.lower() and needle not in names.get(c, "").lower():
            continue
        ck = checked.get(c, {"checked": 0.0, "reserved": 0.0, "free": 0.0})
        unch = round(max(received.get(c, 0.0) - logged.get(c, 0.0), 0.0), 3)
        if ck["checked"] <= _EPS and unch <= _EPS and not include_zero:
            continue
        out.append({
            "pcode": c, "name": names.get(c, ""),
            "physical": round(ck["checked"] + unch, 3), "checked": ck["checked"],
            "unchecked": unch, "so_allocated": ck["reserved"], "advance_reserved": 0.0,
            "free": ck["free"], "opening_pool": 0.0,
            "location": (
                "Zero" if ck["checked"] <= _EPS and unch <= _EPS
                else ("Checked" if ck["checked"] > _EPS and unch <= _EPS
                      else ("Checked + Unchecked" if ck["checked"] > _EPS else "Unchecked"))
            ),
        })
    return out


def stock_by_pcode(
    pcode: str = "", q: str = "", stock_category: str = DEFAULT_STOCK_CATEGORY,
    *,
    include_zero: bool = True,
) -> List[Dict[str, Any]]:
    from ..db import grey_db

    category = norm_category(stock_category)

    code = _norm(pcode)
    # include_zero lists Item Master fabrics with no movements (visibility only; zeros don't change KPIs).
    vis = grey_db.printed_stock_visibility(
        code if code and not q else "", include_zero=bool(include_zero) and category != "grey",
    )
    unchecked: Dict[str, float] = {}
    grey = _open(_grey_path())
    try:
        for r in _rows(
            grey,
            """SELECT UPPER(TRIM(fabric_code)) AS uk, MAX(TRIM(fabric_code)) AS c, SUM(qty) AS q
               FROM printed_fabric_stock
               WHERE status='Unchecked' AND COALESCE(qty,0) > 0.001
               GROUP BY UPPER(TRIM(fabric_code))""",
        ):
            unchecked[str(r["uk"] or "").upper()] = {
                "code": str(r["c"] or ""),
                "qty": round(_f(r["q"]), 3),
            }
    finally:
        if grey is not None:
            grey.close()

    out: List[Dict[str, Any]] = []
    seen: set = set()
    needle = (q or code).strip().lower()
    for v in vis:
        fc = str(v["fabric_code"])
        uk = fc.upper()
        seen.add(uk)
        if needle and needle not in fc.lower() and needle not in str(v.get("fabric_name") or "").lower():
            continue
        unch_info = unchecked.get(uk) or {}
        unch = float(unch_info.get("qty") or 0) if isinstance(unch_info, dict) else float(unch_info or 0)
        # total_stock already = checked free + reserved + opening; do not add opening again.
        checked_physical = round(_f(v["total_stock"]), 3)
        out.append({
            "pcode": fc,
            "name": v.get("fabric_name") or "",
            "physical": round(checked_physical + unch, 3),
            "checked": checked_physical,
            "unchecked": unch,
            "so_allocated": round(_f(v["so_allocated"]), 3),
            "advance_reserved": round(_f(v["advance_reserved"]), 3),
            "free": round(_f(v["free_available"]), 3),
            "opening_pool": round(_f(v.get("opening_free")), 3),
            "location": "Checked" if checked_physical > _EPS and unch <= _EPS
            else ("Checked + Unchecked" if checked_physical > _EPS and unch > _EPS
                  else ("Unchecked" if unch > _EPS else ("Zero" if include_zero else "—"))),
        })
    for uk, unch_info in unchecked.items():
        if uk in seen:
            continue
        fc = unch_info.get("code") if isinstance(unch_info, dict) else uk
        unch = float(unch_info.get("qty") or 0) if isinstance(unch_info, dict) else float(unch_info or 0)
        if needle and needle not in str(fc).lower():
            continue
        seen.add(uk)
        out.append({
            "pcode": fc, "name": "", "physical": unch, "checked": 0.0, "unchecked": unch,
            "so_allocated": 0.0, "advance_reserved": 0.0, "free": 0.0, "opening_pool": 0.0,
            "location": "Unchecked",
        })
    grey2 = _open(_grey_path())
    try:
        grey_codes = _grey_code_set(grey2)
        if category != "sfg":
            have = {r["pcode"].upper() for r in out}
            out += [r for r in _grey_stock_rows(grey2, needle, include_zero=include_zero) if r["pcode"].upper() not in have]
            if include_zero:
                # Item Master GF parents with no grey stock rows yet.
                try:
                    from ..db.item_db import list_items
                    for it in list_items(parent_only=True, limit=20000) or []:
                        if str(it.get("item_type_code") or "").upper() != "GF":
                            continue
                        fc = str(it.get("item_code") or "").strip()
                        if not fc or fc.upper() in have:
                            continue
                        if needle and needle not in fc.lower() and needle not in str(it.get("item_name") or "").lower():
                            continue
                        have.add(fc.upper())
                        out.append({
                            "pcode": fc, "name": it.get("item_name") or fc,
                            "physical": 0.0, "checked": 0.0, "unchecked": 0.0,
                            "so_allocated": 0.0, "advance_reserved": 0.0, "free": 0.0,
                            "opening_pool": 0.0, "location": "Zero",
                        })
                except Exception:
                    pass
    finally:
        if grey2 is not None:
            grey2.close()
    out = [r for r in out if _cat_ok(category_of(r["pcode"], grey_codes), category)]
    for r in out:
        r["category"] = category_of(r["pcode"], grey_codes)
    out.sort(key=lambda r: r["pcode"])
    return out


# ── Dashboard ─────────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def dashboard(
    *,
    from_date: str = "",
    to_date: str = "",
    pcode: str = "",
    warehouse: str = "all",
    txn_type: str = "",
    stock_category: str = DEFAULT_STOCK_CATEGORY,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """KPI strip. Period metrics come from the ledger; Free/SO/Advance/Pending are live stock."""
    warehouse = warehouse if warehouse in WAREHOUSES else "all"
    today = _today()
    frm = from_date or today
    to = to_date or today
    category = norm_category(stock_category)
    evs = events if events is not None else collect_events()
    evs = [e for e in evs if _cat_ok(e.get("category", "sfg"), category)]
    code_f = _norm(pcode).upper()
    types_f = _types(txn_type)

    def _match(e: Dict[str, Any]) -> bool:
        if code_f and e["pcode"].upper() != code_f:
            return False
        if types_f and e["txn_type"] not in types_f:
            return False
        return True

    opening = in_q = out_q = today_in = today_out = 0.0
    returns_in = adj_net = 0.0
    all_time_closing = 0.0
    for e in evs:
        if code_f and e["pcode"].upper() != code_f:
            continue
        d = scope_delta(e, warehouse) if e["physical_move"] else 0.0
        all_time_closing += d
        if not _match(e) or not e["physical_move"]:
            continue
        if e["date"] < frm:
            opening += d
        elif e["date"] <= to:
            if d > _EPS:
                in_q += d
            elif d < -_EPS:
                out_q += -d
            if e["txn_type"] == T_JO_RETURN:
                returns_in += d
            if e["txn_type"] == T_ADJ:
                adj_net += d
            # Period IN/OUT = movements inside the selected date range (not calendar today).
            if d > _EPS:
                today_in += d
            elif d < -_EPS:
                today_out += -d

    # txn_type filter makes opening/closing partial — keep closing arithmetic honest.
    closing = opening + in_q - out_q
    stock = stock_by_pcode(pcode, stock_category=category)
    so_alloc = sum(r["so_allocated"] for r in stock)
    adv = sum(r["advance_reserved"] for r in stock)
    free = sum(r["free"] for r in stock)
    pending = sum(r["unchecked"] for r in stock)
    physical_now = sum(r["physical"] for r in stock)
    checked_now = sum(r["checked"] for r in stock)
    pending_codes = sum(1 for r in stock if r["unchecked"] > _EPS)
    return {
        "from_date": frm,
        "to_date": to,
        "today": today,
        "warehouse": warehouse,
        "stock_category": category,
        "opening": round(opening, 3),
        "in_qty": round(in_q, 3),
        "out_qty": round(out_q, 3),
        "closing": round(closing, 3),
        "today_in": round(today_in, 3),
        "today_out": round(today_out, 3),
        "so_allocated": round(so_alloc, 3),
        "advance_reserved": round(adv, 3),
        "free": round(free, 3),
        "returns_in": round(returns_in, 3),
        "adjustments_net": round(adj_net, 3),
        "returns_adjustments": round(returns_in + adj_net, 3),
        "pending_unchecked": round(pending, 3),
        "pending_unchecked_codes": pending_codes,
        "physical_now": round(physical_now, 3),
        "checked_now": round(checked_now, 3),
        # Info only: ledger (from existing docs) vs live stock tables for the same scope.
        "ledger_closing_all_time": round(all_time_closing, 3),
        "recon_diff": round(
            (physical_now if warehouse == "all" else checked_now if warehouse == "checked" else pending)
            - all_time_closing, 3,
        ),
        "txn_types": ALL_TXN_TYPES,
        "pcode_count": len(stock),
    }


# ── Export ────────────────────────────────────────────────────────────────────

EXPORT_HEADERS = [
    "Date", "P-Code", "Name", "Type", "Ref", "From", "To", "IN", "OUT", "Alloc (Free only)",
    "Running Bal", "SO", "JO", "User", "Remarks",
]


def _export_row(r: Dict[str, Any]) -> List[Any]:
    return [
        r["date"], r["pcode"], r["name"], r["txn_type"], r["ref_no"], r["from_wh"], r["to_wh"],
        r["in_qty"] or "", r["out_qty"] or "", r["alloc_qty"] or "", r["balance"],
        r["so_ref"], r["jo_ref"], r["user"], r["remarks"],
    ]


def export_csv(rows: Iterable[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(EXPORT_HEADERS)
    for r in rows:
        w.writerow(_export_row(r))
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def export_xlsx(rows: Iterable[Dict[str, Any]]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Fabric Ledger"
    ws.append(EXPORT_HEADERS)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append(_export_row(r))
    for col, width in zip("ABCDEFGHIJKLMNO", (11, 14, 24, 28, 16, 16, 16, 10, 10, 12, 12, 14, 14, 12, 30)):
        ws.column_dimensions[col].width = width
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")
