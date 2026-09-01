"""Cutting summary / balance / aging / fabric consumption reports."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..db import production_db
from .helpers import get_parent_sku
from .set_components import parse_component_sku, style_key_for_set_bom

IST = timezone(timedelta(hours=5, minutes=30))
AGING_BUCKETS = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("15+", 16, None),
)
_SIZE_TOKENS = {
    "XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL", "7XL", "8XL",
}


def _today_ist() -> date:
    return datetime.now(IST).date()


def _parse_day(raw: Any) -> date | None:
    s = str(raw or "").strip()[:10]
    if len(s) < 10:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _size_from_sku(sku: str, style: str = "") -> str:
    st = str(style or "").strip()
    if st:
        return st
    s = str(sku or "").strip().upper()
    main, _comp = parse_component_sku(s)
    token = (main or s).rsplit("-", 1)
    if len(token) == 2 and token[1] in _SIZE_TOKENS:
        return token[1]
    return ""


def _aging_days(basis: date | None, today: date) -> int | None:
    if basis is None:
        return None
    return (today - basis).days


def _aging_bucket(days: int | None) -> str:
    if days is None:
        return ""
    if days < 0:
        return "0-2"
    for label, lo, hi in AGING_BUCKETS:
        if hi is None:
            if days >= lo:
                return label
        elif lo <= days <= hi:
            return label
    return "15+"


def _so_lookup() -> dict[str, dict]:
    try:
        from ..db import sales_db

        conn = sales_db._connect()
        rows = conn.execute(
            "SELECT so_number, so_date, delivery_date, buyer FROM sales_orders"
        ).fetchall()
        conn.close()
        out = {}
        for r in rows:
            d = dict(r)
            out[str(d.get("so_number") or "")] = d
        return out
    except Exception:
        return {}


def _qty_status(planned: float, received: float, jo_status: str) -> str:
    if received > planned:
        return "over"
    if received == planned:
        return "exact"
    closed = str(jo_status or "").strip() in {"Completed", "Closed"}
    if closed:
        return "under"
    return "pending"


def _safe_div(num: float, den: float) -> float | None:
    if den is None or abs(float(den)) < 1e-9:
        return None
    return round(float(num) / float(den), 4)


def _allocate_header_receipts(
    rec_line: dict[int, int],
    rec_jo: dict[int, int],
    lines_by_jo: dict[int, list[dict]],
) -> dict[int, int]:
    """Apply JO-level receipts (jo_line_id IS NULL) onto matching lines.

    Receive (JO level) historically stored jo_line_id=NULL. Cutting Report only
    summed line-level receipts, so balance stayed at planned. Allocate leftover
    header qty onto lines by remaining plan, then remainder on the first line.
    """
    extra = dict(rec_line)
    for jid, lines in lines_by_jo.items():
        if not lines:
            continue
        allocated = sum(int(extra.get(int(ln["id"]), 0) or 0) for ln in lines)
        leftover = max(0, int(rec_jo.get(int(jid), 0) or 0) - allocated)
        if leftover <= 0:
            continue
        ordered = sorted(lines, key=lambda ln: int(ln.get("planned_qty") or 0), reverse=True)
        for ln in ordered:
            if leftover <= 0:
                break
            lid = int(ln["id"])
            planned = int(ln.get("planned_qty") or 0)
            already = int(extra.get(lid, 0) or 0)
            room = max(0, planned - already)
            if room <= 0:
                continue
            take = min(leftover, room)
            extra[lid] = already + take
            leftover -= take
        if leftover > 0:
            lid = int(ordered[0]["id"])
            extra[lid] = int(extra.get(lid, 0) or 0) + leftover
    return extra


def _maps(conn, as_of: date | None = None):
    as_of_s = as_of.isoformat() if as_of else None
    rec_line_sql = """SELECT jo_line_id, SUM(received_qty) AS qty
               FROM jo_piece_receipts WHERE jo_line_id IS NOT NULL"""
    rec_jo_sql = "SELECT jo_id, SUM(received_qty) AS qty FROM jo_piece_receipts"
    iss_line_sql = """SELECT jo_line_id, SUM(issued_qty) AS qty
               FROM jo_piece_issues WHERE jo_line_id IS NOT NULL"""
    iss_jo_sql = "SELECT jo_id, SUM(issued_qty) AS qty FROM jo_piece_issues"
    rec_line_params: list = []
    rec_jo_params: list = []
    iss_line_params: list = []
    iss_jo_params: list = []
    if as_of_s:
        rec_line_sql += " AND substr(receipt_date,1,10) <= ?"
        rec_line_params.append(as_of_s)
        rec_jo_sql += " WHERE substr(receipt_date,1,10) <= ?"
        rec_jo_params.append(as_of_s)
        iss_line_sql += " AND substr(issue_date,1,10) <= ?"
        iss_line_params.append(as_of_s)
        iss_jo_sql += " WHERE substr(issue_date,1,10) <= ?"
        iss_jo_params.append(as_of_s)
    rec_line_sql += " GROUP BY jo_line_id"
    rec_jo_sql += " GROUP BY jo_id"
    iss_line_sql += " GROUP BY jo_line_id"
    iss_jo_sql += " GROUP BY jo_id"
    rec_line = {
        int(r["jo_line_id"]): int(r["qty"] or 0)
        for r in conn.execute(rec_line_sql, rec_line_params).fetchall()
        if r["jo_line_id"] is not None
    }
    rec_jo = {
        int(r["jo_id"]): int(r["qty"] or 0)
        for r in conn.execute(rec_jo_sql, rec_jo_params).fetchall()
    }
    iss_line = {
        int(r["jo_line_id"]): int(r["qty"] or 0)
        for r in conn.execute(iss_line_sql, iss_line_params).fetchall()
        if r["jo_line_id"] is not None
    }
    iss_jo = {
        int(r["jo_id"]): int(r["qty"] or 0)
        for r in conn.execute(iss_jo_sql, iss_jo_params).fetchall()
    }
    last_act = {
        int(r["jo_id"]): str(r["last_d"] or "")[:19]
        for r in conn.execute(
            """SELECT jo_id, MAX(last_d) AS last_d FROM (
                 SELECT jo_id, issue_date AS last_d FROM jo_piece_issues
                 UNION ALL SELECT jo_id, receipt_date FROM jo_piece_receipts
                 UNION ALL SELECT jo_id, issue_date FROM jo_fabric_issues
                 UNION ALL SELECT id AS jo_id, updated_at FROM job_orders
               ) GROUP BY jo_id"""
        ).fetchall()
    }
    return rec_line, rec_jo, iss_line, iss_jo, last_act


def _activity_maps(conn, day: date) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Receipts/issues on a single calendar day (line-level and JO-level)."""
    day_s = day.isoformat()
    rec_line = {
        int(r["jo_line_id"]): int(r["qty"] or 0)
        for r in conn.execute(
            """SELECT jo_line_id, SUM(received_qty) AS qty
               FROM jo_piece_receipts
               WHERE jo_line_id IS NOT NULL AND substr(receipt_date,1,10)=?
               GROUP BY jo_line_id""",
            (day_s,),
        ).fetchall()
        if r["jo_line_id"] is not None
    }
    rec_jo = {
        int(r["jo_id"]): int(r["qty"] or 0)
        for r in conn.execute(
            """SELECT jo_id, SUM(received_qty) AS qty
               FROM jo_piece_receipts WHERE substr(receipt_date,1,10)=?
               GROUP BY jo_id""",
            (day_s,),
        ).fetchall()
    }
    iss_line = {
        int(r["jo_line_id"]): int(r["qty"] or 0)
        for r in conn.execute(
            """SELECT jo_line_id, SUM(issued_qty) AS qty
               FROM jo_piece_issues
               WHERE jo_line_id IS NOT NULL AND substr(issue_date,1,10)=?
               GROUP BY jo_line_id""",
            (day_s,),
        ).fetchall()
        if r["jo_line_id"] is not None
    }
    iss_jo = {
        int(r["jo_id"]): int(r["qty"] or 0)
        for r in conn.execute(
            """SELECT jo_id, SUM(issued_qty) AS qty
               FROM jo_piece_issues WHERE substr(issue_date,1,10)=?
               GROUP BY jo_id""",
            (day_s,),
        ).fetchall()
    }
    return rec_line, rec_jo, iss_line, iss_jo


def _match(val: str, needle: str) -> bool:
    if not needle:
        return True
    return needle.lower() in str(val or "").lower()


def _row_search_blob(r: dict) -> str:
    return " ".join(
        str(r.get(k) or "")
        for k in (
            "so_number",
            "jo_number",
            "sku",
            "jo_header_sku",
            "parent_style",
            "jo_header_parent_style",
            "component",
            "fabric_code",
            "size",
        )
    )


def _jo_ids_matching_style_search(
    rows: list[dict],
    *,
    search: str = "",
    parent_style: str = "",
    sku: str = "",
) -> set[int]:
    """If any line/header of a JO matches style/search, include all lines for that JO."""
    q = str(search or "").strip()
    ps = str(parent_style or "").strip()
    sk = str(sku or "").strip()
    if not (q or ps or sk):
        return set()
    matched: set[int] = set()
    for r in rows:
        jid = int(r["jo_id"])
        if q and q.lower() in _row_search_blob(r).lower():
            matched.add(jid)
        if ps and (
            _match(r.get("parent_style") or "", ps)
            or _match(r.get("jo_header_sku") or "", ps)
            or _match(r.get("jo_header_parent_style") or "", ps)
        ):
            matched.add(jid)
        if sk and (
            _match(r.get("sku") or "", sk)
            or _match(r.get("jo_header_sku") or "", sk)
            or _match(r.get("parent_style") or "", sk)
            or _match(r.get("jo_header_parent_style") or "", sk)
        ):
            matched.add(jid)
    return matched


def _parse_components_filter(raw: str) -> list[str]:
    return [c.strip().upper() for c in str(raw or "").split(",") if c.strip()]


def _production_mode_filter(raw: str) -> str:
    from .so_production_path import normalize_production_mode

    s = str(raw or "").strip().lower()
    if s in {"", "all"}:
        return ""
    return normalize_production_mode(raw)


def _is_panel_row(row: dict) -> bool:
    from .component_bom import _looks_like_panel_code

    comp = str(row.get("component") or "")
    sku = str(row.get("sku") or "")
    return _looks_like_panel_code(comp, sku)


def _rollup_set_balance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SKU/set balance — min qty across TOP/PANT/DUPATTA for same SO+style+size."""
    from collections import defaultdict

    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("row_type") == "panel_wip":
            continue
        key = (
            str(r.get("so_number") or ""),
            str(r.get("parent_style") or ""),
            str(r.get("size") or ""),
        )
        buckets[key].append(r)

    out: list[dict[str, Any]] = []
    for (_so, parent, size), items in sorted(buckets.items()):
        base = dict(items[0])
        base["component"] = ""
        base["row_type"] = "set"
        base["balance_level"] = "set"
        base["sku"] = f"{parent}-{size}" if size else parent
        qty_fields = (
            "planned_qty",
            "received_qty",
            "issued_qty",
            "balance_qty",
            "opening_balance",
            "closing_balance",
        )
        for field in qty_fields:
            vals = [int(x.get(field) or 0) for x in items]
            base[field] = min(vals) if vals else 0
        day_fields = ("received_on_date", "issued_on_date")
        for field in day_fields:
            vals = [int(x.get(field) or 0) for x in items if x.get(field) is not None]
            base[field] = min(vals) if vals else items[0].get(field)
        base["qty_variance"] = int(base["received_qty"]) - int(base["planned_qty"])
        base["status"] = _qty_status(
            int(base["planned_qty"]),
            int(base["received_qty"]),
            str(base.get("jo_status") or ""),
        )
        if base["status"] != "pending":
            base["aging_bucket"] = ""
            base["aging_days"] = None
        out.append(base)
    return out


def _panel_wip_balance_rows(*, production_mode: str = "") -> list[dict[str, Any]]:
    """Panel WIP rows from process_stock (FRONT/BACK etc.) — not set/component balance."""
    from .component_bom import _looks_like_panel_code
    from .so_production_path import normalize_production_mode
    from .set_components import parse_component_sku

    mode_want = _production_mode_filter(production_mode)
    conn = production_db._connect()
    rows: list[dict[str, Any]] = []
    try:
        stock = conn.execute(
            """SELECT ps.so_number, ps.sku, ps.process, ps.available_qty, ps.total_in
               FROM process_stock ps
               WHERE ps.available_qty > 0 AND ps.process IN ('Cutting','Embroidery')"""
        ).fetchall()
        jo_modes: dict[tuple[str, str], str] = {}
        for r in conn.execute(
            """SELECT so_number, sku, production_mode FROM job_orders
               WHERE process='Cutting' AND IFNULL(status,'') NOT IN ('Cancelled','Closed')"""
        ).fetchall():
            key = (str(r["so_number"] or ""), str(r["sku"] or "").upper())
            jo_modes[key] = normalize_production_mode(r["production_mode"])

        for r in stock:
            sku = str(r["sku"] or "").strip().upper()
            _main, comp = parse_component_sku(sku)
            if not _looks_like_panel_code(comp or sku, sku):
                continue
            so = str(r["so_number"] or "")
            parent = _main or sku
            mode = jo_modes.get((so, sku), "inhouse")
            if mode_want and mode != mode_want:
                continue
            qty = int(r["available_qty"] or 0)
            if qty <= 0:
                continue
            rows.append(
                {
                    "so_number": so,
                    "so_date": "",
                    "delivery_date": "",
                    "brand": "",
                    "jo_number": "",
                    "jo_id": 0,
                    "jo_date": "",
                    "parent_style": parent,
                    "sku": sku,
                    "size": _size_from_sku(sku),
                    "component": comp or "PANEL",
                    "fabric_code": "",
                    "planned_qty": qty,
                    "issued_qty": 0,
                    "received_qty": qty,
                    "balance_qty": 0,
                    "opening_balance": 0,
                    "closing_balance": 0,
                    "received_on_date": None,
                    "issued_on_date": None,
                    "production_mode": mode,
                    "qty_variance": 0,
                    "status": "exact",
                    "jo_status": "",
                    "last_activity_date": "",
                    "aging_days": None,
                    "aging_bucket": "",
                    "planned_fabric": 0,
                    "actual_fabric": 0,
                    "bom_avg": None,
                    "actual_avg": None,
                    "avg_diff": None,
                    "fabric_saving": None,
                    "fabric_saving_pct": None,
                    "line_id": None,
                    "row_type": "panel_wip",
                    "balance_level": "panel_wip",
                    "panel_process": str(r["process"] or ""),
                }
            )
    finally:
        conn.close()
    return rows


def build_cutting_report(
    *,
    date_from: str = "",
    date_to: str = "",
    so_number: str = "",
    parent_style: str = "",
    sku: str = "",
    size: str = "",
    jo_number: str = "",
    component: str = "",
    fabric_code: str = "",
    status: str = "",
    aging_bucket: str = "",
    aging_basis: str = "jo_date",
    variance: str = "",
    brand: str = "",
    search: str = "",
    group_by: str = "",
    page: int = 1,
    page_size: int = 200,
    export: bool = False,
    as_of_date: str = "",
    activity_date: str = "",
    production_mode: str = "",
    components: str = "",
    balance_level: str = "component",
) -> dict[str, Any]:
    from .so_production_path import normalize_production_mode

    today = _today_ist()
    so_map = _so_lookup()
    as_of = _parse_day(as_of_date) or _parse_day(activity_date)
    activity = _parse_day(activity_date)
    mode_want = _production_mode_filter(production_mode)
    comp_want = _parse_components_filter(components)
    level = str(balance_level or "component").strip().lower()
    if level not in {"component", "set", "panel_wip"}:
        level = "component"
    conn = production_db._connect()
    try:
        jos = [
            dict(r)
            for r in conn.execute(
                """SELECT id, jo_number, jo_date, so_number, sku, sku_name, process, status,
                          planned_qty, issued_qty, received_qty, rejected_qty, balance_qty,
                          expected_completion, fabric_code, fabric_qty, fabric_unit,
                          fabric_issued_qty, fabric_received_qty, fabric_consumption,
                          main_sku, component_code, sku_role, created_at, updated_at,
                          production_mode
                   FROM job_orders
                   WHERE process='Cutting' AND IFNULL(status,'') != 'Cancelled'"""
            ).fetchall()
        ]
        if mode_want:
            jos = [
                j
                for j in jos
                if normalize_production_mode(j.get("production_mode")) == mode_want
            ]
        lines_by_jo: dict[int, list[dict]] = {}
        for r in conn.execute(
            """SELECT id, jo_id, sku, sku_name, style, planned_qty, issued_qty, received_qty,
                      rejected_qty, balance_qty, component_code, parent_sku
               FROM jo_lines"""
        ).fetchall():
            d = dict(r)
            lines_by_jo.setdefault(int(d["jo_id"]), []).append(d)
        rec_line, rec_jo, iss_line, iss_jo, last_act = _maps(conn, as_of=as_of)
        rec_line = _allocate_header_receipts(rec_line, rec_jo, lines_by_jo)
        iss_line = _allocate_header_receipts(iss_line, iss_jo, lines_by_jo)
        opening_rec_line, opening_rec_jo, opening_iss_line, opening_iss_jo, _ = (
            _maps(conn, as_of=(activity - timedelta(days=1)) if activity else None)
        )
        opening_rec_line = _allocate_header_receipts(opening_rec_line, opening_rec_jo, lines_by_jo)
        opening_iss_line = _allocate_header_receipts(opening_iss_line, opening_iss_jo, lines_by_jo)
        # Do NOT independently allocate same-day JO-level receipts onto lines —
        # that mis-assigns cut-today across multi-size lines and breaks
        # opening − cut_today = closing. Derive day qty as closing − opening.
        act_rec_line: dict[int, int] = {}
        act_iss_line: dict[int, int] = {}
        if activity:
            for _jid, lines in lines_by_jo.items():
                for ln in lines:
                    lid = int(ln["id"])
                    act_rec_line[lid] = max(
                        0, int(rec_line.get(lid, 0) or 0) - int(opening_rec_line.get(lid, 0) or 0)
                    )
                    act_iss_line[lid] = max(
                        0, int(iss_line.get(lid, 0) or 0) - int(opening_iss_line.get(lid, 0) or 0)
                    )
    finally:
        conn.close()

    d_from = _parse_day(date_from)
    d_to = _parse_day(date_to)
    basis_key = str(aging_basis or "jo_date").strip().lower()
    if basis_key not in {"jo_date", "so_date", "delivery_date", "due_date"}:
        basis_key = "jo_date"

    rows: list[dict[str, Any]] = []
    for jo in jos:
        jid = int(jo["id"])
        lines = lines_by_jo.get(jid) or []
        so = so_map.get(str(jo.get("so_number") or ""), {})
        so_date = str(so.get("so_date") or "")[:10]
        delivery = str(so.get("delivery_date") or jo.get("expected_completion") or "")[:10]
        jo_date = str(jo.get("jo_date") or "")[:10]
        if d_from or d_to:
            jd = _parse_day(jo_date)
            if jd is None:
                continue
            if d_from and jd < d_from:
                continue
            if d_to and jd > d_to:
                continue

        header_planned = int(jo.get("planned_qty") or 0)
        header_received = int(rec_jo.get(jid, jo.get("received_qty") or 0))
        header_issued = int(iss_jo.get(jid, jo.get("issued_qty") or 0))
        fab_issued = float(jo.get("fabric_issued_qty") or 0)
        fab_ret = float(jo.get("fabric_received_qty") or 0)
        actual_fab = float(jo.get("fabric_consumption") or 0)
        if actual_fab <= 0:
            actual_fab = max(0.0, fab_issued - fab_ret)
        planned_fab = float(jo.get("fabric_qty") or 0)
        header_sku = str(jo.get("sku") or "")
        header_main = str(jo.get("main_sku") or "") or header_sku
        header_parent = style_key_for_set_bom(header_main or header_sku)
        try:
            header_parent = str(get_parent_sku(header_parent) or header_parent)
        except Exception:
            pass
        sources = lines if lines else [None]
        n_src = max(len(lines), 1)

        for ln in sources:
            if ln is None:
                line_planned = header_planned
                line_received = header_received
                line_issued = header_issued
                opening_received = int(opening_rec_jo.get(jid, 0)) if activity else 0
                received_on_day = max(0, line_received - opening_received) if activity else 0
                opening_issued = int(opening_iss_jo.get(jid, 0)) if activity else 0
                issued_on_day = max(0, line_issued - opening_issued) if activity else 0
                line_sku = str(jo.get("sku") or "")
                style = ""
                line_comp = str(jo.get("component_code") or "")
                share = 1.0
            else:
                lid = int(ln["id"])
                line_planned = int(ln.get("planned_qty") or 0)
                line_received = int(rec_line.get(lid, ln.get("received_qty") or 0))
                line_issued = int(iss_line.get(lid, ln.get("issued_qty") or 0))
                opening_received = int(opening_rec_line.get(lid, 0)) if activity else 0
                received_on_day = int(act_rec_line.get(lid, 0)) if activity else 0
                issued_on_day = int(act_iss_line.get(lid, 0)) if activity else 0
                line_sku = str(ln.get("sku") or jo.get("sku") or "")
                style = str(ln.get("style") or "")
                line_comp = str(ln.get("component_code") or jo.get("component_code") or "")
                share = (line_planned / header_planned) if header_planned else (1.0 / n_src)

            # Lifetime view (no activity_date): Open must NOT equal Planned —
            # that made Balance look "wrong". Mirror current balance instead.
            # Daily view: Open = planned − receipts through yesterday; Close = planned − all receipts.
            if activity:
                opening_balance = line_planned - opening_received
                closing_balance = line_planned - line_received
            else:
                opening_balance = closing_balance = line_planned - line_received

            main = str(jo.get("main_sku") or "") or (parse_component_sku(line_sku)[0] or line_sku)
            parent = style_key_for_set_bom(main or line_sku)
            try:
                parent = str(get_parent_sku(parent) or parent)
            except Exception:
                pass
            size_label = _size_from_sku(line_sku, style)
            planned_fab_line = round(planned_fab * share, 4)
            actual_fab_line = round(actual_fab * share, 4)
            bom_avg = _safe_div(planned_fab_line, line_planned) if line_planned else None
            actual_avg = _safe_div(actual_fab_line, line_received) if line_received else None
            planned_req_recv = round((bom_avg or 0) * line_received, 4) if bom_avg is not None else None
            saving = None
            saving_pct = None
            if planned_req_recv is not None and line_received > 0:
                saving = round(planned_req_recv - actual_fab_line, 4)
                saving_pct = _safe_div(saving, planned_req_recv)
                if saving_pct is not None:
                    saving_pct = round(saving_pct * 100, 2)
            variance_qty = line_received - line_planned
            st = _qty_status(line_planned, line_received, str(jo.get("status") or ""))
            basis_date = {
                "jo_date": _parse_day(jo_date),
                "so_date": _parse_day(so_date),
                "delivery_date": _parse_day(delivery),
                "due_date": _parse_day(delivery),
            }.get(basis_key)
            days = _aging_days(basis_date, today)
            bucket = _aging_bucket(days) if st == "pending" else ""
            last = last_act.get(jid) or str(jo.get("updated_at") or "")
            row = {
                "so_number": jo.get("so_number") or "",
                "so_date": so_date,
                "delivery_date": delivery,
                "brand": so.get("buyer") or "",
                "jo_number": jo.get("jo_number") or "",
                "jo_id": jid,
                "jo_date": jo_date,
                "jo_header_sku": header_sku,
                "jo_header_parent_style": header_parent,
                "parent_style": parent,
                "sku": line_sku,
                "size": size_label,
                "component": line_comp,
                "fabric_code": jo.get("fabric_code") or "",
                "planned_qty": line_planned,
                "issued_qty": line_issued,
                "received_qty": line_received,
                "balance_qty": line_planned - line_received,
                "opening_balance": opening_balance,
                "closing_balance": closing_balance,
                "received_on_date": received_on_day if activity else None,
                "issued_on_date": issued_on_day if activity else None,
                "production_mode": jo.get("production_mode") or "",
                "qty_variance": variance_qty,
                "status": st,
                "jo_status": jo.get("status") or "",
                "last_activity_date": last,
                "aging_days": days if st == "pending" else None,
                "aging_bucket": bucket,
                "planned_fabric": planned_fab_line,
                "actual_fabric": actual_fab_line,
                "bom_avg": bom_avg,
                "actual_avg": actual_avg,
                "avg_diff": None if bom_avg is None or actual_avg is None else round(actual_avg - bom_avg, 4),
                "fabric_saving": saving,
                "fabric_saving_pct": saving_pct,
                "line_id": None if ln is None else ln.get("id"),
                "row_type": "component",
                "balance_level": "component",
            }
            rows.append(row)

    if level == "panel_wip":
        rows = _panel_wip_balance_rows(production_mode=production_mode)
    elif level == "set":
        rows = _rollup_set_balance_rows(rows)
    elif level == "component":
        rows = [r for r in rows if not _is_panel_row(r)]

    q = str(search or "").strip()
    jo_style_hits = _jo_ids_matching_style_search(
        rows, search=q, parent_style=parent_style, sku=sku
    )
    filtered = []
    for r in rows:
        jid = int(r["jo_id"])
        if so_number and str(r["so_number"]).lower() != so_number.strip().lower():
            continue
        if parent_style and jid not in jo_style_hits and not (
            _match(r["parent_style"], parent_style)
            or _match(r.get("jo_header_sku") or "", parent_style)
            or _match(r.get("jo_header_parent_style") or "", parent_style)
        ):
            continue
        if sku and jid not in jo_style_hits and not (
            _match(r["sku"], sku)
            or _match(r.get("jo_header_sku") or "", sku)
            or _match(r.get("jo_header_parent_style") or "", sku)
        ):
            continue
        if size and str(r["size"]).upper() != size.strip().upper():
            continue
        if jo_number and not _match(r["jo_number"], jo_number):
            continue
        if component and str(r["component"]).upper() != component.strip().upper():
            continue
        if comp_want and str(r.get("component") or "").upper() not in comp_want:
            continue
        if fabric_code and not _match(r["fabric_code"], fabric_code):
            continue
        if status and r["status"] != status.strip().lower():
            continue
        if variance:
            v = variance.strip().lower()
            if v in {"over", "over_cutting"} and r["status"] != "over":
                continue
            if v in {"under", "under_cutting"} and r["status"] != "under":
                continue
            if v in {"exact", "completed"} and r["status"] != "exact":
                continue
            if v == "pending" and r["status"] != "pending":
                continue
        if aging_bucket and r.get("aging_bucket") != aging_bucket:
            continue
        if brand and not _match(r["brand"], brand):
            continue
        if q:
            if jid not in jo_style_hits and q.lower() not in _row_search_blob(r).lower():
                continue
        filtered.append(r)

    def _sum(key: str) -> float:
        return float(sum(float(x.get(key) or 0) for x in filtered))

    pending_jos = {r["jo_id"] for r in filtered if r["status"] == "pending"}
    kpis = {
        "planned_qty": _sum("planned_qty"),
        "received_qty": _sum("received_qty"),
        "issued_qty": _sum("issued_qty"),
        "balance_qty": _sum("balance_qty"),
        "opening_balance": _sum("opening_balance"),
        "closing_balance": _sum("closing_balance"),
        "received_on_date": _sum("received_on_date") if activity else None,
        "issued_on_date": _sum("issued_on_date") if activity else None,
        "as_of_date": as_of.isoformat() if as_of else "",
        "activity_date": activity.isoformat() if activity else "",
        "pending_jos": len(pending_jos),
        "over_qty": float(sum(max(0, int(r["qty_variance"])) for r in filtered)),
        "under_qty": float(sum(max(0, -int(r["qty_variance"])) for r in filtered if r["status"] in {"under", "pending"})),
        "planned_fabric": _sum("planned_fabric"),
        "actual_fabric": _sum("actual_fabric"),
        "fabric_saving": _sum("fabric_saving"),
    }
    pf, af = kpis["planned_fabric"], kpis["actual_fabric"]
    kpis["fabric_saving_pct"] = None if abs(pf) < 1e-9 else round((pf - af) / pf * 100, 2)

    grouped: list[dict] = []
    gkey = str(group_by or "").strip().lower()
    key_map = {
        "so": "so_number",
        "so_number": "so_number",
        "parent_style": "parent_style",
        "style": "parent_style",
        "sku": "sku",
        "component": "component",
        "size": "size",
        "jo": "jo_number",
    }
    col = key_map.get(gkey)
    if col:
        buckets: dict[str, list] = {}
        for r in filtered:
            buckets.setdefault(str(r.get(col) or "—"), []).append(r)
        for name, items in sorted(buckets.items()):
            grouped.append({
                "group": name,
                "rows": len(items),
                "planned_qty": sum(int(x["planned_qty"]) for x in items),
                "issued_qty": sum(int(x["issued_qty"]) for x in items),
                "received_qty": sum(int(x["received_qty"]) for x in items),
                "balance_qty": sum(int(x["balance_qty"]) for x in items),
                "qty_variance": sum(int(x["qty_variance"]) for x in items),
                "pending_jos": len({x["jo_id"] for x in items if x["status"] == "pending"}),
                "planned_fabric": round(sum(float(x["planned_fabric"] or 0) for x in items), 4),
                "actual_fabric": round(sum(float(x["actual_fabric"] or 0) for x in items), 4),
            })

    total = len(filtered)
    ps = 0 if export else max(0, int(page_size or 0))
    if ps <= 0:
        page_rows = filtered
        page = 1
    else:
        page = max(1, int(page or 1))
        start = (page - 1) * ps
        page_rows = filtered[start:start + ps]

    return {
        "ok": True,
        "aging_basis": basis_key,
        "balance_level": level,
        "production_mode_filter": mode_want or "all",
        "components_filter": comp_want,
        "kpis": kpis,
        "groups": grouped,
        "total": total,
        "page": page,
        "page_size": ps or total,
        "rows": page_rows,
        "definitions": {
            "balance_qty": "planned_qty - received_qty (negative means over-receipt)",
            "qty_variance": "received_qty - planned_qty",
            "status": "over if received>planned; exact if equal; under if closed short; pending if still open short",
            "fabric_saving": "BOM avg × received - actual fabric issued (positive = saving)",
        },
    }
