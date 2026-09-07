"""Date-wise production transactions across all processes.

Lists individual receive / issue / JO-create events with per-transaction qty
(not cumulative JO balances) for daily reconciliation and audit.
"""
from __future__ import annotations

from typing import Any

from ..db import production_db


def _parse_day(raw: Any) -> str:
    s = str(raw or "").strip()[:10]
    return s if len(s) == 10 else ""


def build_process_date_transactions(
    *,
    date_from: str = "",
    date_to: str = "",
    txn_date: str = "",
    process: str = "",
    so_number: str = "",
    sku: str = "",
    jo_number: str = "",
    component: str = "",
    vendor_name: str = "",
    txn_type: str = "",  # created | received | issued | "" (all)
    page: int = 1,
    page_size: int = 200,
    export: bool = False,
) -> dict[str, Any]:
    d_from = _parse_day(date_from)
    d_to = _parse_day(date_to)
    day = _parse_day(txn_date)
    if day:
        d_from = d_to = day

    proc = str(process or "").strip()
    so = str(so_number or "").strip()
    sku_q = str(sku or "").strip()
    jo_q = str(jo_number or "").strip()
    comp = str(component or "").strip()
    vendor = str(vendor_name or "").strip()
    want_type = str(txn_type or "").strip().lower()

    rows: list[dict[str, Any]] = []
    conn = production_db._connect()
    try:
        # 1) JO created on date
        if want_type in ("", "created"):
            sql = """
                SELECT j.id AS jo_id, j.jo_number, j.jo_date, j.process, j.so_number, j.sku,
                       j.component_code, j.planned_qty, j.vendor_name, j.exec_type, j.status,
                       j.main_sku, j.production_mode
                FROM job_orders j
                WHERE IFNULL(j.status,'') != 'Cancelled'
            """
            params: list[Any] = []
            if d_from:
                sql += " AND substr(IFNULL(j.jo_date,''),1,10) >= ?"
                params.append(d_from)
            if d_to:
                sql += " AND substr(IFNULL(j.jo_date,''),1,10) <= ?"
                params.append(d_to)
            if proc:
                sql += " AND j.process = ?"
                params.append(proc)
            if so:
                sql += " AND IFNULL(j.so_number,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{so}%")
            if sku_q:
                sql += " AND (IFNULL(j.sku,'') LIKE ? COLLATE NOCASE OR IFNULL(j.main_sku,'') LIKE ? COLLATE NOCASE)"
                params.extend([f"%{sku_q}%", f"%{sku_q}%"])
            if jo_q:
                sql += " AND IFNULL(j.jo_number,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{jo_q}%")
            if comp:
                sql += " AND IFNULL(j.component_code,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{comp}%")
            if vendor:
                sql += " AND IFNULL(j.vendor_name,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{vendor}%")
            for r in conn.execute(sql, params).fetchall():
                d = dict(r)
                rows.append({
                    "txn_type": "created",
                    "txn_date": str(d.get("jo_date") or "")[:10],
                    "process": d.get("process") or "",
                    "qty": int(d.get("planned_qty") or 0),
                    "so_number": d.get("so_number") or "",
                    "jo_number": d.get("jo_number") or "",
                    "jo_id": d.get("jo_id"),
                    "sku": d.get("sku") or "",
                    "component": d.get("component_code") or "",
                    "vendor_name": d.get("vendor_name") or "",
                    "exec_type": d.get("exec_type") or "",
                    "from_process": "",
                    "to_process": "",
                    "remarks": f"JO created · {d.get('status') or ''}",
                    "ref_id": d.get("jo_id"),
                })

        # 2) Piece receipts (individual qty per date)
        if want_type in ("", "received"):
            sql = """
                SELECT r.id AS ref_id, r.jo_id, r.process, r.so_number, r.sku, r.receipt_date,
                       r.received_qty, r.rejected_qty, r.remarks, r.received_by,
                       j.jo_number, j.vendor_name, j.exec_type, j.component_code, j.main_sku
                FROM jo_piece_receipts r
                LEFT JOIN job_orders j ON j.id = r.jo_id
                WHERE 1=1
            """
            params = []
            if d_from:
                sql += " AND substr(IFNULL(r.receipt_date,''),1,10) >= ?"
                params.append(d_from)
            if d_to:
                sql += " AND substr(IFNULL(r.receipt_date,''),1,10) <= ?"
                params.append(d_to)
            if proc:
                sql += " AND r.process = ?"
                params.append(proc)
            if so:
                sql += " AND IFNULL(r.so_number,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{so}%")
            if sku_q:
                sql += " AND IFNULL(r.sku,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{sku_q}%")
            if jo_q:
                sql += " AND IFNULL(j.jo_number,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{jo_q}%")
            if comp:
                sql += " AND IFNULL(j.component_code,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{comp}%")
            if vendor:
                sql += " AND IFNULL(j.vendor_name,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{vendor}%")
            for r in conn.execute(sql, params).fetchall():
                d = dict(r)
                rows.append({
                    "txn_type": "received",
                    "txn_date": str(d.get("receipt_date") or "")[:10],
                    "process": d.get("process") or "",
                    "qty": int(d.get("received_qty") or 0),
                    "rejected_qty": int(d.get("rejected_qty") or 0),
                    "so_number": d.get("so_number") or "",
                    "jo_number": d.get("jo_number") or "",
                    "jo_id": d.get("jo_id"),
                    "sku": d.get("sku") or "",
                    "component": d.get("component_code") or "",
                    "vendor_name": d.get("vendor_name") or "",
                    "exec_type": d.get("exec_type") or "",
                    "from_process": "",
                    "to_process": "",
                    "remarks": d.get("remarks") or (f"by {d.get('received_by')}" if d.get("received_by") else ""),
                    "ref_id": d.get("ref_id"),
                })

        # 3) Piece issues / moves to next process
        if want_type in ("", "issued"):
            sql = """
                SELECT i.id AS ref_id, i.jo_id, i.from_process, i.to_process, i.so_number, i.sku,
                       i.issue_date, i.issued_qty, i.remarks, i.issued_by,
                       j.jo_number, j.vendor_name, j.exec_type, j.component_code, j.process AS jo_process
                FROM jo_piece_issues i
                LEFT JOIN job_orders j ON j.id = i.jo_id
                WHERE 1=1
            """
            params = []
            if d_from:
                sql += " AND substr(IFNULL(i.issue_date,''),1,10) >= ?"
                params.append(d_from)
            if d_to:
                sql += " AND substr(IFNULL(i.issue_date,''),1,10) <= ?"
                params.append(d_to)
            if proc:
                sql += " AND (i.from_process = ? OR i.to_process = ? OR j.process = ?)"
                params.extend([proc, proc, proc])
            if so:
                sql += " AND IFNULL(i.so_number,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{so}%")
            if sku_q:
                sql += " AND IFNULL(i.sku,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{sku_q}%")
            if jo_q:
                sql += " AND IFNULL(j.jo_number,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{jo_q}%")
            if comp:
                sql += " AND IFNULL(j.component_code,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{comp}%")
            if vendor:
                sql += " AND IFNULL(j.vendor_name,'') LIKE ? COLLATE NOCASE"
                params.append(f"%{vendor}%")
            for r in conn.execute(sql, params).fetchall():
                d = dict(r)
                rows.append({
                    "txn_type": "issued",
                    "txn_date": str(d.get("issue_date") or "")[:10],
                    "process": d.get("from_process") or d.get("jo_process") or "",
                    "qty": int(d.get("issued_qty") or 0),
                    "so_number": d.get("so_number") or "",
                    "jo_number": d.get("jo_number") or "",
                    "jo_id": d.get("jo_id"),
                    "sku": d.get("sku") or "",
                    "component": d.get("component_code") or "",
                    "vendor_name": d.get("vendor_name") or "",
                    "exec_type": d.get("exec_type") or "",
                    "from_process": d.get("from_process") or "",
                    "to_process": d.get("to_process") or "",
                    "remarks": d.get("remarks") or (f"by {d.get('issued_by')}" if d.get("issued_by") else ""),
                    "ref_id": d.get("ref_id"),
                })
    finally:
        conn.close()

    rows.sort(key=lambda x: (x.get("txn_date") or "", x.get("txn_type") or "", x.get("jo_number") or ""))

    def _sum(typ: str) -> float:
        return float(sum(int(r.get("qty") or 0) for r in rows if r.get("txn_type") == typ))

    totals = {
        "created_qty": _sum("created"),
        "received_qty": _sum("received"),
        "issued_qty": _sum("issued"),
        "row_count": len(rows),
        "created_rows": sum(1 for r in rows if r["txn_type"] == "created"),
        "received_rows": sum(1 for r in rows if r["txn_type"] == "received"),
        "issued_rows": sum(1 for r in rows if r["txn_type"] == "issued"),
    }

    total = len(rows)
    ps = 0 if export else max(0, int(page_size or 0))
    if ps <= 0:
        page_rows = rows
        page_out = 1
    else:
        page_out = max(1, int(page or 1))
        start = (page_out - 1) * ps
        page_rows = rows[start:start + ps]

    return {
        "ok": True,
        "date_from": d_from,
        "date_to": d_to,
        "process": proc or "all",
        "totals": totals,
        "column_totals": totals,
        "total": total,
        "page": page_out,
        "page_size": ps or total,
        "rows": page_rows,
    }
