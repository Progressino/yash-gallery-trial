"""Purchase Job Work Order — auto Material Issue Notes (BOM-driven)."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any

from .jo_issue_notes import explode_bom_materials

def _connect():
    from ..db import purchase_db

    return purchase_db._connect()


def _ensure_min_schema(conn) -> None:
    for sql in (
        "ALTER TABLE material_issue_notes ADD COLUMN jwo_id INTEGER",
        "ALTER TABLE material_issue_notes ADD COLUMN jwo_date TEXT DEFAULT ''",
        "ALTER TABLE min_lines ADD COLUMN line_no INTEGER DEFAULT 1",
        "ALTER TABLE min_lines ADD COLUMN output_material TEXT DEFAULT ''",
        "ALTER TABLE min_lines ADD COLUMN output_material_name TEXT DEFAULT ''",
        "ALTER TABLE min_lines ADD COLUMN output_qty REAL DEFAULT 0",
        "ALTER TABLE min_lines ADD COLUMN bom_qty_per_unit REAL DEFAULT 0",
    ):
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass


def _next_min_number(conn) -> str:
    row = conn.execute(
        "SELECT min_number FROM material_issue_notes ORDER BY id DESC LIMIT 1"
    ).fetchone()
    n = 1
    if row:
        try:
            n = int(str(row[0]).split("-")[-1]) + 1
        except ValueError:
            pass
    return f"MIN-{n:04d}"


def _lookup_item_name(code: str) -> str:
    code = (code or "").strip()
    if not code:
        return ""
    try:
        from ..db.purchase_db import _item_db_path

        path = _item_db_path()
        if not path or not os.path.exists(path):
            return ""
        ic = sqlite3.connect(path)
        ic.row_factory = sqlite3.Row
        row = ic.execute(
            "SELECT item_name FROM items WHERE item_code=? LIMIT 1", (code,)
        ).fetchone()
        ic.close()
        return str(row["item_name"]) if row else ""
    except Exception:
        return ""


def _material_lines_from_jwo_line(ln: dict) -> list[dict]:
    """BOM explosion for one JWO process line (output = finished / SFG item)."""
    out_code = str(ln.get("output_material") or "").strip()
    out_qty = float(ln.get("output_qty") or 0)
    if not out_code or out_qty <= 0:
        return []

    out_name = _lookup_item_name(out_code) or out_code
    mats = explode_bom_materials(out_code, out_name, out_qty)

    if not mats:
        inp = str(ln.get("input_material") or "").strip()
        inq = float(ln.get("input_qty") or 0)
        if inp and inq > 0:
            mats = [
                {
                    "finished_item_code": out_code,
                    "finished_item_name": out_name,
                    "finished_planned_qty": out_qty,
                    "material_code": inp,
                    "material_name": _lookup_item_name(inp) or inp,
                    "material_type": "GF",
                    "bom_qty_per_unit": round(inq / out_qty, 4) if out_qty else inq,
                    "required_qty": inq,
                    "unit": ln.get("input_unit") or "MTR",
                }
            ]
    return mats


def create_min_for_jwo(jwoid: int, jwo_number: str, jwo: dict, jwo_lines: list[dict]) -> dict | None:
    """
    Create or refresh the Material Issue Note linked to a Job Work Order.

    Each BOM component becomes a separate issue line with qty scaled to JO output qty.
    """
    if not jwo_lines:
        return None

    material_lines: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for ln in jwo_lines:
        for m in _material_lines_from_jwo_line(ln):
            key = (m["finished_item_code"], m["material_code"])
            if key in seen:
                for ex in material_lines:
                    if (
                        ex["output_material"] == m["finished_item_code"]
                        and ex["material_code"] == m["material_code"]
                    ):
                        ex["issue_qty"] = round(
                            float(ex["issue_qty"]) + float(m["required_qty"]), 3
                        )
                        break
                continue
            seen.add(key)
            material_lines.append(
                {
                    "material_code": m["material_code"],
                    "material_name": m["material_name"],
                    "material_type": m.get("material_type") or "GF",
                    "issue_qty": float(m["required_qty"]),
                    "unit": m.get("unit") or "MTR",
                    "output_material": m["finished_item_code"],
                    "output_material_name": m["finished_item_name"],
                    "output_qty": float(m["finished_planned_qty"]),
                    "bom_qty_per_unit": float(m.get("bom_qty_per_unit") or 0),
                    "remarks": f"For {m['finished_item_name']} ({m['finished_item_code']})",
                }
            )

    conn = _connect()
    _ensure_min_schema(conn)

    existing = conn.execute(
        "SELECT id, min_number FROM material_issue_notes WHERE jwo_id=?",
        (jwoid,),
    ).fetchone()
    if not existing and jwo_number:
        existing = conn.execute(
            "SELECT id, min_number FROM material_issue_notes WHERE jwo_reference=?",
            (jwo_number,),
        ).fetchone()

    min_date = datetime.now().strftime("%Y-%m-%d")
    jwo_date = jwo.get("jwo_date") or min_date
    processor = jwo.get("processor_name") or ""
    remarks = (
        "Auto-generated from BOM"
        if material_lines
        else "No BOM lines — add materials manually"
    )

    if existing:
        min_id = int(existing["id"])
        min_num = str(existing["min_number"])
        conn.execute("DELETE FROM min_lines WHERE min_id=?", (min_id,))
        conn.execute(
            """UPDATE material_issue_notes SET min_date=?, jwo_id=?, jwo_date=?,
               jwo_reference=?, so_reference=?, to_vendor=?, issued_by=?, remarks=?, status='Draft'
               WHERE id=?""",
            (
                min_date,
                jwoid,
                jwo_date,
                jwo_number,
                jwo.get("so_reference") or "",
                processor,
                jwo.get("issued_by") or "",
                remarks,
                min_id,
            ),
        )
    else:
        min_num = _next_min_number(conn)
        conn.execute(
            """INSERT INTO material_issue_notes(
                min_number, min_date, jwo_id, jwo_date, jwo_reference, so_reference,
                from_location, to_location, to_vendor, issued_by, status, remarks)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                min_num,
                min_date,
                jwoid,
                jwo_date,
                jwo_number,
                jwo.get("so_reference") or "",
                "Grey Warehouse",
                "Processor",
                processor,
                jwo.get("issued_by") or "",
                "Draft",
                remarks,
            ),
        )
        min_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    for i, m in enumerate(material_lines, start=1):
        qty = float(m["issue_qty"])
        rate = float(m.get("rate") or 0)
        conn.execute(
            """INSERT INTO min_lines(
                min_id, line_no, material_code, material_name, material_type,
                issue_qty, unit, rate, amount, output_material, output_material_name,
                output_qty, bom_qty_per_unit, remarks)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                min_id,
                i,
                m["material_code"],
                m.get("material_name", ""),
                m.get("material_type", "GF"),
                qty,
                m.get("unit", "MTR"),
                rate,
                qty * rate,
                m.get("output_material", ""),
                m.get("output_material_name", ""),
                m.get("output_qty", 0),
                m.get("bom_qty_per_unit", 0),
                m.get("remarks", ""),
            ),
        )

    conn.commit()
    conn.close()
    return get_min_by_jwo_id(jwoid)


def get_min_by_jwo_id(jwoid: int) -> dict | None:
    conn = _connect()
    _ensure_min_schema(conn)
    row = conn.execute(
        "SELECT * FROM material_issue_notes WHERE jwo_id=? ORDER BY id DESC LIMIT 1",
        (jwoid,),
    ).fetchone()
    if not row:
        conn.close()
        return None
    note = dict(row)
    note["lines"] = [
        dict(l)
        for l in conn.execute(
            "SELECT * FROM min_lines WHERE min_id=? ORDER BY line_no, id",
            (note["id"],),
        ).fetchall()
    ]
    conn.close()
    return note
