"""Production Job Order — material issue notes (BOM-driven)."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any

_PROD_DB = os.environ.get(
    "PRODUCTION_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "production.db"),
)
_ITEM_DB = os.environ.get(
    "ITEM_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "items_dev.db"),
)


def _prod_connect():
    conn = sqlite3.connect(_PROD_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _item_connect():
    for path in (_ITEM_DB, os.path.join(os.path.dirname(__file__), "..", "items_dev.db")):
        if path and os.path.exists(path):
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            return conn
    raise FileNotFoundError("Item Master DB not found for BOM explosion")


def init_issue_note_tables():
    conn = _prod_connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS jo_issue_notes (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        in_number           TEXT UNIQUE NOT NULL,
        in_date             TEXT NOT NULL,
        jo_id               INTEGER NOT NULL REFERENCES job_orders(id) ON DELETE CASCADE,
        jo_number           TEXT NOT NULL,
        jo_date             TEXT DEFAULT '',
        so_number           TEXT DEFAULT '',
        process             TEXT DEFAULT '',
        finished_item_code  TEXT DEFAULT '',
        finished_item_name  TEXT DEFAULT '',
        planned_qty         REAL DEFAULT 0,
        status              TEXT DEFAULT 'Open',
        remarks             TEXT DEFAULT '',
        created_at          TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS jo_issue_note_lines (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_note_id       INTEGER NOT NULL REFERENCES jo_issue_notes(id) ON DELETE CASCADE,
        line_no             INTEGER DEFAULT 1,
        finished_item_code  TEXT DEFAULT '',
        finished_item_name  TEXT DEFAULT '',
        finished_planned_qty REAL DEFAULT 0,
        material_code       TEXT NOT NULL,
        material_name       TEXT DEFAULT '',
        material_type       TEXT DEFAULT '',
        bom_qty_per_unit    REAL DEFAULT 0,
        required_qty        REAL DEFAULT 0,
        unit                TEXT DEFAULT 'PCS',
        issued_qty          REAL DEFAULT 0,
        remarks             TEXT DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_jo_issue_notes_jo ON jo_issue_notes(jo_id);
    """)
    conn.commit()
    conn.close()


def _next_in_number(conn) -> str:
    row = conn.execute("SELECT in_number FROM jo_issue_notes ORDER BY id DESC LIMIT 1").fetchone()
    n = 1
    if row:
        try:
            n = int(str(row[0]).split("-")[-1]) + 1
        except ValueError:
            pass
    return f"IN-{n:04d}"


def _get_item_by_code(conn, code: str):
    row = conn.execute(
        "SELECT i.*, t.code AS type_code FROM items i "
        "JOIN item_types t ON t.id = i.item_type_id WHERE i.item_code = ?",
        (code,),
    ).fetchone()
    return dict(row) if row else None


def _get_item_by_id(conn, iid: int):
    row = conn.execute(
        "SELECT i.*, t.code AS type_code FROM items i "
        "JOIN item_types t ON t.id = i.item_type_id WHERE i.id = ?",
        (iid,),
    ).fetchone()
    return dict(row) if row else None


def _get_default_bom(conn, item_id: int):
    row = conn.execute(
        "SELECT * FROM bom_headers WHERE item_id=? AND is_default=1 LIMIT 1", (item_id,)
    ).fetchone()
    if not row:
        row = conn.execute("SELECT * FROM bom_headers WHERE item_id=? LIMIT 1", (item_id,)).fetchone()
    return dict(row) if row else None


def explode_bom_materials(finished_code: str, finished_name: str, finished_qty: float) -> list[dict]:
    """Return material lines required to produce ``finished_qty`` of ``finished_code``."""
    finished_qty = float(finished_qty or 0)
    if not finished_code or finished_qty <= 0:
        return []

    try:
        iconn = _item_connect()
    except FileNotFoundError:
        return []

    item = _get_item_by_code(iconn, finished_code.strip())
    if not item:
        iconn.close()
        return []

    bom = _get_default_bom(iconn, item["id"])
    if not bom:
        iconn.close()
        return []

    rows = iconn.execute("SELECT * FROM bom_lines WHERE bom_id=?", (bom["id"],)).fetchall()
    out: list[dict] = []
    for ln in rows:
        ln = dict(ln)
        ctype = (ln.get("component_type") or "RM").upper()
        if ctype in ("SVC", "SERVICE", "PROCESS"):
            continue
        comp = None
        code_guess = ""
        if ln.get("component_item_id"):
            comp = _get_item_by_id(iconn, int(ln["component_item_id"]))
        if not comp:
            raw = ln.get("component_name") or ""
            code_guess = raw.split(" — ")[0].strip() if " — " in raw else raw.strip()
            if code_guess:
                comp = _get_item_by_code(iconn, code_guess)
        bom_per = float(ln.get("quantity") or 0)
        shrink = float(ln.get("shrinkage_pct") or 0)
        waste = float(ln.get("wastage_pct") or 0)
        required = round(bom_per * (1 + shrink / 100 + waste / 100) * finished_qty, 3)
        if required <= 0:
            continue
        out.append(
            {
                "finished_item_code": finished_code,
                "finished_item_name": finished_name or item.get("item_name", ""),
                "finished_planned_qty": finished_qty,
                "material_code": (comp["item_code"] if comp else code_guess),
                "material_name": comp["item_name"] if comp else (ln.get("component_name") or ""),
                "material_type": ctype,
                "bom_qty_per_unit": bom_per,
                "required_qty": required,
                "unit": ln.get("unit") or (comp.get("uom") if comp else "PCS"),
            }
        )
    iconn.close()
    return out


def _finished_items_from_jo(jo: dict, line_rows: list[dict]) -> list[dict]:
    if line_rows:
        return [
            {
                "code": ln.get("sku") or jo.get("sku", ""),
                "name": ln.get("sku_name") or jo.get("sku_name", ""),
                "qty": float(ln.get("planned_qty") or 0),
            }
            for ln in line_rows
            if float(ln.get("planned_qty") or 0) > 0
        ]
    sku = (jo.get("sku") or "").strip()
    qty = float(jo.get("planned_qty") or 0)
    if sku and qty > 0:
        return [{"code": sku, "name": jo.get("sku_name", ""), "qty": qty}]
    return []


def create_issue_note_for_jo(joid: int, jo_number: str, jo: dict, jo_lines: list[dict] | None = None) -> dict:
    """Create (or replace draft) issue note with BOM lines for a new job order."""
    init_issue_note_tables()
    jo_lines = jo_lines or []
    finished_items = _finished_items_from_jo(jo, jo_lines)

    material_lines: list[dict] = []
    seen: set[tuple] = set()
    for fin in finished_items:
        for m in explode_bom_materials(fin["code"], fin["name"], fin["qty"]):
            key = (fin["code"], m["material_code"])
            if key in seen:
                for existing in material_lines:
                    if (
                        existing["finished_item_code"] == fin["code"]
                        and existing["material_code"] == m["material_code"]
                    ):
                        existing["required_qty"] = round(
                            float(existing["required_qty"]) + float(m["required_qty"]), 3
                        )
                        break
                continue
            seen.add(key)
            material_lines.append(m)

    # Cutting: add planned fabric when not already in BOM
    fabric_code = (jo.get("fabric_code") or "").strip()
    fabric_qty = float(jo.get("fabric_qty") or 0)
    fabric_unit = jo.get("fabric_unit") or "MTR"
    if fabric_code and fabric_qty > 0:
        if not any(m["material_code"] == fabric_code for m in material_lines):
            primary = finished_items[0] if finished_items else {"code": jo.get("sku", ""), "name": "", "qty": 0}
            material_lines.insert(
                0,
                {
                    "finished_item_code": primary["code"],
                    "finished_item_name": primary["name"],
                    "finished_planned_qty": primary["qty"],
                    "material_code": fabric_code,
                    "material_name": fabric_code,
                    "material_type": "GF",
                    "bom_qty_per_unit": round(fabric_qty / primary["qty"], 4) if primary["qty"] else fabric_qty,
                    "required_qty": fabric_qty,
                    "unit": fabric_unit,
                    "remarks": "From JO fabric plan",
                },
            )

    primary_fin = finished_items[0] if finished_items else {"code": jo.get("sku", ""), "name": jo.get("sku_name", ""), "qty": 0}

    conn = _prod_connect()
    existing = conn.execute("SELECT id FROM jo_issue_notes WHERE jo_id=?", (joid,)).fetchone()
    if existing:
        conn.execute("DELETE FROM jo_issue_note_lines WHERE issue_note_id=?", (existing["id"],))
        in_id = existing["id"]
        in_num = conn.execute("SELECT in_number FROM jo_issue_notes WHERE id=?", (in_id,)).fetchone()[0]
        conn.execute(
            """UPDATE jo_issue_notes SET in_date=?, jo_date=?, so_number=?, process=?,
               finished_item_code=?, finished_item_name=?, planned_qty=?, status='Open',
               remarks=? WHERE id=?""",
            (
                datetime.now().strftime("%Y-%m-%d"),
                jo.get("jo_date", ""),
                jo.get("so_number", ""),
                jo.get("process", ""),
                primary_fin["code"],
                primary_fin["name"],
                primary_fin["qty"],
                "Auto-generated from BOM" if material_lines else "No BOM lines — add materials manually",
                in_id,
            ),
        )
    else:
        in_num = _next_in_number(conn)
        conn.execute(
            """INSERT INTO jo_issue_notes(
                in_number, in_date, jo_id, jo_number, jo_date, so_number, process,
                finished_item_code, finished_item_name, planned_qty, status, remarks)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                in_num,
                datetime.now().strftime("%Y-%m-%d"),
                joid,
                jo_number,
                jo.get("jo_date", ""),
                jo.get("so_number", ""),
                jo.get("process", ""),
                primary_fin["code"],
                primary_fin["name"],
                primary_fin["qty"],
                "Open",
                "Auto-generated from BOM" if material_lines else "No BOM lines — add materials manually",
            ),
        )
        in_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    for i, m in enumerate(material_lines, start=1):
        conn.execute(
            """INSERT INTO jo_issue_note_lines(
                issue_note_id, line_no, finished_item_code, finished_item_name, finished_planned_qty,
                material_code, material_name, material_type, bom_qty_per_unit, required_qty, unit, remarks)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                in_id,
                i,
                m["finished_item_code"],
                m["finished_item_name"],
                m["finished_planned_qty"],
                m["material_code"],
                m["material_name"],
                m.get("material_type", ""),
                m["bom_qty_per_unit"],
                m["required_qty"],
                m.get("unit", "PCS"),
                m.get("remarks", ""),
            ),
        )
    conn.commit()
    conn.close()
    return get_issue_note_by_jo_id(joid) or {"in_number": in_num, "lines": material_lines}


def get_issue_note_by_jo_id(joid: int) -> dict | None:
    init_issue_note_tables()
    conn = _prod_connect()
    row = conn.execute("SELECT * FROM jo_issue_notes WHERE jo_id=?", (joid,)).fetchone()
    if not row:
        conn.close()
        return None
    note = dict(row)
    note["lines"] = [
        dict(l)
        for l in conn.execute(
            "SELECT * FROM jo_issue_note_lines WHERE issue_note_id=? ORDER BY line_no, id",
            (note["id"],),
        ).fetchall()
    ]
    conn.close()
    return note


def list_issue_notes(jo_number: str | None = None, status: str | None = None) -> list[dict]:
    init_issue_note_tables()
    conn = _prod_connect()
    q = "SELECT * FROM jo_issue_notes WHERE 1=1"
    params: list[Any] = []
    if jo_number:
        q += " AND jo_number LIKE ?"
        params.append(f"%{jo_number}%")
    if status:
        q += " AND status=?"
        params.append(status)
    q += " ORDER BY id DESC"
    rows = conn.execute(q, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["lines"] = [
            dict(l)
            for l in conn.execute(
                "SELECT * FROM jo_issue_note_lines WHERE issue_note_id=? ORDER BY line_no, id",
                (d["id"],),
            ).fetchall()
        ]
        d["line_count"] = len(d["lines"])
        d["total_required_qty"] = sum(float(l.get("required_qty") or 0) for l in d["lines"])
        out.append(d)
    conn.close()
    return out
