"""Production Quality — QC reports, defects, rework WIP, debit notes, billing eligibility.

Uses the same SQLite file as production_db (PRODUCTION_DB_PATH).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Optional

from . import production_db


def _connect() -> sqlite3.Connection:
    return production_db._connect()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def init_quality_tables(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    if own:
        conn = _connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS qc_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_no TEXT UNIQUE NOT NULL,
            qc_date TEXT NOT NULL DEFAULT '',
            found_at_process TEXT NOT NULL DEFAULT '',
            so_number TEXT DEFAULT '',
            sku TEXT DEFAULT '',
            component_code TEXT DEFAULT '',
            size_label TEXT DEFAULT '',
            original_jo_id INTEGER,
            checked_qty INTEGER NOT NULL DEFAULT 0,
            pass_qty INTEGER NOT NULL DEFAULT 0,
            rework_qty INTEGER NOT NULL DEFAULT 0,
            reject_qty INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'Completed',
            notes TEXT DEFAULT '',
            created_by TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS quality_defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qc_report_id INTEGER NOT NULL REFERENCES qc_reports(id) ON DELETE CASCADE,
            found_at_process TEXT NOT NULL DEFAULT '',
            defect_source_process TEXT NOT NULL DEFAULT '',
            responsible_vendor TEXT DEFAULT '',
            responsible_jo_id INTEGER,
            action TEXT NOT NULL DEFAULT 'Rework',
            qty INTEGER NOT NULL DEFAULT 0,
            reason TEXT DEFAULT '',
            rework_by_type TEXT DEFAULT 'SameVendor',
            rework_by_vendor TEXT DEFAULT '',
            chargeable INTEGER DEFAULT 0,
            debit_required INTEGER DEFAULT 0,
            with_fabric INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rework_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rework_no TEXT UNIQUE NOT NULL,
            original_jo_id INTEGER NOT NULL,
            defect_id INTEGER,
            qc_report_id INTEGER,
            process TEXT NOT NULL DEFAULT '',
            so_number TEXT DEFAULT '',
            sku TEXT DEFAULT '',
            component_code TEXT DEFAULT '',
            planned_qty INTEGER NOT NULL DEFAULT 0,
            received_qty INTEGER NOT NULL DEFAULT 0,
            pass_qty INTEGER NOT NULL DEFAULT 0,
            reject_qty INTEGER NOT NULL DEFAULT 0,
            balance_qty INTEGER NOT NULL DEFAULT 0,
            rework_by_type TEXT DEFAULT 'SameVendor',
            rework_by_vendor TEXT DEFAULT '',
            responsible_vendor TEXT DEFAULT '',
            defect_source_process TEXT DEFAULT '',
            found_at_process TEXT DEFAULT '',
            chargeable INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'Open',
            notes TEXT DEFAULT '',
            created_by TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rework_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rework_id INTEGER NOT NULL REFERENCES rework_orders(id) ON DELETE CASCADE,
            receipt_date TEXT NOT NULL DEFAULT '',
            received_qty INTEGER NOT NULL DEFAULT 0,
            pass_qty INTEGER NOT NULL DEFAULT 0,
            reject_qty INTEGER NOT NULL DEFAULT 0,
            remarks TEXT DEFAULT '',
            created_by TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS debit_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debit_no TEXT UNIQUE NOT NULL,
            debit_date TEXT NOT NULL DEFAULT '',
            defect_id INTEGER,
            qc_report_id INTEGER,
            original_jo_id INTEGER,
            rework_id INTEGER,
            responsible_vendor TEXT NOT NULL DEFAULT '',
            process TEXT DEFAULT '',
            so_number TEXT DEFAULT '',
            sku TEXT DEFAULT '',
            component_code TEXT DEFAULT '',
            qty INTEGER NOT NULL DEFAULT 0,
            with_fabric INTEGER DEFAULT 0,
            workmanship_rate REAL DEFAULT 0,
            workmanship_amount REAL DEFAULT 0,
            fabric_amount REAL DEFAULT 0,
            total_amount REAL DEFAULT 0,
            rate_basis TEXT DEFAULT 'weighted_avg',
            fabric_lines_json TEXT DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'Open',
            notes TEXT DEFAULT '',
            created_by TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS qc_billing_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            production_mode TEXT NOT NULL DEFAULT 'default',
            qc_process TEXT NOT NULL DEFAULT 'Finishing',
            UNIQUE(production_mode)
        );
        """
    )
    for ddl in (
        "CREATE INDEX IF NOT EXISTS idx_qc_reports_jo ON qc_reports(original_jo_id)",
        "CREATE INDEX IF NOT EXISTS idx_qc_reports_date ON qc_reports(qc_date)",
        "CREATE INDEX IF NOT EXISTS idx_quality_defects_report ON quality_defects(qc_report_id)",
        "CREATE INDEX IF NOT EXISTS idx_rework_orders_jo ON rework_orders(original_jo_id)",
        "CREATE INDEX IF NOT EXISTS idx_rework_orders_status ON rework_orders(status)",
        "CREATE INDEX IF NOT EXISTS idx_debit_notes_vendor ON debit_notes(responsible_vendor)",
        "CREATE INDEX IF NOT EXISTS idx_debit_notes_jo ON debit_notes(original_jo_id)",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass
    # Seed default billing QC process
    conn.execute(
        """INSERT OR IGNORE INTO qc_billing_config(production_mode, qc_process)
           VALUES('default','Finishing'),('inhouse','Finishing'),
                 ('cut_to_pack','Finishing'),('stitch_to_pack','Finishing')"""
    )
    if own:
        conn.commit()
        conn.close()


def _next_no(conn: sqlite3.Connection, table: str, prefix: str) -> str:
    row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
    n = int(row["c"] if row else 0) + 1
    return f"{prefix}-{n:05d}"


def resolve_responsible_jo(
    *,
    so_number: str,
    sku: str,
    defect_source_process: str,
) -> dict | None:
    """Best-effort: find the JO that performed the defect-source process."""
    conn = _connect()
    row = conn.execute(
        """SELECT id, jo_number, process, vendor_name, exec_type, planned_qty, received_qty, vendor_rate
           FROM job_orders
           WHERE IFNULL(status,'') != 'Cancelled'
             AND process=?
             AND IFNULL(so_number,'')=?
             AND (IFNULL(sku,'')=? OR IFNULL(main_sku,'')=?)
           ORDER BY id DESC LIMIT 1""",
        (defect_source_process, so_number, sku, sku),
    ).fetchone()
    if not row and so_number:
        row = conn.execute(
            """SELECT id, jo_number, process, vendor_name, exec_type, planned_qty, received_qty, vendor_rate
               FROM job_orders
               WHERE IFNULL(status,'') != 'Cancelled'
                 AND process=? AND IFNULL(so_number,'')=?
               ORDER BY id DESC LIMIT 1""",
            (defect_source_process, so_number),
        ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── QC Reports ────────────────────────────────────────────────────────────────


def create_qc_report(data: dict) -> dict:
    init_quality_tables()
    found_at = str(data.get("found_at_process") or "").strip()
    if not found_at:
        raise ValueError("found_at_process is required")
    checked = int(data.get("checked_qty") or 0)
    if checked <= 0:
        raise ValueError("checked_qty must be > 0")
    pass_qty = int(data.get("pass_qty") or 0)
    rework_qty = int(data.get("rework_qty") or 0)
    reject_qty = int(data.get("reject_qty") or 0)
    if pass_qty + rework_qty + reject_qty > checked:
        raise ValueError("pass + rework + reject cannot exceed checked_qty")

    defects = data.get("defects") or []
    defect_sum = sum(int(d.get("qty") or 0) for d in defects)
    if defects and defect_sum != rework_qty + reject_qty:
        # Allow if caller sets rework/reject from defects
        if rework_qty == 0 and reject_qty == 0:
            rework_qty = sum(
                int(d.get("qty") or 0)
                for d in defects
                if str(d.get("action") or "").lower() in ("rework", "alteration")
            )
            reject_qty = sum(
                int(d.get("qty") or 0)
                for d in defects
                if str(d.get("action") or "").lower() in ("finalreject", "final_reject", "reject")
            )
            if pass_qty == 0:
                pass_qty = max(0, checked - rework_qty - reject_qty)
        elif defect_sum != rework_qty + reject_qty:
            raise ValueError(
                f"Defect qtys ({defect_sum}) must equal rework+reject ({rework_qty + reject_qty})"
            )

    jo_id = data.get("original_jo_id")
    try:
        jo_id = int(jo_id) if jo_id not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        jo_id = None

    so_number = str(data.get("so_number") or "").strip()
    sku = str(data.get("sku") or "").strip()
    component = str(data.get("component_code") or "").strip()
    if jo_id:
        jo = production_db.get_jo(jo_id)
        if not jo:
            raise ValueError("original_jo_id not found")
        so_number = so_number or jo.get("so_number") or ""
        sku = sku or jo.get("sku") or ""
        component = component or jo.get("component_code") or ""

    conn = _connect()
    report_no = _next_no(conn, "qc_reports", "QC")
    qc_date = str(data.get("qc_date") or _today())[:10]
    cur = conn.execute(
        """INSERT INTO qc_reports(
            report_no, qc_date, found_at_process, so_number, sku, component_code, size_label,
            original_jo_id, checked_qty, pass_qty, rework_qty, reject_qty, status, notes, created_by, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            report_no,
            qc_date,
            found_at,
            so_number,
            sku,
            component,
            str(data.get("size_label") or ""),
            jo_id,
            checked,
            pass_qty,
            rework_qty,
            reject_qty,
            str(data.get("status") or "Completed"),
            str(data.get("notes") or ""),
            str(data.get("created_by") or ""),
            _now(),
        ),
    )
    report_id = int(cur.lastrowid)
    created_defects: list[dict] = []
    created_reworks: list[dict] = []
    created_debits: list[dict] = []

    for d in defects:
        action = str(d.get("action") or "Rework").strip()
        al = action.lower().replace(" ", "").replace("_", "")
        if al in ("finalreject", "reject", "rejected"):
            action = "FinalReject"
        elif al in ("alteration", "alter"):
            action = "Alteration"
        else:
            action = "Rework"
        qty = int(d.get("qty") or 0)
        if qty <= 0:
            continue
        source = str(d.get("defect_source_process") or "").strip()
        if not source:
            raise ValueError("Each defect needs defect_source_process")
        resp_vendor = str(d.get("responsible_vendor") or "").strip()
        resp_jo_id = d.get("responsible_jo_id")
        try:
            resp_jo_id = int(resp_jo_id) if resp_jo_id not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            resp_jo_id = None
        if not resp_jo_id or not resp_vendor:
            hint = resolve_responsible_jo(
                so_number=so_number, sku=sku, defect_source_process=source
            )
            if hint:
                resp_jo_id = resp_jo_id or int(hint["id"])
                resp_vendor = resp_vendor or str(hint.get("vendor_name") or "")

        rework_by_type = str(d.get("rework_by_type") or "SameVendor").strip()
        if rework_by_type not in ("SameVendor", "OtherVendor", "Inhouse"):
            rework_by_type = "SameVendor"
        rework_by_vendor = str(d.get("rework_by_vendor") or "").strip()
        if rework_by_type == "SameVendor" and not rework_by_vendor:
            rework_by_vendor = resp_vendor
        if rework_by_type == "Inhouse":
            rework_by_vendor = rework_by_vendor or "In-house"

        debit_required = int(
            d.get("debit_required")
            if d.get("debit_required") is not None
            else (1 if rework_by_type == "Inhouse" or action == "FinalReject" else 0)
        )
        with_fabric = int(d.get("with_fabric") or 0)
        if action == "FinalReject" and d.get("with_fabric") is None:
            with_fabric = int(bool(d.get("with_fabric_default") or False))

        chargeable = int(d.get("chargeable") or 0)
        if rework_by_type == "SameVendor":
            chargeable = 0

        cur_d = conn.execute(
            """INSERT INTO quality_defects(
                qc_report_id, found_at_process, defect_source_process, responsible_vendor,
                responsible_jo_id, action, qty, reason, rework_by_type, rework_by_vendor,
                chargeable, debit_required, with_fabric
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                report_id,
                found_at,
                source,
                resp_vendor,
                resp_jo_id,
                action,
                qty,
                str(d.get("reason") or ""),
                rework_by_type,
                rework_by_vendor,
                chargeable,
                debit_required,
                with_fabric,
            ),
        )
        defect_id = int(cur_d.lastrowid)
        defect_row = {
            "id": defect_id,
            "action": action,
            "qty": qty,
            "defect_source_process": source,
            "responsible_vendor": resp_vendor,
            "responsible_jo_id": resp_jo_id,
            "rework_by_type": rework_by_type,
            "rework_by_vendor": rework_by_vendor,
            "chargeable": chargeable,
            "debit_required": debit_required,
            "with_fabric": with_fabric,
            "reason": str(d.get("reason") or ""),
        }
        created_defects.append(defect_row)

        # Auto-create rework order for Rework/Alteration
        if action in ("Rework", "Alteration") and jo_id:
            rw = _create_rework_order_conn(
                conn,
                {
                    "original_jo_id": jo_id,
                    "defect_id": defect_id,
                    "qc_report_id": report_id,
                    "process": source,
                    "so_number": so_number,
                    "sku": sku,
                    "component_code": component,
                    "planned_qty": qty,
                    "rework_by_type": rework_by_type,
                    "rework_by_vendor": rework_by_vendor,
                    "responsible_vendor": resp_vendor,
                    "defect_source_process": source,
                    "found_at_process": found_at,
                    "chargeable": chargeable,
                    "created_by": data.get("created_by") or "",
                    "notes": d.get("reason") or "",
                },
            )
            created_reworks.append(rw)

        # Auto debit note when required
        if debit_required and resp_vendor:
            dn = _create_debit_note_conn(
                conn,
                {
                    "defect_id": defect_id,
                    "qc_report_id": report_id,
                    "original_jo_id": resp_jo_id or jo_id,
                    "responsible_vendor": resp_vendor,
                    "process": source,
                    "so_number": so_number,
                    "sku": sku,
                    "component_code": component,
                    "qty": qty,
                    "with_fabric": with_fabric if action == "FinalReject" else 0,
                    "notes": d.get("reason") or "",
                    "created_by": data.get("created_by") or "",
                    "workmanship_rate": d.get("workmanship_rate"),
                },
            )
            created_debits.append(dn)

    conn.commit()
    conn.close()
    report = get_qc_report(report_id)
    report["created_reworks"] = created_reworks
    report["created_debits"] = created_debits
    return report


def get_qc_report(report_id: int) -> dict:
    conn = _connect()
    row = conn.execute("SELECT * FROM qc_reports WHERE id=?", (report_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("QC report not found")
    d = dict(row)
    defects = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM quality_defects WHERE qc_report_id=? ORDER BY id",
            (report_id,),
        ).fetchall()
    ]
    conn.close()
    d["defects"] = defects
    return d


def list_qc_reports(
    *,
    so_number: str = "",
    process: str = "",
    jo_id: int | None = None,
    date_from: str = "",
    date_to: str = "",
    limit: int = 200,
) -> list[dict]:
    init_quality_tables()
    conn = _connect()
    sql = "SELECT * FROM qc_reports WHERE 1=1"
    params: list[Any] = []
    if so_number:
        sql += " AND IFNULL(so_number,'') LIKE ? COLLATE NOCASE"
        params.append(f"%{so_number}%")
    if process:
        sql += " AND found_at_process=?"
        params.append(process)
    if jo_id:
        sql += " AND original_jo_id=?"
        params.append(int(jo_id))
    if date_from:
        sql += " AND substr(qc_date,1,10) >= ?"
        params.append(date_from[:10])
    if date_to:
        sql += " AND substr(qc_date,1,10) <= ?"
        params.append(date_to[:10])
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(int(limit or 200), 2000)))
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    for r in rows:
        r["defects"] = [
            dict(x)
            for x in conn.execute(
                "SELECT * FROM quality_defects WHERE qc_report_id=?",
                (r["id"],),
            ).fetchall()
        ]
    conn.close()
    return rows


# ── Rework orders ─────────────────────────────────────────────────────────────


def _create_rework_order_conn(conn: sqlite3.Connection, data: dict) -> dict:
    planned = int(data.get("planned_qty") or 0)
    if planned <= 0:
        raise ValueError("rework planned_qty must be > 0")
    original_jo_id = int(data["original_jo_id"])
    rework_no = _next_no(conn, "rework_orders", "RW")
    cur = conn.execute(
        """INSERT INTO rework_orders(
            rework_no, original_jo_id, defect_id, qc_report_id, process, so_number, sku,
            component_code, planned_qty, received_qty, pass_qty, reject_qty, balance_qty,
            rework_by_type, rework_by_vendor, responsible_vendor, defect_source_process,
            found_at_process, chargeable, status, notes, created_by, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,0,0,0,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            rework_no,
            original_jo_id,
            data.get("defect_id"),
            data.get("qc_report_id"),
            str(data.get("process") or ""),
            str(data.get("so_number") or ""),
            str(data.get("sku") or ""),
            str(data.get("component_code") or ""),
            planned,
            planned,  # balance = planned initially
            str(data.get("rework_by_type") or "SameVendor"),
            str(data.get("rework_by_vendor") or ""),
            str(data.get("responsible_vendor") or ""),
            str(data.get("defect_source_process") or ""),
            str(data.get("found_at_process") or ""),
            int(data.get("chargeable") or 0),
            "Open",
            str(data.get("notes") or ""),
            str(data.get("created_by") or ""),
            _now(),
        ),
    )
    rid = int(cur.lastrowid)
    return dict(conn.execute("SELECT * FROM rework_orders WHERE id=?", (rid,)).fetchone())


def create_rework_order(data: dict) -> dict:
    init_quality_tables()
    jo_id = int(data.get("original_jo_id") or 0)
    if not jo_id:
        raise ValueError("original_jo_id is required")
    jo = production_db.get_jo(jo_id)
    if not jo:
        raise ValueError("original JO not found")
    payload = {
        **data,
        "so_number": data.get("so_number") or jo.get("so_number") or "",
        "sku": data.get("sku") or jo.get("sku") or "",
        "component_code": data.get("component_code") or jo.get("component_code") or "",
        "process": data.get("process") or data.get("defect_source_process") or jo.get("process") or "",
        "responsible_vendor": data.get("responsible_vendor") or jo.get("vendor_name") or "",
    }
    conn = _connect()
    row = _create_rework_order_conn(conn, payload)
    conn.commit()
    conn.close()
    return row


def receive_rework(rework_id: int, data: dict) -> dict:
    init_quality_tables()
    conn = _connect()
    row = conn.execute("SELECT * FROM rework_orders WHERE id=?", (rework_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Rework order not found")
    rw = dict(row)
    if str(rw.get("status") or "") in ("Closed", "Cancelled"):
        conn.close()
        raise ValueError("Rework order is closed")
    received = int(data.get("received_qty") or 0)
    pass_qty = int(data.get("pass_qty") or received)
    reject_qty = int(data.get("reject_qty") or 0)
    if received <= 0:
        conn.close()
        raise ValueError("received_qty must be > 0")
    if pass_qty + reject_qty > received:
        conn.close()
        raise ValueError("pass + reject cannot exceed received")
    remaining = int(rw["planned_qty"]) - int(rw["received_qty"])
    if received > remaining:
        conn.close()
        raise ValueError(f"Cannot receive more than remaining rework qty ({remaining})")

    conn.execute(
        """INSERT INTO rework_receipts(rework_id, receipt_date, received_qty, pass_qty, reject_qty, remarks, created_by)
           VALUES(?,?,?,?,?,?,?)""",
        (
            rework_id,
            str(data.get("receipt_date") or _today())[:10],
            received,
            pass_qty,
            reject_qty,
            str(data.get("remarks") or ""),
            str(data.get("created_by") or ""),
        ),
    )
    new_rec = int(rw["received_qty"]) + received
    new_pass = int(rw["pass_qty"]) + pass_qty
    new_rej = int(rw["reject_qty"]) + reject_qty
    new_bal = int(rw["planned_qty"]) - new_rec
    status = "Completed" if new_bal <= 0 else "In Progress"
    if new_bal <= 0 and new_rej > 0 and new_pass + new_rej >= int(rw["planned_qty"]):
        status = "Completed"
    conn.execute(
        """UPDATE rework_orders
           SET received_qty=?, pass_qty=?, reject_qty=?, balance_qty=?, status=?, updated_at=?
           WHERE id=?""",
        (new_rec, new_pass, new_rej, max(0, new_bal), status, _now(), rework_id),
    )
    conn.commit()
    out = dict(conn.execute("SELECT * FROM rework_orders WHERE id=?", (rework_id,)).fetchone())
    receipts = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM rework_receipts WHERE rework_id=? ORDER BY id",
            (rework_id,),
        ).fetchall()
    ]
    conn.close()
    out["receipts"] = receipts
    return out


def list_rework_orders(
    *,
    status: str = "",
    process: str = "",
    original_jo_id: int | None = None,
    vendor: str = "",
    limit: int = 200,
) -> list[dict]:
    init_quality_tables()
    conn = _connect()
    sql = "SELECT * FROM rework_orders WHERE 1=1"
    params: list[Any] = []
    if status:
        sql += " AND status=?"
        params.append(status)
    if process:
        sql += " AND process=?"
        params.append(process)
    if original_jo_id:
        sql += " AND original_jo_id=?"
        params.append(int(original_jo_id))
    if vendor:
        sql += " AND (IFNULL(responsible_vendor,'') LIKE ? COLLATE NOCASE OR IFNULL(rework_by_vendor,'') LIKE ? COLLATE NOCASE)"
        params.extend([f"%{vendor}%", f"%{vendor}%"])
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(int(limit or 200), 2000)))
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def get_rework_order(rework_id: int) -> dict:
    conn = _connect()
    row = conn.execute("SELECT * FROM rework_orders WHERE id=?", (rework_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Rework not found")
    d = dict(row)
    d["receipts"] = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM rework_receipts WHERE rework_id=? ORDER BY id",
            (rework_id,),
        ).fetchall()
    ]
    conn.close()
    return d


def rework_wip_summary(*, so_number: str = "", sku: str = "") -> list[dict]:
    """Accountable split: original JO qty vs rework WIP by process (does not inflate planned)."""
    init_quality_tables()
    conn = _connect()
    sql = """
        SELECT process, so_number, sku, component_code,
               SUM(balance_qty) AS rework_pending,
               SUM(planned_qty) AS rework_issued,
               SUM(pass_qty) AS rework_pass,
               SUM(reject_qty) AS rework_reject,
               COUNT(*) AS open_orders
        FROM rework_orders
        WHERE status IN ('Open','In Progress') AND balance_qty > 0
    """
    params: list[Any] = []
    if so_number:
        sql += " AND IFNULL(so_number,'') LIKE ? COLLATE NOCASE"
        params.append(f"%{so_number}%")
    if sku:
        sql += " AND IFNULL(sku,'') LIKE ? COLLATE NOCASE"
        params.append(f"%{sku}%")
    sql += " GROUP BY process, so_number, sku, component_code ORDER BY so_number, process"
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


# ── Debit notes ───────────────────────────────────────────────────────────────


def _fabric_recovery_lines(sku: str, qty: int, process: str = "") -> tuple[list[dict], float]:
    """BOM materials × reject qty. Amount uses qty as proxy until rate enrichment."""
    try:
        from ..services.jo_issue_notes import explode_bom_materials
    except Exception:
        return [], 0.0
    lines = explode_bom_materials(sku, "", float(qty), process=process or None) or []
    out = []
    total = 0.0
    for ln in lines:
        adj = float(ln.get("adj_qty") or ln.get("qty") or 0)
        rate = float(ln.get("rate") or ln.get("unit_cost") or ln.get("avg_rate") or 0)
        amt = round(adj * rate, 2) if rate else 0.0
        total += amt
        out.append(
            {
                "material_code": ln.get("material_code") or ln.get("code") or "",
                "material_name": ln.get("material_name") or ln.get("name") or "",
                "unit": ln.get("unit") or "",
                "qty": adj,
                "rate": rate,
                "amount": amt,
                "rate_basis": "bom_rate" if rate else "qty_only",
            }
        )
    return out, round(total, 2)


def _create_debit_note_conn(conn: sqlite3.Connection, data: dict) -> dict:
    qty = int(data.get("qty") or 0)
    if qty <= 0:
        raise ValueError("debit qty must be > 0")
    vendor = str(data.get("responsible_vendor") or "").strip()
    if not vendor:
        raise ValueError("responsible_vendor is required for debit note")
    with_fabric = int(data.get("with_fabric") or 0)
    process = str(data.get("process") or "")
    sku = str(data.get("sku") or "")
    # Workmanship rate from JO vendor_rate when available
    rate = data.get("workmanship_rate")
    if rate is None and data.get("original_jo_id"):
        jo = conn.execute(
            "SELECT vendor_rate FROM job_orders WHERE id=?",
            (int(data["original_jo_id"]),),
        ).fetchone()
        if jo:
            rate = jo["vendor_rate"]
    try:
        rate = float(rate or 0)
    except (TypeError, ValueError):
        rate = 0.0
    workmanship = round(rate * qty, 2)
    fabric_lines: list[dict] = []
    fabric_amt = 0.0
    if with_fabric and sku:
        fabric_lines, fabric_amt = _fabric_recovery_lines(sku, qty, process)
    total = round(workmanship + fabric_amt, 2)
    debit_no = _next_no(conn, "debit_notes", "DN")
    cur = conn.execute(
        """INSERT INTO debit_notes(
            debit_no, debit_date, defect_id, qc_report_id, original_jo_id, rework_id,
            responsible_vendor, process, so_number, sku, component_code, qty, with_fabric,
            workmanship_rate, workmanship_amount, fabric_amount, total_amount, rate_basis,
            fabric_lines_json, status, notes, created_by, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            debit_no,
            str(data.get("debit_date") or _today())[:10],
            data.get("defect_id"),
            data.get("qc_report_id"),
            data.get("original_jo_id"),
            data.get("rework_id"),
            vendor,
            process,
            str(data.get("so_number") or ""),
            sku,
            str(data.get("component_code") or ""),
            qty,
            with_fabric,
            rate,
            workmanship,
            fabric_amt,
            total,
            str(data.get("rate_basis") or "weighted_avg"),
            json.dumps(fabric_lines),
            "Open",
            str(data.get("notes") or ""),
            str(data.get("created_by") or ""),
            _now(),
        ),
    )
    did = int(cur.lastrowid)
    row = dict(conn.execute("SELECT * FROM debit_notes WHERE id=?", (did,)).fetchone())
    row["fabric_lines"] = fabric_lines
    return row


def create_debit_note(data: dict) -> dict:
    init_quality_tables()
    conn = _connect()
    row = _create_debit_note_conn(conn, data)
    conn.commit()
    conn.close()
    return row


def list_debit_notes(
    *,
    vendor: str = "",
    jo_id: int | None = None,
    status: str = "",
    limit: int = 200,
) -> list[dict]:
    init_quality_tables()
    conn = _connect()
    sql = "SELECT * FROM debit_notes WHERE 1=1"
    params: list[Any] = []
    if vendor:
        sql += " AND IFNULL(responsible_vendor,'') LIKE ? COLLATE NOCASE"
        params.append(f"%{vendor}%")
    if jo_id:
        sql += " AND original_jo_id=?"
        params.append(int(jo_id))
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(int(limit or 200), 2000)))
    rows = []
    for r in conn.execute(sql, params).fetchall():
        d = dict(r)
        try:
            d["fabric_lines"] = json.loads(d.get("fabric_lines_json") or "[]")
        except Exception:
            d["fabric_lines"] = []
        rows.append(d)
    conn.close()
    return rows


def update_debit_note_status(debit_id: int, status: str, notes: str = "") -> dict:
    init_quality_tables()
    st = str(status or "").strip()
    if st not in ("Open", "Posted", "Cancelled", "Settled"):
        raise ValueError("Invalid debit note status")
    conn = _connect()
    conn.execute(
        "UPDATE debit_notes SET status=?, notes=CASE WHEN ?!='' THEN ? ELSE notes END, updated_at=? WHERE id=?",
        (st, notes, notes, _now(), debit_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM debit_notes WHERE id=?", (debit_id,)).fetchone()
    conn.close()
    if not row:
        raise ValueError("Debit note not found")
    return dict(row)


# ── Billing eligibility ───────────────────────────────────────────────────────


def get_billing_qc_process(production_mode: str = "") -> str:
    init_quality_tables()
    mode = str(production_mode or "default").strip().lower() or "default"
    conn = _connect()
    row = conn.execute(
        "SELECT qc_process FROM qc_billing_config WHERE production_mode=?",
        (mode,),
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT qc_process FROM qc_billing_config WHERE production_mode='default'"
        ).fetchone()
    conn.close()
    return str(row["qc_process"] if row else "Finishing")


def set_billing_qc_process(production_mode: str, qc_process: str) -> dict:
    init_quality_tables()
    mode = str(production_mode or "default").strip().lower() or "default"
    proc = str(qc_process or "Finishing").strip() or "Finishing"
    conn = _connect()
    conn.execute(
        """INSERT INTO qc_billing_config(production_mode, qc_process) VALUES(?,?)
           ON CONFLICT(production_mode) DO UPDATE SET qc_process=excluded.qc_process""",
        (mode, proc),
    )
    conn.commit()
    conn.close()
    return {"production_mode": mode, "qc_process": proc}


def jo_billing_eligibility(jo_id: int) -> dict:
    """QC-linked billing position for a vendor JO (receive ≠ billable)."""
    init_quality_tables()
    jo = production_db.get_jo(jo_id)
    if not jo:
        raise ValueError("JO not found")
    mode = str(jo.get("production_mode") or "default")
    billing_qc = get_billing_qc_process(mode)

    conn = _connect()
    # Prefer QC reports linked to this JO; else SO+SKU at billing QC process
    qcs = [
        dict(r)
        for r in conn.execute(
            """SELECT * FROM qc_reports
               WHERE original_jo_id=? OR (
                 IFNULL(so_number,'')=? AND IFNULL(sku,'')=? AND found_at_process=?
               )
               ORDER BY id DESC""",
            (jo_id, jo.get("so_number") or "", jo.get("sku") or "", billing_qc),
        ).fetchall()
    ]
    # Deduplicate by id
    seen = set()
    uniq = []
    for q in qcs:
        if q["id"] in seen:
            continue
        seen.add(q["id"])
        uniq.append(q)
    qcs = uniq

    checked = sum(int(q.get("checked_qty") or 0) for q in qcs)
    passed = sum(int(q.get("pass_qty") or 0) for q in qcs)
    rework = sum(int(q.get("rework_qty") or 0) for q in qcs)
    rejected = sum(int(q.get("reject_qty") or 0) for q in qcs)

    reworks = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM rework_orders WHERE original_jo_id=? OR responsible_vendor=?",
            (jo_id, jo.get("vendor_name") or "___none___"),
        ).fetchall()
    ]
    reworks = [
        r
        for r in reworks
        if int(r.get("original_jo_id") or 0) == jo_id
        or (
            str(r.get("responsible_vendor") or "") == str(jo.get("vendor_name") or "")
            and str(r.get("defect_source_process") or "") == str(jo.get("process") or "")
        )
    ]
    # Prefer original_jo linked; also include defect_source matching this JO process via responsible_jo
    reworks_src = [
        dict(r)
        for r in conn.execute(
            """SELECT rw.* FROM rework_orders rw
               JOIN quality_defects qd ON qd.id = rw.defect_id
               WHERE qd.responsible_jo_id=?""",
            (jo_id,),
        ).fetchall()
    ]
    by_id = {r["id"]: r for r in reworks}
    for r in reworks_src:
        by_id[r["id"]] = r
    reworks = list(by_id.values())

    rework_pending = sum(int(r.get("balance_qty") or 0) for r in reworks)
    rework_pass = sum(int(r.get("pass_qty") or 0) for r in reworks)
    rework_reject = sum(int(r.get("reject_qty") or 0) for r in reworks)

    debits = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM debit_notes WHERE original_jo_id=? AND status!='Cancelled'",
            (jo_id,),
        ).fetchall()
    ]
    debit_qty = sum(int(d.get("qty") or 0) for d in debits)
    debit_amount = sum(float(d.get("total_amount") or 0) for d in debits)
    conn.close()

    received = int(jo.get("received_qty") or 0)
    planned = int(jo.get("planned_qty") or 0)
    qc_done = len(qcs) > 0 and checked > 0
    # Eligible = QC pass (net of permanent rejects already in reject_qty). Rework pending not billable yet.
    # If no QC yet → hold (eligible 0, status Hold)
    if not qc_done:
        eligible = 0
        bill_status = "Hold — QC pending"
    else:
        eligible = max(0, passed)
        if rework_pending > 0:
            bill_status = "Partial — rework pending"
        elif eligible >= received and received > 0:
            bill_status = "Clear"
        elif eligible > 0:
            bill_status = "Partial"
        else:
            bill_status = "Hold — no QC pass"

    return {
        "jo_id": jo_id,
        "jo_number": jo.get("jo_number"),
        "process": jo.get("process"),
        "vendor_name": jo.get("vendor_name") or "",
        "exec_type": jo.get("exec_type") or "",
        "so_number": jo.get("so_number") or "",
        "sku": jo.get("sku") or "",
        "planned_qty": planned,
        "received_qty": received,
        "billing_qc_process": billing_qc,
        "qc_status": "Completed" if qc_done else "Pending",
        "qc_reports": [
            {
                "id": q["id"],
                "report_no": q["report_no"],
                "found_at_process": q["found_at_process"],
                "checked_qty": q["checked_qty"],
                "pass_qty": q["pass_qty"],
                "rework_qty": q["rework_qty"],
                "reject_qty": q["reject_qty"],
                "qc_date": q["qc_date"],
            }
            for q in qcs
        ],
        "qc_checked_qty": checked,
        "qc_pass_qty": passed,
        "qc_rework_qty": rework,
        "qc_reject_qty": rejected,
        "rework_pending_qty": rework_pending,
        "rework_pass_qty": rework_pass,
        "rework_reject_qty": rework_reject,
        "debit_qty": debit_qty,
        "debit_amount": round(debit_amount, 2),
        "eligible_billing_qty": eligible,
        "billing_status": bill_status,
        "rework_orders": [
            {
                "id": r["id"],
                "rework_no": r["rework_no"],
                "planned_qty": r["planned_qty"],
                "balance_qty": r["balance_qty"],
                "pass_qty": r["pass_qty"],
                "reject_qty": r["reject_qty"],
                "status": r["status"],
                "chargeable": r["chargeable"],
                "rework_by_type": r["rework_by_type"],
                "rework_by_vendor": r["rework_by_vendor"],
            }
            for r in reworks
        ],
        "debit_notes": [
            {
                "id": d["id"],
                "debit_no": d["debit_no"],
                "qty": d["qty"],
                "total_amount": d["total_amount"],
                "with_fabric": d["with_fabric"],
                "status": d["status"],
            }
            for d in debits
        ],
    }
