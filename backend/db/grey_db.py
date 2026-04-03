"""Grey Fabric Module DB — lifecycle: PO → transit → transport → factory/printer → QC → job work."""
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

_DB = os.environ.get("GREY_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "grey.db"))


def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _table_cols(conn, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _add_col(conn, table: str, name: str, decl: str) -> None:
    if name not in _table_cols(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


# Canonical status flow (Item Type Grey Fabric — not a normal PO line)
GREY_STATUSES = [
    "PO Created",
    "Vendor Dispatch",
    "In Transit",
    "At Transport Location",
    "Sent to Factory",
    "At Factory",
    "Sent to Printer",
    "At Printer",
    "Printed Fabric Received",
    "QC Pending",
    "QC Done",
    "Rejected",
    "Return to Vendor",
    "Rework",
    "Closed",
]


def init_db():
    conn = _connect()
    conn.executescript(
        """
    CREATE TABLE IF NOT EXISTS grey_tracker (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_key      TEXT UNIQUE NOT NULL,
        po_number        TEXT,
        material_code    TEXT,
        material_name    TEXT,
        supplier         TEXT,
        so_reference     TEXT,
        ordered_qty      REAL DEFAULT 0,
        dispatched_qty   REAL DEFAULT 0,
        received_qty     REAL DEFAULT 0,
        transport_qty    REAL DEFAULT 0,
        factory_qty      REAL DEFAULT 0,
        printer_qty      REAL DEFAULT 0,
        checked_qty      REAL DEFAULT 0,
        rejected_qty     REAL DEFAULT 0,
        rework_qty       REAL DEFAULT 0,
        dispatch_date    TEXT,
        bilty_no         TEXT,
        vendor_invoice   TEXT,
        vendor_challan   TEXT,
        vehicle_no       TEXT,
        transporter      TEXT,
        expected_arrival TEXT,
        status           TEXT DEFAULT 'PO Created',
        qc_status        TEXT DEFAULT 'Pending',
        qc_checked_by    TEXT,
        qc_date          TEXT,
        qc_remarks       TEXT,
        created_at       TEXT DEFAULT (datetime('now')),
        updated_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS grey_ledger (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_date       TEXT NOT NULL,
        tracker_id       INTEGER REFERENCES grey_tracker(id),
        material_code    TEXT,
        material_name    TEXT,
        transaction_type TEXT,
        qty              REAL DEFAULT 0,
        unit             TEXT DEFAULT 'MTR',
        from_location    TEXT,
        to_location      TEXT,
        reference_no     TEXT,
        remarks          TEXT,
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS hard_reservations (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        fabric_code      TEXT NOT NULL,
        fabric_name      TEXT,
        so_number        TEXT,
        sku              TEXT,
        qty              REAL DEFAULT 0,
        unit             TEXT DEFAULT 'MTR',
        reserved_date    TEXT DEFAULT (datetime('now')),
        status           TEXT DEFAULT 'Active',
        remarks          TEXT
    );
    CREATE TABLE IF NOT EXISTS grey_mrp_requirement (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        run_label        TEXT DEFAULT '',
        material_code    TEXT NOT NULL,
        material_name    TEXT DEFAULT '',
        so_number        TEXT DEFAULT '',
        sku              TEXT DEFAULT '',
        qty_required     REAL NOT NULL DEFAULT 0,
        notes            TEXT DEFAULT '',
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS grey_printer_issue (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_id       INTEGER REFERENCES grey_tracker(id),
        job_order_no     TEXT NOT NULL DEFAULT '',
        material_code    TEXT DEFAULT '',
        issue_qty        REAL DEFAULT 0,
        from_location    TEXT DEFAULT '',
        to_vendor        TEXT DEFAULT '',
        issue_date       TEXT DEFAULT '',
        challan_no       TEXT DEFAULT '',
        gate_pass        TEXT DEFAULT '',
        received_back_qty REAL DEFAULT 0,
        remarks          TEXT DEFAULT '',
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS grey_conversion (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_id       INTEGER REFERENCES grey_tracker(id),
        printer_issue_id INTEGER REFERENCES grey_printer_issue(id),
        grey_input_mtr   REAL DEFAULT 0,
        printed_item_code TEXT DEFAULT '',
        printed_output_mtr REAL DEFAULT 0,
        wastage_mtr      REAL DEFAULT 0,
        conversion_date  TEXT DEFAULT '',
        remarks          TEXT DEFAULT '',
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS grey_qc_event (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_id       INTEGER NOT NULL REFERENCES grey_tracker(id),
        received_qty     REAL DEFAULT 0,
        checked_qty      REAL DEFAULT 0,
        passed_qty       REAL DEFAULT 0,
        rejected_qty     REAL DEFAULT 0,
        rework_qty       REAL DEFAULT 0,
        outcome          TEXT DEFAULT '',
        qc_remarks       TEXT DEFAULT '',
        qc_by            TEXT DEFAULT '',
        qc_date          TEXT DEFAULT '',
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS grey_return_vendor (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_id       INTEGER NOT NULL REFERENCES grey_tracker(id),
        return_qty       REAL DEFAULT 0,
        debit_note_no    TEXT DEFAULT '',
        return_challan   TEXT DEFAULT '',
        return_date      TEXT DEFAULT '',
        remarks          TEXT DEFAULT '',
        created_at       TEXT DEFAULT (datetime('now'))
    );
    """
    )

    # Migrate grey_tracker columns
    for name, decl in [
        ("delivery_location", "TEXT DEFAULT ''"),
        ("rate", "REAL DEFAULT 0"),
        ("in_transit_qty", "REAL DEFAULT 0"),
        ("passed_qty", "REAL DEFAULT 0"),
        ("return_to_vendor_qty", "REAL DEFAULT 0"),
        ("rework_issue_qty", "REAL DEFAULT 0"),
        ("rework_receive_qty", "REAL DEFAULT 0"),
        ("debit_note_no", "TEXT DEFAULT ''"),
        ("return_challan_no", "TEXT DEFAULT ''"),
        ("gate_pass_no", "TEXT DEFAULT ''"),
        ("job_work_order_no", "TEXT DEFAULT ''"),
        ("printed_fabric_qty", "REAL DEFAULT 0"),
    ]:
        _add_col(conn, "grey_tracker", name, decl)

    _add_col(conn, "grey_ledger", "tracker_id", "INTEGER")

    conn.commit()
    conn.close()


# Legacy alias
STATUSES = GREY_STATUSES


def _next_key(conn, po_number, material_code):
    existing = conn.execute(
        "SELECT id FROM grey_tracker WHERE po_number=? AND material_code=?",
        (po_number, material_code),
    ).fetchone()
    if existing:
        return None
    row = conn.execute("SELECT COUNT(*) FROM grey_tracker").fetchone()
    return f"GT-{int(row[0]) + 1:04d}"


def _add_ledger_entry(
    conn,
    tracker_id,
    material_code,
    material_name,
    txn_type,
    qty,
    unit,
    from_loc,
    to_loc,
    ref,
    remarks,
):
    conn.execute(
        """INSERT INTO grey_ledger(entry_date,tracker_id,material_code,material_name,transaction_type,qty,unit,from_location,to_location,reference_no,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().strftime("%Y-%m-%d"),
            tracker_id,
            material_code,
            material_name,
            txn_type,
            qty,
            unit,
            from_loc,
            to_loc,
            ref,
            remarks,
        ),
    )


def list_grey(status=None):
    conn = _connect()
    q = (
        "SELECT * FROM grey_tracker WHERE status=? ORDER BY id DESC"
        if status
        else "SELECT * FROM grey_tracker ORDER BY id DESC"
    )
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_grey(gid: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (gid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_grey_entry(data: dict):
    conn = _connect()
    key = _next_key(conn, data.get("po_number", ""), data.get("material_code", ""))
    if not key:
        conn.close()
        return None
    conn.execute(
        """INSERT INTO grey_tracker(tracker_key,po_number,material_code,material_name,supplier,so_reference,
        ordered_qty,rate,delivery_location,status) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            key,
            data.get("po_number", ""),
            data.get("material_code", ""),
            data.get("material_name", ""),
            data.get("supplier", ""),
            data.get("so_reference", ""),
            float(data.get("ordered_qty", 0) or 0),
            float(data.get("rate", 0) or 0),
            data.get("delivery_location", ""),
            "PO Created",
        ),
    )
    tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    _add_ledger_entry(
        conn,
        tid,
        data.get("material_code", ""),
        data.get("material_name", ""),
        "PO Created",
        float(data.get("ordered_qty", 0) or 0),
        "MTR",
        "",
        "Ordered",
        data.get("po_number", ""),
        "PO created",
    )
    conn.commit()
    conn.close()
    return key


_TRACKER_PATCHABLE = {
    "status",
    "dispatched_qty",
    "received_qty",
    "transport_qty",
    "factory_qty",
    "printer_qty",
    "checked_qty",
    "rejected_qty",
    "rework_qty",
    "passed_qty",
    "dispatch_date",
    "bilty_no",
    "vendor_invoice",
    "vendor_challan",
    "vehicle_no",
    "transporter",
    "expected_arrival",
    "qc_status",
    "qc_checked_by",
    "qc_date",
    "qc_remarks",
    "delivery_location",
    "rate",
    "in_transit_qty",
    "return_to_vendor_qty",
    "rework_issue_qty",
    "rework_receive_qty",
    "debit_note_no",
    "return_challan_no",
    "gate_pass_no",
    "job_work_order_no",
    "printed_fabric_qty",
}


def update_grey_status(gid: int, data: dict):
    allowed = {k: v for k, v in data.items() if k in _TRACKER_PATCHABLE and v is not None}
    if not allowed:
        return
    sets = ", ".join(f"{k}=?" for k in allowed)
    vals = list(allowed.values()) + [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gid]
    conn = _connect()
    conn.execute(f"UPDATE grey_tracker SET {sets}, updated_at=? WHERE id=?", vals)
    if "status" in allowed:
        row = conn.execute(
            "SELECT material_code,material_name FROM grey_tracker WHERE id=?", (gid,)
        ).fetchone()
        if row:
            qty = float(
                data.get("dispatched_qty")
                or data.get("in_transit_qty")
                or data.get("transport_qty")
                or data.get("factory_qty")
                or 0
            )
            _add_ledger_entry(
                conn,
                gid,
                row["material_code"],
                row["material_name"],
                str(allowed["status"]),
                qty,
                "MTR",
                "",
                allowed["status"],
                "",
                f"Status → {allowed['status']}",
            )
    conn.commit()
    conn.close()


def vendor_dispatch(
    gid: int,
    bilty_no: str,
    transporter: str,
    dispatch_date: str,
    expected_arrival: str,
    dispatched_qty: float,
    vehicle_no: str = "",
):
    """Bilty captured → In Transit + qty on road."""
    conn = _connect()
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (gid,)).fetchone()
    if not row:
        conn.close()
        return False
    q = float(dispatched_qty or 0)
    conn.execute(
        """UPDATE grey_tracker SET bilty_no=?, transporter=?, dispatch_date=?, expected_arrival=?,
        vehicle_no=?, dispatched_qty=?, in_transit_qty=?, status=?, updated_at=?
        WHERE id=?""",
        (
            bilty_no,
            transporter,
            dispatch_date,
            expected_arrival,
            vehicle_no,
            q,
            q,
            "In Transit",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gid,
        ),
    )
    _add_ledger_entry(
        conn,
        gid,
        row["material_code"],
        row["material_name"],
        "Vendor Dispatch / In Transit",
        q,
        "MTR",
        "Supplier",
        "In Transit",
        bilty_no,
        f"Transporter: {transporter}",
    )
    conn.commit()
    conn.close()
    return True


def arrive_at_transport(gid: int, qty: Optional[float] = None):
    """In Transit → At Transport Location; move qty from in_transit to transport."""
    conn = _connect()
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (gid,)).fetchone()
    if not row:
        conn.close()
        return False
    move = float(qty if qty is not None else row["in_transit_qty"] or row["dispatched_qty"] or 0)
    new_trans = float(row["transport_qty"] or 0) + move
    new_it = max(float(row["in_transit_qty"] or 0) - move, 0)
    conn.execute(
        """UPDATE grey_tracker SET transport_qty=?, in_transit_qty=?, status=?, updated_at=? WHERE id=?""",
        (
            new_trans,
            new_it,
            "At Transport Location",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gid,
        ),
    )
    _add_ledger_entry(
        conn,
        gid,
        row["material_code"],
        row["material_name"],
        "At Transport Location",
        move,
        "MTR",
        "In Transit",
        "Transport Location",
        "",
        "Arrived at transport",
    )
    conn.commit()
    conn.close()
    return True


def transfer_qty(gid: int, to_location: str, qty: float):
    """
    to_location: 'factory' | 'printer'
    Deducts from transport_qty, adds to factory_qty or printer_qty; updates status.
    """
    to_location = to_location.strip().lower()
    if to_location not in ("factory", "printer"):
        return False
    conn = _connect()
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (gid,)).fetchone()
    if not row:
        conn.close()
        return False
    q = float(qty or 0)
    tq = float(row["transport_qty"] or 0)
    if q > tq:
        conn.close()
        return False
    fq = float(row["factory_qty"] or 0)
    pq = float(row["printer_qty"] or 0)
    new_t = tq - q
    if to_location == "factory":
        fq += q
        status = "Sent to Factory"
        dest = "Factory / Inhouse"
        ledger_to = "Factory"
    else:
        pq += q
        status = "Sent to Printer"
        dest = "Job Work / Printer"
        ledger_to = "Printer"
    conn.execute(
        """UPDATE grey_tracker SET transport_qty=?, factory_qty=?, printer_qty=?, status=?, updated_at=? WHERE id=?""",
        (
            new_t,
            fq,
            pq,
            status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gid,
        ),
    )
    _add_ledger_entry(
        conn,
        gid,
        row["material_code"],
        row["material_name"],
        status,
        q,
        "MTR",
        "Transport Location",
        ledger_to,
        "",
        f"Transfer to {dest}",
    )
    conn.commit()
    conn.close()
    return True


def record_qc(
    gid: int,
    received_qty: float,
    checked_qty: float,
    passed_qty: float,
    rejected_qty: float,
    rework_qty: float,
    outcome: str,
    qc_remarks: str,
    qc_by: str,
    qc_date: str,
):
    """Partial QC supported; logs grey_qc_event and updates tracker aggregates."""
    conn = _connect()
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (gid,)).fetchone()
    if not row:
        conn.close()
        return False
    conn.execute(
        """INSERT INTO grey_qc_event(tracker_id,received_qty,checked_qty,passed_qty,rejected_qty,rework_qty,outcome,qc_remarks,qc_by,qc_date)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            gid,
            received_qty,
            checked_qty,
            passed_qty,
            rejected_qty,
            rework_qty,
            outcome,
            qc_remarks,
            qc_by,
            qc_date,
        ),
    )
    st = row["status"]
    if outcome in ("Pass", "QC Done", "Partial Pass"):
        st = "QC Done"
    elif outcome == "Reject":
        st = "Rejected"
    elif outcome == "Rework":
        st = "Rework"
    conn.execute(
        """UPDATE grey_tracker SET received_qty=?, checked_qty=?, passed_qty=?, rejected_qty=?, rework_qty=?,
        qc_status=?, qc_remarks=?, qc_checked_by=?, qc_date=?, status=?, updated_at=? WHERE id=?""",
        (
            received_qty,
            checked_qty,
            passed_qty,
            rejected_qty,
            rework_qty,
            outcome,
            qc_remarks,
            qc_by,
            qc_date,
            st,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gid,
        ),
    )
    _add_ledger_entry(
        conn,
        gid,
        row["material_code"],
        row["material_name"],
        f"QC: {outcome}",
        checked_qty,
        "MTR",
        "",
        "",
        "",
        qc_remarks or "",
    )
    conn.commit()
    conn.close()
    return True


def return_to_vendor(
    gid: int, return_qty: float, debit_note_no: str, return_challan: str, return_date: str, remarks: str = ""
):
    conn = _connect()
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (gid,)).fetchone()
    if not row:
        conn.close()
        return False
    rq = float(return_qty or 0)
    conn.execute(
        """INSERT INTO grey_return_vendor(tracker_id,return_qty,debit_note_no,return_challan,return_date,remarks)
        VALUES(?,?,?,?,?,?)""",
        (gid, rq, debit_note_no, return_challan, return_date, remarks),
    )
    prev = float(row["return_to_vendor_qty"] or 0)
    conn.execute(
        """UPDATE grey_tracker SET return_to_vendor_qty=?, debit_note_no=?, return_challan_no=?, status=?, updated_at=? WHERE id=?""",
        (
            prev + rq,
            debit_note_no,
            return_challan,
            "Return to Vendor",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gid,
        ),
    )
    _add_ledger_entry(
        conn,
        gid,
        row["material_code"],
        row["material_name"],
        "Return to Vendor",
        rq,
        "MTR",
        "Internal",
        "Vendor",
        debit_note_no,
        remarks or "Debit note / return challan",
    )
    conn.commit()
    conn.close()
    return True


def create_printer_issue(data: dict) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO grey_printer_issue(tracker_id,job_order_no,material_code,issue_qty,from_location,to_vendor,issue_date,challan_no,gate_pass,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            int(data["tracker_id"]),
            data.get("job_order_no", ""),
            data.get("material_code", ""),
            float(data.get("issue_qty", 0) or 0),
            data.get("from_location", "Transport Location"),
            data.get("to_vendor", ""),
            data.get("issue_date", ""),
            data.get("challan_no", ""),
            data.get("gate_pass", ""),
            data.get("remarks", ""),
        ),
    )
    iid = cur.lastrowid
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (int(data["tracker_id"]),)).fetchone()
    if row:
        conn.execute(
            """UPDATE grey_tracker SET job_work_order_no=?, gate_pass_no=?, status=?, updated_at=? WHERE id=?""",
            (
                data.get("job_order_no", ""),
                data.get("gate_pass", ""),
                "At Printer",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                int(data["tracker_id"]),
            ),
        )
        _add_ledger_entry(
            conn,
            int(data["tracker_id"]),
            row["material_code"],
            row["material_name"],
            "Grey Issue to Printer",
            float(data.get("issue_qty", 0) or 0),
            "MTR",
            data.get("from_location", ""),
            data.get("to_vendor", ""),
            data.get("challan_no", ""),
            data.get("remarks", ""),
        )
    conn.commit()
    conn.close()
    return iid


def list_printer_issues(tracker_id: Optional[int] = None):
    conn = _connect()
    if tracker_id:
        rows = conn.execute(
            "SELECT * FROM grey_printer_issue WHERE tracker_id=? ORDER BY id DESC",
            (tracker_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM grey_printer_issue ORDER BY id DESC LIMIT 500").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["balance_qty"] = float(d.get("issue_qty") or 0) - float(d.get("received_back_qty") or 0)
        out.append(d)
    return out


def receive_printed_fabric(issue_id: int, received_back_qty: float, conversion: Optional[dict] = None):
    """Printer returns printed metres; optional grey→printed conversion row."""
    conn = _connect()
    iss = conn.execute("SELECT * FROM grey_printer_issue WHERE id=?", (issue_id,)).fetchone()
    if not iss:
        conn.close()
        return False
    new_rb = float(iss["received_back_qty"] or 0) + float(received_back_qty or 0)
    conn.execute(
        "UPDATE grey_printer_issue SET received_back_qty=? WHERE id=?",
        (new_rb, issue_id),
    )
    tid = iss["tracker_id"]
    if conversion:
        conn.execute(
            """INSERT INTO grey_conversion(tracker_id,printer_issue_id,grey_input_mtr,printed_item_code,printed_output_mtr,wastage_mtr,conversion_date,remarks)
            VALUES(?,?,?,?,?,?,?,?)""",
            (
                tid,
                issue_id,
                float(conversion.get("grey_input_mtr", 0) or 0),
                conversion.get("printed_item_code", ""),
                float(conversion.get("printed_output_mtr", 0) or 0),
                float(conversion.get("wastage_mtr", 0) or 0),
                conversion.get("conversion_date", ""),
                conversion.get("remarks", ""),
            ),
        )
        pf = conn.execute("SELECT printed_fabric_qty FROM grey_tracker WHERE id=?", (tid,)).fetchone()
        prev_pf = float(pf["printed_fabric_qty"] or 0) if pf else 0
        conn.execute(
            "UPDATE grey_tracker SET printed_fabric_qty=?, status=?, updated_at=? WHERE id=?",
            (
                prev_pf + float(conversion.get("printed_output_mtr", 0) or 0),
                "Printed Fabric Received",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tid,
            ),
        )
    row = conn.execute("SELECT * FROM grey_tracker WHERE id=?", (tid,)).fetchone()
    if row:
        _add_ledger_entry(
            conn,
            tid,
            row["material_code"],
            row["material_name"],
            "Printed Fabric Received",
            float(received_back_qty or 0),
            "MTR",
            iss["to_vendor"],
            "Stock",
            dict(iss).get("challan_no") or "",
            "",
        )
    conn.commit()
    conn.close()
    return True


def list_conversions(tracker_id: Optional[int] = None):
    conn = _connect()
    if tracker_id:
        rows = conn.execute(
            "SELECT * FROM grey_conversion WHERE tracker_id=? ORDER BY id DESC",
            (tracker_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM grey_conversion ORDER BY id DESC LIMIT 300").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_qc_events(tracker_id: Optional[int] = None):
    conn = _connect()
    if tracker_id:
        rows = conn.execute(
            "SELECT * FROM grey_qc_event WHERE tracker_id=? ORDER BY id DESC",
            (tracker_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM grey_qc_event ORDER BY id DESC LIMIT 300").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_returns():
    conn = _connect()
    rows = conn.execute("SELECT * FROM grey_return_vendor ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── MRP requirements (manual / import; link SO + SKU) ─────────────────────────
def list_mrp_requirements(material_code: Optional[str] = None):
    conn = _connect()
    if material_code:
        rows = conn.execute(
            "SELECT * FROM grey_mrp_requirement WHERE material_code=? ORDER BY id DESC",
            (material_code,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM grey_mrp_requirement ORDER BY id DESC LIMIT 1000").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_mrp_requirement(data: dict) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO grey_mrp_requirement(run_label,material_code,material_name,so_number,sku,qty_required,notes)
        VALUES(?,?,?,?,?,?,?)""",
        (
            data.get("run_label", ""),
            data["material_code"],
            data.get("material_name", ""),
            data.get("so_number", ""),
            data.get("sku", ""),
            float(data.get("qty_required", 0) or 0),
            data.get("notes", ""),
        ),
    )
    iid = cur.lastrowid
    conn.commit()
    conn.close()
    return iid


def mrp_drilldown_by_material(material_code: str):
    """SO/SKU-wise lines for one grey fabric code."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM grey_mrp_requirement WHERE material_code=? ORDER BY so_number, sku",
        (material_code,),
    ).fetchall()
    conn.close()
    lines = [dict(r) for r in rows]
    total = sum(float(x["qty_required"] or 0) for x in lines)
    return {"material_code": material_code, "total_qty_required": total, "lines": lines}


def mrp_totals_by_material():
    conn = _connect()
    rows = conn.execute(
        """SELECT material_code,
        MAX(material_name) AS material_name,
        SUM(qty_required) AS total_required
        FROM grey_mrp_requirement GROUP BY material_code ORDER BY total_required DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Reservations ───────────────────────────────────────────────────────────────
def list_hard_reservations(status="Active"):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM hard_reservations WHERE status=? ORDER BY id DESC", (status,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def reserved_qty_for_fabric(fabric_code: str) -> float:
    conn = _connect()
    row = conn.execute(
        "SELECT COALESCE(SUM(qty),0) FROM hard_reservations WHERE fabric_code=? AND status='Active'",
        (fabric_code,),
    ).fetchone()
    conn.close()
    return float(row[0] or 0)


def available_for_other_so(fabric_code: str) -> dict:
    """Rough availability: physical buckets minus active reservations (MRP planning aid)."""
    loc = get_location_summary()
    by_mat = loc.get("by_material", {}).get(fabric_code, {})
    total_phys = sum(
        float(by_mat.get(k, 0) or 0)
        for k in ("in_transit_qty", "transport_qty", "factory_qty", "printer_qty")
    )
    res = reserved_qty_for_fabric(fabric_code)
    return {
        "fabric_code": fabric_code,
        "physical_pipeline_mtr": round(total_phys, 3),
        "hard_reserved_mtr": round(res, 3),
        "not_available_for_other_so_mtr": round(res, 3),
        "unreserved_pipeline_mtr": round(max(total_phys - res, 0), 3),
    }


def create_hard_reservation(data: dict):
    conn = _connect()
    conn.execute(
        """INSERT INTO hard_reservations(fabric_code,fabric_name,so_number,sku,qty,unit,status,remarks)
        VALUES(?,?,?,?,?,?,?,?)""",
        (
            data["fabric_code"],
            data.get("fabric_name", ""),
            data.get("so_number", ""),
            data.get("sku", ""),
            data.get("qty", 0),
            data.get("unit", "MTR"),
            "Active",
            data.get("remarks", ""),
        ),
    )
    conn.commit()
    conn.close()


def release_hard_reservation(rid: int):
    conn = _connect()
    conn.execute("UPDATE hard_reservations SET status='Released' WHERE id=?", (rid,))
    conn.commit()
    conn.close()


def list_ledger(material_code=None):
    conn = _connect()
    if material_code:
        rows = conn.execute(
            "SELECT * FROM grey_ledger WHERE material_code=? ORDER BY id DESC",
            (material_code,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM grey_ledger ORDER BY id DESC LIMIT 500").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_location_summary():
    conn = _connect()
    rows = conn.execute("SELECT * FROM grey_tracker").fetchall()
    conn.close()
    agg = {
        "in_transit_mtr": 0.0,
        "transport_mtr": 0.0,
        "factory_mtr": 0.0,
        "printer_mtr": 0.0,
        "rejected_recorded_mtr": 0.0,
        "return_vendor_mtr": 0.0,
        "rework_mtr": 0.0,
        "printed_fabric_mtr": 0.0,
    }
    by_material: Dict[str, Dict[str, float]] = {}
    for r in rows:
        d = dict(r)
        code = d.get("material_code") or ""
        agg["in_transit_mtr"] += float(d.get("in_transit_qty") or 0)
        agg["transport_mtr"] += float(d.get("transport_qty") or 0)
        agg["factory_mtr"] += float(d.get("factory_qty") or 0)
        agg["printer_mtr"] += float(d.get("printer_qty") or 0)
        agg["rejected_recorded_mtr"] += float(d.get("rejected_qty") or 0)
        agg["return_vendor_mtr"] += float(d.get("return_to_vendor_qty") or 0)
        agg["rework_mtr"] += float(d.get("rework_qty") or 0)
        agg["printed_fabric_mtr"] += float(d.get("printed_fabric_qty") or 0)
        if code:
            if code not in by_material:
                by_material[code] = {
                    "in_transit_qty": 0.0,
                    "transport_qty": 0.0,
                    "factory_qty": 0.0,
                    "printer_qty": 0.0,
                    "rejected_qty": 0.0,
                    "return_to_vendor_qty": 0.0,
                }
            bm = by_material[code]
            bm["in_transit_qty"] += float(d.get("in_transit_qty") or 0)
            bm["transport_qty"] += float(d.get("transport_qty") or 0)
            bm["factory_qty"] += float(d.get("factory_qty") or 0)
            bm["printer_qty"] += float(d.get("printer_qty") or 0)
            bm["rejected_qty"] += float(d.get("rejected_qty") or 0)
            bm["return_to_vendor_qty"] += float(d.get("return_to_vendor_qty") or 0)
    return {"totals": agg, "by_material": by_material}


def get_grey_stats():
    conn = _connect()
    loc = get_location_summary()
    stats = {
        "total_trackers": conn.execute("SELECT COUNT(*) FROM grey_tracker").fetchone()[0],
        "in_transit": conn.execute(
            "SELECT COUNT(*) FROM grey_tracker WHERE status='In Transit'"
        ).fetchone()[0],
        "at_transport": conn.execute(
            "SELECT COUNT(*) FROM grey_tracker WHERE status='At Transport Location'"
        ).fetchone()[0],
        "at_factory": conn.execute(
            "SELECT COUNT(*) FROM grey_tracker WHERE status IN ('At Factory','Sent to Factory')"
        ).fetchone()[0],
        "at_printer": conn.execute(
            "SELECT COUNT(*) FROM grey_tracker WHERE status IN ('At Printer','Sent to Printer','Printed Fabric Received')"
        ).fetchone()[0],
        "pending_qc": conn.execute(
            "SELECT COUNT(*) FROM grey_tracker WHERE qc_status='Pending' OR status='QC Pending'"
        ).fetchone()[0],
        "hard_reserved": conn.execute(
            "SELECT COUNT(*) FROM hard_reservations WHERE status='Active'"
        ).fetchone()[0],
        "transit_meters": loc["totals"]["in_transit_mtr"],
        "location_totals": loc["totals"],
    }
    conn.close()
    return stats


def report_transit():
    return list_grey("In Transit")


def report_stock_by_location():
    return get_location_summary()


def report_qc():
    conn = _connect()
    rows = conn.execute("SELECT * FROM grey_qc_event ORDER BY id DESC LIMIT 500").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def report_rejects_returns():
    conn = _connect()
    r1 = conn.execute(
        "SELECT * FROM grey_tracker WHERE status IN ('Rejected','Return to Vendor') ORDER BY id DESC"
    ).fetchall()
    r2 = conn.execute("SELECT * FROM grey_return_vendor ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
    return {"trackers": [dict(r) for r in r1], "return_documents": [dict(r) for r in r2]}


def report_printer_issues():
    return list_printer_issues(None)


def report_grey_consumption():
    conn = _connect()
    rows = conn.execute(
        """SELECT material_code, SUM(qty) AS total_mtr FROM grey_ledger
        WHERE transaction_type LIKE '%Printer%' OR transaction_type LIKE '%Factory%'
        GROUP BY material_code ORDER BY total_mtr DESC LIMIT 200"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mrp_stock_snapshot():
    """For MRP: expose buckets + reservations per material (no live OMS link in this module)."""
    loc = get_location_summary()
    conn = _connect()
    res = conn.execute(
        """SELECT fabric_code, SUM(qty) AS reserved FROM hard_reservations WHERE status='Active' GROUP BY fabric_code"""
    ).fetchall()
    conn.close()
    res_map = {r["fabric_code"]: float(r["reserved"] or 0) for r in res}
    materials = set(loc["by_material"].keys()) | set(res_map.keys())
    out = []
    for m in sorted(materials):
        bm = loc["by_material"].get(m, {})
        out.append(
            {
                "material_code": m,
                "in_transit": bm.get("in_transit_qty", 0),
                "transport": bm.get("transport_qty", 0),
                "factory": bm.get("factory_qty", 0),
                "printer_job_work": bm.get("printer_qty", 0),
                "hard_reserved": res_map.get(m, 0),
            }
        )
    return out
