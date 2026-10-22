"""Vendor portal accounts — map an ERP login to one existing supplier or processor.

Does not duplicate vendor masters, purchase orders, job-work orders, or stock.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

_DB = os.environ.get(
    "PURCHASE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "purchase.db"),
)
_PROD_DB = os.environ.get(
    "PRODUCTION_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "production.db"),
)

PARTY_KINDS = frozenset({"supplier", "processor"})
PORTAL_ROLES = frozenset({"vendor_admin", "vendor_ops", "vendor_accounts"})
_COMMERCIAL_KEYS = frozenset({
    "rate", "amount", "gst_pct", "gst_amount", "subtotal", "total",
    "vendor_rate", "process_cost", "total_cost",
})


def _connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or _DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS vendor_portal_links (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER NOT NULL UNIQUE,
        party_kind   TEXT NOT NULL,
        party_id     INTEGER NOT NULL,
        portal_role  TEXT NOT NULL DEFAULT 'vendor_admin',
        active       INTEGER NOT NULL DEFAULT 1,
        created_at   TEXT DEFAULT (datetime('now')),
        updated_at   TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_vendor_portal_party
        ON vendor_portal_links(party_kind, party_id);
    CREATE TABLE IF NOT EXISTS vendor_audit_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER,
        username     TEXT,
        party_kind   TEXT,
        party_id     INTEGER,
        action       TEXT NOT NULL,
        record_type  TEXT,
        record_id    TEXT,
        details      TEXT,
        created_at   TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_vendor_audit_party
        ON vendor_audit_events(party_kind, party_id, id);
    CREATE TABLE IF NOT EXISTS vendor_doc_responses (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        party_kind  TEXT NOT NULL,
        party_id    INTEGER NOT NULL,
        doc_type    TEXT NOT NULL,
        doc_id      INTEGER NOT NULL,
        action      TEXT NOT NULL,
        reason      TEXT DEFAULT '',
        user_id     INTEGER,
        username    TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_vendor_doc_resp
        ON vendor_doc_responses(party_kind, party_id, doc_type, doc_id, id);
    CREATE TABLE IF NOT EXISTS vendor_material_acks (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        party_kind   TEXT NOT NULL,
        party_id     INTEGER NOT NULL,
        source_type  TEXT NOT NULL,
        source_id    INTEGER NOT NULL,
        line_id      INTEGER NOT NULL,
        issued_qty   REAL DEFAULT 0,
        received_qty REAL DEFAULT 0,
        variance_qty REAL DEFAULT 0,
        reason       TEXT DEFAULT '',
        user_id      INTEGER,
        username     TEXT,
        created_at   TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS vendor_delivery_intimations (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        intimation_no TEXT UNIQUE NOT NULL,
        party_kind    TEXT NOT NULL,
        party_id      INTEGER NOT NULL,
        doc_type      TEXT NOT NULL,
        doc_id        INTEGER NOT NULL,
        doc_number    TEXT DEFAULT '',
        qty           REAL DEFAULT 0,
        challan_no    TEXT DEFAULT '',
        delivery_date TEXT DEFAULT '',
        remarks       TEXT DEFAULT '',
        status        TEXT NOT NULL DEFAULT 'Submitted',
        user_id       INTEGER,
        username      TEXT,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def party_exists(party_kind: str, party_id: int) -> dict | None:
    if party_kind not in PARTY_KINDS:
        return None
    table = "suppliers" if party_kind == "supplier" else "processors"
    name_col = "supplier_name" if party_kind == "supplier" else "processor_name"
    code_col = "supplier_code" if party_kind == "supplier" else "processor_code"
    conn = _connect()
    row = conn.execute(
        f"SELECT id, {code_col} AS code, {name_col} AS name, active FROM {table} WHERE id=?",
        (int(party_id),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_link_for_user(user_id: int, *, active_only: bool = True) -> dict | None:
    conn = _connect()
    q = "SELECT * FROM vendor_portal_links WHERE user_id=?"
    if active_only:
        q += " AND active=1"
    row = conn.execute(q, (int(user_id),)).fetchone()
    conn.close()
    if not row:
        return None
    link = dict(row)
    party = party_exists(link["party_kind"], link["party_id"])
    if party:
        link["party_name"] = party["name"]
        link["party_code"] = party["code"]
    return link


def list_links() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM vendor_portal_links ORDER BY id DESC"
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        link = dict(row)
        party = party_exists(link["party_kind"], link["party_id"])
        if party:
            link["party_name"] = party["name"]
            link["party_code"] = party["code"]
        out.append(link)
    return out


def upsert_link(*, user_id: int, party_kind: str, party_id: int, portal_role: str) -> dict:
    if party_kind not in PARTY_KINDS:
        raise ValueError("party_kind must be supplier or processor")
    if portal_role not in PORTAL_ROLES:
        raise ValueError("portal_role must be vendor_admin, vendor_ops, or vendor_accounts")
    if not party_exists(party_kind, party_id):
        raise ValueError("Supplier or processor record was not found")
    conn = _connect()
    existing = conn.execute(
        "SELECT id FROM vendor_portal_links WHERE user_id=?", (int(user_id),)
    ).fetchone()
    now = _now()
    if existing:
        conn.execute(
            """UPDATE vendor_portal_links
               SET party_kind=?, party_id=?, portal_role=?, active=1, updated_at=?
               WHERE user_id=?""",
            (party_kind, int(party_id), portal_role, now, int(user_id)),
        )
    else:
        conn.execute(
            """INSERT INTO vendor_portal_links(user_id, party_kind, party_id, portal_role, active, created_at, updated_at)
               VALUES(?,?,?,?,1,?,?)""",
            (int(user_id), party_kind, int(party_id), portal_role, now, now),
        )
    conn.commit()
    conn.close()
    link = get_link_for_user(user_id, active_only=False)
    if not link:
        raise ValueError("Could not save vendor portal link")
    return link


def set_link_active(link_id: int, active: bool) -> dict | None:
    conn = _connect()
    conn.execute(
        "UPDATE vendor_portal_links SET active=?, updated_at=? WHERE id=?",
        (1 if active else 0, _now(), int(link_id)),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM vendor_portal_links WHERE id=?", (int(link_id),)).fetchone()
    conn.close()
    return dict(row) if row else None


def log_event(
    *,
    user_id: int | None,
    username: str,
    party_kind: str,
    party_id: int,
    action: str,
    record_type: str = "",
    record_id: str = "",
    details: dict | None = None,
) -> None:
    conn = _connect()
    conn.execute(
        """INSERT INTO vendor_audit_events(
            user_id, username, party_kind, party_id, action, record_type, record_id, details
        ) VALUES(?,?,?,?,?,?,?,?)""",
        (
            user_id,
            username or "",
            party_kind,
            int(party_id),
            action,
            record_type,
            str(record_id or ""),
            json.dumps(details or {}, default=str),
        ),
    )
    conn.commit()
    conn.close()


def _strip(row: dict, *, show_commercial: bool) -> dict:
    if show_commercial:
        return row
    return {k: v for k, v in row.items() if k not in _COMMERCIAL_KEYS}


def show_commercial(portal_role: str) -> bool:
    return portal_role in ("vendor_admin", "vendor_accounts")


def _latest_response(conn: sqlite3.Connection, party_kind: str, party_id: int, doc_type: str, doc_id: int) -> dict | None:
    row = conn.execute(
        """SELECT action, reason, username, created_at FROM vendor_doc_responses
           WHERE party_kind=? AND party_id=? AND doc_type=? AND doc_id=?
           ORDER BY id DESC LIMIT 1""",
        (party_kind, int(party_id), doc_type, int(doc_id)),
    ).fetchone()
    return dict(row) if row else None


def list_purchase_orders(supplier_id: int, *, show_rates: bool) -> list[dict]:
    conn = _connect()
    headers = conn.execute(
        "SELECT * FROM po_headers WHERE supplier_id=? ORDER BY id DESC",
        (int(supplier_id),),
    ).fetchall()
    out = []
    for h in headers:
        d = _strip(dict(h), show_commercial=show_rates)
        lines = [
            _strip(dict(ln), show_commercial=show_rates)
            for ln in conn.execute("SELECT * FROM po_lines WHERE po_id=?", (h["id"],)).fetchall()
        ]
        d["lines"] = lines
        d["vendor_response"] = _latest_response(conn, "supplier", int(supplier_id), "po", int(h["id"]))
        out.append(d)
    conn.close()
    return out


def get_purchase_order(supplier_id: int, po_id: int, *, show_rates: bool) -> dict | None:
    conn = _connect()
    h = conn.execute(
        "SELECT * FROM po_headers WHERE id=? AND supplier_id=?",
        (int(po_id), int(supplier_id)),
    ).fetchone()
    if not h:
        conn.close()
        return None
    d = _strip(dict(h), show_commercial=show_rates)
    d["lines"] = [
        _strip(dict(ln), show_commercial=show_rates)
        for ln in conn.execute("SELECT * FROM po_lines WHERE po_id=?", (h["id"],)).fetchall()
    ]
    d["vendor_response"] = _latest_response(conn, "supplier", int(supplier_id), "po", int(h["id"]))
    conn.close()
    return d


def list_job_work_orders(processor_id: int, *, show_rates: bool) -> list[dict]:
    conn = _connect()
    headers = conn.execute(
        "SELECT * FROM jwo_headers WHERE processor_id=? ORDER BY id DESC",
        (int(processor_id),),
    ).fetchall()
    out = []
    for h in headers:
        d = _strip(dict(h), show_commercial=show_rates)
        d["lines"] = [
            _strip(dict(ln), show_commercial=show_rates)
            for ln in conn.execute("SELECT * FROM jwo_lines WHERE jwo_id=?", (h["id"],)).fetchall()
        ]
        d["vendor_response"] = _latest_response(conn, "processor", int(processor_id), "jwo", int(h["id"]))
        out.append(d)
    conn.close()
    return out


def get_job_work_order(processor_id: int, jwo_id: int, *, show_rates: bool) -> dict | None:
    conn = _connect()
    h = conn.execute(
        "SELECT * FROM jwo_headers WHERE id=? AND processor_id=?",
        (int(jwo_id), int(processor_id)),
    ).fetchone()
    if not h:
        conn.close()
        return None
    d = _strip(dict(h), show_commercial=show_rates)
    d["lines"] = [
        _strip(dict(ln), show_commercial=show_rates)
        for ln in conn.execute("SELECT * FROM jwo_lines WHERE jwo_id=?", (h["id"],)).fetchall()
    ]
    d["vendor_response"] = _latest_response(conn, "processor", int(processor_id), "jwo", int(h["id"]))
    conn.close()
    return d


def list_production_job_orders(vendor_name: str, *, show_rates: bool) -> list[dict]:
    """Production JOs store the vendor as a name, not a foreign key."""
    name = (vendor_name or "").strip()
    if not name or not os.path.exists(_PROD_DB):
        return []
    conn = _connect(_PROD_DB)
    try:
        rows = conn.execute(
            """SELECT id, jo_number, jo_date, so_number, sku, sku_name, process, stage,
                      exec_type, vendor_name, vendor_rate, planned_qty, issued_qty,
                      received_qty, rejected_qty, balance_qty, status,
                      expected_completion, completed_date
               FROM job_orders
               WHERE lower(trim(vendor_name)) = lower(trim(?))
               ORDER BY id DESC""",
            (name,),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []
    out = [_strip(dict(r), show_commercial=show_rates) for r in rows]
    conn.close()
    return out


def dashboard_for_link(link: dict) -> dict:
    rates = show_commercial(link.get("portal_role") or "")
    kind = link["party_kind"]
    pid = int(link["party_id"])
    name = link.get("party_name") or ""
    pos = list_purchase_orders(pid, show_rates=False) if kind == "supplier" else []
    jwos = list_job_work_orders(pid, show_rates=False) if kind == "processor" else []
    jos = list_production_job_orders(name, show_rates=False)

    def _count(rows: list[dict], status: str) -> int:
        return sum(1 for r in rows if str(r.get("status") or "") == status)

    return {
        "party_kind": kind,
        "party_id": pid,
        "party_name": name,
        "party_code": link.get("party_code") or "",
        "portal_role": link.get("portal_role"),
        "show_commercial": rates,
        "purchase_orders": len(pos),
        "job_work_orders": len(jwos),
        "production_job_orders": len(jos),
        "po_by_status": {s: _count(pos, s) for s in sorted({str(r.get("status") or "") for r in pos})},
        "jwo_by_status": {s: _count(jwos, s) for s in sorted({str(r.get("status") or "") for r in jwos})},
        "note": (
            "Factory GRN, stock, and accounts stay with the ERP team. "
            "Accepting an order or reporting a delivery here does not receive stock or approve a bill."
        ),
    }


_DOC_ACTIONS = frozenset({"accept", "reject", "clarify"})


def _owns_document(link: dict, doc_type: str, doc_id: int) -> dict | None:
    kind = link["party_kind"]
    pid = int(link["party_id"])
    name = (link.get("party_name") or "").strip()
    if doc_type == "po" and kind == "supplier":
        return get_purchase_order(pid, doc_id, show_rates=False)
    if doc_type == "jwo" and kind == "processor":
        return get_job_work_order(pid, doc_id, show_rates=False)
    if doc_type == "jo":
        for row in list_production_job_orders(name, show_rates=False):
            if int(row["id"]) == int(doc_id):
                return row
    return None


def record_document_response(link: dict, *, doc_type: str, doc_id: int, action: str, reason: str, user_id: int | None, username: str) -> dict:
    action = (action or "").strip().lower()
    if action not in _DOC_ACTIONS:
        raise ValueError("Action must be accept, reject, or clarify")
    if action in ("reject", "clarify") and not (reason or "").strip():
        raise ValueError("A reason is required")
    if doc_type not in ("po", "jwo", "jo"):
        raise ValueError("Unknown document type")
    owned = _owns_document(link, doc_type, doc_id)
    if not owned:
        raise LookupError("Document not found")
    conn = _connect()
    conn.execute(
        """INSERT INTO vendor_doc_responses(
            party_kind, party_id, doc_type, doc_id, action, reason, user_id, username
        ) VALUES(?,?,?,?,?,?,?,?)""",
        (link["party_kind"], int(link["party_id"]), doc_type, int(doc_id), action, (reason or "").strip(), user_id, username or ""),
    )
    conn.commit()
    row = _latest_response(conn, link["party_kind"], int(link["party_id"]), doc_type, int(doc_id))
    conn.close()
    log_event(
        user_id=user_id,
        username=username,
        party_kind=link["party_kind"],
        party_id=int(link["party_id"]),
        action=f"document_{action}",
        record_type=doc_type,
        record_id=str(doc_id),
        details={"reason": (reason or "").strip()},
    )
    return {"ok": True, "doc_type": doc_type, "doc_id": int(doc_id), "vendor_response": row, "erp_status_unchanged": True}


def list_material_issues(party_name: str) -> list[dict]:
    name = (party_name or "").strip()
    if not name:
        return []
    conn = _connect()
    try:
        notes = conn.execute(
            """SELECT * FROM material_issue_notes
               WHERE lower(trim(COALESCE(to_vendor,''))) = lower(trim(?))
               ORDER BY id DESC""",
            (name,),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []
    out = []
    for note in notes:
        d = dict(note)
        lines = []
        for ln in conn.execute("SELECT * FROM min_lines WHERE min_id=?", (note["id"],)).fetchall():
            line = dict(ln)
            ack = conn.execute(
                """SELECT received_qty, variance_qty, reason, created_at
                   FROM vendor_material_acks
                   WHERE source_type='min' AND source_id=? AND line_id=?
                   ORDER BY id DESC LIMIT 1""",
                (note["id"], ln["id"]),
            ).fetchone()
            line["latest_ack"] = dict(ack) if ack else None
            lines.append(line)
        d["lines"] = lines
        out.append(d)
    conn.close()
    return out


def acknowledge_material_issue(
    link: dict,
    *,
    min_id: int,
    lines: list[dict],
    user_id: int | None,
    username: str,
) -> dict:
    name = (link.get("party_name") or "").strip()
    conn = _connect()
    note = conn.execute(
        """SELECT * FROM material_issue_notes
           WHERE id=? AND lower(trim(COALESCE(to_vendor,''))) = lower(trim(?))""",
        (int(min_id), name),
    ).fetchone()
    if not note:
        conn.close()
        raise LookupError("Material issue not found")
    saved = []
    for item in lines:
        line_id = int(item["line_id"])
        src = conn.execute(
            "SELECT * FROM min_lines WHERE id=? AND min_id=?",
            (line_id, int(min_id)),
        ).fetchone()
        if not src:
            conn.close()
            raise ValueError(f"Line {line_id} is not on this material issue")
        issued = float(src["issue_qty"] or 0)
        received = float(item.get("received_qty") or 0)
        if received < 0:
            conn.close()
            raise ValueError("Received quantity cannot be negative")
        variance = round(issued - received, 4)
        reason = (item.get("reason") or "").strip()
        if abs(variance) > 0.0001 and not reason:
            conn.close()
            raise ValueError("A reason is required when received quantity differs from issued quantity")
        conn.execute(
            """INSERT INTO vendor_material_acks(
                party_kind, party_id, source_type, source_id, line_id,
                issued_qty, received_qty, variance_qty, reason, user_id, username
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                link["party_kind"], int(link["party_id"]), "min", int(min_id), line_id,
                issued, received, variance, reason, user_id, username or "",
            ),
        )
        saved.append({"line_id": line_id, "issued_qty": issued, "received_qty": received, "variance_qty": variance})
    unchanged = conn.execute("SELECT issue_qty FROM min_lines WHERE min_id=?", (int(min_id),)).fetchall()
    conn.commit()
    conn.close()
    log_event(
        user_id=user_id,
        username=username,
        party_kind=link["party_kind"],
        party_id=int(link["party_id"]),
        action="material_acknowledged",
        record_type="material_issue",
        record_id=str(min_id),
        details={"lines": saved},
    )
    return {
        "ok": True,
        "min_id": int(min_id),
        "lines": saved,
        "stock_unchanged": True,
        "issued_qty_after": [float(r["issue_qty"] or 0) for r in unchanged],
    }


def create_delivery_intimation(
    link: dict,
    *,
    doc_type: str,
    doc_id: int,
    qty: float,
    challan_no: str,
    delivery_date: str,
    remarks: str,
    user_id: int | None,
    username: str,
) -> dict:
    if qty <= 0:
        raise ValueError("Quantity must be greater than zero")
    owned = _owns_document(link, doc_type, doc_id)
    if not owned:
        raise LookupError("Document not found")
    doc_number = owned.get("po_number") or owned.get("jwo_number") or owned.get("jo_number") or ""
    conn = _connect()
    grn_before = 0
    try:
        grn_before = int(conn.execute("SELECT COUNT(*) AS n FROM grn_headers").fetchone()["n"])
    except sqlite3.OperationalError:
        grn_before = 0
    seq = int(conn.execute("SELECT COUNT(*) AS n FROM vendor_delivery_intimations").fetchone()["n"]) + 1
    number = f"VDI-{seq:05d}"
    conn.execute(
        """INSERT INTO vendor_delivery_intimations(
            intimation_no, party_kind, party_id, doc_type, doc_id, doc_number,
            qty, challan_no, delivery_date, remarks, status, user_id, username
        ) VALUES(?,?,?,?,?,?,?,?,?,?, 'Submitted', ?, ?)""",
        (
            number, link["party_kind"], int(link["party_id"]), doc_type, int(doc_id), doc_number,
            float(qty), challan_no or "", delivery_date or "", remarks or "", user_id, username or "",
        ),
    )
    try:
        grn_after = int(conn.execute("SELECT COUNT(*) AS n FROM grn_headers").fetchone()["n"])
    except sqlite3.OperationalError:
        grn_after = 0
    conn.commit()
    conn.close()
    log_event(
        user_id=user_id,
        username=username,
        party_kind=link["party_kind"],
        party_id=int(link["party_id"]),
        action="delivery_intimation",
        record_type=doc_type,
        record_id=str(doc_id),
        details={"intimation_no": number, "qty": qty},
    )
    return {
        "ok": True,
        "intimation_no": number,
        "status": "Submitted",
        "grn_created": grn_after != grn_before,
        "stock_unchanged": True,
    }


def list_delivery_intimations(link: dict) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT * FROM vendor_delivery_intimations
           WHERE party_kind=? AND party_id=? ORDER BY id DESC""",
        (link["party_kind"], int(link["party_id"])),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
