"""
Shared Accounts document verification registry.

Overlay on operational docs (PO/JWO/GRN/MIN/JO). Does not replace module statuses.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

DB_PATH = os.environ.get("DOCUMENT_AUDIT_DB_PATH") or os.environ.get(
    "PURCHASE_DB_PATH", "/data/purchase.db"
)


def _connect() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:
        conn = sqlite3.connect("./purchase_dev.db")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    conn = _connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS document_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            doc_number TEXT NOT NULL DEFAULT '',
            module TEXT NOT NULL DEFAULT '',
            so_reference TEXT DEFAULT '',
            party_name TEXT DEFAULT '',
            process_name TEXT DEFAULT '',
            doc_date TEXT DEFAULT '',
            audit_status TEXT NOT NULL DEFAULT 'Pending',
            created_by TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            verified_by TEXT DEFAULT '',
            verified_at TEXT DEFAULT '',
            unverified_by TEXT DEFAULT '',
            unverified_at TEXT DEFAULT '',
            last_reason TEXT DEFAULT '',
            UNIQUE(doc_type, doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_doc_audit_status ON document_audit(audit_status, doc_date);
        CREATE INDEX IF NOT EXISTS idx_doc_audit_type ON document_audit(doc_type, doc_number);

        CREATE TABLE IF NOT EXISTS document_audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT DEFAULT '',
            reason TEXT DEFAULT '',
            detail TEXT DEFAULT '',
            event_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_doc_audit_ev ON document_audit_events(doc_type, doc_id, id);
        """
    )
    conn.commit()
    conn.close()


DOC_TYPES = ("PO", "JWO", "GRN", "MIN", "JO", "GIN")


def enroll_document(
    doc_type: str,
    doc_id: int,
    *,
    doc_number: str = "",
    module: str = "",
    so_reference: str = "",
    party_name: str = "",
    process_name: str = "",
    doc_date: str = "",
    created_by: str = "",
) -> None:
    dt = str(doc_type or "").strip().upper()
    if dt not in DOC_TYPES or not doc_id:
        return
    init_db()
    now = _now()
    conn = _connect()
    existing = conn.execute(
        "SELECT id, audit_status FROM document_audit WHERE doc_type=? AND doc_id=?",
        (dt, int(doc_id)),
    ).fetchone()
    if existing:
        # Keep status; refresh display fields
        conn.execute(
            """UPDATE document_audit SET
                   doc_number=COALESCE(NULLIF(?,''), doc_number),
                   so_reference=COALESCE(NULLIF(?,''), so_reference),
                   party_name=COALESCE(NULLIF(?,''), party_name),
                   process_name=COALESCE(NULLIF(?,''), process_name),
                   doc_date=COALESCE(NULLIF(?,''), doc_date)
               WHERE doc_type=? AND doc_id=?""",
            (
                doc_number,
                so_reference,
                party_name,
                process_name,
                doc_date,
                dt,
                int(doc_id),
            ),
        )
    else:
        conn.execute(
            """INSERT INTO document_audit(
                   doc_type, doc_id, doc_number, module, so_reference, party_name,
                   process_name, doc_date, audit_status, created_by, created_at
               ) VALUES(?,?,?,?,?,?,?,?,'Pending',?,?)""",
            (
                dt,
                int(doc_id),
                doc_number or "",
                module or dt,
                so_reference or "",
                party_name or "",
                process_name or "",
                doc_date or "",
                created_by or "",
                now,
            ),
        )
        conn.execute(
            """INSERT INTO document_audit_events(
                   doc_type, doc_id, event_type, actor, reason, detail, event_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (dt, int(doc_id), "created", created_by or "", "", "", now),
        )
    conn.commit()
    conn.close()


def get_audit(doc_type: str, doc_id: int) -> Optional[dict]:
    init_db()
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM document_audit WHERE doc_type=? AND doc_id=?",
        (str(doc_type).upper(), int(doc_id)),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def is_verified(doc_type: str, doc_id: int) -> bool:
    row = get_audit(doc_type, doc_id)
    return bool(row and str(row.get("audit_status") or "") == "Verified")


def list_audit_events(doc_type: str, doc_id: int) -> list[dict]:
    init_db()
    conn = _connect()
    rows = conn.execute(
        """SELECT * FROM document_audit_events
           WHERE doc_type=? AND doc_id=? ORDER BY id""",
        (str(doc_type).upper(), int(doc_id)),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_documents(
    *,
    audit_status: str = "",
    doc_type: str = "",
    date_from: str = "",
    date_to: str = "",
    search: str = "",
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    init_db()
    conn = _connect()
    q = "SELECT * FROM document_audit WHERE 1=1"
    params: list[Any] = []
    if audit_status:
        q += " AND audit_status=?"
        params.append(audit_status)
    if doc_type:
        q += " AND doc_type=?"
        params.append(doc_type.upper())
    if date_from:
        q += " AND IFNULL(NULLIF(doc_date,''), substr(created_at,1,10)) >= ?"
        params.append(date_from[:10])
    if date_to:
        q += " AND IFNULL(NULLIF(doc_date,''), substr(created_at,1,10)) <= ?"
        params.append(date_to[:10])
    if search:
        like = f"%{search.strip()}%"
        q += " AND (doc_number LIKE ? OR so_reference LIKE ? OR party_name LIKE ? OR process_name LIKE ?)"
        params.extend([like, like, like, like])
    count = conn.execute(
        f"SELECT COUNT(*) AS c FROM ({q})", params
    ).fetchone()["c"]
    q += " ORDER BY IFNULL(NULLIF(doc_date,''), created_at) DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    pending = conn.execute(
        "SELECT COUNT(*) AS c FROM document_audit WHERE audit_status='Pending'"
    ).fetchone()["c"]
    verified = conn.execute(
        "SELECT COUNT(*) AS c FROM document_audit WHERE audit_status='Verified'"
    ).fetchone()["c"]
    conn.close()
    return {
        "rows": rows,
        "total": int(count),
        "pending": int(pending),
        "verified": int(verified),
    }


def _append_event(
    conn,
    doc_type: str,
    doc_id: int,
    event_type: str,
    actor: str,
    reason: str = "",
    detail: str = "",
) -> None:
    conn.execute(
        """INSERT INTO document_audit_events(
               doc_type, doc_id, event_type, actor, reason, detail, event_at
           ) VALUES(?,?,?,?,?,?,?)""",
        (
            doc_type,
            int(doc_id),
            event_type,
            actor or "",
            reason or "",
            detail or "",
            _now(),
        ),
    )


def verify_document(doc_type: str, doc_id: int, *, actor: str = "") -> dict:
    init_db()
    dt = str(doc_type).upper()
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM document_audit WHERE doc_type=? AND doc_id=?",
        (dt, int(doc_id)),
    ).fetchone()
    if not row:
        conn.close()
        raise ValueError("Document not enrolled in audit registry")
    if str(row["audit_status"]) == "Verified":
        conn.close()
        return dict(row)
    now = _now()
    conn.execute(
        """UPDATE document_audit SET
               audit_status='Verified', verified_by=?, verified_at=?,
               unverified_by='', unverified_at='', last_reason=''
           WHERE doc_type=? AND doc_id=?""",
        (actor or "", now, dt, int(doc_id)),
    )
    _append_event(conn, dt, int(doc_id), "verified", actor)
    conn.commit()
    out = dict(
        conn.execute(
            "SELECT * FROM document_audit WHERE doc_type=? AND doc_id=?",
            (dt, int(doc_id)),
        ).fetchone()
    )
    conn.close()
    return out


def unverify_document(
    doc_type: str,
    doc_id: int,
    *,
    actor: str = "",
    reason: str = "",
    force: bool = False,
) -> dict:
    init_db()
    dt = str(doc_type).upper()
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("Reason is required to unverify")
    blockers = dependency_blockers(dt, int(doc_id))
    if blockers and not force:
        raise ValueError(
            "Downstream transactions exist — Admin force unverify required: "
            + "; ".join(blockers[:5])
        )
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM document_audit WHERE doc_type=? AND doc_id=?",
        (dt, int(doc_id)),
    ).fetchone()
    if not row:
        conn.close()
        raise ValueError("Document not enrolled in audit registry")
    now = _now()
    conn.execute(
        """UPDATE document_audit SET
               audit_status='Pending', unverified_by=?, unverified_at=?,
               last_reason=?, verified_by='', verified_at=''
           WHERE doc_type=? AND doc_id=?""",
        (actor or "", now, reason, dt, int(doc_id)),
    )
    detail = "force=1; " + "; ".join(blockers) if force and blockers else "; ".join(blockers)
    _append_event(conn, dt, int(doc_id), "unverified", actor, reason, detail)
    conn.commit()
    out = dict(
        conn.execute(
            "SELECT * FROM document_audit WHERE doc_type=? AND doc_id=?",
            (dt, int(doc_id)),
        ).fetchone()
    )
    conn.close()
    return out


def record_edit_event(doc_type: str, doc_id: int, *, actor: str = "", detail: str = "") -> None:
    if not get_audit(doc_type, doc_id):
        return
    init_db()
    conn = _connect()
    _append_event(conn, str(doc_type).upper(), int(doc_id), "edited", actor, "", detail)
    conn.commit()
    conn.close()


def dependency_blockers(doc_type: str, doc_id: int) -> list[str]:
    """Shallow downstream checks — block casual unverify/edit when chain exists."""
    dt = str(doc_type).upper()
    blockers: list[str] = []
    try:
        if dt == "PO":
            from ..db import purchase_db as pdb

            conn = pdb._connect()
            n = conn.execute(
                """SELECT COUNT(*) AS c FROM grn_headers g
                   JOIN po_headers p ON p.po_number = g.reference_number
                   WHERE p.id=? AND IFNULL(g.status,'') NOT IN ('Cancelled','Draft')""",
                (int(doc_id),),
            ).fetchone()["c"]
            conn.close()
            if int(n) > 0:
                blockers.append(f"{n} GRN(s) posted/verified against this PO")
        elif dt == "JWO":
            from ..db import purchase_db as pdb

            conn = pdb._connect()
            n = conn.execute(
                """SELECT COUNT(*) AS c FROM grn_headers g
                   JOIN jwo_headers j ON j.jwo_number = g.reference_number
                   WHERE j.id=? AND IFNULL(g.status,'') NOT IN ('Cancelled','Draft')""",
                (int(doc_id),),
            ).fetchone()["c"]
            conn.close()
            if int(n) > 0:
                blockers.append(f"{n} GRN(s) against this JWO")
        elif dt == "JO":
            from ..db import production_db as prdb

            conn = prdb._connect()
            iss = conn.execute(
                "SELECT COUNT(*) AS c FROM jo_piece_issues WHERE jo_id=?",
                (int(doc_id),),
            ).fetchone()["c"]
            nxt = conn.execute(
                """SELECT COUNT(*) AS c FROM job_orders
                   WHERE parent_jo_id=? AND IFNULL(status,'') != 'Cancelled'""",
                (int(doc_id),),
            ).fetchone()["c"]
            conn.close()
            if int(iss) > 0:
                blockers.append(f"{iss} piece issue(s) to next process")
            if int(nxt) > 0:
                blockers.append(f"{nxt} child/next-stage JO(s)")
        elif dt == "GRN":
            # Once inventory posted, casual unverify is risky
            from ..db import purchase_db as pdb

            conn = pdb._connect()
            row = conn.execute(
                "SELECT status, inventory_posted FROM grn_headers WHERE id=?",
                (int(doc_id),),
            ).fetchone()
            conn.close()
            if row and int(row["inventory_posted"] or 0) == 1:
                blockers.append("GRN inventory already posted — use Cancel/reverse flow")
        elif dt == "MIN":
            from ..db import purchase_db as pdb

            conn = pdb._connect()
            row = conn.execute(
                "SELECT status FROM material_issue_notes WHERE id=?",
                (int(doc_id),),
            ).fetchone()
            conn.close()
            if row and str(row["status"] or "") == "Confirmed":
                blockers.append("MIN confirmed with stock posting — cancel/reverse first")
    except Exception as exc:
        blockers.append(f"dependency check error: {exc}")
    return blockers


def assert_doc_editable(
    doc_type: str,
    doc_id: int,
    *,
    allow_admin: bool = False,
) -> None:
    """Raise ValueError if Verified (unless admin override)."""
    if allow_admin:
        return
    if is_verified(doc_type, doc_id):
        raise ValueError(
            f"{doc_type} #{doc_id} is Accounts-Verified and locked. "
            "Ask Accounts to Unverify with reason before editing."
        )
