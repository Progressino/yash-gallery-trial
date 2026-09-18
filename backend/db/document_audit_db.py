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


_INITIALIZED_PATH: str | None = None


def init_db() -> None:
    global _INITIALIZED_PATH
    if _INITIALIZED_PATH == DB_PATH:
        return
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
        CREATE INDEX IF NOT EXISTS idx_doc_audit_created ON document_audit(created_at);
        CREATE INDEX IF NOT EXISTS idx_doc_audit_status_id ON document_audit(audit_status, id DESC);

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
    _INITIALIZED_PATH = DB_PATH


# Operational docs Accounts verifies daily (issue/receive are piece movement docs).
DOC_TYPES = ("PO", "JWO", "GRN", "MIN", "JO", "GIN", "JO_ISSUE", "JO_RECEIVE")
DOC_TYPE_LABELS = {
    "PO": "Purchase Order",
    "JWO": "Job Work Order",
    "GRN": "Goods Receipt Note",
    "MIN": "Material Issue Note",
    "JO": "Process Job Order",
    "GIN": "Gate Inward Note",
    "JO_ISSUE": "Issue to Next Process",
    "JO_RECEIVE": "Piece Receipt / Receive",
}


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
    where = " WHERE 1=1"
    params: list[Any] = []
    if audit_status:
        where += " AND audit_status=?"
        params.append(audit_status)
    if doc_type:
        where += " AND doc_type=?"
        params.append(doc_type.upper())
    # Prefer indexed doc_date; fall back to created_at date only when doc_date empty
    if date_from:
        where += " AND COALESCE(NULLIF(doc_date,''), substr(created_at,1,10)) >= ?"
        params.append(date_from[:10])
    if date_to:
        where += " AND COALESCE(NULLIF(doc_date,''), substr(created_at,1,10)) <= ?"
        params.append(date_to[:10])
    if search:
        like = f"%{search.strip()}%"
        where += " AND (doc_number LIKE ? OR so_reference LIKE ? OR party_name LIKE ? OR process_name LIKE ?)"
        params.extend([like, like, like, like])

    count = int(conn.execute(f"SELECT COUNT(*) AS c FROM document_audit{where}", params).fetchone()["c"])
    # Fast path: status-only list uses status+id index
    if not date_from and not date_to and not search:
        order = " ORDER BY id DESC"
    else:
        order = " ORDER BY COALESCE(NULLIF(doc_date,''), created_at) DESC, id DESC"
    rows = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM document_audit{where}{order} LIMIT ? OFFSET ?",
            [*params, int(limit), int(offset)],
        ).fetchall()
    ]
    pending = int(
        conn.execute("SELECT COUNT(*) AS c FROM document_audit WHERE audit_status='Pending'").fetchone()["c"]
    )
    verified = int(
        conn.execute("SELECT COUNT(*) AS c FROM document_audit WHERE audit_status='Verified'").fetchone()["c"]
    )
    conn.close()
    return {
        "rows": rows,
        "total": count,
        "pending": pending,
        "verified": verified,
    }


def backfill_from_modules(*, limit_per_type: int = 5000, actor: str = "system-backfill") -> dict[str, int]:
    """Enroll existing operational docs that were created before audit hooks."""
    init_db()
    counts: dict[str, int] = {t: 0 for t in DOC_TYPES}
    lim = max(1, min(int(limit_per_type or 5000), 20000))

    def _enroll_many(rows: list[dict], doc_type: str, mapper) -> None:
        for r in rows:
            meta = mapper(r)
            if not meta.get("doc_id"):
                continue
            before = get_audit(doc_type, int(meta["doc_id"]))
            enroll_document(doc_type, int(meta["doc_id"]), created_by=actor, **{k: v for k, v in meta.items() if k != "doc_id"})
            if not before:
                counts[doc_type] = counts.get(doc_type, 0) + 1

    try:
        from ..db import purchase_db as pdb

        conn = pdb._connect()
        pos = [dict(r) for r in conn.execute(
            "SELECT id, po_number, supplier_name, so_reference, po_date FROM po_headers ORDER BY id DESC LIMIT ?",
            (lim,),
        ).fetchall()]
        jwos = [dict(r) for r in conn.execute(
            "SELECT id, jwo_number, processor_name, so_reference, jwo_date FROM jwo_headers ORDER BY id DESC LIMIT ?",
            (lim,),
        ).fetchall()]
        grns = [dict(r) for r in conn.execute(
            "SELECT id, grn_number, party_name, reference_number, grn_date FROM grn_headers ORDER BY id DESC LIMIT ?",
            (lim,),
        ).fetchall()]
        try:
            mins = [dict(r) for r in conn.execute(
                "SELECT id, min_number, to_vendor, so_reference, jwo_reference, min_date FROM material_issue_notes ORDER BY id DESC LIMIT ?",
                (lim,),
            ).fetchall()]
        except Exception:
            mins = []
        try:
            gins = [dict(r) for r in conn.execute(
                "SELECT id, gin_number, party_name, source_number, gin_date, stage FROM gin_headers ORDER BY id DESC LIMIT ?",
                (lim,),
            ).fetchall()]
        except Exception:
            gins = []
        conn.close()

        _enroll_many(pos, "PO", lambda r: {
            "doc_id": r["id"], "doc_number": r.get("po_number") or "", "module": "purchase",
            "so_reference": r.get("so_reference") or "", "party_name": r.get("supplier_name") or "",
            "doc_date": r.get("po_date") or "",
        })
        _enroll_many(jwos, "JWO", lambda r: {
            "doc_id": r["id"], "doc_number": r.get("jwo_number") or "", "module": "purchase",
            "so_reference": r.get("so_reference") or "", "party_name": r.get("processor_name") or "",
            "doc_date": r.get("jwo_date") or "",
        })
        _enroll_many(grns, "GRN", lambda r: {
            "doc_id": r["id"], "doc_number": r.get("grn_number") or "", "module": "purchase",
            "so_reference": r.get("reference_number") or "", "party_name": r.get("party_name") or "",
            "doc_date": r.get("grn_date") or "",
        })
        _enroll_many(mins, "MIN", lambda r: {
            "doc_id": r["id"], "doc_number": r.get("min_number") or "", "module": "purchase",
            "so_reference": r.get("so_reference") or r.get("jwo_reference") or "",
            "party_name": r.get("to_vendor") or "", "doc_date": r.get("min_date") or "",
            "process_name": "Material Issue",
        })
        _enroll_many(gins, "GIN", lambda r: {
            "doc_id": r["id"], "doc_number": r.get("gin_number") or "", "module": "gate",
            "so_reference": r.get("source_number") or "", "party_name": r.get("party_name") or "",
            "doc_date": r.get("gin_date") or "", "process_name": r.get("stage") or "Gate Inward",
        })
    except Exception:
        pass

    try:
        from ..db import production_db as prdb

        conn = prdb._connect()
        jos = [dict(r) for r in conn.execute(
            """SELECT id, jo_number, vendor_name, so_number, process, jo_date
               FROM job_orders WHERE IFNULL(status,'') != 'Cancelled'
               ORDER BY id DESC LIMIT ?""",
            (lim,),
        ).fetchall()]
        try:
            issues = [dict(r) for r in conn.execute(
                """SELECT id, jo_id, from_process, to_process, so_number, sku, issue_date, issued_qty
                   FROM jo_piece_issues ORDER BY id DESC LIMIT ?""",
                (lim,),
            ).fetchall()]
        except Exception:
            issues = []
        try:
            receipts = [dict(r) for r in conn.execute(
                """SELECT id, jo_id, process, so_number, sku, receipt_date, received_qty
                   FROM jo_piece_receipts ORDER BY id DESC LIMIT ?""",
                (lim,),
            ).fetchall()]
        except Exception:
            receipts = []
        jo_nums = {}
        if issues or receipts:
            ids = {int(r["jo_id"]) for r in issues + receipts if r.get("jo_id")}
            if ids:
                qmarks = ",".join("?" * len(ids))
                for row in conn.execute(
                    f"SELECT id, jo_number FROM job_orders WHERE id IN ({qmarks})",
                    tuple(ids),
                ).fetchall():
                    jo_nums[int(row["id"])] = row["jo_number"]
        conn.close()

        _enroll_many(jos, "JO", lambda r: {
            "doc_id": r["id"], "doc_number": r.get("jo_number") or "", "module": "production",
            "so_reference": r.get("so_number") or "", "party_name": r.get("vendor_name") or "",
            "process_name": r.get("process") or "", "doc_date": r.get("jo_date") or "",
        })
        _enroll_many(issues, "JO_ISSUE", lambda r: {
            "doc_id": r["id"],
            "doc_number": f"ISS-{r['id']}",
            "module": "production",
            "so_reference": r.get("so_number") or "",
            "party_name": jo_nums.get(int(r.get("jo_id") or 0), f"JO#{r.get('jo_id')}"),
            "process_name": f"{r.get('from_process') or ''} → {r.get('to_process') or ''}".strip(" →"),
            "doc_date": r.get("issue_date") or "",
        })
        _enroll_many(receipts, "JO_RECEIVE", lambda r: {
            "doc_id": r["id"],
            "doc_number": f"RCV-{r['id']}",
            "module": "production",
            "so_reference": r.get("so_number") or "",
            "party_name": jo_nums.get(int(r.get("jo_id") or 0), f"JO#{r.get('jo_id')}"),
            "process_name": r.get("process") or "Receive",
            "doc_date": r.get("receipt_date") or "",
        })
    except Exception:
        pass

    counts["total_new"] = sum(v for k, v in counts.items() if k != "total_new")
    return counts


def get_document_detail(doc_type: str, doc_id: int, *, include_blockers: bool = True) -> dict:
    row = get_audit(doc_type, doc_id)
    if not row:
        raise ValueError("Not found in audit registry")
    out = {
        **row,
        "events": list_audit_events(doc_type, doc_id),
        "editable": row.get("audit_status") != "Verified",
        "dependency_blockers": [],
    }
    if include_blockers:
        out["dependency_blockers"] = dependency_blockers(doc_type, doc_id)
    return out


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
