"""Job Order returns, mandatory material reconciliation and return traceability.

One mechanism for every job-work process (Cutting, Stitching, Kaj Button,
Handwork, Embroidery, Finishing, ...):

* **Processed** pieces are posted through the normal ``receive_pieces`` flow and
  move on to the next process as usual.
* **Unprocessed** pieces release the JO's commitment: the JO line / header
  ``planned_qty`` drops by the returned qty and ``unprocessed_return_qty`` keeps
  the original visible. Ready-To for a process is ``stock − planned on open JOs``
  (JO creation never moves stock), so the pieces reappear on the same process's
  Ready-To list and can be given to another vendor via the normal JO flow.
* Every issued material (pieces, fabric, accessories from the BOM issue note) is
  reconciled as Issued = Consumed + Returned + Wastage + Balance-with-vendor.
  ``job_orders.reconciliation_status`` is ``Pending`` until balances are zero.
* ``jo_return_links`` / ``jo_return_allocations`` tie the original JO → return →
  Ready-To availability → subsequent JO (and vendor) created for that qty.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

_log = logging.getLogger(__name__)

RECON_PENDING = "Pending"
RECON_COMPLETED = "Completed"
KIND_PIECES = "PIECES"
KIND_FABRIC = "FABRIC"
KIND_ACCESSORY = "ACCESSORY"
_EPS = 1e-3


def _pdb():
    from ..db import production_db

    return production_db


def _connect():
    return _pdb()._connect()


def init_jo_return_tables(conn=None) -> None:
    own = conn is None
    if own:
        conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jo_returns (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                return_number   TEXT UNIQUE NOT NULL,
                return_date     TEXT NOT NULL,
                jo_id           INTEGER NOT NULL REFERENCES job_orders(id),
                jo_number       TEXT NOT NULL,
                process         TEXT DEFAULT '',
                so_number       TEXT DEFAULT '',
                vendor_name     TEXT DEFAULT '',
                processed_qty   INTEGER DEFAULT 0,
                unprocessed_qty INTEGER DEFAULT 0,
                rejected_qty    INTEGER DEFAULT 0,
                reason          TEXT DEFAULT '',
                remarks         TEXT DEFAULT '',
                returned_by     TEXT DEFAULT '',
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS jo_return_lines (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                return_id       INTEGER NOT NULL REFERENCES jo_returns(id) ON DELETE CASCADE,
                jo_line_id      INTEGER,
                sku             TEXT DEFAULT '',
                processed_qty   INTEGER DEFAULT 0,
                unprocessed_qty INTEGER DEFAULT 0,
                rejected_qty    INTEGER DEFAULT 0,
                receipt_id      INTEGER
            );
            CREATE TABLE IF NOT EXISTS jo_return_links (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                return_id       INTEGER NOT NULL REFERENCES jo_returns(id) ON DELETE CASCADE,
                return_line_id  INTEGER REFERENCES jo_return_lines(id) ON DELETE CASCADE,
                source_jo_id    INTEGER NOT NULL,
                source_jo_number TEXT DEFAULT '',
                source_vendor   TEXT DEFAULT '',
                process         TEXT DEFAULT '',
                so_number       TEXT DEFAULT '',
                sku             TEXT DEFAULT '',
                unprocessed_qty INTEGER DEFAULT 0,
                allocated_qty   INTEGER DEFAULT 0,
                ready_at        TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS jo_return_allocations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                link_id         INTEGER NOT NULL REFERENCES jo_return_links(id) ON DELETE CASCADE,
                new_jo_id       INTEGER NOT NULL,
                new_jo_number   TEXT DEFAULT '',
                new_vendor      TEXT DEFAULT '',
                qty             INTEGER DEFAULT 0,
                released        INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS jo_material_reconciliation (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                jo_id           INTEGER NOT NULL REFERENCES job_orders(id) ON DELETE CASCADE,
                material_kind   TEXT NOT NULL,
                material_code   TEXT NOT NULL,
                material_name   TEXT DEFAULT '',
                unit            TEXT DEFAULT '',
                issued_qty      REAL,
                consumed_qty    REAL,
                returned_qty    REAL,
                wastage_qty     REAL DEFAULT 0,
                remarks         TEXT DEFAULT '',
                updated_by      TEXT DEFAULT '',
                updated_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(jo_id, material_kind, material_code)
            );
            CREATE INDEX IF NOT EXISTS idx_jo_returns_jo ON jo_returns(jo_id);
            CREATE INDEX IF NOT EXISTS idx_jo_return_links_key
                ON jo_return_links(process, so_number, sku);
            CREATE INDEX IF NOT EXISTS idx_jo_return_links_src ON jo_return_links(source_jo_id);
            CREATE INDEX IF NOT EXISTS idx_jo_return_alloc_new ON jo_return_allocations(new_jo_id);
            """
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def _next_return_number(conn) -> str:
    row = conn.execute(
        "SELECT return_number FROM jo_returns ORDER BY id DESC LIMIT 1"
    ).fetchone()
    n = 1
    if row:
        try:
            n = int(str(row[0]).split("-")[-1]) + 1
        except (TypeError, ValueError):
            n = int(conn.execute("SELECT COUNT(*) FROM jo_returns").fetchone()[0]) + 1
    return f"RET-{n:05d}"


def _as_int(v, field: str) -> int:
    try:
        n = int(float(v or 0))
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number") from None
    if n < 0:
        raise ValueError(f"{field} cannot be negative")
    return n


def _as_qty(v, field: str) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        q = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number") from None
    if q < 0:
        raise ValueError(f"{field} cannot be negative")
    return round(q, 4)


def _assert_editable(joid: int) -> None:
    try:
        from ..db.document_audit_db import assert_doc_editable

        assert_doc_editable("JO", int(joid))
    except ValueError:
        raise
    except Exception:
        pass


# ── Return ────────────────────────────────────────────────────────────────────

def _scaled(qty: int, num: float, den: float) -> float:
    return qty * num / den if den > 0 else 0.0


def _release_commitment(conn, *, joid: int, line: Optional[dict], qty: int) -> None:
    """Lower planned qty (JO units) by ``qty`` on the line and keep garment/measurement in step."""
    if line is not None:
        planned = int(line.get("planned_qty") or 0)
        garment = float(line.get("garment_qty") or 0)
        meas = float(line.get("measurement_qty") or 0)
        conn.execute(
            """UPDATE jo_lines SET
                   planned_qty = planned_qty - ?,
                   unprocessed_return_qty = COALESCE(unprocessed_return_qty,0) + ?,
                   garment_qty = MAX(0, COALESCE(garment_qty,0) - ?),
                   measurement_qty = MAX(0, COALESCE(measurement_qty,0) - ?),
                   balance_qty = (planned_qty - ?) - COALESCE(received_qty,0)
               WHERE id=? AND jo_id=?""",
            (
                qty,
                qty,
                int(round(_scaled(qty, garment, planned))),
                round(_scaled(qty, meas, planned), 4),
                qty,
                int(line["id"]),
                joid,
            ),
        )
    hdr = dict(conn.execute(
        "SELECT planned_qty, garment_qty, measurement_qty FROM job_orders WHERE id=?", (joid,)
    ).fetchone())
    h_planned = int(hdr.get("planned_qty") or 0)
    conn.execute(
        """UPDATE job_orders SET
               planned_qty = MAX(0, COALESCE(planned_qty,0) - ?),
               unprocessed_return_qty = COALESCE(unprocessed_return_qty,0) + ?,
               garment_qty = MAX(0, COALESCE(garment_qty,0) - ?),
               measurement_qty = MAX(0, COALESCE(measurement_qty,0) - ?)
           WHERE id=?""",
        (
            qty,
            qty,
            int(round(_scaled(qty, float(hdr.get("garment_qty") or 0), h_planned))),
            round(_scaled(qty, float(hdr.get("measurement_qty") or 0), h_planned), 4),
            joid,
        ),
    )


def record_return(joid: int, data: dict) -> dict:
    """Post a vendor return: processed pieces → receive; unprocessed → back to Ready-To.

    ``data.lines``: ``[{jo_line_id, processed_qty, unprocessed_qty, rejected_qty}]``
    (``jo_line_id`` may be omitted for JOs without size lines). Qty is in JO units.
    """
    init_jo_return_tables()
    pdb = _pdb()
    _assert_editable(joid)
    jo = pdb.get_jo(int(joid))
    if not jo:
        raise ValueError("JO not found")
    if str(jo.get("status") or "") == "Cancelled":
        raise ValueError("Job order is cancelled — return is not allowed")

    lines_by_id = {int(ln["id"]): ln for ln in (jo.get("lines") or [])}
    entries: list[dict] = []
    seen: set = set()
    for raw in data.get("lines") or []:
        lid = raw.get("jo_line_id")
        lid = int(lid) if lid not in (None, "", 0, "0") else None
        processed = _as_int(raw.get("processed_qty"), "Processed qty")
        unprocessed = _as_int(raw.get("unprocessed_qty"), "Unprocessed qty")
        rejected = _as_int(raw.get("rejected_qty"), "Rejected qty")
        if processed + unprocessed + rejected == 0:
            continue
        if rejected and not processed:
            raise ValueError("Rejected qty applies to processed pieces — enter processed qty too")
        if lid is None and lines_by_id:
            if len(lines_by_id) != 1:
                raise ValueError("Select the JO line (size/SKU) for each returned qty")
            lid = next(iter(lines_by_id))
        if lid is not None and lid not in lines_by_id:
            raise ValueError("JO line not found for this job order")
        if lid in seen:
            raise ValueError("Each JO line may appear only once per return")
        seen.add(lid)
        line = lines_by_id.get(lid) if lid is not None else None
        planned = int((line or jo).get("planned_qty") or 0)
        received = int((line or jo).get("received_qty") or 0)
        pending = max(planned - received, 0)
        sku = str((line or jo).get("sku") or jo.get("sku") or "").strip()
        if processed + unprocessed > pending:
            raise ValueError(
                f"{sku or 'JO'}: returning {processed + unprocessed} exceeds {pending} pending with vendor"
            )
        entries.append({
            "line": line,
            "jo_line_id": lid,
            "sku": sku,
            "processed": processed,
            "unprocessed": unprocessed,
            "rejected": rejected,
        })
    if not entries:
        raise ValueError("Enter processed and/or unprocessed qty to return")

    process = str(jo.get("process") or "")
    so_number = str(jo.get("so_number") or "")
    return_date = str(data.get("return_date") or datetime.now().strftime("%Y-%m-%d"))
    returned_by = str(data.get("returned_by") or "")
    if any(e["processed"] for e in entries):
        conn = _connect()
        try:
            for e in entries:
                if e["processed"]:
                    pdb._assert_set_receive_allowed(conn, so_number, e["sku"], process)
        finally:
            conn.close()

    # Processed pieces follow the normal receive → next-process flow.
    for e in entries:
        if not e["processed"]:
            continue
        res = pdb.receive_pieces(
            int(joid),
            {
                "received_qty": e["processed"],
                "rejected_qty": e["rejected"],
                "jo_line_id": e["jo_line_id"],
                "sku": e["sku"],
                "process": process,
                "receipt_date": return_date,
                "received_by": returned_by,
                "remarks": "JO return — processed",
            },
        )
        e["receipt_id"] = (res or {}).get("receipt_id")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tot_p = sum(e["processed"] for e in entries)
    tot_u = sum(e["unprocessed"] for e in entries)
    tot_r = sum(e["rejected"] for e in entries)
    conn = _connect()
    try:
        ret_no = _next_return_number(conn)
        conn.execute(
            """INSERT INTO jo_returns(return_number, return_date, jo_id, jo_number, process,
                   so_number, vendor_name, processed_qty, unprocessed_qty, rejected_qty,
                   reason, remarks, returned_by)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ret_no, return_date, int(joid), jo.get("jo_number") or "", process, so_number,
                jo.get("vendor_name") or "", tot_p, tot_u, tot_r,
                str(data.get("reason") or ""), str(data.get("remarks") or ""), returned_by,
            ),
        )
        ret_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        for e in entries:
            conn.execute(
                """INSERT INTO jo_return_lines(return_id, jo_line_id, sku, processed_qty,
                       unprocessed_qty, rejected_qty, receipt_id)
                   VALUES(?,?,?,?,?,?,?)""",
                (ret_id, e["jo_line_id"], e["sku"], e["processed"], e["unprocessed"],
                 e["rejected"], e.get("receipt_id")),
            )
            rl_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            if not e["unprocessed"]:
                continue
            before = int((e["line"] or jo).get("planned_qty") or 0)
            _release_commitment(conn, joid=int(joid), line=e["line"], qty=e["unprocessed"])
            pdb._record_jo_qty_history(
                conn, int(joid), "planned_qty", before, before - e["unprocessed"],
                returned_by, f"Unprocessed return {ret_no} → Ready To {process}",
                jo_line_id=e["jo_line_id"],
            )
            conn.execute(
                """INSERT INTO jo_return_links(return_id, return_line_id, source_jo_id,
                       source_jo_number, source_vendor, process, so_number, sku,
                       unprocessed_qty, ready_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (ret_id, rl_id, int(joid), jo.get("jo_number") or "",
                 jo.get("vendor_name") or "", process, so_number,
                 e["sku"].upper(), e["unprocessed"], now),
            )
        if lines_by_id:
            conn.execute(
                """UPDATE job_orders SET planned_qty =
                       (SELECT COALESCE(SUM(planned_qty),0) FROM jo_lines WHERE jo_id=?)
                   WHERE id=?""",
                (int(joid), int(joid)),
            )
        hdr = dict(conn.execute(
            "SELECT planned_qty, received_qty, status FROM job_orders WHERE id=?", (int(joid),)
        ).fetchone())
        planned_now = int(hdr.get("planned_qty") or 0)
        received_now = int(hdr.get("received_qty") or 0)
        close = received_now >= planned_now and hdr.get("status") != "Closed"
        sets = ["balance_qty = MAX(0, ? - ?)", "reconciliation_status = ?"]
        vals: list[Any] = [planned_now, received_now, RECON_PENDING]
        if close:
            sets += ["status = 'Closed'", "completed_date = ?"]
            vals.append(datetime.now().strftime("%Y-%m-%d"))
        conn.execute(
            f"UPDATE job_orders SET {', '.join(sets)}, updated_at = datetime('now') WHERE id=?",
            (*vals, int(joid)),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    recon = _auto_complete_if_nothing_to_reconcile(int(joid))
    try:
        from ..db import document_audit_db as _audit

        _audit.enroll_document(
            "JO_RETURN",
            ret_id,
            doc_number=ret_no,
            module="production",
            so_reference=so_number,
            party_name=str(jo.get("vendor_name") or jo.get("jo_number") or ""),
            process_name=f"{process} return",
            doc_date=return_date,
            created_by=returned_by,
        )
    except Exception:
        pass
    return {
        "ok": True,
        "return_id": ret_id,
        "return_number": ret_no,
        "processed_qty": tot_p,
        "unprocessed_qty": tot_u,
        "rejected_qty": tot_r,
        "receipt_ids": [e["receipt_id"] for e in entries if e.get("receipt_id")],
        "jo_status": "Closed" if close else hdr.get("status"),
        "reconciliation_status": recon,
    }


# ── Reconciliation ────────────────────────────────────────────────────────────

def _saved_rows(conn, joid: int) -> dict[tuple[str, str], dict]:
    return {
        (r["material_kind"], r["material_code"]): dict(r)
        for r in conn.execute(
            "SELECT * FROM jo_material_reconciliation WHERE jo_id=?", (joid,)
        ).fetchall()
    }


def _pick(saved: Optional[dict], key: str, default: float) -> float:
    if saved and saved.get(key) is not None:
        return float(saved[key])
    return float(default)


def _row(kind, code, name, unit, *, issued, consumed, returned, wastage,
         expected=None, editable=(), saved=None) -> dict:
    balance = round(issued - consumed - returned - wastage, 4)
    settled = balance <= _EPS if kind == KIND_PIECES else abs(balance) <= _EPS
    return {
        "material_kind": kind,
        "material_code": code,
        "material_name": name,
        "unit": unit,
        "issued_qty": round(issued, 4),
        "consumed_qty": round(consumed, 4),
        "returned_qty": round(returned, 4),
        "wastage_qty": round(wastage, 4),
        "balance_qty": balance,
        "expected_consumption": None if expected is None else round(expected, 4),
        "settled": settled,
        "editable": list(editable),
        "remarks": (saved or {}).get("remarks") or "",
        "confirmed": saved is not None,
    }


def _build_rows(conn, jo: dict) -> list[dict]:
    joid = int(jo["id"])
    saved = _saved_rows(conn, joid)
    rows: list[dict] = []
    received_total = int(jo.get("received_qty") or 0)

    if str(jo.get("process") or "").strip().lower() != "cutting":
        lines = jo.get("lines") or []
        piece_srcs = lines or [jo]
        for src in piece_srcs:
            sku = str(src.get("sku") or jo.get("sku") or "").strip().upper() or "PIECES"
            returned = float(src.get("unprocessed_return_qty") or 0)
            issued = float(src.get("planned_qty") or 0) + returned
            consumed = float(src.get("received_qty") or 0)
            s = saved.get((KIND_PIECES, sku))
            rows.append(_row(
                KIND_PIECES, sku, f"Pieces {sku}", "PCS",
                issued=issued, consumed=consumed, returned=returned,
                wastage=_pick(s, "wastage_qty", 0), editable=("wastage_qty",), saved=s,
            ))

    fabric_issued: dict[str, dict] = {}
    for fi in jo.get("fabric_issues") or []:
        code = str(fi.get("fabric_code") or "").strip().upper()
        if not code:
            continue
        d = fabric_issued.setdefault(code, {
            "name": fi.get("fabric_name") or code, "unit": fi.get("unit") or "MTR", "qty": 0.0,
        })
        d["qty"] += float(fi.get("issued_qty") or 0)
    fabric_returned: dict[str, float] = {}
    for fr in jo.get("fabric_returns") or []:
        code = str(fr.get("fabric_code") or "").strip().upper()
        fabric_returned[code] = fabric_returned.get(code, 0.0) + float(fr.get("returned_qty") or 0)
    original_planned = float(jo.get("planned_qty") or 0) + float(jo.get("unprocessed_return_qty") or 0)
    for code, d in fabric_issued.items():
        s = saved.get((KIND_FABRIC, code))
        expected = None
        fq = float(jo.get("fabric_qty") or 0)
        if fq > 0 and original_planned > 0:
            expected = fq / original_planned * received_total
        rows.append(_row(
            KIND_FABRIC, code, d["name"], d["unit"],
            issued=d["qty"],
            consumed=_pick(s, "consumed_qty", expected or 0),
            returned=fabric_returned.get(code, 0.0),
            wastage=_pick(s, "wastage_qty", 0),
            expected=expected, editable=("consumed_qty", "wastage_qty"), saved=s,
        ))

    note = jo.get("issue_note") or {}
    acc: dict[str, dict] = {}
    for ln in note.get("lines") or []:
        code = str(ln.get("material_code") or "").strip().upper()
        if not code or code in fabric_issued:
            continue
        d = acc.setdefault(code, {
            "name": ln.get("material_name") or code, "unit": ln.get("unit") or "",
            "required": 0.0, "per_unit": 0.0,
        })
        d["required"] += float(ln.get("issued_qty") or 0) or float(ln.get("required_qty") or 0)
        d["per_unit"] += float(ln.get("bom_qty_per_unit") or 0)
    for code, d in acc.items():
        s = saved.get((KIND_ACCESSORY, code))
        expected = d["per_unit"] * received_total if d["per_unit"] > 0 else None
        rows.append(_row(
            KIND_ACCESSORY, code, d["name"], d["unit"],
            issued=_pick(s, "issued_qty", d["required"]),
            consumed=_pick(s, "consumed_qty", expected or 0),
            returned=_pick(s, "returned_qty", 0),
            wastage=_pick(s, "wastage_qty", 0),
            expected=expected,
            editable=("issued_qty", "consumed_qty", "returned_qty", "wastage_qty"),
            saved=s,
        ))
    return rows


def _load_jo(joid: int) -> dict:
    jo = _pdb().get_jo(int(joid))
    if not jo:
        raise ValueError("JO not found")
    return jo


def get_reconciliation(joid: int) -> dict:
    init_jo_return_tables()
    jo = _load_jo(joid)
    conn = _connect()
    try:
        rows = _build_rows(conn, jo)
    finally:
        conn.close()
    pending = [r for r in rows if not r["settled"]]
    return {
        "jo_id": int(joid),
        "jo_number": jo.get("jo_number"),
        "process": jo.get("process"),
        "vendor_name": jo.get("vendor_name") or "",
        "status": jo.get("status"),
        "reconciliation_status": jo.get("reconciliation_status") or "",
        "reconciled_at": jo.get("reconciled_at") or "",
        "reconciled_by": jo.get("reconciled_by") or "",
        "rows": rows,
        "pending_materials": [r["material_name"] for r in pending],
        "can_complete": not pending,
    }


def _needs_manual_reconciliation(rows: list[dict]) -> bool:
    return any(r["material_kind"] != KIND_PIECES or not r["settled"] for r in rows)


def _auto_complete_if_nothing_to_reconcile(joid: int) -> str:
    """Pieces-only JOs with zero balance have nothing for a person to confirm."""
    jo = _load_jo(joid)
    status = str(jo.get("reconciliation_status") or "")
    if status != RECON_PENDING:
        return status
    conn = _connect()
    try:
        rows = _build_rows(conn, jo)
        if _needs_manual_reconciliation(rows):
            return status
        conn.execute(
            """UPDATE job_orders SET reconciliation_status=?, reconciled_at=datetime('now'),
                   reconciled_by='system (nothing to reconcile)' WHERE id=?""",
            (RECON_COMPLETED, int(joid)),
        )
        conn.commit()
        return RECON_COMPLETED
    finally:
        conn.close()


def mark_reconciliation_required(joid: int) -> str:
    """Called when a JO closes: start reconciliation unless already started/done."""
    init_jo_return_tables()
    conn = _connect()
    try:
        conn.execute(
            """UPDATE job_orders SET reconciliation_status=?
               WHERE id=? AND IFNULL(reconciliation_status,'')=''""",
            (RECON_PENDING, int(joid)),
        )
        conn.commit()
    finally:
        conn.close()
    return _auto_complete_if_nothing_to_reconcile(int(joid))


def save_reconciliation(joid: int, data: dict) -> dict:
    """Save user-entered reconciliation values; ``complete=True`` requires zero balances."""
    init_jo_return_tables()
    _assert_editable(joid)
    jo = _load_jo(joid)
    if str(jo.get("status") or "") == "Cancelled":
        raise ValueError("Job order is cancelled — reconciliation is not allowed")
    user = str(data.get("reconciled_by") or "")
    conn = _connect()
    try:
        allowed = {
            (r["material_kind"], r["material_code"]): r for r in _build_rows(conn, jo)
        }
        for raw in data.get("rows") or []:
            key = (str(raw.get("material_kind") or "").upper(),
                   str(raw.get("material_code") or "").strip().upper())
            base = allowed.get(key)
            if base is None:
                raise ValueError(f"{key[1] or 'Material'} was not issued against this JO")
            vals = {
                f: _as_qty(raw.get(f), f.replace("_", " ").capitalize())
                for f in ("issued_qty", "consumed_qty", "returned_qty", "wastage_qty")
                if f in base["editable"]
            }
            cols = ["jo_id", "material_kind", "material_code", "material_name", "unit",
                    "remarks", "updated_by"]
            params: list[Any] = [int(joid), key[0], key[1], base["material_name"], base["unit"],
                                 str(raw.get("remarks") or ""), user]
            for f, v in vals.items():
                cols.append(f)
                params.append(v)
            updates = ", ".join(f"{c}=excluded.{c}" for c in cols[3:])
            conn.execute(
                f"""INSERT INTO jo_material_reconciliation({", ".join(cols)}, updated_at)
                    VALUES({", ".join("?" * len(cols))}, datetime('now'))
                    ON CONFLICT(jo_id, material_kind, material_code)
                    DO UPDATE SET {updates}, updated_at=datetime('now')""",
                params,
            )
        rows = _build_rows(conn, jo)
        complete = bool(data.get("complete"))
        if complete:
            bad = [r for r in rows if not r["settled"]]
            if bad:
                conn.rollback()
                detail = "; ".join(
                    f"{r['material_name']} balance {r['balance_qty']:g} {r['unit']}".strip()
                    for r in bad[:8]
                )
                raise ValueError(f"Reconciliation incomplete — balance with vendor remains: {detail}")
            conn.execute(
                """UPDATE job_orders SET reconciliation_status=?, reconciled_at=datetime('now'),
                       reconciled_by=?, updated_at=datetime('now') WHERE id=?""",
                (RECON_COMPLETED, user, int(joid)),
            )
            pieces = [r for r in rows if r["material_kind"] == KIND_PIECES]
            if pieces and str(jo.get("status") or "") != "Closed":
                conn.execute(
                    """UPDATE job_orders SET status='Closed', completed_date=?,
                           updated_at=datetime('now') WHERE id=?""",
                    (datetime.now().strftime("%Y-%m-%d"), int(joid)),
                )
        else:
            conn.execute(
                """UPDATE job_orders SET reconciliation_status=?
                   WHERE id=? AND IFNULL(reconciliation_status,'')=''""",
                (RECON_PENDING, int(joid)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_reconciliation(joid)


# ── Traceability ──────────────────────────────────────────────────────────────

def allocate_new_jo(joid: int) -> list[dict]:
    """Link a newly created JO to earlier unprocessed returns for the same SO/SKU/process (FIFO)."""
    init_jo_return_tables()
    conn = _connect()
    out: list[dict] = []
    try:
        jo = conn.execute("SELECT * FROM job_orders WHERE id=?", (int(joid),)).fetchone()
        if not jo:
            return out
        jo = dict(jo)
        lines = [dict(r) for r in conn.execute(
            "SELECT sku, planned_qty FROM jo_lines WHERE jo_id=?", (int(joid),)
        ).fetchall()] or [{"sku": jo.get("sku"), "planned_qty": jo.get("planned_qty")}]
        for ln in lines:
            remaining = int(ln.get("planned_qty") or 0)
            sku = str(ln.get("sku") or "").strip().upper()
            if remaining <= 0 or not sku:
                continue
            links = conn.execute(
                """SELECT id, unprocessed_qty, allocated_qty FROM jo_return_links
                   WHERE UPPER(TRIM(process))=UPPER(TRIM(?)) AND so_number=? AND sku=?
                     AND source_jo_id != ? AND unprocessed_qty > allocated_qty
                   ORDER BY id""",
                (jo.get("process") or "", jo.get("so_number") or "", sku, int(joid)),
            ).fetchall()
            for lk in links:
                if remaining <= 0:
                    break
                take = min(remaining, int(lk["unprocessed_qty"]) - int(lk["allocated_qty"]))
                conn.execute(
                    """INSERT INTO jo_return_allocations(link_id, new_jo_id, new_jo_number,
                           new_vendor, qty) VALUES(?,?,?,?,?)""",
                    (int(lk["id"]), int(joid), jo.get("jo_number") or "",
                     jo.get("vendor_name") or "", take),
                )
                conn.execute(
                    "UPDATE jo_return_links SET allocated_qty = allocated_qty + ? WHERE id=?",
                    (take, int(lk["id"])),
                )
                remaining -= take
                out.append({"link_id": int(lk["id"]), "qty": take, "sku": sku})
        conn.commit()
    finally:
        conn.close()
    return out


def release_allocations(joid: int) -> None:
    """A cancelled follow-up JO hands its returned qty back to the open pool."""
    init_jo_return_tables()
    conn = _connect()
    try:
        for a in conn.execute(
            "SELECT id, link_id, qty FROM jo_return_allocations WHERE new_jo_id=? AND released=0",
            (int(joid),),
        ).fetchall():
            conn.execute(
                "UPDATE jo_return_links SET allocated_qty = MAX(0, allocated_qty - ?) WHERE id=?",
                (int(a["qty"]), int(a["link_id"])),
            )
            conn.execute("UPDATE jo_return_allocations SET released=1 WHERE id=?", (int(a["id"]),))
        conn.commit()
    finally:
        conn.close()


def _returns_with_lines(conn, where: str, params: tuple) -> list[dict]:
    rets = [dict(r) for r in conn.execute(
        f"SELECT * FROM jo_returns WHERE {where} ORDER BY id DESC", params
    ).fetchall()]
    for r in rets:
        r["lines"] = [dict(x) for x in conn.execute(
            "SELECT * FROM jo_return_lines WHERE return_id=? ORDER BY id", (r["id"],)
        ).fetchall()]
    return rets


def get_return_history(joid: int) -> dict:
    """Original JO → returns → Ready-To → follow-up JOs (and where this JO's qty came from)."""
    init_jo_return_tables()
    jo = _load_jo(joid)
    conn = _connect()
    try:
        returns = _returns_with_lines(conn, "jo_id=?", (int(joid),))
        outgoing = []
        for lk in conn.execute(
            """SELECT l.*, r.return_number, r.return_date FROM jo_return_links l
               JOIN jo_returns r ON r.id = l.return_id
               WHERE l.source_jo_id=? ORDER BY l.id""",
            (int(joid),),
        ).fetchall():
            d = dict(lk)
            d["allocations"] = [dict(a) for a in conn.execute(
                "SELECT * FROM jo_return_allocations WHERE link_id=? ORDER BY id", (d["id"],)
            ).fetchall()]
            d["pending_in_ready_qty"] = max(0, int(d["unprocessed_qty"]) - int(d["allocated_qty"]))
            outgoing.append(d)
        incoming = [dict(a) for a in conn.execute(
            """SELECT a.id, a.qty, a.released, a.created_at AS allocated_at,
                      l.source_jo_id, l.source_jo_number, l.source_vendor, l.process,
                      l.so_number, l.sku, l.ready_at, r.return_number, r.return_date
               FROM jo_return_allocations a
               JOIN jo_return_links l ON l.id = a.link_id
               JOIN jo_returns r ON r.id = l.return_id
               WHERE a.new_jo_id=? ORDER BY a.id""",
            (int(joid),),
        ).fetchall()]
    finally:
        conn.close()
    unprocessed = int(jo.get("unprocessed_return_qty") or 0)
    return {
        "jo_id": int(joid),
        "jo_number": jo.get("jo_number"),
        "vendor_name": jo.get("vendor_name") or "",
        "process": jo.get("process"),
        "summary": {
            "original_qty": int(jo.get("planned_qty") or 0) + unprocessed,
            "processed_qty": int(jo.get("received_qty") or 0),
            "unprocessed_returned_qty": unprocessed,
            "reallocated_qty": sum(
                int(a["qty"]) for lk in outgoing for a in lk["allocations"] if not a["released"]
            ),
            "pending_in_ready_qty": sum(lk["pending_in_ready_qty"] for lk in outgoing),
        },
        "returns": returns,
        "outgoing": outgoing,
        "incoming": incoming,
    }


def list_returns(
    *,
    so_number: Optional[str] = None,
    process: Optional[str] = None,
    vendor: Optional[str] = None,
    limit: int = 500,
) -> list[dict]:
    init_jo_return_tables()
    where, params = ["1=1"], []
    if so_number:
        where.append("so_number=?")
        params.append(so_number)
    if process:
        where.append("process=?")
        params.append(process)
    if vendor:
        where.append("vendor_name LIKE ? COLLATE NOCASE")
        params.append(f"%{vendor}%")
    conn = _connect()
    try:
        rets = _returns_with_lines(conn, " AND ".join(where), tuple(params))[: max(1, int(limit))]
        for r in rets:
            r["links"] = []
            for lk in conn.execute(
                "SELECT * FROM jo_return_links WHERE return_id=? ORDER BY id", (r["id"],)
            ).fetchall():
                d = dict(lk)
                d["allocations"] = [dict(a) for a in conn.execute(
                    "SELECT new_jo_id, new_jo_number, new_vendor, qty, released, created_at "
                    "FROM jo_return_allocations WHERE link_id=? ORDER BY id", (d["id"],)
                ).fetchall()]
                r["links"].append(d)
        return rets
    finally:
        conn.close()
