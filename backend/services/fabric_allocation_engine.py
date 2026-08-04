"""Grey Fabric Planning, Allocation & Reallocation engine.

Hierarchy (planning truth):
  Grey Fabric → Printed Fabric (P-Code / SFG) → FG SKU → Sales Order

Stages (never merged):
  1 GREY_ALLOCATED — free grey → reserved against a Printed Fabric (+ optional SO/FG intent)
  2 PRINTING_JO — optional link via jwo_ref
  3 PF_RECEIVED — checked printed stock
  4 PF_ALLOCATED — printed meters reserved to FG+SO (printed_fabric_reservations)
  5 CUTTING_ISSUED — lock; reallocation forbidden

Core principle: Grey is planned/printed against P-code, not FG.
Final FG status uses *current* printed allocation until Cutting Issue locks it.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from ..db import grey_db as gdb

# ── stages / event types ─────────────────────────────────────────────────────

STAGE_GREY = "GREY_ALLOCATED"
STAGE_PRINT_JO = "PRINTING_JO"
STAGE_PF_RECEIVED = "PF_RECEIVED"
STAGE_PF_ALLOC = "RESERVED"
STAGE_JO_CREATED = "JO_CREATED"
STAGE_CUT_LOCKED = "CUTTING_ISSUED"

EVENT_GREY_ALLOC = "GREY_ALLOCATE"
EVENT_GREY_RELEASE = "GREY_RELEASE"
EVENT_PF_ALLOC = "PF_ALLOCATE"
EVENT_PF_REALLOC = "PF_REALLOCATE"
EVENT_PF_RELEASE = "PF_RELEASE"
EVENT_PF_LOCK = "PF_LOCK_CUTTING"
EVENT_PF_STATUS = "PF_STATUS"


class FabricAllocationError(ValueError):
    """Business-rule violation (allocation / reallocation / lock)."""


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _conn():
    return gdb._connect()


def _audit(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    entity_type: str = "",
    entity_id: int | None = None,
    grey_code: str = "",
    printed_code: str = "",
    from_so: str = "",
    from_sku: str = "",
    to_so: str = "",
    to_sku: str = "",
    qty: float = 0,
    old_status: str = "",
    new_status: str = "",
    user_name: str = "",
    reason: str = "",
    document_ref: str = "",
) -> None:
    conn.execute(
        """INSERT INTO fabric_allocation_history(
            event_type, entity_type, entity_id, grey_code, printed_code,
            from_so, from_sku, to_so, to_sku, qty, old_status, new_status,
            user_name, reason, document_ref, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            event_type,
            entity_type,
            entity_id,
            grey_code,
            printed_code,
            from_so,
            from_sku,
            to_so,
            to_sku,
            float(qty or 0),
            old_status,
            new_status,
            user_name,
            reason,
            document_ref,
            _now(),
        ),
    )


# ── stock helpers ────────────────────────────────────────────────────────────

def grey_stock_snapshot(grey_code: str | None = None) -> list[dict[str, Any]]:
    """Free vs allocated grey meters by fabric code (checked grey warehouse)."""
    conn = _conn()
    try:
        q = """SELECT fabric_code, fabric_name,
                      COALESCE(passed_qty, 0) AS passed_qty,
                      COALESCE(reserved_qty, 0) AS reserved_qty,
                      COALESCE(available_qty, 0) AS available_qty
               FROM fabric_checked_stock"""
        params: tuple = ()
        if grey_code:
            q += " WHERE TRIM(fabric_code)=?"
            params = (grey_code.strip(),)
        q += " ORDER BY fabric_code"
        rows = [dict(r) for r in conn.execute(q, params).fetchall()]
        # Also sum tracker pipeline free if no checked stock
        if not rows:
            tq = """SELECT material_code AS fabric_code,
                           MAX(material_name) AS fabric_name,
                           SUM(COALESCE(passed_qty, checked_qty, factory_qty, 0)) AS available_qty
                    FROM grey_tracker WHERE 1=1"""
            if grey_code:
                tq += " AND TRIM(material_code)=?"
            tq += " GROUP BY material_code"
            rows = []
            for r in conn.execute(tq, params).fetchall():
                d = dict(r)
                d["passed_qty"] = float(d.get("available_qty") or 0)
                d["reserved_qty"] = 0.0
                d["available_qty"] = float(d.get("available_qty") or 0)
                rows.append(d)
        # Overlay grey_fabric_allocations Active totals as "allocated"
        for d in rows:
            code = (d.get("fabric_code") or "").strip()
            alloc = conn.execute(
                """SELECT COALESCE(SUM(qty),0) FROM grey_fabric_allocations
                   WHERE status='Active' AND TRIM(grey_code)=?""",
                (code,),
            ).fetchone()
            allocated = float(alloc[0] if alloc else 0)
            free = float(d.get("available_qty") or 0)
            d["grey_allocated_qty"] = allocated
            d["grey_free_qty"] = max(0.0, free)  # free is warehouse free after hard reserve
            d["status_color"] = (
                "green" if allocated > 0 and free <= 0.001 else
                "red" if free <= 0.001 and allocated <= 0 else
                "orange" if allocated > 0 else
                "grey"
            )
        return rows
    finally:
        conn.close()


def _grey_free_qty(conn: sqlite3.Connection, grey_code: str) -> float:
    row = conn.execute(
        "SELECT available_qty FROM fabric_checked_stock WHERE TRIM(fabric_code)=?",
        (grey_code.strip(),),
    ).fetchone()
    if row:
        return max(0.0, float(row["available_qty"] or 0))
    # Fallback: factory/checked pipeline not yet in checked stock table
    tr = conn.execute(
        """SELECT SUM(COALESCE(passed_qty, checked_qty, factory_qty, 0))
           FROM grey_tracker WHERE TRIM(material_code)=?""",
        (grey_code.strip(),),
    ).fetchone()
    return max(0.0, float(tr[0] if tr and tr[0] is not None else 0))


def _is_pf_locked(conn: sqlite3.Connection, reservation_id: int) -> bool:
    row = conn.execute(
        "SELECT status, stage, cutting_issued_qty FROM printed_fabric_reservations WHERE id=?",
        (reservation_id,),
    ).fetchone()
    if not row:
        return False
    st = (row["status"] or "").strip()
    stage = (row["stage"] or "").strip()
    if st in ("Issued", "Consumed", STAGE_CUT_LOCKED) or stage == STAGE_CUT_LOCKED:
        return True
    if float(row["cutting_issued_qty"] or 0) > 0:
        return True
    return False


def _pf_reservation_locked_for_so_sku(
    conn: sqlite3.Connection, so_number: str, sku: str, printed_code: str | None = None
) -> bool:
    q = """SELECT id FROM printed_fabric_reservations
           WHERE TRIM(so_number)=? AND TRIM(sku)=?
             AND (
               status IN ('Issued','Consumed','CUTTING_ISSUED')
               OR stage = 'CUTTING_ISSUED'
               OR COALESCE(cutting_issued_qty,0) > 0
             )"""
    params: list[Any] = [so_number.strip(), sku.strip()]
    if printed_code:
        q += " AND TRIM(fabric_code)=?"
        params.append(printed_code.strip())
    return conn.execute(q, tuple(params)).fetchone() is not None


# ── Grey allocation ──────────────────────────────────────────────────────────

def allocate_grey(data: dict[str, Any]) -> dict[str, Any]:
    """Stage 1: allocate free grey meters to a Printed Fabric (P-code).

    Optional so_number / fg_sku is *intent only* until printed fabric is allocated.
    """
    grey_code = (data.get("grey_code") or data.get("fabric_code") or "").strip()
    printed_code = (data.get("printed_code") or data.get("p_code") or "").strip()
    qty = float(data.get("qty") or 0)
    so_number = (data.get("so_number") or "").strip()
    fg_sku = (data.get("fg_sku") or data.get("sku") or "").strip()
    user_name = (data.get("user_name") or data.get("created_by") or "").strip()
    reason = (data.get("reason") or data.get("remarks") or "").strip()

    if not grey_code:
        raise FabricAllocationError("grey_code is required")
    if not printed_code:
        raise FabricAllocationError("printed_code (P-code / SFG) is required")
    if qty <= 0:
        raise FabricAllocationError("qty must be greater than 0")

    conn = _conn()
    try:
        # Serialize concurrent allocators on the same DB connection batch
        conn.execute("BEGIN IMMEDIATE")
        free = _grey_free_qty(conn, grey_code)
        if qty > free + 0.001:
            raise FabricAllocationError(
                f"Cannot allocate {qty} m — only {free:.1f} m free grey for {grey_code}"
            )
        grey_name = (data.get("grey_name") or "").strip()
        printed_name = (data.get("printed_name") or "").strip()
        row = conn.execute(
            "SELECT fabric_name FROM fabric_checked_stock WHERE TRIM(fabric_code)=?",
            (grey_code,),
        ).fetchone()
        if row and not grey_name:
            grey_name = row["fabric_name"] or grey_code
        # Move free → reserved on grey checked stock (if present)
        if row:
            cur = conn.execute(
                """UPDATE fabric_checked_stock
                   SET reserved_qty = COALESCE(reserved_qty,0) + ?,
                       available_qty = available_qty - ?
                   WHERE TRIM(fabric_code)=? AND available_qty >= ?""",
                (qty, qty, grey_code, qty),
            )
            if cur.rowcount != 1:
                raise FabricAllocationError(
                    f"Cannot allocate {qty} m — free grey changed (concurrent allocation)"
                )
        cur = conn.execute(
            """INSERT INTO grey_fabric_allocations(
                grey_code, grey_name, printed_code, printed_name, so_number, fg_sku,
                qty, unit, stage, status, jwo_ref, tracker_id, created_by, remarks, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                grey_code,
                grey_name,
                printed_code,
                printed_name or printed_code,
                so_number,
                fg_sku,
                qty,
                (data.get("unit") or "MTR"),
                STAGE_GREY,
                "Active",
                (data.get("jwo_ref") or ""),
                data.get("tracker_id"),
                user_name,
                reason,
                _now(),
            ),
        )
        alloc_id = int(cur.lastrowid)
        # Legacy hard_reservations row for MRP/availability compatibility
        conn.execute(
            """INSERT INTO hard_reservations(fabric_code, fabric_name, so_number, sku, qty, unit, status, remarks)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                grey_code,
                grey_name,
                so_number or f"P:{printed_code}",
                fg_sku or printed_code,
                qty,
                "MTR",
                "Active",
                f"grey_alloc:{alloc_id}→{printed_code}",
            ),
        )
        _audit(
            conn,
            event_type=EVENT_GREY_ALLOC,
            entity_type="grey_allocation",
            entity_id=alloc_id,
            grey_code=grey_code,
            printed_code=printed_code,
            to_so=so_number,
            to_sku=fg_sku,
            qty=qty,
            new_status="Active",
            user_name=user_name,
            reason=reason,
            document_ref=data.get("document_ref") or "",
        )
        conn.commit()
        return {
            "ok": True,
            "allocation_id": alloc_id,
            "grey_code": grey_code,
            "printed_code": printed_code,
            "qty": qty,
            "stage": STAGE_GREY,
            "free_remaining": max(0.0, free - qty),
        }
    except FabricAllocationError:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def release_grey_allocation(
    allocation_id: int,
    *,
    user_name: str = "",
    reason: str = "",
) -> dict[str, Any]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM grey_fabric_allocations WHERE id=?", (allocation_id,)
        ).fetchone()
        if not row:
            raise FabricAllocationError("Grey allocation not found")
        if (row["status"] or "") != "Active":
            raise FabricAllocationError(f"Allocation status is {row['status']}, cannot release")
        qty = float(row["qty"] or 0)
        grey_code = (row["grey_code"] or "").strip()
        conn.execute(
            "UPDATE grey_fabric_allocations SET status='Released' WHERE id=?",
            (allocation_id,),
        )
        conn.execute(
            """UPDATE fabric_checked_stock
               SET reserved_qty = MAX(0, COALESCE(reserved_qty,0) - ?),
                   available_qty = COALESCE(available_qty,0) + ?
               WHERE TRIM(fabric_code)=?""",
            (qty, qty, grey_code),
        )
        conn.execute(
            """UPDATE hard_reservations SET status='Released'
               WHERE status='Active' AND remarks LIKE ?""",
            (f"grey_alloc:{allocation_id}%",),
        )
        _audit(
            conn,
            event_type=EVENT_GREY_RELEASE,
            entity_type="grey_allocation",
            entity_id=allocation_id,
            grey_code=grey_code,
            printed_code=row["printed_code"] or "",
            from_so=row["so_number"] or "",
            from_sku=row["fg_sku"] or "",
            qty=qty,
            old_status="Active",
            new_status="Released",
            user_name=user_name,
            reason=reason,
        )
        conn.commit()
        return {"ok": True, "allocation_id": allocation_id, "released_qty": qty}
    except FabricAllocationError:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Printed fabric allocation / reallocation ───────────────────────────────

def allocate_printed(data: dict[str, Any]) -> dict[str, Any]:
    """Stage 4: reserve checked printed fabric to FG SKU + SO."""
    # Reuse validated path then audit
    payload = {
        "fabric_code": (data.get("printed_code") or data.get("fabric_code") or "").strip(),
        "fabric_name": data.get("printed_name") or data.get("fabric_name") or "",
        "so_number": (data.get("so_number") or "").strip(),
        "sku": (data.get("fg_sku") or data.get("sku") or "").strip(),
        "qty": float(data.get("qty") or 0),
        "remarks": data.get("remarks") or data.get("reason") or "",
    }
    gdb.reserve_printed_fabric(payload)
    conn = _conn()
    try:
        row = conn.execute(
            """SELECT id FROM printed_fabric_reservations
               WHERE status='Active' AND TRIM(fabric_code)=? AND TRIM(so_number)=? AND TRIM(sku)=?
               ORDER BY id DESC LIMIT 1""",
            (payload["fabric_code"], payload["so_number"], payload["sku"]),
        ).fetchone()
        rid = int(row["id"]) if row else None
        if rid:
            conn.execute(
                "UPDATE printed_fabric_reservations SET stage=? WHERE id=?",
                (STAGE_PF_ALLOC, rid),
            )
            _audit(
                conn,
                event_type=EVENT_PF_ALLOC,
                entity_type="printed_reservation",
                entity_id=rid,
                printed_code=payload["fabric_code"],
                to_so=payload["so_number"],
                to_sku=payload["sku"],
                qty=payload["qty"],
                new_status="Active",
                user_name=data.get("user_name") or "",
                reason=payload["remarks"],
            )
            conn.commit()
        return {"ok": True, "reservation_id": rid, "stage": STAGE_PF_ALLOC, **payload}
    finally:
        conn.close()


def reallocate_printed(data: dict[str, Any]) -> dict[str, Any]:
    """Move printed allocation from one FG/SO to another without touching grey/print history.

    Forbidden after Cutting Issue (lock).
    """
    reservation_id = data.get("reservation_id") or data.get("from_reservation_id")
    from_so = (data.get("from_so") or "").strip()
    from_sku = (data.get("from_sku") or "").strip()
    to_so = (data.get("to_so") or data.get("so_number") or "").strip()
    to_sku = (data.get("to_sku") or data.get("fg_sku") or data.get("sku") or "").strip()
    qty = data.get("qty")
    printed_code = (data.get("printed_code") or data.get("fabric_code") or "").strip()
    user_name = (data.get("user_name") or "").strip()
    reason = (data.get("reason") or data.get("remarks") or "").strip()
    if not reason:
        raise FabricAllocationError("reason is required for reallocation audit trail")
    if not to_so or not to_sku:
        raise FabricAllocationError("to_so and to_sku (FG) are required")

    conn = _conn()
    try:
        row = None
        if reservation_id:
            row = conn.execute(
                "SELECT * FROM printed_fabric_reservations WHERE id=?",
                (int(reservation_id),),
            ).fetchone()
        elif from_so and from_sku:
            q = """SELECT * FROM printed_fabric_reservations
                   WHERE status='Active' AND TRIM(so_number)=? AND TRIM(sku)=?"""
            params: list[Any] = [from_so, from_sku]
            if printed_code:
                q += " AND TRIM(fabric_code)=?"
                params.append(printed_code)
            q += " ORDER BY id DESC LIMIT 1"
            row = conn.execute(q, tuple(params)).fetchone()
        if not row:
            raise FabricAllocationError("Source printed allocation not found (must be Active)")
        rid = int(row["id"])
        if _is_pf_locked(conn, rid):
            raise FabricAllocationError(
                "Cannot reallocate: printed fabric already issued to Cutting (locked)"
            )
        fabric_code = (row["fabric_code"] or "").strip()
        fabric_name = row["fabric_name"] or fabric_code
        src_qty = float(row["qty"] or 0)
        move_qty = float(qty) if qty is not None else src_qty
        if move_qty <= 0:
            raise FabricAllocationError("qty must be greater than 0")
        if move_qty > src_qty + 0.001:
            raise FabricAllocationError(
                f"Cannot move {move_qty} m — source allocation only has {src_qty:.1f} m"
            )
        # Destination must not already be locked
        if _pf_reservation_locked_for_so_sku(conn, to_so, to_sku, fabric_code):
            raise FabricAllocationError(
                f"Destination {to_sku} on {to_so} is locked after cutting"
            )
        # Destination Active check — allow merge into existing Active same fabric
        dest = conn.execute(
            """SELECT * FROM printed_fabric_reservations
               WHERE status='Active' AND TRIM(fabric_code)=? AND TRIM(so_number)=? AND TRIM(sku)=?
               LIMIT 1""",
            (fabric_code, to_so, to_sku),
        ).fetchone()

        old_so, old_sku = row["so_number"] or "", row["sku"] or ""
        # Reduce or close source
        if abs(move_qty - src_qty) < 0.001:
            conn.execute(
                """UPDATE printed_fabric_reservations
                   SET status='Released', stage='REALLOCATED'
                   WHERE id=?""",
                (rid,),
            )
        else:
            conn.execute(
                "UPDATE printed_fabric_reservations SET qty = qty - ? WHERE id=?",
                (move_qty, rid),
            )
        # Note: reserved_qty on stock already holds total reserved; reallocation
        # between SKUs does not change warehouse reserved total when full move.
        # Partial leave source reserved, dest needs +reserved from free.
        if dest:
            conn.execute(
                "UPDATE printed_fabric_reservations SET qty = qty + ? WHERE id=?",
                (move_qty, int(dest["id"])),
            )
            dest_id = int(dest["id"])
        else:
            # Full reallocation of reserved qty: stock reserved_qty unchanged.
            # Create Active dest reservation.
            cur = conn.execute(
                """INSERT INTO printed_fabric_reservations(
                    fabric_code, fabric_name, so_number, sku, qty, unit, status, remarks, stage
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    fabric_code,
                    fabric_name,
                    to_so,
                    to_sku,
                    move_qty,
                    "MTR",
                    "Active",
                    f"reallocated from {old_sku}/{old_so}: {reason}",
                    STAGE_PF_ALLOC,
                ),
            )
            dest_id = int(cur.lastrowid)
            # reserved_qty already counted for source; no warehouse delta on reallocation

        _audit(
            conn,
            event_type=EVENT_PF_REALLOC,
            entity_type="printed_reservation",
            entity_id=dest_id,
            printed_code=fabric_code,
            from_so=old_so,
            from_sku=old_sku,
            to_so=to_so,
            to_sku=to_sku,
            qty=move_qty,
            old_status="Active",
            new_status="Active",
            user_name=user_name,
            reason=reason,
            document_ref=data.get("document_ref") or f"src:{rid}",
        )
        conn.commit()
        return {
            "ok": True,
            "from_reservation_id": rid,
            "to_reservation_id": dest_id,
            "printed_code": fabric_code,
            "qty": move_qty,
            "from": {"so_number": old_so, "sku": old_sku},
            "to": {"so_number": to_so, "sku": to_sku},
        }
    except FabricAllocationError:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def release_printed_allocation(
    reservation_id: int,
    *,
    user_name: str = "",
    reason: str = "",
) -> dict[str, Any]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM printed_fabric_reservations WHERE id=?", (reservation_id,)
        ).fetchone()
        if not row:
            raise FabricAllocationError("Printed reservation not found")
        if _is_pf_locked(conn, reservation_id):
            raise FabricAllocationError(
                "Cannot release: printed fabric already issued to Cutting (locked)"
            )
        if (row["status"] or "") != "Active":
            raise FabricAllocationError(f"Status is {row['status']}, cannot release")
        qty = float(row["qty"] or 0)
        fabric_code = (row["fabric_code"] or "").strip()
        conn.execute(
            """UPDATE printed_fabric_reservations SET status='Released', stage='RELEASED'
               WHERE id=?""",
            (reservation_id,),
        )
        conn.execute(
            """UPDATE printed_fabric_checked_stock
               SET reserved_qty = MAX(0, COALESCE(reserved_qty,0) - ?),
                   available_qty = COALESCE(available_qty,0) + ?
               WHERE TRIM(fabric_code)=?""",
            (qty, qty, fabric_code),
        )
        _audit(
            conn,
            event_type=EVENT_PF_RELEASE,
            entity_type="printed_reservation",
            entity_id=reservation_id,
            printed_code=fabric_code,
            from_so=row["so_number"] or "",
            from_sku=row["sku"] or "",
            qty=qty,
            old_status="Active",
            new_status="Released",
            user_name=user_name,
            reason=reason,
        )
        conn.commit()
        return {"ok": True, "reservation_id": reservation_id, "released_qty": qty}
    except FabricAllocationError:
        conn.rollback()
        raise
    finally:
        conn.close()


def lock_printed_on_cutting_issue(
    *,
    jo_id: int,
    fabric_code: str,
    so_number: str = "",
    sku: str = "",
    issued_qty: float = 0,
    user_name: str = "",
) -> dict[str, Any]:
    """Stage 5: mark printed allocations locked when fabric is issued to Cutting."""
    fabric_code = (fabric_code or "").strip()
    so_number = (so_number or "").strip()
    sku = (sku or "").strip()
    issued_qty = float(issued_qty or 0)
    if not fabric_code or issued_qty <= 0:
        return {"ok": False, "message": "skip"}

    conn = _conn()
    try:
        q = """SELECT * FROM printed_fabric_reservations
               WHERE TRIM(fabric_code)=? AND status IN ('Active','JO Created')"""
        params: list[Any] = [fabric_code]
        if so_number:
            q += " AND TRIM(so_number)=?"
            params.append(so_number)
        if sku:
            q += " AND TRIM(sku)=?"
            params.append(sku)
        rows = conn.execute(q, tuple(params)).fetchall()
        locked_ids = []
        remaining = issued_qty
        for row in rows:
            rid = int(row["id"])
            # Deduct reserved warehouse first, then any free if legacy issue path
            r_qty = float(row["qty"] or 0)
            deduct = min(remaining, r_qty) if remaining > 0 else 0
            conn.execute(
                """UPDATE printed_fabric_reservations SET
                    status='CUTTING_ISSUED',
                    stage='CUTTING_ISSUED',
                    locked_at=?,
                    locked_reason='Cutting issue',
                    jo_id=COALESCE(jo_id, ?),
                    cutting_issued_qty = COALESCE(cutting_issued_qty,0) + ?
                   WHERE id=?""",
                (_now(), jo_id, deduct or issued_qty, rid),
            )
            if deduct > 0:
                conn.execute(
                    """UPDATE printed_fabric_checked_stock
                       SET reserved_qty = MAX(0, COALESCE(reserved_qty,0) - ?)
                       WHERE TRIM(fabric_code)=?""",
                    (deduct, fabric_code),
                )
                remaining = max(0.0, remaining - deduct)
            _audit(
                conn,
                event_type=EVENT_PF_LOCK,
                entity_type="printed_reservation",
                entity_id=rid,
                printed_code=fabric_code,
                from_so=row["so_number"] or "",
                from_sku=row["sku"] or "",
                qty=deduct or issued_qty,
                old_status=row["status"] or "",
                new_status="CUTTING_ISSUED",
                user_name=user_name,
                reason="Cutting issue",
                document_ref=f"JO:{jo_id}",
            )
            locked_ids.append(rid)
        conn.commit()
        return {"ok": True, "locked_reservation_ids": locked_ids, "jo_id": jo_id}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Planning tree (MRP hierarchy) ────────────────────────────────────────────

def _bom_children(parent_code: str) -> list[dict[str, Any]]:
    try:
        from ..db import item_db

        item = item_db.get_item_by_code(parent_code)
        if not item:
            return []
        bom = item_db.get_default_bom(item["id"]) if hasattr(item_db, "get_default_bom") else None
        if not bom:
            # try list bom lines
            if hasattr(item_db, "get_bom_for_item"):
                bom = item_db.get_bom_for_item(item["id"])
        lines = []
        if isinstance(bom, dict):
            lines = bom.get("lines") or bom.get("items") or []
        elif isinstance(bom, list):
            lines = bom
        out = []
        for ln in lines:
            code = (ln.get("component_code") or ln.get("item_code") or ln.get("material_code") or "").strip()
            if not code:
                continue
            out.append(
                {
                    "code": code,
                    "name": ln.get("component_name") or ln.get("item_name") or ln.get("name") or code,
                    "qty_per": float(ln.get("qty") or ln.get("qty_per") or ln.get("quantity") or 1),
                    "type": (ln.get("item_type") or ln.get("type") or "").upper(),
                }
            )
        return out
    except Exception:
        return []


def build_planning_tree(*, so_numbers: list[str] | None = None) -> dict[str, Any]:
    """Expand Grey → Printed (SFG) → FG → SO with allocation status colors."""
    try:
        from ..db import sales_db

        orders = sales_db.list_orders() or []
    except Exception:
        orders = []
    if so_numbers:
        want = {s.strip() for s in so_numbers if s and str(s).strip()}
        orders = [o for o in orders if (o.get("so_number") or "").strip() in want]

    # Collect FG SKUs from open SOs
    fg_lines: list[dict[str, Any]] = []
    for so in orders:
        status = (so.get("status") or "").strip()
        if status in ("Closed", "Cancelled"):
            continue
        so_n = (so.get("so_number") or "").strip()
        for ln in so.get("lines") or []:
            sku = (ln.get("sku") or "").strip()
            if not sku:
                continue
            fg_lines.append(
                {
                    "so_number": so_n,
                    "buyer": so.get("buyer") or "",
                    "sku": sku,
                    "sku_name": ln.get("sku_name") or "",
                    "qty": float(ln.get("qty") or 0),
                }
            )

    # Map SKU → printed SFG → grey via BOM (1-level and 2-level)
    tree: dict[str, Any] = {}  # grey_code → node

    def ensure_grey(code: str, name: str = "") -> dict:
        if code not in tree:
            tree[code] = {
                "type": "grey",
                "code": code,
                "name": name or code,
                "children": {},  # printed_code → node
                "color": "grey",
            }
        return tree[code]

    def ensure_printed(gnode: dict, code: str, name: str = "") -> dict:
        ch = gnode["children"]
        if code not in ch:
            ch[code] = {
                "type": "printed",
                "code": code,
                "name": name or code,
                "children": {},  # sku → node
                "color": "blue",
            }
        return ch[code]

    for fl in fg_lines:
        sku = fl["sku"]
        children = _bom_children(sku)
        printed_items = [c for c in children if c.get("type") in ("SFG", "PRINTED", "") or c["code"].upper().startswith("P")]
        if not printed_items:
            # treat first BOM line as printed fabric proxy
            printed_items = children[:1] if children else [{"code": "UNMAPPED-PF", "name": "Unmapped Printed", "qty_per": 1, "type": "SFG"}]
        for pi in printed_items:
            greys = _bom_children(pi["code"])
            grey_items = [c for c in greys if c.get("type") in ("GF", "RM", "")] or greys
            if not grey_items:
                grey_items = [{"code": "UNMAPPED-GREY", "name": "Unmapped Grey", "qty_per": 1, "type": "GF"}]
            for gi in grey_items:
                gnode = ensure_grey(gi["code"], gi.get("name") or "")
                pnode = ensure_printed(gnode, pi["code"], pi.get("name") or "")
                key = f"{fl['sku']}::{fl['so_number']}"
                if key not in pnode["children"]:
                    pnode["children"][key] = {
                        "type": "fg_so",
                        "sku": fl["sku"],
                        "sku_name": fl["sku_name"],
                        "so_number": fl["so_number"],
                        "buyer": fl["buyer"],
                        "so_qty": fl["qty"],
                        "req_printed": float(fl["qty"]) * float(pi.get("qty_per") or 1),
                        "req_grey": float(fl["qty"]) * float(pi.get("qty_per") or 1) * float(gi.get("qty_per") or 1),
                        "color": "grey",
                        "allocated_printed": 0.0,
                        "allocated_grey": 0.0,
                        "status": "Pending",
                    }

    conn = _conn()
    try:
        # annotate allocations
        for g_code, gnode in tree.items():
            for p_code, pnode in gnode["children"].items():
                for _k, leaf in pnode["children"].items():
                    so, sku = leaf["so_number"], leaf["sku"]
                    r = conn.execute(
                        """SELECT COALESCE(SUM(qty),0) FROM printed_fabric_reservations
                           WHERE status IN ('Active','JO Created','CUTTING_ISSUED')
                             AND TRIM(fabric_code)=? AND TRIM(so_number)=? AND TRIM(sku)=?""",
                        (p_code, so, sku),
                    ).fetchone()
                    pf_alloc = float(r[0] if r else 0)
                    leaf["allocated_printed"] = pf_alloc
                    ga = conn.execute(
                        """SELECT COALESCE(SUM(qty),0) FROM grey_fabric_allocations
                           WHERE status='Active' AND TRIM(grey_code)=? AND TRIM(printed_code)=?
                             AND (TRIM(so_number)=? OR so_number='' OR so_number IS NULL)
                             AND (TRIM(fg_sku)=? OR fg_sku='' OR fg_sku IS NULL)""",
                        (g_code, p_code, so, sku),
                    ).fetchone()
                    leaf["allocated_grey"] = float(ga[0] if ga else 0)
                    locked = _pf_reservation_locked_for_so_sku(conn, so, sku, p_code)
                    if locked:
                        leaf["status"] = "Locked-Cut"
                        leaf["color"] = "blue"
                    elif pf_alloc + 0.001 >= leaf["req_printed"] and leaf["req_printed"] > 0:
                        leaf["status"] = "Printed Available"
                        leaf["color"] = "green"
                    elif pf_alloc > 0:
                        leaf["status"] = "Partial Printed"
                        leaf["color"] = "orange"
                    elif leaf["allocated_grey"] > 0:
                        leaf["status"] = "Grey Allocated"
                        leaf["color"] = "green"
                    else:
                        leaf["status"] = "Printed Pending"
                        leaf["color"] = "red" if leaf["req_printed"] > 0 else "grey"
                # printed free stock
                st = conn.execute(
                    "SELECT available_qty, reserved_qty FROM printed_fabric_checked_stock WHERE TRIM(fabric_code)=?",
                    (p_code,),
                ).fetchone()
                pnode["available_qty"] = float(st["available_qty"] or 0) if st else 0.0
                pnode["reserved_qty"] = float(st["reserved_qty"] or 0) if st else 0.0
                pnode["color"] = "blue" if pnode["available_qty"] > 0 or pnode["reserved_qty"] > 0 else "grey"
            snap = grey_stock_snapshot(g_code)
            if snap:
                gnode["free_qty"] = snap[0].get("grey_free_qty") or snap[0].get("available_qty") or 0
                gnode["allocated_qty"] = snap[0].get("grey_allocated_qty") or 0
                gnode["color"] = snap[0].get("status_color") or "grey"
            else:
                gnode["free_qty"] = 0
                gnode["allocated_qty"] = 0
    finally:
        conn.close()

    # serialize children dicts → lists
    def ser_printed(pnode: dict) -> dict:
        return {
            **{k: v for k, v in pnode.items() if k != "children"},
            "fg_lines": list(pnode["children"].values()),
        }

    def ser_grey(gnode: dict) -> dict:
        return {
            **{k: v for k, v in gnode.items() if k != "children"},
            "printed": [ser_printed(p) for p in gnode["children"].values()],
        }

    nodes = [ser_grey(g) for g in tree.values()]
    return {
        "ok": True,
        "hierarchy": "Grey → Printed (P-Code) → FG SKU → Sales Order",
        "nodes": nodes,
        "so_count": len({f["so_number"] for f in fg_lines}),
        "fg_line_count": len(fg_lines),
    }


def list_grey_allocations(status: str = "Active") -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            """SELECT * FROM grey_fabric_allocations
               WHERE (? = '' OR status = ?)
               ORDER BY id DESC LIMIT 500""",
            (status, status),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_printed_allocations(status: str = "") -> list[dict]:
    conn = _conn()
    try:
        if status:
            rows = conn.execute(
                """SELECT * FROM printed_fabric_reservations WHERE status=?
                   ORDER BY id DESC LIMIT 500""",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM printed_fabric_reservations ORDER BY id DESC LIMIT 500"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_allocation_history(limit: int = 200, printed_code: str = "", grey_code: str = "") -> list[dict]:
    conn = _conn()
    try:
        q = "SELECT * FROM fabric_allocation_history WHERE 1=1"
        params: list[Any] = []
        if printed_code:
            q += " AND TRIM(printed_code)=?"
            params.append(printed_code.strip())
        if grey_code:
            q += " AND TRIM(grey_code)=?"
            params.append(grey_code.strip())
        q += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in conn.execute(q, tuple(params)).fetchall()]
    finally:
        conn.close()


def printing_jo_grey_status(printed_code: str = "") -> list[dict[str, Any]]:
    """What grey is allocated per P-code — for Printing JO display."""
    conn = _conn()
    try:
        q = """SELECT printed_code, printed_name, grey_code, grey_name,
                      SUM(qty) AS grey_allocated, COUNT(*) AS lines
               FROM grey_fabric_allocations
               WHERE status='Active'"""
        params: list[Any] = []
        if printed_code:
            q += " AND TRIM(printed_code)=?"
            params.append(printed_code.strip())
        q += " GROUP BY printed_code, grey_code ORDER BY printed_code, grey_code"
        rows = [dict(r) for r in conn.execute(q, tuple(params)).fetchall()]
        for r in rows:
            r["ready_to_issue"] = float(r.get("grey_allocated") or 0) > 0
            r["status_label"] = "Ready to Issue" if r["ready_to_issue"] else "No Grey"
        return rows
    finally:
        conn.close()


def fg_status_report(printed_code: str = "") -> list[dict[str, Any]]:
    """Reporting must use *current* printed allocation (not original grey intent)."""
    conn = _conn()
    try:
        q = """SELECT fabric_code AS printed_code, fabric_name, so_number, sku AS fg_sku,
                      qty, status, stage, cutting_issued_qty, locked_at
               FROM printed_fabric_reservations
               WHERE status IN ('Active','JO Created','CUTTING_ISSUED','Issued','Consumed')"""
        params: list[Any] = []
        if printed_code:
            q += " AND TRIM(fabric_code)=?"
            params.append(printed_code.strip())
        q += " ORDER BY fabric_code, so_number, sku"
        out = []
        for r in conn.execute(q, tuple(params)).fetchall():
            d = dict(r)
            st = (d.get("status") or "")
            stage = (d.get("stage") or "")
            if st in ("CUTTING_ISSUED", "Issued", "Consumed") or stage == STAGE_CUT_LOCKED:
                d["report_status"] = "Cut / Locked"
                d["color"] = "blue"
            elif st in ("Active", "JO Created"):
                d["report_status"] = "Printed Available"
                d["color"] = "green"
            else:
                d["report_status"] = "Pending"
                d["color"] = "grey"
            out.append(d)
        return out
    finally:
        conn.close()


def _material_is_printed(mat_type: str, mat_code: str) -> bool:
    t = (mat_type or "").upper()
    c = (mat_code or "").strip().upper()
    if t in ("SFG", "PRINTED", "PRINTED FABRIC", "PF"):
        return True
    return bool(c) and (c.startswith("P") and any(ch.isdigit() for ch in c[:6]))


def _material_is_grey_or_fabric(mat_type: str, mat_code: str, unit: str = "") -> bool:
    t = (mat_type or "").upper()
    if t in ("GF", "GREY", "GREY FABRIC", "RM", "FABRIC"):
        return True
    if (unit or "").upper() in ("MTR", "M", "METER", "METRE"):
        return True
    return False


def resolve_p_code_for_fg(fg_sku: str, preferred: str = "") -> str:
    """Best-effort FG → P-Code via Item Master BOM (first printed SFG child)."""
    pref = (preferred or "").strip()
    if pref:
        return pref
    sku = (fg_sku or "").strip()
    if not sku:
        return ""
    children = _bom_children(sku)
    printed = [
        c
        for c in children
        if (c.get("type") or "").upper() in ("SFG", "PRINTED", "PRINTED FABRIC", "PF")
        or str(c.get("code") or "").upper().startswith("P")
    ]
    if printed:
        return str(printed[0].get("code") or "").strip()
    if children:
        return str(children[0].get("code") or "").strip()
    return ""


def annotate_mrp_breakdown_with_allocations(materials: dict[str, Any] | None) -> dict[str, Any]:
    """Enrich Production MRP material breakdowns with Grey→P-Code→FG visibility.

    Mutates and returns ``materials``. Each breakdown line gains:
      - ``p_code`` / ``printed_code``
      - ``allocated_qty`` (grey intent for GF material rows, PF reservation for SFG)
      - ``status`` + ``color`` (Allocated / Partial / Pending / Locked-Cut / …)

    Grey is still planned at **P-Code** level; FG remains on each line so SKU-level
    status dashboards can trace requirement → allocation → issue.
    """
    if not isinstance(materials, dict) or not materials:
        return materials or {}

    conn = None
    try:
        conn = _conn()
    except Exception:
        conn = None

    def _pf_alloc(p_code: str, so: str, sku: str) -> float:
        if conn is None or not p_code:
            return 0.0
        r = conn.execute(
            """SELECT COALESCE(SUM(qty),0) FROM printed_fabric_reservations
               WHERE status IN ('Active','JO Created','CUTTING_ISSUED','Issued','Consumed')
                 AND TRIM(fabric_code)=? AND TRIM(so_number)=? AND TRIM(sku)=?""",
            (p_code, so, sku),
        ).fetchone()
        return float(r[0] if r else 0)

    def _grey_alloc(grey_code: str, p_code: str, so: str, sku: str) -> float:
        if conn is None or not grey_code:
            return 0.0
        # Prefer FG-intent rows first so multi-SKU P-codes do not double-count.
        r = conn.execute(
            """SELECT COALESCE(SUM(qty),0) FROM grey_fabric_allocations
               WHERE status='Active' AND TRIM(grey_code)=?
                 AND (? = '' OR TRIM(printed_code)=?)
                 AND TRIM(so_number)=? AND TRIM(fg_sku)=?""",
            (grey_code, p_code, p_code, so, sku),
        ).fetchone()
        specific = float(r[0] if r else 0)
        if specific > 0:
            return specific
        # Fall back to P-code-level grey only when no FG-specific intent exists
        # for this SO+SKU (shared pool — report for status, not exclusive claim).
        r2 = conn.execute(
            """SELECT COALESCE(SUM(qty),0) FROM grey_fabric_allocations
               WHERE status='Active' AND TRIM(grey_code)=?
                 AND (? = '' OR TRIM(printed_code)=?)
                 AND (so_number IS NULL OR TRIM(so_number)='' OR TRIM(so_number)=?)
                 AND (fg_sku IS NULL OR TRIM(fg_sku)='')""",
            (grey_code, p_code, p_code, so),
        ).fetchone()
        return float(r2[0] if r2 else 0)

    def _locked(p_code: str, so: str, sku: str) -> bool:
        if conn is None or not p_code:
            return False
        try:
            return bool(_pf_reservation_locked_for_so_sku(conn, so, sku, p_code))
        except Exception:
            return False

    try:
        for mat_code, mat in materials.items():
            if not isinstance(mat, dict):
                continue
            mat_type = str(mat.get("type") or "")
            unit = str(mat.get("unit") or "")
            is_printed = _material_is_printed(mat_type, mat_code)
            is_fabric = is_printed or _material_is_grey_or_fabric(mat_type, mat_code, unit)
            breakdown = mat.get("breakdown") or []
            if not isinstance(breakdown, list):
                continue
            for bd in breakdown:
                if not isinstance(bd, dict):
                    continue
                so = str(bd.get("so_no") or bd.get("so_number") or "").strip()
                sku = str(bd.get("sku") or bd.get("fg_sku") or "").strip()
                qty_req = float(bd.get("qty_req") or 0)
                p_code = str(bd.get("p_code") or bd.get("printed_code") or "").strip()
                if not p_code and is_printed:
                    p_code = str(mat_code or "").strip()
                if not p_code and is_fabric and sku:
                    p_code = resolve_p_code_for_fg(sku)
                bd["p_code"] = p_code
                bd["printed_code"] = p_code
                bd.setdefault("fg_sku", sku)

                if not is_fabric:
                    bd["allocated_qty"] = float(bd.get("allocated_qty") or 0)
                    bd["status"] = bd.get("status") or "—"
                    bd["color"] = bd.get("color") or "grey"
                    continue

                pf_alloc = _pf_alloc(p_code, so, sku)
                grey_alloc = _grey_alloc(str(mat_code), p_code, so, sku) if not is_printed else 0.0
                # Row lives on the material being exploded: use the matching pool.
                allocated = pf_alloc if is_printed else (grey_alloc if grey_alloc > 0 else 0.0)
                # If grey row has no grey intent but PF is already reserved for FG, surface that.
                if not is_printed and allocated <= 0 and pf_alloc > 0:
                    allocated = pf_alloc

                bd["allocated_qty"] = round(float(allocated), 3)
                bd["allocated_printed"] = round(float(pf_alloc), 3)
                bd["allocated_grey"] = round(float(grey_alloc), 3)

                locked = _locked(p_code, so, sku)
                if locked:
                    bd["status"] = "Locked-Cut"
                    bd["color"] = "blue"
                elif qty_req > 0 and allocated + 0.001 >= qty_req:
                    bd["status"] = "Allocated"
                    bd["color"] = "green"
                elif allocated > 0:
                    bd["status"] = "Partial"
                    bd["color"] = "orange"
                elif is_printed and pf_alloc <= 0:
                    bd["status"] = "Pending"
                    bd["color"] = "red" if qty_req > 0 else "grey"
                else:
                    bd["status"] = "Pending"
                    bd["color"] = "red" if qty_req > 0 else "grey"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return materials
