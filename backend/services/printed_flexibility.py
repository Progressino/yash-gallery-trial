"""Style-wise printed fabric block, and a size revision when a Cutting job is created.

Both use the existing checked-stock row and reservation rows. Metres move once.
"""
from __future__ import annotations

from ..db import grey_db as gdb


def list_advance_reservations(fabric_code: str = "") -> list[dict]:
    conn = gdb._connect()
    code = (fabric_code or "").strip().upper()
    q = """SELECT * FROM printed_fabric_reservations
           WHERE COALESCE(reservation_kind,'so')='advance' AND status='Active'"""
    params: tuple = ()
    if code:
        q += " AND UPPER(TRIM(fabric_code))=?"
        params = (code,)
    q += " ORDER BY id DESC"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


def _style_uses_fabric(style_code: str, fabric_code: str) -> None:
    try:
        from .fabric_sku_matching import sku_uses_fabric
        if not sku_uses_fabric(style_code, fabric_code):
            raise ValueError(f"Style {style_code} does not use fabric {fabric_code}")
    except ValueError:
        raise
    except Exception:
        return


def _ensure_checked_free_for_advance(fabric_code: str, qty: float, *, user_name: str = "", reason: str = "") -> None:
    """Pull free Opening Stock into checked free when advance reserve needs more metres."""
    from .fabric_allocation_engine import (
        FabricAllocationError,
        convert_printed_opening_to_checked,
        printed_opening_allocatable,
    )

    conn = gdb._connect()
    try:
        stock = conn.execute(
            "SELECT available_qty FROM printed_fabric_checked_stock WHERE UPPER(TRIM(fabric_code))=?",
            (fabric_code,),
        ).fetchone()
        free = float(stock["available_qty"] or 0) if stock else 0.0
    finally:
        conn.close()
    if free + 1e-6 >= qty:
        return
    opening = float(printed_opening_allocatable(fabric_code).get("allocatable") or 0)
    need = round(qty - free, 3)
    if need > opening + 1e-6:
        raise ValueError(
            f"Only {free + opening:.1f} m is free for {fabric_code} "
            f"({free:.1f} m checked + {opening:.1f} m opening). "
            "A style reservation cannot use metres already reserved."
        )
    if need > 0:
        try:
            convert_printed_opening_to_checked(
                fabric_code,
                need,
                user_name=user_name,
                reason=reason or "Opening stock treated as checked (advance style reserve)",
            )
        except FabricAllocationError as e:
            raise ValueError(str(e)) from e


def reserve_for_style(data: dict) -> int:
    fabric_code = (data.get("fabric_code") or "").strip().upper()
    style_code = (data.get("style_code") or "").strip().upper()
    sku = (data.get("sku") or "").strip().upper()
    qty = float(data.get("qty") or 0)
    reason = (data.get("reason") or data.get("remarks") or "").strip()
    user_name = (data.get("user_name") or data.get("reserved_by") or "").strip()
    if not fabric_code or not style_code or qty <= 0:
        raise ValueError("Fabric, style, and metres are required")
    if not reason:
        raise ValueError("A reason is required to reserve fabric for a style")
    # Size-level: SKU must belong to the parent style and use this fabric.
    if sku:
        if not (sku == style_code or sku.startswith(style_code + "-")):
            raise ValueError(f"Size SKU {sku} does not belong to style {style_code}")
        _style_uses_fabric(sku, fabric_code)
    else:
        _style_uses_fabric(style_code, fabric_code)
    _ensure_checked_free_for_advance(fabric_code, qty, user_name=user_name, reason=reason)
    conn = gdb._connect()
    stock = conn.execute(
        "SELECT * FROM printed_fabric_checked_stock WHERE UPPER(TRIM(fabric_code))=?",
        (fabric_code,),
    ).fetchone()
    if not stock:
        conn.close()
        raise ValueError(f"No checked or opening stock for fabric {fabric_code}")
    if float(stock["available_qty"] or 0) + 1e-6 < qty:
        free = float(stock["available_qty"] or 0)
        conn.close()
        raise ValueError(
            f"Only {free:.1f} m is free for {fabric_code}. "
            "A style reservation cannot use metres already reserved."
        )
    existing = conn.execute(
        """SELECT id FROM printed_fabric_reservations
           WHERE UPPER(TRIM(fabric_code))=? AND UPPER(TRIM(style_code))=?
             AND UPPER(TRIM(COALESCE(sku,'')))=?
             AND COALESCE(reservation_kind,'so')='advance' AND status='Active'""",
        (fabric_code, style_code, sku),
    ).fetchone()
    conn.execute(
        "UPDATE printed_fabric_checked_stock SET available_qty = available_qty - ?, reserved_qty = reserved_qty + ? WHERE id=?",
        (qty, qty, stock["id"]),
    )
    if existing:
        rid = int(existing["id"])
        conn.execute(
            "UPDATE printed_fabric_reservations SET qty = qty + ?, reserved_by=? WHERE id=?",
            (qty, user_name, rid),
        )
    else:
        cur = conn.execute(
            """INSERT INTO printed_fabric_reservations(
                fabric_code, fabric_name, so_number, sku, qty, unit, status, remarks,
                reservation_kind, style_code, reserved_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fabric_code, stock["fabric_name"] or data.get("fabric_name") or "",
                "", sku, qty, "MTR", "Active", reason,
                "advance", style_code, user_name,
            ),
        )
        rid = int(cur.lastrowid)
    gdb._audit_fabric(
        conn, event_type="ADVANCE_RESERVE", fabric_code=fabric_code, qty=qty, reason=reason,
        user_name=user_name, to_sku=sku or style_code, status_to="Active",
    )
    conn.commit()
    conn.close()
    return rid


def _advance_row(conn, reservation_id: int):
    row = conn.execute(
        "SELECT * FROM printed_fabric_reservations WHERE id=?",
        (reservation_id,),
    ).fetchone()
    if not row or str(row["reservation_kind"] or "") != "advance" or str(row["status"] or "") != "Active":
        raise ValueError("Active style reservation not found")
    return row


def release_advance(reservation_id: int, qty: float, reason: str, user_name: str = "") -> dict:
    qty = float(qty or 0)
    reason = (reason or "").strip()
    if qty <= 0:
        raise ValueError("Release quantity must be greater than zero")
    if not reason:
        raise ValueError("A reason is required to release a style reservation")
    conn = gdb._connect()
    row = _advance_row(conn, reservation_id)
    held = float(row["qty"] or 0)
    if qty > held + 1e-6:
        conn.close()
        raise ValueError(f"Only {held:.1f} m is still reserved for this style")
    fabric = str(row["fabric_code"] or "")
    conn.execute(
        """UPDATE printed_fabric_checked_stock
           SET available_qty = available_qty + ?, reserved_qty = MAX(0, reserved_qty - ?)
           WHERE UPPER(TRIM(fabric_code))=UPPER(TRIM(?))""",
        (qty, qty, fabric),
    )
    left = held - qty
    status = "Active" if left > 0.001 else "Released"
    conn.execute(
        "UPDATE printed_fabric_reservations SET qty=?, status=? WHERE id=?",
        (max(0, left), status, reservation_id),
    )
    gdb._audit_fabric(
        conn, event_type="ADVANCE_RELEASE", fabric_code=fabric, qty=qty, reason=reason,
        user_name=user_name, from_sku=str(row["style_code"] or ""), status_to=status,
    )
    conn.commit()
    conn.close()
    return {"id": reservation_id, "remaining_qty": round(max(0, left), 3), "status": status}


def transfer_advance(reservation_id: int, to_style: str, qty: float, reason: str, user_name: str = "") -> dict:
    to_style = (to_style or "").strip().upper()
    qty = float(qty or 0)
    reason = (reason or "").strip()
    if not to_style or qty <= 0:
        raise ValueError("Target style and metres are required")
    if not reason:
        raise ValueError("A reason is required to transfer a style reservation")
    conn = gdb._connect()
    try:
        row = _advance_row(conn, reservation_id)
        held = float(row["qty"] or 0)
        if qty > held + 1e-6:
            raise ValueError(f"Only {held:.1f} m can be transferred")
        fabric = str(row["fabric_code"] or "")
        from_style = str(row["style_code"] or "")
        if to_style == from_style:
            raise ValueError("Choose a different style")
        _style_uses_fabric(to_style, fabric)
        left = held - qty
        status = "Active" if left > 0.001 else "Transferred"
        conn.execute(
            "UPDATE printed_fabric_reservations SET qty=?, status=? WHERE id=?",
            (max(0, left), status, reservation_id),
        )
        existing = conn.execute(
            """SELECT id FROM printed_fabric_reservations
               WHERE UPPER(TRIM(fabric_code))=? AND UPPER(TRIM(style_code))=?
                 AND COALESCE(reservation_kind,'so')='advance' AND status='Active'""",
            (fabric.upper(), to_style),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE printed_fabric_reservations SET qty = qty + ? WHERE id=?",
                (qty, int(existing["id"])),
            )
            new_id = int(existing["id"])
        else:
            cur = conn.execute(
                """INSERT INTO printed_fabric_reservations(
                    fabric_code, fabric_name, so_number, sku, qty, unit, status, remarks,
                    reservation_kind, style_code, reserved_by)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (fabric, row["fabric_name"] or "", "", "", qty, "MTR", "Active", reason, "advance", to_style, user_name),
            )
            new_id = int(cur.lastrowid)
        gdb._audit_fabric(
            conn, event_type="ADVANCE_TRANSFER", fabric_code=fabric, qty=qty, reason=reason,
            user_name=user_name, from_sku=from_style, to_sku=to_style, status_to="Active",
        )
        conn.commit()
        return {"from_id": reservation_id, "to_id": new_id, "remaining_qty": round(max(0, left), 3)}
    finally:
        conn.close()


def convert_advance_to_so(reservation_id: int, lines: list, reason: str = "", user_name: str = "") -> dict:
    """Move metres from a style block onto SO/SKU rows. Free and reserved totals do not change.

    Supports partial conversion: leftover metres stay on the advance reservation.
    Parent-style holds may split across several SO/size lines; size-level holds
    may only convert to that same size SKU.
    """
    reason = (reason or "").strip() or "Converted style reservation to sales order"
    clean = []
    for ln in lines or []:
        so = str(ln.get("so_number") or "").strip()
        sku = str(ln.get("sku") or "").strip().upper()
        qty = float(ln.get("qty") or 0)
        if so and sku and qty > 0:
            clean.append({"so_number": so, "sku": sku, "qty": qty})
    if not clean:
        raise ValueError("Add at least one sales order, size, and metres")
    need = sum(x["qty"] for x in clean)
    conn = gdb._connect()
    try:
        row = _advance_row(conn, reservation_id)
        held = float(row["qty"] or 0)
        if need > held + 1e-6:
            raise ValueError(f"Style reservation has {held:.1f} m. Conversion asks for {need:.1f} m.")
        fabric = str(row["fabric_code"] or "")
        style = str(row["style_code"] or "").strip().upper()
        held_sku = str(row["sku"] or "").strip().upper()
        for ln in clean:
            if held_sku and ln["sku"] != held_sku:
                raise ValueError(
                    f"This reservation is for size {held_sku}. Convert only to that size SKU."
                )
            if not held_sku and not (
                ln["sku"] == style or ln["sku"].startswith(style + "-")
            ):
                raise ValueError(
                    f"Size {ln['sku']} does not belong to style {style}"
                )
            _style_uses_fabric(ln["sku"], fabric)
        created = []
        for ln in clean:
            existing = conn.execute(
                """SELECT id FROM printed_fabric_reservations
                   WHERE UPPER(TRIM(fabric_code))=? AND TRIM(so_number)=? AND UPPER(TRIM(sku))=?
                     AND status='Active' AND COALESCE(reservation_kind,'so')!='advance'""",
                (fabric.upper(), ln["so_number"], ln["sku"]),
            ).fetchone()
            if existing:
                rid = int(existing["id"])
                conn.execute("UPDATE printed_fabric_reservations SET qty = qty + ? WHERE id=?", (ln["qty"], rid))
            else:
                cur = conn.execute(
                    """INSERT INTO printed_fabric_reservations(
                        fabric_code, fabric_name, so_number, sku, qty, unit, status, remarks, reservation_kind, style_code)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (fabric, row["fabric_name"] or "", ln["so_number"], ln["sku"], ln["qty"], "MTR", "Active", reason, "so", style),
                )
                rid = int(cur.lastrowid)
            created.append({"id": rid, **ln})
        left = held - need
        status = "Active" if left > 0.001 else "Converted"
        conn.execute(
            "UPDATE printed_fabric_reservations SET qty=?, status=? WHERE id=?",
            (max(0, left), status, reservation_id),
        )
        gdb._audit_fabric(
            conn, event_type="ADVANCE_CONVERT", fabric_code=fabric, qty=need, reason=reason,
            user_name=user_name, from_sku=style, to_so=clean[0]["so_number"], to_sku=clean[0]["sku"],
            status_to=status,
        )
        conn.commit()
        return {"id": reservation_id, "remaining_qty": round(max(0, left), 3), "status": status, "lines": created}
    finally:
        conn.close()


def advance_options(fabric_code: str = "") -> dict:
    """Fabrics with free (checked + opening), BOM-linked parent styles / size SKUs, and matching SOs."""
    fabric = (fabric_code or "").strip().upper()
    visibility = gdb.printed_stock_visibility(fabric)
    fabrics = [
        {
            "fabric_code": v["fabric_code"],
            "fabric_name": v.get("fabric_name") or v["fabric_code"],
            "free_available": v.get("free_available") or 0,
            "checked_free": v.get("checked_free") or 0,
            "opening_free": v.get("opening_free") or 0,
        }
        for v in visibility
        if float(v.get("free_available") or 0) > 0
    ]
    parent_styles: list[dict] = []
    size_skus: list[dict] = []
    sales_orders: list[dict] = []
    fabric_filter_active = False
    if fabric:
        try:
            from .fabric_sku_matching import skus_using_fabric

            mapped = sorted(skus_using_fabric(fabric))
            fabric_filter_active = bool(mapped)
            parents: set[str] = set()
            sizes: set[str] = set()
            for code in mapped:
                u = code.strip().upper()
                if not u:
                    continue
                if "-" in u:
                    sizes.add(u)
                    parents.add(u.rsplit("-", 1)[0])
                else:
                    parents.add(u)
            # Prefer real Item Master parents when available
            try:
                from ..db import item_db

                for p in list(parents):
                    rows = item_db.list_items(search=p, parent_only=True, limit=5, rank_search=True)
                    for r in rows:
                        if str(r.get("item_code") or "").strip().upper() == p:
                            break
            except Exception:
                pass
            parent_styles = [{"style_code": p, "label": p} for p in sorted(parents)]
            size_skus = [
                {
                    "sku": s,
                    "style_code": s.rsplit("-", 1)[0],
                    "label": s,
                }
                for s in sorted(sizes)
            ]
        except Exception:
            parent_styles = []
            size_skus = []
            fabric_filter_active = False
        try:
            opts = gdb.printed_fabric_reserve_options(fabric_code=fabric)
            sales_orders = opts.get("sales_orders") or []
            fabric_filter_active = fabric_filter_active or bool(opts.get("fabric_filter_active"))
        except Exception:
            sales_orders = []
    return {
        "fabrics": fabrics,
        "parent_styles": parent_styles,
        "size_skus": size_skus,
        "sales_orders": sales_orders,
        "fabric_code": fabric or None,
        "fabric_filter_active": fabric_filter_active,
    }


def style_base(sku: str) -> str:
    s = (sku or "").strip().upper()
    return s.rsplit("-", 1)[0] if "-" in s else s


def same_style(anchor: str, sku: str) -> bool:
    a = style_base(anchor)
    b = style_base(sku)
    if not a or not b:
        return False
    return a == b


def fabric_metres_per_piece(sku: str, fabric_code: str) -> float:
    """BOM average for one piece: quantity × (1 + shrinkage + wastage), for this fabric only."""
    try:
        from .jo_issue_notes import _get_default_bom, _get_item_by_code, _get_item_by_id
    except Exception:
        return 0.0
    fabric = (fabric_code or "").strip().upper()
    code = (sku or "").strip()
    if not fabric or not code:
        return 0.0
    seen = set()
    current = code
    for _ in range(6):
        key = current.upper()
        if key in seen:
            break
        seen.add(key)
        item = _get_item_by_code(current)
        if not item:
            break
        bom = _get_default_bom(int(item["id"]))
        if bom:
            total = 0.0
            for ln in bom.get("lines") or []:
                comp = _get_item_by_id(int(ln["component_item_id"])) if ln.get("component_item_id") else None
                comp_code = str((comp or {}).get("item_code") or ln.get("component_name") or "").strip().upper()
                if comp_code != fabric:
                    continue
                qty = float(ln.get("quantity") or 0)
                shrink = float(ln.get("shrinkage_pct") or 0) / 100.0
                waste = float(ln.get("wastage_pct") or 0) / 100.0
                total += qty * (1.0 + shrink + waste)
            if total > 0:
                return round(total, 4)
        parent_id = item.get("parent_id")
        if not parent_id:
            break
        parent = _get_item_by_id(int(parent_id))
        if not parent:
            break
        current = str(parent.get("item_code") or "")
    return 0.0


def so_reserved_metres(so_number: str, fabric_code: str, skus: list[str]) -> float:
    conn = gdb._connect()
    total = 0.0
    fabric = (fabric_code or "").strip().upper()
    so = (so_number or "").strip()
    bases = {style_base(s) for s in skus if style_base(s)}
    rows = conn.execute(
        """SELECT sku, qty FROM printed_fabric_reservations
           WHERE TRIM(so_number)=? AND UPPER(TRIM(fabric_code))=?
             AND status='Active' AND COALESCE(reservation_kind,'so')!='advance'""",
        (so, fabric),
    ).fetchall()
    conn.close()
    wanted = {str(s or "").strip().upper() for s in skus}
    for row in rows:
        sku = str(row["sku"] or "").strip().upper()
        if sku in wanted or style_base(sku) in bases:
            total += float(row["qty"] or 0)
    return round(total, 3)


def evaluate_cutting_plan(so_number: str, fabric_code: str, lines: list, anchor_sku: str = "") -> dict:
    """Compare revised pieces × BOM with metres already reserved for this order and fabric."""
    fabric = (fabric_code or "").strip().upper()
    so = (so_number or "").strip()
    anchor = (anchor_sku or "").strip().upper()
    detail = []
    changed = False
    for ln in lines or []:
        sku = str(ln.get("sku") or "").strip().upper()
        if not sku:
            continue
        revised = int(round(float(ln.get("planned_qty") if ln.get("planned_qty") is not None else ln.get("revised_qty") or 0)))
        if revised < 0:
            raise ValueError(f"Cutting quantity for {sku} cannot be negative")
        raw_so = ln.get("so_qty")
        so_qty = int(round(float(raw_so))) if raw_so is not None and str(raw_so) != "" else revised
        if revised != so_qty:
            changed = True
        if not anchor:
            anchor = sku
        if anchor and not same_style(anchor, sku):
            raise ValueError(f"{sku} is not a size of style {style_base(anchor)}")
        _style_uses_fabric(sku, fabric)
        per = fabric_metres_per_piece(sku, fabric) if revised > 0 or so_qty > 0 else 0.0
        if revised > 0 and per <= 0:
            raise ValueError(f"No BOM average for {sku} and fabric {fabric}")
        detail.append({
            "sku": sku,
            "so_qty": so_qty,
            "revised_qty": revised,
            "bom_per_pc": per,
            "fabric_metres": round(revised * per, 3),
            "original_metres": round(so_qty * per, 3),
        })
    if not detail:
        raise ValueError("Add at least one size")
    family = [d["sku"] for d in detail]
    reserved = so_reserved_metres(so, fabric, family)
    need = round(sum(d["fabric_metres"] for d in detail), 3)
    original = round(sum(d["original_metres"] for d in detail), 3)
    free = _free(fabric)
    gap = round(max(0.0, need - reserved), 3)
    ok = gap <= free + 1e-6
    return {
        "changed": changed,
        "style_code": style_base(anchor),
        "fabric_code": fabric,
        "so_number": so,
        "lines": detail,
        "original_metres": original,
        "revised_metres": need,
        "reserved_metres": reserved,
        "free_available": free,
        "extra_metres": gap,
        "ok": ok,
        "message": (
            "Revised plan fits the fabric already reserved for this order."
            if gap <= 0.001
            else (
                f"Revised plan needs {gap:.1f} m more from free stock."
                if ok
                else f"Revised plan needs {gap:.1f} m more, but only {free:.1f} m is free."
            )
        ),
    }


def _free(fabric_code: str) -> float:
    conn = gdb._connect()
    row = conn.execute(
        "SELECT available_qty FROM printed_fabric_checked_stock WHERE UPPER(TRIM(fabric_code))=?",
        ((fabric_code or "").strip().upper(),),
    ).fetchone()
    conn.close()
    return round(float(row["available_qty"] or 0), 3) if row else 0.0


def _plan_changed(lines: list) -> bool:
    for ln in lines or []:
        if not str(ln.get("sku") or "").strip():
            continue
        revised = int(round(float(ln.get("planned_qty") if ln.get("planned_qty") is not None else ln.get("revised_qty") or 0)))
        raw_so = ln.get("so_qty")
        if raw_so is None or str(raw_so) == "":
            continue
        so_qty = int(round(float(raw_so)))
        if revised != so_qty:
            return True
    return False


def apply_cutting_plan(so_number: str, fabric_code: str, lines: list, reason: str, anchor_sku: str = "", user_name: str = "") -> dict | None:
    """Validate a size revision and, if it needs more metres, reserve them from free stock once.

    An unchanged size mix does not touch stock or the MRP check.
    """
    if not _plan_changed(lines):
        return None
    plan = evaluate_cutting_plan(so_number, fabric_code, lines, anchor_sku=anchor_sku)
    if not (reason or "").strip():
        raise ValueError("Enter a reason when the cutting quantities differ from the sales order")
    if not plan["ok"]:
        raise ValueError(plan["message"])
    topup = float(plan["extra_metres"] or 0)
    topup_sku = ""
    if topup > 0.001:
        topup_sku = next((ln["sku"] for ln in plan["lines"] if ln["revised_qty"] > 0), plan["lines"][0]["sku"])
        gdb.reserve_printed_fabric({
            "fabric_code": plan["fabric_code"],
            "so_number": so_number,
            "sku": topup_sku,
            "qty": topup,
            "remarks": (reason or "").strip(),
            "per_fabric": True,
        })
    plan["reason"] = reason.strip()
    plan["user_name"] = user_name or ""
    plan["topped_up_metres"] = topup
    plan["topup_sku"] = topup_sku
    return plan


def undo_cutting_topup(plan: dict) -> None:
    qty = float(plan.get("topped_up_metres") or 0)
    sku = str(plan.get("topup_sku") or "")
    if qty <= 0 or not sku:
        return
    conn = gdb._connect()
    row = conn.execute(
        """SELECT id, qty FROM printed_fabric_reservations
           WHERE TRIM(so_number)=? AND UPPER(TRIM(fabric_code))=? AND UPPER(TRIM(sku))=?
             AND status='Active' AND COALESCE(reservation_kind,'so')!='advance'""",
        (plan.get("so_number") or "", plan.get("fabric_code") or "", sku),
    ).fetchone()
    if not row:
        conn.close()
        return
    left = float(row["qty"] or 0) - qty
    if left > 0.001:
        conn.execute("UPDATE printed_fabric_reservations SET qty=? WHERE id=?", (left, row["id"]))
    else:
        conn.execute("UPDATE printed_fabric_reservations SET qty=0, status='Released' WHERE id=?", (row["id"],))
    conn.execute(
        """UPDATE printed_fabric_checked_stock
           SET available_qty = available_qty + ?, reserved_qty = MAX(0, reserved_qty - ?)
           WHERE UPPER(TRIM(fabric_code))=UPPER(TRIM(?))""",
        (qty, qty, plan.get("fabric_code") or ""),
    )
    conn.commit()
    conn.close()


def save_cutting_plan(plan: dict, jo_numbers: str) -> int:
    conn = gdb._connect()
    cur = conn.execute(
        """INSERT INTO cutting_plan_revisions(
            so_number, fabric_code, style_code, reason, user_name, jo_numbers,
            original_metres, revised_metres, reserved_metres, topped_up_metres)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            plan.get("so_number") or "", plan.get("fabric_code") or "", plan.get("style_code") or "",
            plan.get("reason") or "", plan.get("user_name") or "", jo_numbers or "",
            float(plan.get("original_metres") or 0), float(plan.get("revised_metres") or 0),
            float(plan.get("reserved_metres") or 0), float(plan.get("topped_up_metres") or 0),
        ),
    )
    rid = int(cur.lastrowid)
    for ln in plan.get("lines") or []:
        conn.execute(
            """INSERT INTO cutting_plan_revision_lines(revision_id, sku, so_qty, revised_qty, bom_per_pc, fabric_metres)
               VALUES (?,?,?,?,?,?)""",
            (rid, ln.get("sku") or "", float(ln.get("so_qty") or 0), float(ln.get("revised_qty") or 0),
             float(ln.get("bom_per_pc") or 0), float(ln.get("fabric_metres") or 0)),
        )
    gdb._audit_fabric(
        conn, event_type="CUTTING_PLAN", fabric_code=plan.get("fabric_code") or "",
        qty=float(plan.get("revised_metres") or 0), reason=plan.get("reason") or "",
        user_name=plan.get("user_name") or "", from_so=plan.get("so_number") or "",
        to_sku=plan.get("style_code") or "", status_to="JO Created",
    )
    conn.commit()
    conn.close()
    return rid
