"""Fabric ↔ SKU relationships for printed fabric reserve options/validation.

Uses Item Master BOM (FG → SFG/P-code materials) and Production Set BOM
material lines. Does not require SQL FKs between grey and sales.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def _norm(code: str) -> str:
    return str(code or "").strip().upper()


def _row_code(row: Any) -> str:
    if row is None:
        return ""
    if isinstance(row, dict):
        return _norm(row.get("item_code"))
    try:
        return _norm(row["item_code"])
    except Exception:
        pass
    try:
        return _norm(row[0])
    except Exception:
        return ""


def _fabric_match_sql() -> str:
    """Match bom_lines to a fabric by component item_code or bare component_name."""
    return """
        UPPER(TRIM(COALESCE(ci.item_code, ''))) = ?
        OR (
            l.component_item_id IS NULL
            AND UPPER(TRIM(
                CASE
                    WHEN INSTR(COALESCE(l.component_name, ''), ' — ') > 0
                    THEN SUBSTR(l.component_name, 1, INSTR(l.component_name, ' — ') - 1)
                    ELSE COALESCE(l.component_name, '')
                END
            )) = ?
        )
    """


def item_master_skus_using_fabric(fabric_code: str) -> set[str]:
    """Item codes whose **any** BOM (default or size-wise) consumes ``fabric_code``.

    Scans all ``bom_headers`` (not only ``is_default=1``), includes one SFG/FG hop,
    then inherits size variants via ``parent_id``.
    """
    fabric = _norm(fabric_code)
    if not fabric:
        return set()
    try:
        from .jo_issue_notes import _item_connect
    except Exception:
        return set()
    try:
        conn = _item_connect()
    except Exception:
        return set()
    out: set[str] = set()
    try:
        match = _fabric_match_sql()
        # Direct fabric on any BOM header for the item.
        for r in conn.execute(
            f"""
            SELECT DISTINCT i.item_code
            FROM bom_lines l
            JOIN bom_headers h ON h.id = l.bom_id
            JOIN items i ON i.id = h.item_id
            LEFT JOIN items ci ON ci.id = l.component_item_id
            WHERE UPPER(COALESCE(l.component_type, 'RM')) NOT IN ('SVC', 'SERVICE', 'PROCESS')
              AND ({match})
            """,
            (fabric, fabric),
        ).fetchall():
            code = _row_code(r)
            if code:
                out.add(code)

        # One hop: FG/style BOM consumes an SFG/FG that itself uses this fabric.
        sfg_ids = [
            int(r["id"] if hasattr(r, "keys") else r[0])
            for r in conn.execute(
                f"""
                SELECT DISTINCT i.id
                FROM bom_lines l
                JOIN bom_headers h ON h.id = l.bom_id
                JOIN items i ON i.id = h.item_id
                LEFT JOIN items ci ON ci.id = l.component_item_id
                WHERE UPPER(COALESCE(l.component_type, 'RM')) NOT IN ('SVC', 'SERVICE', 'PROCESS')
                  AND ({match})
                """,
                (fabric, fabric),
            ).fetchall()
            if r is not None
        ]
        if sfg_ids:
            placeholders = ",".join("?" * len(sfg_ids))
            for r in conn.execute(
                f"""
                SELECT DISTINCT i.item_code
                FROM bom_lines l
                JOIN bom_headers h ON h.id = l.bom_id
                JOIN items i ON i.id = h.item_id
                WHERE l.component_item_id IN ({placeholders})
                  AND UPPER(COALESCE(l.component_type, 'RM')) IN ('SFG', 'FG', 'SEMI')
                """,
                tuple(sfg_ids),
            ).fetchall():
                code = _row_code(r)
                if code:
                    out.add(code)

        # Size SKUs usually have no BOM of their own — inherit parent style fabric.
        if out:
            pending = set(out)
            while pending:
                placeholders = ",".join("?" * len(pending))
                children = conn.execute(
                    f"""
                    SELECT child.item_code
                    FROM items child
                    JOIN items parent ON parent.id = child.parent_id
                    WHERE UPPER(TRIM(parent.item_code)) IN ({placeholders})
                    """,
                    tuple(pending),
                ).fetchall()
                pending = set()
                for r in children:
                    code = _row_code(r)
                    if code and code not in out:
                        out.add(code)
                        pending.add(code)
    except Exception:
        _log.exception("item_master_skus_using_fabric failed")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def set_bom_skus_using_fabric(fabric_code: str) -> set[str]:
    """Style keys / main SKUs whose set BOM consumes fabric on any set-component material line."""
    fabric = _norm(fabric_code)
    if not fabric:
        return set()
    try:
        from ..db.production_db import list_set_boms, get_set_bom
    except Exception:
        return set()
    out: set[str] = set()
    try:
        headers = list_set_boms() or []
        for h in headers:
            sk = _norm(h.get("style_key") or h.get("style") or "")
            if not sk:
                continue
            bom = get_set_bom(sk)
            if not bom:
                continue
            for ln in bom.get("lines") or []:
                for m in ln.get("materials") or []:
                    if _norm(m.get("material_code")) == fabric:
                        out.add(sk)
                        break
                else:
                    continue
                break
    except Exception:
        _log.exception("set_bom_skus_using_fabric failed")
    return out


def skus_using_fabric(fabric_code: str) -> set[str]:
    """Union of item-master and set-BOM style keys that use the fabric."""
    fabric = _norm(fabric_code)
    if not fabric:
        return set()
    return item_master_skus_using_fabric(fabric) | set_bom_skus_using_fabric(fabric)


def sku_in_fabric_map(sku: str, mapped: set[str]) -> bool:
    """Membership check against a precomputed ``skus_using_fabric`` result."""
    sku_u = _norm(sku)
    if not sku_u:
        return False
    if not mapped:
        return True
    if sku_u in mapped:
        return True
    for m in mapped:
        if sku_u == m or sku_u.startswith(m + "-") or m.startswith(sku_u + "-"):
            return True
        if sku_u.startswith(m) or m.startswith(sku_u.split("-")[0] if "-" in sku_u else sku_u):
            base = sku_u.rsplit("-", 1)[0] if "-" in sku_u else sku_u
            if base == m or base.startswith(m) or m.startswith(base):
                return True
    return False


def sku_uses_fabric(sku: str, fabric_code: str) -> bool:
    """True when the FG/size SKU (or its style root) is linked to the fabric.

    When **no** mapping exists in either BOM system for this fabric, returns True
    (open reserve) so shops without BOM data can still operate. When the fabric
    has **any** mapped SKUs, enforces membership.
    """
    fabric = _norm(fabric_code)
    sku_u = _norm(sku)
    if not fabric or not sku_u:
        return False
    return sku_in_fabric_map(sku_u, skus_using_fabric(fabric))
