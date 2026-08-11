"""Fabric ↔ SKU relationships for printed fabric reserve options/validation.

Uses Item Master BOM (FG → SFG/P-code materials) and Production Set BOM
material lines. Does not require SQL FKs between grey and sales.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

_log = logging.getLogger(__name__)


def _norm(code: str) -> str:
    return str(code or "").strip().upper()


def item_master_skus_using_fabric(fabric_code: str) -> set[str]:
    """FG / SFG item codes whose default BOM consumes ``fabric_code`` (direct or via one SFG hop)."""
    fabric = _norm(fabric_code)
    if not fabric:
        return set()
    try:
        from .jo_issue_notes import (
            _get_default_bom,
            _get_item_by_code,
            _get_item_by_id,
            _item_connect,
        )
    except Exception:
        return set()
    try:
        conn = _item_connect()
    except Exception:
        return set()
    out: set[str] = set()
    try:
        items = conn.execute(
            "SELECT id, item_code FROM items WHERE COALESCE(active,1)=1"
        ).fetchall()
        for it in items:
            it = dict(it)
            code = _norm(it.get("item_code"))
            if not code:
                continue
            bom = _get_default_bom(conn, int(it["id"]))
            if not bom:
                continue
            lines = conn.execute(
                "SELECT * FROM bom_lines WHERE bom_id=?",
                (int(bom["id"]),),
            ).fetchall()
            for ln in lines:
                ln = dict(ln)
                ctype = str(ln.get("component_type") or "RM").upper()
                if ctype in ("SVC", "SERVICE", "PROCESS"):
                    continue
                comp = None
                if ln.get("component_item_id"):
                    comp = _get_item_by_id(conn, int(ln["component_item_id"]))
                raw = str(ln.get("component_name") or "").strip()
                guess = raw.split(" — ")[0].strip() if " — " in raw else raw
                if not comp and guess:
                    comp = _get_item_by_code(conn, guess)
                ccode = _norm((comp or {}).get("item_code") or guess)
                if not ccode:
                    continue
                if ccode == fabric:
                    out.add(code)
                    continue
                # P-code SFG often sits between FG and grey — match fabric two hops when common.
                if ctype in ("SFG", "FG", "SEMI"):
                    try:
                        sub_item = comp or _get_item_by_code(conn, ccode)
                        if not sub_item:
                            continue
                        sub_bom = _get_default_bom(conn, int(sub_item["id"]))
                        if not sub_bom:
                            continue
                        for sl in conn.execute(
                            "SELECT * FROM bom_lines WHERE bom_id=?",
                            (int(sub_bom["id"]),),
                        ).fetchall():
                            sl = dict(sl)
                            sc = None
                            if sl.get("component_item_id"):
                                sc = _get_item_by_id(conn, int(sl["component_item_id"]))
                            sraw = str(sl.get("component_name") or "").strip()
                            sguess = sraw.split(" — ")[0].strip() if " — " in sraw else sraw
                            if not sc and sguess:
                                sc = _get_item_by_code(conn, sguess)
                            scode = _norm((sc or {}).get("item_code") or sguess)
                            if scode == fabric:
                                out.add(code)
                                break
                    except Exception:
                        continue
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
        from ..db.production_db import _connect, list_set_boms, get_set_bom
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
    except Exception:
        _log.exception("set_bom_skus_using_fabric failed")
    return out


def skus_using_fabric(fabric_code: str) -> set[str]:
    """Union of item-master and set-BOM style keys that use the fabric."""
    fabric = _norm(fabric_code)
    if not fabric:
        return set()
    return item_master_skus_using_fabric(fabric) | set_bom_skus_using_fabric(fabric)


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
    mapped = skus_using_fabric(fabric)
    if not mapped:
        return True
    if sku_u in mapped:
        return True
    # Strip size suffix STYLE-3XL → match STYLE base or progressive prefixes
    for m in mapped:
        if sku_u == m or sku_u.startswith(m + "-") or m.startswith(sku_u + "-"):
            return True
        # common: style key without size vs size SKU
        if sku_u.startswith(m) or m.startswith(sku_u.split("-")[0] if "-" in sku_u else sku_u):
            # tighter: first token style code overlap
            base = sku_u.rsplit("-", 1)[0] if "-" in sku_u else sku_u
            if base == m or base.startswith(m) or m.startswith(base):
                return True
    return False
