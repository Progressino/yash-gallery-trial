"""Canonical printed/grey fabric codes — case-insensitive identity.

Item Master casing wins when present; otherwise the most-used stock-row casing.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

_log = logging.getLogger(__name__)


def _norm_key(code: str) -> str:
    return str(code or "").strip().upper()


def canonical_printed_fabric_code(code: str, *, conn=None) -> str:
    """Prefer Item Master item_code casing; else existing checked-stock casing; else trimmed input."""
    raw = str(code or "").strip()
    if not raw:
        return ""
    key = _norm_key(raw)
    try:
        from ..db.item_db import opening_stock_for_code

        im = opening_stock_for_code(raw)
        if im.get("found") and im.get("item_code"):
            return str(im["item_code"]).strip()
    except Exception:
        pass
    owns = conn is None
    if owns:
        from ..db import grey_db as gdb

        conn = gdb._connect()
    try:
        row = conn.execute(
            """SELECT fabric_code FROM printed_fabric_checked_stock
               WHERE UPPER(TRIM(fabric_code))=?
               ORDER BY id LIMIT 1""",
            (key,),
        ).fetchone()
        if row and row["fabric_code"]:
            return str(row["fabric_code"]).strip()
    except Exception:
        pass
    finally:
        if owns:
            try:
                conn.close()
            except Exception:
                pass
    return raw


def merge_printed_fabric_case_duplicates(*, dry_run: bool = False) -> dict[str, Any]:
    """Merge printed_fabric_checked_stock (and related) rows that differ only by case.

    Keeps canonical Item Master casing when available. Sums qty columns; rewrites
    reservations / conversions / unchecked stock to the survivor code; deletes extras.
    """
    from ..db import grey_db as gdb

    conn = gdb._connect()
    report: dict[str, Any] = {"ok": True, "groups": [], "merged": 0, "dry_run": dry_run}
    try:
        groups = conn.execute(
            """
            SELECT UPPER(TRIM(fabric_code)) AS ukey,
                   GROUP_CONCAT(id) AS ids,
                   GROUP_CONCAT(fabric_code, ' || ') AS codes,
                   COUNT(*) AS n
            FROM printed_fabric_checked_stock
            GROUP BY UPPER(TRIM(fabric_code))
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for g in groups:
            ukey = str(g["ukey"] or "")
            ids = [int(x) for x in str(g["ids"] or "").split(",") if x.strip()]
            if len(ids) < 2:
                continue
            canonical = canonical_printed_fabric_code(ukey, conn=conn)
            rows = conn.execute(
                f"SELECT * FROM printed_fabric_checked_stock WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            ).fetchall()
            rows = [dict(r) for r in rows]
            survivor = next((r for r in rows if str(r["fabric_code"]).strip() == canonical), None)
            if not survivor:
                survivor = rows[0]
                canonical = str(survivor["fabric_code"]).strip()
            sid = int(survivor["id"])
            checked = passed = reserved = available = 0.0
            for r in rows:
                checked += float(r.get("checked_qty") or 0)
                passed += float(r.get("passed_qty") or 0)
                reserved += float(r.get("reserved_qty") or 0)
                available += float(r.get("available_qty") or 0)
            detail = {
                "canonical": canonical,
                "from_codes": [str(r["fabric_code"]) for r in rows],
                "checked_qty": round(checked, 3),
                "reserved_qty": round(reserved, 3),
                "available_qty": round(available, 3),
            }
            report["groups"].append(detail)
            if dry_run:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """UPDATE printed_fabric_checked_stock
                       SET fabric_code=?, fabric_name=?,
                           checked_qty=?, passed_qty=?, reserved_qty=?, available_qty=?
                       WHERE id=?""",
                    (
                        canonical,
                        survivor.get("fabric_name") or canonical,
                        round(checked, 3),
                        round(passed, 3),
                        round(reserved, 3),
                        round(available, 3),
                        sid,
                    ),
                )
                for r in rows:
                    if int(r["id"]) == sid:
                        continue
                    old = str(r["fabric_code"])
                    for tbl, col in (
                        ("printed_fabric_reservations", "fabric_code"),
                        ("printed_fabric_opening_conversions", "fabric_code"),
                        ("printed_fabric_stock", "fabric_code"),
                        ("printed_fabric_audit", "fabric_code"),
                    ):
                        try:
                            conn.execute(
                                f"UPDATE {tbl} SET {col}=? WHERE TRIM({col})=?",
                                (canonical, old),
                            )
                        except Exception:
                            pass
                    conn.execute(
                        "DELETE FROM printed_fabric_checked_stock WHERE id=?",
                        (int(r["id"]),),
                    )
                # Also rewrite any remaining UPPER mismatches for this key
                conn.execute(
                    """UPDATE printed_fabric_reservations SET fabric_code=?
                       WHERE UPPER(TRIM(fabric_code))=? AND TRIM(fabric_code)!=?""",
                    (canonical, ukey, canonical),
                )
                conn.execute(
                    """UPDATE printed_fabric_opening_conversions SET fabric_code=?
                       WHERE UPPER(TRIM(fabric_code))=? AND TRIM(fabric_code)!=?""",
                    (canonical, ukey, canonical),
                )
                conn.commit()
                report["merged"] += 1
            except Exception:
                conn.rollback()
                raise
        # Rewrite reservation codes that have no checked-stock case split but wrong casing
        if not dry_run:
            try:
                from ..db.item_db import opening_stock_for_code

                for r in conn.execute(
                    "SELECT DISTINCT fabric_code FROM printed_fabric_reservations WHERE status='Active'"
                ).fetchall():
                    fc = str(r["fabric_code"] or "").strip()
                    if not fc:
                        continue
                    im = opening_stock_for_code(fc)
                    canon = str(im.get("item_code") or "").strip() if im.get("found") else ""
                    if canon and canon != fc and _norm_key(canon) == _norm_key(fc):
                        conn.execute(
                            "UPDATE printed_fabric_reservations SET fabric_code=? WHERE TRIM(fabric_code)=?",
                            (canon, fc),
                        )
                conn.commit()
            except Exception:
                _log.exception("reservation casing rewrite failed")
    except Exception as e:
        _log.exception("merge_printed_fabric_case_duplicates failed")
        report["ok"] = False
        report["error"] = str(e)
    finally:
        conn.close()
    return report
