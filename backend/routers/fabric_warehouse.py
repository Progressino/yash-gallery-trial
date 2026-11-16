"""Fabric Warehouse API — read-only dashboard, stock and ledger over existing records.

Mounted at ``/api/fabric-warehouse``. No stock is posted here; see
``backend/services/fabric_warehouse.py`` for the source map.
"""
from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ..services import fabric_warehouse as fw

router = APIRouter()

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _d(value: Optional[str], name: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if not _DATE_RE.match(v):
        raise HTTPException(status_code=400, detail=f"{name} must be YYYY-MM-DD")
    return v


def _wh(value: str) -> str:
    v = (value or "all").strip().lower()
    if v not in fw.WAREHOUSES:
        raise HTTPException(status_code=400, detail=f"warehouse must be one of {', '.join(fw.WAREHOUSES)}")
    return v


def _dir(value: str) -> str:
    v = (value or "all").strip().lower()
    if v not in fw.DIRECTIONS:
        raise HTTPException(status_code=400, detail=f"direction must be one of {', '.join(fw.DIRECTIONS)}")
    return v


@router.get("/dashboard")
def get_dashboard(
    from_date: str = "",
    to_date: str = "",
    pcode: str = "",
    warehouse: str = "all",
    txn_type: str = "",
):
    """Opening, Today IN/OUT, Closing, SO Allocated, Advance Reserved, Free, Returns/Adj, Pending Unchecked."""
    return fw.dashboard(
        from_date=_d(from_date, "from_date"),
        to_date=_d(to_date, "to_date"),
        pcode=pcode,
        warehouse=_wh(warehouse),
        txn_type=txn_type,
    )


@router.get("/stock")
def get_stock(pcode: str = "", q: str = ""):
    """Per P-Code: physical, checked, unchecked, SO allocated, advance reserved, free, location."""
    rows = fw.stock_by_pcode(pcode=pcode, q=q)
    return {"rows": rows, "count": len(rows)}


@router.get("/ledger")
def get_ledger(
    from_date: str = "",
    to_date: str = "",
    pcode: str = "",
    warehouse: str = "all",
    direction: str = "all",
    txn_type: str = "",
    q: str = "",
    order: str = "desc",
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """Date-wise register — union of GRN, QC, opening, MIN, JO issue/return, adjustments, reservations."""
    res = fw.build_ledger(
        from_date=_d(from_date, "from_date"),
        to_date=_d(to_date, "to_date"),
        pcode=pcode,
        warehouse=_wh(warehouse),
        direction=_dir(direction),
        txn_type=txn_type,
        q=q,
    )
    rows = res["rows"]
    if (order or "desc").lower() != "asc":
        rows = list(reversed(rows))
    total = len(rows)
    return {
        **{k: v for k, v in res.items() if k != "rows"},
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": rows[offset: offset + limit],
        "txn_types": fw.ALL_TXN_TYPES,
    }


@router.get("/ledger/export")
def export_ledger(
    format: str = "xlsx",
    from_date: str = "",
    to_date: str = "",
    pcode: str = "",
    warehouse: str = "all",
    direction: str = "all",
    txn_type: str = "",
    q: str = "",
):
    fmt = (format or "xlsx").strip().lower()
    if fmt not in ("csv", "xlsx"):
        raise HTTPException(status_code=400, detail="format must be csv or xlsx")
    res = fw.build_ledger(
        from_date=_d(from_date, "from_date"),
        to_date=_d(to_date, "to_date"),
        pcode=pcode,
        warehouse=_wh(warehouse),
        direction=_dir(direction),
        txn_type=txn_type,
        q=q,
    )
    rows = res["rows"]
    name = f"fabric_ledger_{fw.stamp()}"
    if fmt == "csv":
        return Response(
            content=fw.export_csv(rows),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
        )
    return Response(
        content=fw.export_xlsx(rows),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'},
    )
