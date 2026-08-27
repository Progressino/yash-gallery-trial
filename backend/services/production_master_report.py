"""
Master Production Status Report — quantity-level WIP across all stages.

Source of truth: ``process_stock`` (SO × SKU × process) plus open ``job_orders``.
Stages are discovered from Item Master routing_steps and live stock — never hardcoded.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..db.production_db import _connect, get_all_routing_steps

# Component suffix on size SKUs: STYLE-SIZE-TOP / …-FRONT etc.
_COMPONENT_SUFFIX_RE = re.compile(
    r"^(.+?)-(TOP|PANT|BOTTOM|DUPATTA|DUPATA|FRONT|BACK|SLEEVE|NECK|YOKE)$",
    re.I,
)


def _parse_main_and_component(sku: str) -> tuple[str, str]:
    s = str(sku or "").strip().upper()
    if not s:
        return "", ""
    m = _COMPONENT_SUFFIX_RE.match(s)
    if m:
        return m.group(1), m.group(2).upper()
    return s, ""


def list_production_stage_names() -> list[str]:
    """Ordered stage names from Item Master; include live stock processes not in catalog."""
    steps = get_all_routing_steps() or []
    names: list[str] = []
    seen: set[str] = set()
    for s in steps:
        if isinstance(s, dict):
            n = str(s.get("name") or "").strip()
        else:
            n = str(s or "").strip()
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT process FROM process_stock WHERE TRIM(COALESCE(process,'')) <> '' "
            "ORDER BY process"
        ).fetchall()
        for r in rows:
            n = str(r["process"] if hasattr(r, "keys") else r[0]).strip()
            if n and n not in seen:
                seen.add(n)
                names.append(n)
    finally:
        conn.close()
    return names


def query_master_production_status(
    *,
    so_number: str = "",
    sku: str = "",
    main_sku: str = "",
    component: str = "",
    jo_number: str = "",
    process: str = "",
    q: str = "",
    status: str = "",
    min_available: int = 0,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Flat quantity-level lines: one row per (SO, SKU, process) with stock + open JO qty.

    Same SKU can appear on many stages — that is intentional.
    """
    so_number = str(so_number or "").strip()
    sku = str(sku or "").strip().upper()
    main_sku = str(main_sku or "").strip().upper()
    component = str(component or "").strip().upper()
    jo_number = str(jo_number or "").strip().upper()
    process = str(process or "").strip()
    q = str(q or "").strip().upper()
    status = str(status or "").strip().lower()  # stock|in_jo|any
    limit = max(1, min(int(limit or 200), 5000))
    offset = max(0, int(offset or 0))
    min_available = max(0, int(min_available or 0))

    stages = list_production_stage_names()
    conn = _connect()
    try:
        # Open JO aggregation per SO+SKU+process
        jo_sql = """
            SELECT so_number, sku, process,
                   GROUP_CONCAT(DISTINCT jo_number) AS jo_numbers,
                   MAX(COALESCE(main_sku,'')) AS main_sku,
                   MAX(COALESCE(component_code,'')) AS component_code,
                   MAX(COALESCE(sku_role,'')) AS sku_role,
                   SUM(COALESCE(planned_qty,0)) AS jo_planned,
                   SUM(COALESCE(issued_qty,0)) AS jo_issued,
                   SUM(COALESCE(received_qty,0)) AS jo_received,
                   SUM(COALESCE(balance_qty,
                        COALESCE(planned_qty,0) - COALESCE(received_qty,0))) AS jo_balance,
                   COUNT(*) AS jo_count
            FROM job_orders
            WHERE status NOT IN ('Cancelled','Closed')
            GROUP BY so_number, sku, process
        """
        jo_rows = conn.execute(jo_sql).fetchall()
        jo_map: dict[tuple[str, str, str], dict] = {}
        for r in jo_rows:
            key = (str(r["so_number"]), str(r["sku"]).upper(), str(r["process"]))
            jo_map[key] = dict(r)

        stock_where = ["1=1"]
        params: list[Any] = []
        if so_number:
            stock_where.append("so_number = ?")
            params.append(so_number)
        if sku:
            stock_where.append("UPPER(sku) = ?")
            params.append(sku)
        if process:
            stock_where.append("process = ?")
            params.append(process)
        if jo_number:
            stock_where.append("UPPER(COALESCE(jo_number,'')) LIKE ?")
            params.append(f"%{jo_number}%")
        if min_available > 0:
            stock_where.append("available_qty >= ?")
            params.append(min_available)

        stock_sql = f"""
            SELECT so_number, sku, process, available_qty, total_in, total_out,
                   COALESCE(batch,'') AS batch,
                   COALESCE(vendor_name,'') AS vendor_name,
                   COALESCE(jo_number,'') AS stock_jo_number,
                   updated_at
            FROM process_stock
            WHERE {' AND '.join(stock_where)}
            ORDER BY so_number, sku, process
        """
        stock_rows = [dict(r) for r in conn.execute(stock_sql, params).fetchall()]
    finally:
        conn.close()

    # Merge keys present only on JO side (open JO with no stock row yet)
    keys_seen: set[tuple[str, str, str]] = set()
    lines: list[dict[str, Any]] = []

    def _match_filters(line: dict) -> bool:
        if main_sku:
            ms = str(line.get("main_sku") or "").upper()
            sk = str(line.get("sku") or "").upper()
            if main_sku not in ms and not sk.startswith(main_sku):
                return False
        if component:
            cc = str(line.get("component_code") or "").upper()
            sk = str(line.get("sku") or "").upper()
            if component not in cc and not sk.endswith(f"-{component}"):
                return False
        if q:
            blob = " ".join(
                str(line.get(k) or "")
                for k in (
                    "so_number",
                    "sku",
                    "main_sku",
                    "component_code",
                    "process",
                    "jo_numbers",
                    "stock_jo_number",
                )
            ).upper()
            if q not in blob:
                return False
        if status == "stock" and int(line.get("available_qty") or 0) <= 0:
            return False
        if status == "in_jo" and int(line.get("jo_balance") or 0) <= 0:
            return False
        return True

    for r in stock_rows:
        so = str(r["so_number"])
        sk = str(r["sku"]).upper()
        proc = str(r["process"])
        key = (so, sk, proc)
        keys_seen.add(key)
        jo = jo_map.get(key) or {}
        parsed_main, parsed_comp = _parse_main_and_component(sk)
        main = str(jo.get("main_sku") or "").strip().upper() or parsed_main
        comp = str(jo.get("component_code") or "").strip().upper() or parsed_comp
        line = {
            "so_number": so,
            "sku": sk,
            "main_sku": main,
            "component_code": comp,
            "sku_role": str(jo.get("sku_role") or ("COMPONENT" if comp else "MAIN")),
            "process": proc,
            "available_qty": int(r.get("available_qty") or 0),
            "total_in": int(r.get("total_in") or 0),
            "total_out": int(r.get("total_out") or 0),
            "batch": r.get("batch") or "",
            "vendor_name": r.get("vendor_name") or "",
            "stock_jo_number": r.get("stock_jo_number") or "",
            "updated_at": r.get("updated_at") or "",
            "jo_numbers": str(jo.get("jo_numbers") or r.get("stock_jo_number") or ""),
            "jo_planned": int(jo.get("jo_planned") or 0),
            "jo_issued": int(jo.get("jo_issued") or 0),
            "jo_received": int(jo.get("jo_received") or 0),
            "jo_balance": int(jo.get("jo_balance") or 0),
            "jo_count": int(jo.get("jo_count") or 0),
        }
        if _match_filters(line):
            lines.append(line)

    # JO-only lines (no process_stock yet) — still show stage position via open JO
    if not so_number and not sku and not process:
        jo_candidates = jo_map.items()
    else:
        jo_candidates = jo_map.items()
    for key, jo in jo_candidates:
        if key in keys_seen:
            continue
        so, sk, proc = key
        if so_number and so != so_number:
            continue
        if sku and sk != sku:
            continue
        if process and proc != process:
            continue
        if jo_number and jo_number not in str(jo.get("jo_numbers") or "").upper():
            continue
        parsed_main, parsed_comp = _parse_main_and_component(sk)
        main = str(jo.get("main_sku") or "").strip().upper() or parsed_main
        comp = str(jo.get("component_code") or "").strip().upper() or parsed_comp
        line = {
            "so_number": so,
            "sku": sk,
            "main_sku": main,
            "component_code": comp,
            "sku_role": str(jo.get("sku_role") or ("COMPONENT" if comp else "MAIN")),
            "process": proc,
            "available_qty": 0,
            "total_in": 0,
            "total_out": 0,
            "batch": "",
            "vendor_name": "",
            "stock_jo_number": "",
            "updated_at": "",
            "jo_numbers": str(jo.get("jo_numbers") or ""),
            "jo_planned": int(jo.get("jo_planned") or 0),
            "jo_issued": int(jo.get("jo_issued") or 0),
            "jo_received": int(jo.get("jo_received") or 0),
            "jo_balance": int(jo.get("jo_balance") or 0),
            "jo_count": int(jo.get("jo_count") or 0),
        }
        if _match_filters(line):
            lines.append(line)

    lines.sort(key=lambda x: (x["so_number"], x["main_sku"], x["sku"], x["process"]))
    total = len(lines)
    page = lines[offset : offset + limit]

    # SKU overview: pivot available qty by process (same SKU, multiple stages)
    overview: dict[tuple[str, str], dict[str, Any]] = {}
    for ln in lines:
        ok = (ln["so_number"], ln["sku"])
        if ok not in overview:
            overview[ok] = {
                "so_number": ln["so_number"],
                "sku": ln["sku"],
                "main_sku": ln["main_sku"],
                "component_code": ln["component_code"],
                "stage_qty": {},
                "total_available": 0,
                "total_jo_balance": 0,
                "jo_numbers": set(),
            }
        g = overview[ok]
        g["stage_qty"][ln["process"]] = {
            "available": ln["available_qty"],
            "jo_balance": ln["jo_balance"],
        }
        g["total_available"] += ln["available_qty"]
        g["total_jo_balance"] += ln["jo_balance"]
        for part in str(ln.get("jo_numbers") or "").split(","):
            p = part.strip()
            if p:
                g["jo_numbers"].add(p)

    overview_rows = []
    for g in overview.values():
        overview_rows.append(
            {
                **{k: v for k, v in g.items() if k != "jo_numbers"},
                "jo_numbers": ",".join(sorted(g["jo_numbers"])),
            }
        )
    overview_rows.sort(key=lambda x: (x["so_number"], x["main_sku"], x["sku"]))

    return {
        "ok": True,
        "stages": stages,
        "total": total,
        "limit": limit,
        "offset": offset,
        "lines": page,
        "overview": overview_rows[: min(500, len(overview_rows))],
        "overview_total": len(overview_rows),
        "filters": {
            "so_number": so_number,
            "sku": sku,
            "main_sku": main_sku,
            "component": component,
            "jo_number": jo_number,
            "process": process,
            "q": q,
            "status": status,
            "min_available": min_available,
        },
    }


def stage_report_config() -> dict[str, Any]:
    """
    Flexible stage-report column registry — KPIs finalized later without schema rewrites.
    Columns reference existing JO / process_stock fields only.
    """
    stages = list_production_stage_names()
    base_columns = [
        {"id": "so_number", "label": "SO No.", "source": "job_orders.so_number"},
        {"id": "sku", "label": "SKU", "source": "job_orders.sku / process_stock.sku"},
        {"id": "main_sku", "label": "Main / Parent SKU", "source": "job_orders.main_sku"},
        {"id": "component_code", "label": "Component", "source": "job_orders.component_code"},
        {"id": "jo_number", "label": "Job Order", "source": "job_orders.jo_number"},
        {"id": "process", "label": "Stage", "source": "process_stock.process"},
        {"id": "available_qty", "label": "At stage (stock)", "source": "process_stock.available_qty"},
        {"id": "total_in", "label": "Received into stage", "source": "process_stock.total_in"},
        {"id": "total_out", "label": "Moved out of stage", "source": "process_stock.total_out"},
        {"id": "jo_planned", "label": "JO planned", "source": "job_orders.planned_qty"},
        {"id": "jo_issued", "label": "JO issued", "source": "job_orders.issued_qty"},
        {"id": "jo_received", "label": "JO received", "source": "job_orders.received_qty"},
        {"id": "jo_balance", "label": "JO pending", "source": "job_orders.balance_qty"},
        {"id": "updated_at", "label": "Last stock update", "source": "process_stock.updated_at"},
    ]
    return {
        "ok": True,
        "stages": stages,
        "columns": base_columns,
        "notes": (
            "Stage KPIs are intentionally generic. Add/rename columns here later; "
            "do not duplicate transactional ledgers."
        ),
    }
