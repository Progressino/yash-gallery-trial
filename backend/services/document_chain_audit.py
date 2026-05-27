"""Audit trail: MRP commitments → PO / JO → GRN → grey ledger for one SO."""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional


def _purchase_db() -> str:
    return os.environ.get(
        "PURCHASE_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "purchase.db"),
    )


def _production_db() -> str:
    return os.environ.get(
        "PRODUCTION_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "production.db"),
    )


def _grey_db() -> str:
    return os.environ.get(
        "GREY_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "grey.db"),
    )


def get_document_chain_audit(so_number: str) -> Dict[str, Any]:
    so = (so_number or "").strip()
    if not so:
        return {"so_number": "", "materials": [], "error": "so_number required"}

    materials: Dict[str, Dict[str, Any]] = {}

    # MRP commitments + production JOs
    if os.path.exists(_production_db()):
        conn = sqlite3.connect(_production_db())
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT * FROM mrp_material_commitments WHERE so_number=? ORDER BY material_code",
            (so,),
        ):
            d = dict(row)
            code = d["material_code"]
            mrp = float(d.get("mrp_qty") or 0)
            po_c = float(d.get("po_committed_qty") or 0)
            jo_c = float(d.get("jo_committed_qty") or 0)
            materials[code] = {
                "material_code": code,
                "material_name": d.get("material_name", ""),
                "unit": d.get("unit", "PCS"),
                "mrp_qty": mrp,
                "po_committed_qty": po_c,
                "jo_committed_qty": jo_c,
                "remaining_qty": round(max(0.0, mrp - po_c - jo_c), 6),
                "commitment_status": d.get("status", "Open"),
                "pos": [],
                "jwos": [],
                "grns": [],
                "job_orders": [],
                "grey_trackers": [],
                "grey_ledger": [],
            }
        for jo in conn.execute(
            """SELECT id, jo_number, process, status, sku, fabric_code, fabric_qty, fabric_unit,
               planned_qty, received_qty, created_at FROM job_orders WHERE so_number=? ORDER BY id DESC""",
            (so,),
        ):
            j = dict(jo)
            fc = (j.get("fabric_code") or "").strip()
            if fc:
                materials.setdefault(
                    fc,
                    {
                        "material_code": fc,
                        "material_name": "",
                        "unit": j.get("fabric_unit") or "MTR",
                        "mrp_qty": 0,
                        "po_committed_qty": 0,
                        "jo_committed_qty": 0,
                        "remaining_qty": 0,
                        "commitment_status": "Open",
                        "pos": [],
                        "jwos": [],
                        "grns": [],
                        "job_orders": [],
                        "grey_trackers": [],
                        "grey_ledger": [],
                    },
                )
                materials[fc]["job_orders"].append(
                    {
                        "jo_number": j["jo_number"],
                        "process": j.get("process"),
                        "status": j.get("status"),
                        "sku": j.get("sku"),
                        "fabric_qty": float(j.get("fabric_qty") or 0),
                        "planned_qty": int(j.get("planned_qty") or 0),
                        "received_qty": int(j.get("received_qty") or 0),
                    }
                )
        conn.close()

    if os.path.exists(_purchase_db()):
        conn = sqlite3.connect(_purchase_db())
        conn.row_factory = sqlite3.Row
        for po in conn.execute(
            "SELECT * FROM po_headers WHERE so_reference=? ORDER BY id DESC",
            (so,),
        ):
            po_d = dict(po)
            lines = [
                dict(l)
                for l in conn.execute("SELECT * FROM po_lines WHERE po_id=?", (po_d["id"],)).fetchall()
            ]
            for ln in lines:
                code = ln["material_code"]
                materials.setdefault(
                    code,
                    {
                        "material_code": code,
                        "material_name": ln.get("material_name", ""),
                        "unit": ln.get("unit", "PCS"),
                        "mrp_qty": 0,
                        "po_committed_qty": 0,
                        "jo_committed_qty": 0,
                        "remaining_qty": 0,
                        "commitment_status": "Open",
                        "pos": [],
                        "jwos": [],
                        "grns": [],
                        "job_orders": [],
                        "grey_trackers": [],
                        "grey_ledger": [],
                    },
                )
                materials[code]["pos"].append(
                    {
                        "po_number": po_d["po_number"],
                        "status": po_d.get("status"),
                        "po_qty": float(ln.get("po_qty") or 0),
                        "grn_accepted_qty": float(ln.get("grn_accepted_qty") or 0),
                        "po_date": po_d.get("po_date"),
                    }
                )
        for jwo in conn.execute(
            "SELECT * FROM jwo_headers WHERE so_reference=? ORDER BY id DESC",
            (so,),
        ):
            jwo_d = dict(jwo)
            lines = [
                dict(l)
                for l in conn.execute("SELECT * FROM jwo_lines WHERE jwo_id=?", (jwo_d["id"],)).fetchall()
            ]
            for ln in lines:
                out_code = ln.get("output_material") or ""
                if out_code:
                    materials.setdefault(
                        out_code,
                        {
                            "material_code": out_code,
                            "material_name": "",
                            "unit": ln.get("output_unit", "MTR"),
                            "mrp_qty": 0,
                            "po_committed_qty": 0,
                            "jo_committed_qty": 0,
                            "remaining_qty": 0,
                            "commitment_status": "Open",
                            "pos": [],
                            "jwos": [],
                            "grns": [],
                            "job_orders": [],
                            "grey_trackers": [],
                            "grey_ledger": [],
                        },
                    )
                    materials[out_code]["jwos"].append(
                        {
                            "jwo_number": jwo_d["jwo_number"],
                            "status": jwo_d.get("status"),
                            "output_qty": float(ln.get("output_qty") or 0),
                            "input_material": ln.get("input_material"),
                            "input_qty": float(ln.get("input_qty") or 0),
                        }
                    )
        for grn in conn.execute(
            """SELECT * FROM grn_headers WHERE so_reference=? OR reference_number IN (
                SELECT po_number FROM po_headers WHERE so_reference=?
            ) ORDER BY id DESC""",
            (so, so),
        ):
            g = dict(grn)
            g_lines = [
                dict(l)
                for l in conn.execute("SELECT * FROM grn_lines WHERE grn_id=?", (g["id"],)).fetchall()
            ]
            for ln in g_lines:
                code = ln.get("material_code") or ""
                if not code:
                    continue
                materials.setdefault(
                    code,
                    {
                        "material_code": code,
                        "material_name": ln.get("material_name", ""),
                        "unit": ln.get("unit", "PCS"),
                        "mrp_qty": 0,
                        "po_committed_qty": 0,
                        "jo_committed_qty": 0,
                        "remaining_qty": 0,
                        "commitment_status": "Open",
                        "pos": [],
                        "jwos": [],
                        "grns": [],
                        "job_orders": [],
                        "grey_trackers": [],
                        "grey_ledger": [],
                    },
                )
                materials[code]["grns"].append(
                    {
                        "grn_number": g["grn_number"],
                        "grn_type": g.get("grn_type"),
                        "reference_number": g.get("reference_number"),
                        "status": g.get("status"),
                        "accepted_qty": float(ln.get("accepted_qty") or 0),
                        "inventory_posted": bool(g.get("inventory_posted")),
                        "grn_date": g.get("grn_date"),
                    }
                )
        conn.close()

    if os.path.exists(_grey_db()):
        gconn = sqlite3.connect(_grey_db())
        gconn.row_factory = sqlite3.Row
        for tr in gconn.execute(
            "SELECT * FROM grey_tracker WHERE so_reference=? ORDER BY id DESC",
            (so,),
        ):
            t = dict(tr)
            code = t.get("material_code") or ""
            if code:
                materials.setdefault(
                    code,
                    {
                        "material_code": code,
                        "material_name": t.get("material_name", ""),
                        "unit": "MTR",
                        "mrp_qty": 0,
                        "po_committed_qty": 0,
                        "jo_committed_qty": 0,
                        "remaining_qty": 0,
                        "commitment_status": "Open",
                        "pos": [],
                        "jwos": [],
                        "grns": [],
                        "job_orders": [],
                        "grey_trackers": [],
                        "grey_ledger": [],
                    },
                )
                materials[code]["grey_trackers"].append(
                    {
                        "tracker_key": t.get("tracker_key"),
                        "po_number": t.get("po_number"),
                        "status": t.get("status"),
                        "ordered_qty": float(t.get("ordered_qty") or 0),
                        "in_transit_qty": float(t.get("in_transit_qty") or 0),
                        "transport_qty": float(t.get("transport_qty") or 0),
                        "factory_qty": float(t.get("factory_qty") or 0),
                    }
                )
            tid = t.get("id")
            if tid:
                for le in gconn.execute(
                    "SELECT * FROM grey_ledger WHERE tracker_id=? ORDER BY id DESC LIMIT 50",
                    (tid,),
                ):
                    ld = dict(le)
                    mc = ld.get("material_code") or code
                    if mc and mc in materials:
                        materials[mc]["grey_ledger"].append(
                            {
                                "entry_date": ld.get("entry_date"),
                                "transaction_type": ld.get("transaction_type"),
                                "qty": float(ld.get("qty") or 0),
                                "from_location": ld.get("from_location"),
                                "to_location": ld.get("to_location"),
                                "reference_no": ld.get("reference_no"),
                                "remarks": ld.get("remarks"),
                            }
                        )
        gconn.close()

    return {
        "so_number": so,
        "materials": sorted(materials.values(), key=lambda m: m["material_code"]),
    }
