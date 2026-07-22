#!/usr/bin/env python3
"""Prod smoke: operation-based partial WIP + stitching bundle gate."""
from __future__ import annotations

import sys

sys.path.insert(0, "/srv")

from backend.db import item_db, production_db


def main() -> int:
    production_db.init_db()
    item_db.init_db()
    bom = production_db.upsert_set_bom(
        {
            "style_key": "WIPROUTEDEMO",
            "style_name": "WIP Route Demo",
            "stitching_requires_complete_set": True,
            "bundle_gate_process": "Cutting",
            "lines": [
                {
                    "component_code": "FRONT",
                    "component_name": "Front",
                    "qty_per_set": 1,
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                    "default_next_process": "Embroidery",
                },
                {
                    "component_code": "BACK",
                    "component_name": "Back",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "default_next_process": "Stitching",
                },
            ],
        }
    )
    assert bom and bom.get("lines"), "set bom save failed"
    print("front_route", production_db.get_component_routing("WIPROUTEDEMO-S-FRONT"))
    print("next_cut", production_db.get_next_process("WIPROUTEDEMO-S-FRONT", "Cutting"))
    print("next_emb", production_db.get_next_process("WIPROUTEDEMO-S-FRONT", "Embroidery"))

    created = production_db.create_jo(
        {
            "jo_date": "2026-07-22",
            "so_number": "SO-WIP-DEMO",
            "sku": "WIPROUTEDEMO-S",
            "process": "Cutting",
            "planned_qty": 5,
            "create_component_jos": False,
            "lines": [{"sku": "WIPROUTEDEMO-S", "style": "S", "planned_qty": 5}],
        }
    )
    joid = created.get("id") if isinstance(created, dict) else None
    if not joid:
        rows = production_db.list_jos()
        jo = next(
            r
            for r in rows
            if r.get("so_number") == "SO-WIP-DEMO" and r.get("sku") == "WIPROUTEDEMO-S"
        )
        joid = jo["id"]
        detail = production_db.get_jo(joid)
    else:
        detail = production_db.get_jo(joid)
    line_id = detail["lines"][0]["id"]

    production_db.receive_pieces(
        joid,
        {
            "received_qty": 5,
            "process": "Cutting",
            "sku": "WIPROUTEDEMO-S",
            "jo_line_id": line_id,
            "split_components": True,
        },
    )
    production_db.issue_pieces(
        joid,
        {
            "issued_qty": 5,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "WIPROUTEDEMO-S-FRONT",
        },
    )
    board = production_db.get_partial_wip_board("SO-WIP-DEMO", "WIPROUTEDEMO-S")
    print("bundle_while_emb", board["bundle_complete"], board["message"])
    try:
        production_db.issue_pieces(
            joid,
            {
                "issued_qty": 1,
                "from_process": "Cutting",
                "to_process": "Stitching",
                "sku": "WIPROUTEDEMO-S-BACK",
            },
        )
        print("FAIL stitching not blocked")
        return 2
    except ValueError as e:
        print("stitch_blocked_ok", str(e)[:100])

    production_db.issue_pieces(
        joid,
        {
            "issued_qty": 5,
            "from_process": "Embroidery",
            "to_process": "Cutting",
            "sku": "WIPROUTEDEMO-S-FRONT",
        },
    )
    ready = production_db.preview_bundle_ready("SO-WIP-DEMO", "WIPROUTEDEMO-S")
    print("bundle_after_return", ready["bundle_complete"], ready["complete_sets"])
    production_db.issue_pieces(
        joid,
        {
            "issued_qty": 5,
            "from_process": "Cutting",
            "to_process": "Stitching",
            "sku": "WIPROUTEDEMO-S-BACK",
        },
    )
    print("PROD_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
