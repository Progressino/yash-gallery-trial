#!/usr/bin/env python3
"""Prod smoke: TEST SKU-TOP Cutting JO panel WIP (UI + stock flow)."""
from __future__ import annotations

import sys

sys.path.insert(0, "/srv")

from backend.db import production_db


def main() -> int:
    production_db.init_db()
    jo = next(
        (
            j
            for j in production_db.list_jos(so_number="SO-0005")
            if str(j.get("sku") or "").upper() == "TEST SKU-TOP"
        ),
        None,
    )
    if not jo:
        print("FAIL no TEST SKU-TOP JO for SO-0005")
        return 1

    ctx = production_db.get_jo_panel_wip(int(jo["id"]))
    print("jo", jo["jo_number"], jo["sku"], "received", jo.get("received_qty"))
    print("has_panels", ctx.get("has_panels"), "panels", [p["component_code"] for p in ctx.get("panels") or []])
    if not ctx.get("has_panels"):
        print("FAIL panel context missing")
        return 2

    line = (jo.get("lines") or [{}])[0]
  # receive if not yet done
    if int(jo.get("received_qty") or 0) < 10:
        production_db.receive_pieces(
            int(jo["id"]),
            {
                "received_qty": 10,
                "process": "Cutting",
                "sku": "TEST SKU-TOP",
                "jo_line_id": line.get("id"),
                "split_components": True,
            },
        )
        print("received 10 on TEST SKU-TOP")

    ctx2 = production_db.get_jo_panel_wip(int(jo["id"]))
    front = next(p for p in ctx2["panels"] if p["component_code"] == "FRONT")
    back = next(p for p in ctx2["panels"] if p["component_code"] == "BACK")
    print("front_cutting", front.get("issueable_qty"), "back_cutting", back.get("issueable_qty"))
    if front.get("issueable_qty", 0) < 10 or back.get("issueable_qty", 0) < 10:
        print("FAIL panel stock not created after receive")
        return 3

    production_db.issue_pieces(
        int(jo["id"]),
        {
            "issued_qty": 10,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "TEST SKU-FRONT",
        },
    )
    ctx3 = production_db.get_jo_panel_wip(int(jo["id"]))
    front3 = next(p for p in ctx3["panels"] if p["component_code"] == "FRONT")
    print("front_after_emb_issue", front3.get("embroidery_outstanding"), front3.get("current_location"))
    if front3.get("embroidery_outstanding", 0) != 10:
        print("FAIL FRONT not in embroidery")
        return 4

    try:
        production_db.issue_pieces(
            int(jo["id"]),
            {
                "issued_qty": 1,
                "from_process": "Cutting",
                "to_process": "Stitching",
                "sku": "TEST SKU-BACK",
            },
        )
        print("FAIL stitching should be blocked while FRONT in embroidery")
        return 5
    except ValueError as e:
        print("stitch_blocked_ok", str(e)[:80])

    production_db.issue_pieces(
        int(jo["id"]),
        {
            "issued_qty": 10,
            "from_process": "Embroidery",
            "to_process": "Cutting",
            "sku": "TEST SKU-FRONT",
        },
    )
    production_db.issue_pieces(
        int(jo["id"]),
        {
            "issued_qty": 10,
            "from_process": "Cutting",
            "to_process": "Stitching",
            "sku": "TEST SKU-BACK",
        },
    )
    ready = production_db.preview_bundle_ready("SO-0005", "TEST SKU")
    print("bundle_complete", ready.get("bundle_complete"), ready.get("complete_sets"))
    print("PROD_TEST_SKU_PANEL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
