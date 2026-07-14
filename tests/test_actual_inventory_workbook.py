"""Actual inventory.xlsx must map 1:1 onto OMS_SKU Available Inventory."""
from pathlib import Path

import pandas as pd

from backend.services.inventory import (
    parse_actual_inventory_workbook,
    recompute_inventory_totals,
    sync_manual_overlay_from_inventory_frame,
)
from backend.session import AppSession


def test_parse_actual_inventory_workbook_matches_available():
    path = Path("/Users/samraisinghani/Downloads/Actual inventory.xlsx")
    if not path.exists():
        import pytest

        pytest.skip("Actual inventory.xlsx not present")
    raw = path.read_bytes()
    out = parse_actual_inventory_workbook(raw)
    assert not out.empty
    assert bool(out.attrs.get("actual_inventory_workbook"))

    src = pd.read_excel(path)
    src = src[src["Sku"].astype(str).str.strip().str.lower() != "total"].copy()
    src["OMS_SKU"] = (
        src["Sku"].astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True)
    )
    src = src.rename(columns={"Available Inventory": "Available"})
    m = out.merge(src[["OMS_SKU", "Available"]], on="OMS_SKU", how="inner")
    assert len(m) >= 10000
    eq = (
        pd.to_numeric(m["Total_Inventory"], errors="coerce").fillna(0).round(6)
        == pd.to_numeric(m["Available"], errors="coerce").fillna(0).round(6)
    ).sum()
    assert eq == len(m), f"Available mismatch on {len(m) - eq} SKUs"

    for sku, avail in [
        ("1001YKBEIGE-3XL", 46),
        ("1096YKBLUE-M", 294),
        ("206YK324MUSTARD-XL", 336),
        ("1303YKBLACK-L", 243),
    ]:
        hit = out[out["OMS_SKU"] == sku]
        assert len(hit) == 1
        assert float(hit.iloc[0]["Total_Inventory"]) == float(avail)


def test_sync_overlay_from_actual_frame_preserves_totals():
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["A", "B"],
            "OMS_Inventory": [10, 20],
            "Amazon_Inventory": [5, 0],
            "Manual_InTransit": [0, 2],
            "Not_In_Inventory_Qty": [3, 0],
        }
    )
    inv = recompute_inventory_totals(inv)
    sess = AppSession()
    sess.inventory_df_variant = inv.copy()
    # Stale overlay that would corrupt A if applied alone as source of truth
    sess.manual_intransit_overlay_df = pd.DataFrame(
        {"OMS_SKU": ["A"], "Manual_InTransit": [99], "Not_In_Inventory_Qty": [99]}
    )
    sync_manual_overlay_from_inventory_frame(sess, inv)
    from backend.services.manual_intransit_sheet import apply_manual_intransit_overlay_to_inventory

    apply_manual_intransit_overlay_to_inventory(sess)
    out = sess.inventory_df_variant.set_index("OMS_SKU")
    assert float(out.loc["A", "Total_Inventory"]) == 18.0
    assert float(out.loc["A", "Not_In_Inventory_Qty"]) == 3.0
    assert float(out.loc["B", "Manual_InTransit"]) == 2.0
