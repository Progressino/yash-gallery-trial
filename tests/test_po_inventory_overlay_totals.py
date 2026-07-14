"""PO Total_Inventory must match Inventory tab (manual overlay + source recompute)."""
import pandas as pd

from backend.services.daily_inventory_history import overlay_inventory_variant_from_history
from backend.services.inventory import recompute_inventory_totals
from backend.services.manual_intransit_sheet import apply_manual_intransit_overlay_to_inventory
from backend.session import AppSession


def test_stale_baked_not_in_qty_cleared_by_overlay_apply():
    """Warm parquet sometimes bakes wrong Not_In_Inventory_Qty into Total_Inventory."""
    sess = AppSession()
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": ["206YK324MUSTARD-XL", "1096YKBLUE-M"],
            "OMS_Inventory": [43.0, 241.0],
            "Amazon_Inventory": [18.0, 40.0],
            "Myntra_Other_Inventory": [1.0, 8.0],
            "Flipkart_Inventory": [0.0, 0.0],
            # Stale baked values (corrupt warm) — should be replaced by overlay.
            "Not_In_Inventory_Qty": [255.0, 0.0],
            "Manual_InTransit": [0.0, 0.0],
            "Marketplace_Total": [274.0, 48.0],
            "Total_Inventory": [317.0, 289.0],
        }
    )
    sess.manual_intransit_overlay_df = pd.DataFrame(
        {
            "OMS_SKU": ["1096YKBLUE-M"],
            "Manual_InTransit": [3],
            "Not_In_Inventory_Qty": [201],
        }
    )
    apply_manual_intransit_overlay_to_inventory(sess)
    out = sess.inventory_df_variant.set_index("OMS_SKU")
    assert float(out.loc["206YK324MUSTARD-XL", "Total_Inventory"]) == 62.0
    assert float(out.loc["206YK324MUSTARD-XL", "Not_In_Inventory_Qty"]) == 0.0
    assert float(out.loc["1096YKBLUE-M", "Total_Inventory"]) == 493.0
    assert float(out.loc["1096YKBLUE-M", "Not_In_Inventory_Qty"]) == 201.0


def test_history_overlay_recomputes_marketplace_from_source_cols():
    """History OMS refresh must not freeze a stale (Total−OMS) marketplace gap."""
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "OMS_Inventory": [40.0],
            "Amazon_Inventory": [10.0],
            "Not_In_Inventory_Qty": [0.0],
            "Marketplace_Total": [200.0],  # stale gap vs channels
            "Total_Inventory": [240.0],
        }
    )
    hist = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"] * 3,
            "Date": pd.to_datetime(["2026-07-12", "2026-07-13", "2026-07-14"]),
            "Qty": [40.0, 40.0, 43.0],
        }
    )
    out, meta = overlay_inventory_variant_from_history(
        inv,
        hist,
        snapshot_date="2026-07-10",
        reference_date="2026-07-14",
    )
    assert meta["applied"] is True
    row = out.iloc[0]
    assert float(row["OMS_Inventory"]) == 43.0
    assert float(row["Marketplace_Total"]) == 10.0
    assert float(row["Total_Inventory"]) == 53.0


def test_recompute_includes_manual_not_in():
    df = pd.DataFrame(
        {
            "OMS_SKU": ["A"],
            "OMS_Inventory": [66],
            "Amazon_Inventory": [17],
            "Flipkart_Inventory": [4],
            "Manual_InTransit": [6],
            "Not_In_Inventory_Qty": [14],
        }
    )
    out = recompute_inventory_totals(df)
    assert float(out.iloc[0]["Marketplace_Total"]) == 41.0
    assert float(out.iloc[0]["Total_Inventory"]) == 107.0
