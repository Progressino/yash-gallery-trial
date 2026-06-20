"""Inventory clear endpoint and Myntra duplicate-snapshot handling."""

import pandas as pd

from backend.session import AppSession
from backend.services.inventory import (
    _parse_myntra_other,
    clear_inventory_snapshot,
    load_inventory_consolidated,
    recompute_inventory_totals,
)


def test_clear_inventory_snapshot_empties_session():
    sess = AppSession()
    sess.inventory_df_variant = pd.DataFrame({"OMS_SKU": ["A"], "Total_Inventory": [99]})
    sess.inventory_df_parent = sess.inventory_df_variant.copy()
    sess.inventory_debug = {"oms": "1 SKUs"}
    sess.inventory_snapshot_date = "2026-06-18"
    sess.inventory_snapshot_date_label = "18 Jun 2026"
    sess.inventory_upload_status = "done"

    clear_inventory_snapshot(sess)

    assert sess.inventory_df_variant.empty
    assert sess.inventory_df_parent.empty
    assert sess.inventory_debug == {}
    assert sess.inventory_snapshot_date == ""
    assert sess.inventory_upload_status == "idle"


def test_myntra_duplicate_snapshots_use_max_not_sum():
    mapping = {"SKU-A": "SKU-A"}
    csv_hi = b"seller sku code,inventory count\nSKU-A,180\n"
    csv_lo = b"seller sku code,inventory count\nSKU-A,90\n"
    parts = [_parse_myntra_other(csv_hi, mapping), _parse_myntra_other(csv_lo, mapping)]
    m_all = pd.concat(parts, ignore_index=True)
    part = m_all.groupby("OMS_SKU")["Myntra_Other_Inventory"].max().reset_index()
    assert int(part["Myntra_Other_Inventory"].iloc[0]) == 180


def test_myntra_warehouse_filter_other_only():
    mapping = {"SKU-A": "SKU-A"}
    csv_bytes = (
        b"seller sku code,inventory count,warehouse name\n"
        b"SKU-A,50,Myntra Other Warehouse\n"
        b"SKU-A,200,Myntra Primary Warehouse\n"
    )
    out = _parse_myntra_other(csv_bytes, mapping)
    assert len(out) == 1
    assert int(out["Myntra_Other_Inventory"].iloc[0]) == 50


def test_clear_inventory_platform_endpoint(client, session_for_client, monkeypatch):
    """DELETE /upload/clear/inventory clears snapshot without touching sales."""
    import backend.main as main_mod

    monkeypatch.setattr(main_mod, "merge_inventory_into_warm_cache", lambda _s: None)
    monkeypatch.setattr(
        "backend.services.upload_policy.may_delete_upload_data",
        lambda _role, _username=None: True,
    )

    _, sess = session_for_client
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["A"], "OMS_Inventory": [10], "Total_Inventory": [10]}
    )
    sess.sales_df = pd.DataFrame({"Sku": ["A"], "Quantity": [1]})

    r = client.delete("/api/upload/clear/inventory")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert sess.inventory_df_variant.empty
    assert not sess.sales_df.empty


def test_load_inventory_myntra_parts_max_across_files():
    mapping = {"SKU-A": "SKU-A"}
    csv1 = b"seller sku code,inventory count\nSKU-A,100\n"
    csv2 = b"seller sku code,inventory count\nSKU-A,40\n"
    df, dbg = load_inventory_consolidated(
        None, None, [csv1, csv2], None, mapping, return_debug=True
    )
    assert int(df["Myntra_Other_Inventory"].sum()) == 100
    assert "2 file payload" in dbg.get("myntra", "")
