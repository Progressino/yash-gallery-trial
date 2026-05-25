"""Marketplace presence in snapshot inventory."""

from pathlib import Path

import pandas as pd

from backend.services.inventory import (
    _content_sniff_csv_kind,
    inventory_marketplace_breakdown,
    inventory_missing_marketplace_warnings,
    load_inventory_consolidated,
)


def test_sniff_myntra_ppmp_by_warehouse_name():
    p = Path(
        "/Users/samraisinghani/Downloads/Office/Progressino/Progressino Technologies Pvt. Ltd."
        "/Projects/Ongoing/Yash gallery ERP-CRM/Sample sheets/E- commerce Portal Sales/Inventory/04-02-2026/Myntra.csv"
    )
    if not p.is_file():
        return
    assert _content_sniff_csv_kind(p.read_bytes()) == "myntra"


def test_marketplace_breakdown_flags_missing_myntra():
    df = pd.DataFrame({
        "OMS_SKU": ["A"],
        "OMS_Inventory": [10],
        "Flipkart_Inventory": [5],
        "Amazon_Inventory": [3],
    })
    dbg = {"flipkart": "100 SKUs", "myntra": "0 SKUs (no Myntra PPMP)", "amz": "50 SKUs", "oms": "1 SKUs"}
    rows = inventory_marketplace_breakdown(df, dbg)
    myntra = next(r for r in rows if r["key"] == "Myntra_Other_Inventory")
    fk = next(r for r in rows if r["key"] == "Flipkart_Inventory")
    assert fk["included"] is True
    assert myntra["included"] is False
    hints = inventory_missing_marketplace_warnings(dbg)
    assert any("Myntra" in h for h in hints)


def test_rar_classifies_seller_inventory_report_as_myntra():
    p = Path("/Users/samraisinghani/Downloads/Inventory 25-May-26.rar")
    if not p.is_file():
        return
    from backend.services.inventory import _extract_all_from_rar

    _, man = _extract_all_from_rar(p.read_bytes())
    sir = [m for m in man if "seller_inventory_report" in m["filename"].lower()]
    assert sir
    assert all(m["category"] == "myntra" for m in sir)


def test_rar_loads_myntra_and_flipkart_columns():
    p = Path("/Users/samraisinghani/Downloads/Inventory 25-May-26.rar")
    if not p.is_file():
        return
    df, dbg = load_inventory_consolidated(None, None, None, p.read_bytes(), {}, return_debug=True)
    assert "Flipkart_Inventory" in df.columns
    assert int(df["Flipkart_Inventory"].sum()) > 0
    assert "Myntra_Other_Inventory" in df.columns
    assert int(df["Myntra_Other_Inventory"].sum()) > 0
    assert dbg.get("flipkart", "").startswith("0 SKUs") is False
    assert dbg.get("myntra", "").startswith("0 SKUs") is False
