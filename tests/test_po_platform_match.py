"""Platform match export — SKU × marketplace × quarter for File reconciliation."""

from __future__ import annotations

import pandas as pd

from backend.services.po_platform_match import (
    PLATFORM_DISPLAY_ORDER,
    build_platform_match_export_bytes,
    build_platform_match_frames,
    normalize_platform_display,
)


def test_normalize_platform_display():
    assert normalize_platform_display("Amazon") == "Amazon"
    assert normalize_platform_display("flipkart-india") == "Flipkart"
    assert normalize_platform_display("Meesho") == "Meesho"
    assert normalize_platform_display("") == "Other"
    assert normalize_platform_display("Return_Sheet") == "Other"


def test_sold_splits_platforms_and_ignores_returns():
    today = pd.Timestamp.today().normalize()
    # Place dates inside current FY quarter window (always within n_quarters=8)
    sales = pd.DataFrame(
        {
            "Sku": ["1003YKMUSTARD-L", "1003YKMUSTARD-L", "1003YKMUSTARD-L", "1592YKBLUE-XL"],
            "TxnDate": [today - pd.Timedelta(days=2)] * 3 + [today - pd.Timedelta(days=1)],
            "Quantity": [10, 3, 5, 7],
            "Transaction Type": ["Shipment", "Refund", "Shipment", "Shipment"],
            "Source": ["Amazon", "Amazon", "Meesho", "Flipkart"],
        }
    )
    frames = build_platform_match_frames(
        sales, {}, n_quarters=8, demand_basis="Sold", group_by_parent=False
    )
    long = frames["long"]
    assert not long.empty
    # Sold(Gross): Amazon shipment 10 only (refund ignored); Meesho 5; Flipkart 7
    amz = long[(long["OMS_SKU"] == "1003YKMUSTARD-L") & (long["Platform"] == "Amazon")]
    mee = long[(long["OMS_SKU"] == "1003YKMUSTARD-L") & (long["Platform"] == "Meesho")]
    flp = long[(long["OMS_SKU"] == "1592YKBLUE-XL") & (long["Platform"] == "Flipkart")]
    assert int(amz["Units"].sum()) == 10
    assert int(mee["Units"].sum()) == 5
    assert int(flp["Units"].sum()) == 7

    wide = frames["wide_total"]
    row = wide[wide["OMS_SKU"] == "1003YKMUSTARD-L"].iloc[0]
    assert int(row["Amazon"]) == 10
    assert int(row["Meesho"]) == 5
    assert int(row["Total"]) == 15


def test_net_subtracts_amazon_refund():
    today = pd.Timestamp.today().normalize()
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A"],
            "TxnDate": [today - pd.Timedelta(days=1), today - pd.Timedelta(days=1)],
            "Quantity": [10, 4],
            "Transaction Type": ["Shipment", "Refund"],
            "Source": ["Amazon", "Amazon"],
        }
    )
    frames = build_platform_match_frames(sales, {}, n_quarters=4, demand_basis="Net")
    long = frames["long"]
    assert int(long["Units"].sum()) == 6


def test_combo_listing_stays_on_listing_not_component():
    today = pd.Timestamp.today().normalize()
    sales = pd.DataFrame(
        {
            "Sku": ["1003DPT21MULTI-L"],
            "TxnDate": [today - pd.Timedelta(days=1)],
            "Quantity": [2],
            "Transaction Type": ["Shipment"],
            "Source": ["Meesho"],
        }
    )
    mapping = {"1003DPT21MULTI-L": "1003YKMUSTARD-L"}
    bom = {
        "1003DPT21MULTI-L": [
            ("1003YKMUSTARD-L", 1.0),
            ("DPT21MULTI", 1.0),
        ]
    }
    frames = build_platform_match_frames(
        sales, mapping, n_quarters=4, demand_basis="Sold", combo_sku_map=bom
    )
    skus = set(frames["long"]["OMS_SKU"].astype(str))
    assert "1003DPT21MULTI-L" in skus
    assert "1003YKMUSTARD-L" not in skus


def test_xlsx_export_bytes_nonempty():
    today = pd.Timestamp.today().normalize()
    sales = pd.DataFrame(
        {
            "Sku": ["A-L", "B-M"],
            "TxnDate": [today - pd.Timedelta(days=1)] * 2,
            "Quantity": [1, 2],
            "Transaction Type": ["Shipment", "Shipment"],
            "Source": ["Amazon", "Myntra"],
        }
    )
    body, media, fname = build_platform_match_export_bytes(
        sales, {}, n_quarters=4, demand_basis="Sold", fmt="xlsx"
    )
    assert len(body) > 100
    assert "spreadsheet" in media or media.endswith("sheet")
    assert fname.endswith(".xlsx")
    assert set(PLATFORM_DISPLAY_ORDER)  # sanity
