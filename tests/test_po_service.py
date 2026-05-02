"""PO engine smoke test with minimal sales + inventory."""

import pandas as pd

from backend.services.po_engine import calculate_po_base, calculate_quarterly_history


def _minimal_sales():
    return pd.DataFrame(
        {
            "Sku": ["TEST-SKU-1"] * 30,
            "TxnDate": pd.date_range("2025-11-01", periods=30, freq="D"),
            "Transaction Type": ["Shipment"] * 30,
            "Quantity": [2] * 30,
            "Units_Effective": [2] * 30,
            "Source": ["Myntra"] * 30,
        }
    )


def _minimal_inventory():
    return pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "Total_Inventory": [50],
        }
    )


def test_calculate_quarterly_history_returns_rows():
    sales = _minimal_sales()
    pivot = calculate_quarterly_history(
        sales_df=sales,
        mtr_df=None,
        myntra_df=None,
        sku_mapping=None,
        group_by_parent=False,
        n_quarters=4,
    )
    assert not pivot.empty
    assert "OMS_SKU" in pivot.columns


def test_calculate_po_base_non_empty():
    sales = _minimal_sales()
    inv = _minimal_inventory()
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
    )
    assert not po.empty
    assert "PO_Qty" in po.columns
    assert "ADS" in po.columns
    row = po[po["OMS_SKU"] == "TEST-SKU-1"].iloc[0]
    assert row["Sold_Units"] >= 30


def _sales_two_sizes_same_parent():
    """Shared parent CUTPARENT; L moves more units than XL."""
    days = pd.date_range("2025-11-01", periods=30, freq="D")
    rows = []
    for d in days:
        rows.append(
            {"Sku": "CUTPARENT-L", "TxnDate": d, "Transaction Type": "Shipment", "Quantity": 5, "Units_Effective": 5}
        )
        rows.append(
            {"Sku": "CUTPARENT-XL", "TxnDate": d, "Transaction Type": "Shipment", "Quantity": 1, "Units_Effective": 1}
        )
    return pd.DataFrame(rows)


def test_cutting_ratio_single_gross_size_no_ads_split():
    """One size has gross PO need and net PO is zero: cut ratio stays on that size only."""
    sales = _sales_two_sizes_same_parent()
    # L is understocked (gross PO > 0); XL is flush — only L has gross requirement.
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["CUTPARENT-L", "CUTPARENT-XL"],
            "Total_Inventory": [0, 99_999],
        }
    )
    existing = pd.DataFrame(
        {
            "OMS_SKU": ["CUTPARENT-L", "CUTPARENT-XL"],
            "PO_Pipeline_Total": [2_000_000, 0],
        }
    )
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        existing_po_df=existing,
        enforce_two_size_minimum=False,
    )
    row_l = po[po["OMS_SKU"] == "CUTPARENT-L"].iloc[0]
    row_xl = po[po["OMS_SKU"] == "CUTPARENT-XL"].iloc[0]
    assert row_l["PO_Qty"] == 0 and row_xl["PO_Qty"] == 0
    assert row_l["Gross_PO_Qty"] > 0
    assert row_xl["Gross_PO_Qty"] == 0
    # Only one variant should carry the cutting share (not ADS-split across both).
    assert float(row_l["Cutting_Ratio"]) == 1.0
    assert float(row_xl["Cutting_Ratio"]) == 0.0


def test_cutting_ratio_two_gross_sizes_uses_demand_split():
    """Two sizes with gross need and no net PO: split by ADS (or gross) across the family."""
    sales = _sales_two_sizes_same_parent()
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["CUTPARENT-L", "CUTPARENT-XL"],
            "Total_Inventory": [0, 0],
        }
    )
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        enforce_two_size_minimum=False,
    )
    row_l = po[po["OMS_SKU"] == "CUTPARENT-L"].iloc[0]
    row_xl = po[po["OMS_SKU"] == "CUTPARENT-XL"].iloc[0]
    assert row_l["Gross_PO_Qty"] > 0 and row_xl["Gross_PO_Qty"] > 0
    assert row_l["Cutting_Ratio"] > row_xl["Cutting_Ratio"]
    assert abs(float(row_l["Cutting_Ratio"]) + float(row_xl["Cutting_Ratio"]) - 1.0) < 0.02


def test_sku_status_parse_detects_closed_and_lead():
    from backend.services.sku_status_lead import parse_sku_status_lead_dataframe

    raw = pd.DataFrame(
        {
            "SKU": ["ABC-L", "XYZ-M"],
            "Status": ["Open", "Closed SKU"],
            "Lead Time": [10, 22],
        }
    )
    out = parse_sku_status_lead_dataframe(raw, None)
    assert len(out) == 2
    r = out.set_index("OMS_SKU").loc["XYZ-M"]
    assert bool(r["SKU_Sheet_Closed"]) is True
    assert int(r["Lead_Time_From_Sheet"]) == 22


def test_closed_sku_sheet_suppresses_po_qty():
    sales = _minimal_sales()
    inv = _minimal_inventory()
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "SKU_Sheet_Status": ["Closed SKU"],
            "SKU_Sheet_Closed": [True],
            "Lead_Time_From_Sheet": [float("nan")],
        }
    )
    po = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=sheet)
    assert int(po.iloc[0]["Gross_PO_Qty"]) == 0
    assert "Closed" in str(po.iloc[0]["PO_Block_Reason"])


def test_single_size_minimum_zeros_gross_when_only_one_variant_needs_stock():
    sales = _sales_two_sizes_same_parent()
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["CUTPARENT-L", "CUTPARENT-XL"],
            "Total_Inventory": [0, 99_999],
        }
    )
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        enforce_two_size_minimum=True,
    )
    row_l = po[po["OMS_SKU"] == "CUTPARENT-L"].iloc[0]
    assert int(row_l["Gross_PO_Qty"]) == 0
    assert "Single size" in str(row_l["PO_Block_Reason"])


def test_sheet_lead_time_applied_per_row():
    sales = _minimal_sales()
    inv = _minimal_inventory()
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "SKU_Sheet_Status": ["Active"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [45.0],
        }
    )
    po = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=sheet)
    assert int(po.iloc[0]["Lead_Time_Days"]) == 45
