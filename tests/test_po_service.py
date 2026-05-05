"""PO engine smoke test with minimal sales + inventory."""

import pandas as pd
import pytest

from backend.services.helpers import get_parent_sku
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


def test_calculate_quarterly_history_shipment_type_case_insensitive():
    days = pd.date_range("2025-06-01", periods=10, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["CASE-SHIP"] * 10,
            "TxnDate": days,
            "Transaction Type": ["SHIPMENT"] * 10,
            "Quantity": [2] * 10,
        }
    )
    pivot = calculate_quarterly_history(
        sales_df=sales,
        mtr_df=None,
        myntra_df=None,
        sku_mapping=None,
        group_by_parent=False,
        n_quarters=2,
    )
    assert not pivot.empty
    assert (pivot["OMS_SKU"] == "CASE-SHIP").any()


def test_sheet_lead_on_parent_applies_to_variant_inventory_sku():
    """Style-level lead row should propagate to size variants (e.g. STYLE-M)."""
    days = pd.date_range("2025-11-01", periods=25, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["PARENTLEAD-M"] * 25,
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * 25,
            "Quantity": [2] * 25,
            "Units_Effective": [2] * 25,
            "Source": ["Amazon"] * 25,
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["PARENTLEAD-M"], "Total_Inventory": [80]})
    sku_status = pd.DataFrame(
        {
            "OMS_SKU": ["PARENTLEAD"],
            "SKU_Sheet_Status": ["Open"],
            "Lead_Time_From_Sheet": [33.0],
            "SKU_Sheet_Closed": [False],
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
        sku_status_df=sku_status,
    )
    row = po.iloc[0]
    assert int(row["Lead_Time_Days"]) == 33


def test_sheet_lead_numeric_style_row_applies_to_full_oms_sku():
    """Sheet lists style ``1657`` only; inventory / sales use ``1657YKWHITE-M`` (common in Excel)."""
    days = pd.date_range("2025-11-01", periods=25, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["1657YKWHITE-M"] * 25,
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * 25,
            "Quantity": [2] * 25,
            "Units_Effective": [2] * 25,
            "Source": ["Amazon"] * 25,
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["1657YKWHITE-M"], "Total_Inventory": [80]})
    sku_status = pd.DataFrame(
        {
            "OMS_SKU": ["1657"],
            "SKU_Sheet_Status": ["Open"],
            "Lead_Time_From_Sheet": [30.0],
            "SKU_Sheet_Closed": [False],
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
        sku_status_df=sku_status,
    )
    row = po[po["OMS_SKU"] == "1657YKWHITE-M"].iloc[0]
    assert int(row["Lead_Time_Days"]) == 30


def test_eff_days_uses_active_demand_span_not_trailing_calendar():
    """Eff_Days counts first→last shipment in the ADS window, not empty days to max_date."""
    rows = []
    for d in pd.date_range("2025-11-01", periods=10, freq="D"):
        rows.append(
            {
                "Sku": "ACTIVEDAYS-SKU",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 3,
                "Units_Effective": 3,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["ACTIVEDAYS-SKU"], "Total_Inventory": [500]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        min_denominator=7,
    )
    row = po.iloc[0]
    assert int(row["Eff_Days"]) == 10
    # 10 days × 3 units = 30 sold; Recent_ADS = 30 / 10
    assert abs(float(row["Recent_ADS"]) - 3.0) < 0.02


def test_eff_days_not_forced_to_min_denominator_for_single_active_day():
    """One active shipment day should keep Eff_Days=1 (not forced to 7/30)."""
    sales = pd.DataFrame(
        [
            {
                "Sku": "ONEDAY-SKU",
                "TxnDate": pd.Timestamp("2025-11-10"),
                "Transaction Type": "Shipment",
                "Quantity": 21,
                "Units_Effective": 21,
                "Source": "Amazon",
            }
        ]
    )
    inv = pd.DataFrame({"OMS_SKU": ["ONEDAY-SKU"], "Total_Inventory": [100]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        min_denominator=7,
    )
    row = po.iloc[0]
    assert int(row["Eff_Days"]) == 1
    assert abs(float(row["Recent_ADS"]) - 21.0) < 0.02


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


def test_sku_status_pick_lead_time_days_column_name():
    from backend.services.sku_status_lead import parse_sku_status_lead_dataframe

    raw = pd.DataFrame(
        {
            "SKU": ["A-1"],
            "Status": ["Open"],
            "Lead_Time_Days": [33],
        }
    )
    out = parse_sku_status_lead_dataframe(raw, None)
    assert int(out.iloc[0]["Lead_Time_From_Sheet"]) == 33


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


def test_closed_sku_sheet_does_not_change_po_only_lead_time_matters():
    """Sheet 'closed' is informational; PO qty matches the engine without sheet (same global lead)."""
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
    po_sheet = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=sheet)
    po_none = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=None)
    assert int(po_sheet.iloc[0]["Gross_PO_Qty"]) == int(po_none.iloc[0]["Gross_PO_Qty"])
    assert int(po_sheet.iloc[0]["PO_Qty"]) == int(po_none.iloc[0]["PO_Qty"])
    assert bool(po_sheet.iloc[0]["SKU_Sheet_Closed"]) is True


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


def test_sheet_without_positive_lead_matches_no_sheet_po():
    """Status-only row (no numeric lead) keeps global lead_time → same PO as no upload."""
    sales = _minimal_sales()
    inv = _minimal_inventory()
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "SKU_Sheet_Status": ["Open"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [float("nan")],
        }
    )
    po_sheet = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=sheet)
    po_none = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=None)
    assert int(po_sheet.iloc[0]["Gross_PO_Qty"]) == int(po_none.iloc[0]["Gross_PO_Qty"])
    assert float(po_sheet.iloc[0]["ADS"]) == float(po_none.iloc[0]["ADS"])


def test_sheet_lead_applies_with_inventory_sku_spacing_and_case():
    """Inventory / sheet SKUs are canonicalized so per-SKU lead matches (not stuck on global default)."""
    sales = _minimal_sales()
    inv = pd.DataFrame({"OMS_SKU": ["  test-sku-1\t"], "Total_Inventory": [50]})
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "SKU_Sheet_Status": ["Open"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [88.0],
        }
    )
    po = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=sheet)
    assert int(po.iloc[0]["Lead_Time_Days"]) == 88


def test_sheet_lead_rollup_to_parent_when_group_by_parent():
    """Lead sheet lists size-level SKUs; parent-level PO rows pick up max positive lead per parent."""
    days = pd.date_range("2025-11-01", periods=30, freq="D")
    rows = []
    for d in days:
        rows.append(
            {
                "Sku": "MYSTYLE-L",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 2,
                "Units_Effective": 2,
                "Source": "Amz",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["MYSTYLE"], "Total_Inventory": [50]})
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["MYSTYLE-L", "MYSTYLE-XL"],
            "SKU_Sheet_Status": ["Open", "Open"],
            "SKU_Sheet_Closed": [False, False],
            "Lead_Time_From_Sheet": [15.0, 25.0],
        }
    )
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        safety_pct=0.0,
        sku_status_df=sheet,
        group_by_parent=True,
    )
    row = po[po["OMS_SKU"] == "MYSTYLE"].iloc[0]
    assert int(row["Lead_Time_Days"]) == 25
    # Sales were uploaded as ``MYSTYLE-L``; parent inventory row must inherit rolled-up units.
    assert int(row["Sold_Units"]) == 60  # 30 days × 2 units/day, all in-period shipments


def test_days_left_uses_total_inventory_when_column_present():
    """Days cover reflects Total_Inventory when OMS warehouse column is zero but FBA/total stock exists."""
    sales = _minimal_sales()
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "Total_Inventory": [60],
            "OMS_Inventory": [0],
        }
    )
    po = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0)
    assert float(po.iloc[0]["ADS"]) > 0
    assert float(po.iloc[0]["Days_Left"]) > 0


def test_sheet_lead_increases_gross_vs_shorter_global_lead():
    """Larger uploaded lead increases lead_demand vs default global lead (same ADS/inventory)."""
    sales = _minimal_sales()
    inv = _minimal_inventory()
    short = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=None)
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "SKU_Sheet_Status": ["Active"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [90.0],
        }
    )
    long_lead = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, sku_status_df=sheet)
    assert int(long_lead.iloc[0]["Lead_Time_Days"]) == 90
    assert int(long_lead.iloc[0]["Gross_PO_Qty"]) > int(short.iloc[0]["Gross_PO_Qty"])
    assert float(long_lead.iloc[0]["ADS"]) == float(short.iloc[0]["ADS"])


def test_get_parent_sku_preserves_two_part_numeric_style_code():
    """``AK-1394`` style IDs must not collapse to ``AK`` (digit-only segment is not a size)."""
    assert get_parent_sku("AK-1394") == "AK-1394"
    assert get_parent_sku("AK-1394BROWN-L") == "AK-1394BROWN"
    assert get_parent_sku("1657-M") == "1657"


def test_sheet_lead_from_style_code_applies_to_color_size_variant():
    """Lead sheet lists ``1394``; inventory ``AK-1394BROWN-L`` inherits that lead."""
    days = pd.date_range("2025-11-01", periods=30, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["AK-1394BROWN-L"] * 30,
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * 30,
            "Quantity": [1] * 30,
            "Units_Effective": [1] * 30,
            "Source": ["Amazon"] * 30,
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["AK-1394BROWN-L"], "Total_Inventory": [21]})
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["1394"],
            "SKU_Sheet_Status": ["Open"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [55.0],
        }
    )
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=30,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        sku_status_df=sheet,
    )
    row = po.iloc[0]
    assert int(row["Lead_Time_Days"]) == 55
    ads = float(row["ADS"])
    assert ads > 0
    assert float(row["Projected_Running_Days"]) == pytest.approx(round(21 / ads, 1), abs=0.05)
    assert float(row["Days_Left"]) == pytest.approx(float(row["Projected_Running_Days"]), abs=0.01)


def test_projected_running_days_is_total_inventory_plus_pipeline_over_ads():
    """Sheet formula: (PO_Pipeline_Total + Total_Inventory) / ADS."""
    sales = _minimal_sales()
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["TEST-SKU-1"],
            "Total_Inventory": [40],
            "OMS_Inventory": [10],
        }
    )
    existing = pd.DataFrame({"OMS_SKU": ["TEST-SKU-1"], "PO_Pipeline_Total": [20]})
    po = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, existing_po_df=existing)
    row = po.iloc[0]
    ads = float(row["ADS"])
    assert ads > 0
    expected = round((40 + 20) / ads, 1)
    expected_days_left = round(40 / ads, 1)
    assert float(row["Projected_Running_Days"]) == pytest.approx(expected, abs=0.05)
    assert float(row["Days_Left"]) == pytest.approx(expected_days_left, abs=0.05)
    # Uses Total_Inventory (40) in numerator, not OMS_Inventory (10) alone.
    assert float(row["Projected_Running_Days"]) != pytest.approx(round((10 + 20) / ads, 1), abs=0.05)


def test_ads_equals_sales_over_effective_days():
    """ADS uses Sales / Eff_Days (no seasonal/flat overrides)."""
    rows = []
    for d in pd.date_range("2025-11-01", periods=10, freq="D"):
        rows.append(
            {
                "Sku": "ADS-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 3,
                "Units_Effective": 3,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["ADS-SKU-1"], "Total_Inventory": [20]})
    po = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, demand_basis="Sold")
    r = po.iloc[0]
    assert int(r["Eff_Days"]) == 10
    assert float(r["Sold_Units"]) == pytest.approx(30.0, abs=0.01)
    assert float(r["ADS"]) == pytest.approx(3.0, abs=0.02)
    assert float(r["Days_Left"]) == pytest.approx(round(20 / 3.0, 1), abs=0.05)


def test_variant_mode_parent_inventory_sku_falls_back_to_child_sales_for_ads():
    """
    Some inventory files contain parent-like OMS_SKU (no size suffix) even in variant mode.
    When exact sales key is missing but child sizes sold, PO should inherit parent rollup.
    """
    rows = []
    for d in pd.date_range("2025-11-01", periods=30, freq="D"):
        rows.append(
            {
                "Sku": "1007YKBLACK-XL",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 2,
                "Units_Effective": 2,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["1007YKBLACK"], "Total_Inventory": [40]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        group_by_parent=False,
    )
    row = po.iloc[0]
    assert int(row["Sold_Units"]) == 60
    assert float(row["ADS"]) == pytest.approx(2.0, abs=0.02)
    assert float(row["Flat30_ADS"]) > 0
    assert float(row["LY_ADS"]) >= 0


def test_variant_mode_parent_fallback_handles_non_delimited_size_suffix():
    """Sales SKU like ``1007YKBLACKXXL`` should roll up to parent ``1007YKBLACK``."""
    rows = []
    for d in pd.date_range("2025-11-01", periods=30, freq="D"):
        rows.append(
            {
                "Sku": "1007YKBLACKXXL",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["1007YKBLACK"], "Total_Inventory": [100]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        group_by_parent=False,
    )
    row = po.iloc[0]
    assert int(row["Sold_Units"]) == 30
    assert float(row["ADS"]) == pytest.approx(1.0, abs=0.02)
    assert int(row["Eff_Days"]) == 30


def test_variant_mode_parent_token_uses_full_style_rollup_even_with_exact_row():
    """
    If inventory row is parent-style (1007YKBLACK), it should include exact row sales
    plus child size sales, not just the exact-key line.
    """
    rows = []
    # One exact parent-key sale
    rows.append(
        {
            "Sku": "1007YKBLACK",
            "TxnDate": pd.Timestamp("2025-11-01"),
            "Transaction Type": "Shipment",
            "Quantity": 1,
            "Units_Effective": 1,
            "Source": "Amazon",
        }
    )
    # Child-size sales for same style
    for d in pd.date_range("2025-11-02", periods=29, freq="D"):
        rows.append(
            {
                "Sku": "1007YKBLACK-5XL",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["1007YKBLACK"], "Total_Inventory": [447]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=90,
        lead_time=30,
        target_days=210,
        demand_basis="Sold",
        safety_pct=0.0,
        group_by_parent=False,
    )
    row = po.iloc[0]
    assert int(row["Sold_Units"]) == 30
    assert float(row["ADS"]) == pytest.approx(1.0, abs=0.02)


def test_ship_units_150d_shows_broader_context_than_period_sold_units():
    """Sold_Units stays period-based; Ship_Units_150d reflects broader shipment context."""
    rows = []
    # Older shipments outside 30d window but inside 150d window.
    for d in pd.date_range("2025-07-01", periods=20, freq="D"):
        rows.append(
            {
                "Sku": "WIDE-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    # Recent 30d window shipments.
    for d in pd.date_range("2025-11-01", periods=10, freq="D"):
        rows.append(
            {
                "Sku": "WIDE-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["WIDE-SKU-1"], "Total_Inventory": [50]})
    po = calculate_po_base(sales, inv, 30, 7, 60, safety_pct=0.0, group_by_parent=False)
    row = po.iloc[0]
    assert int(row["Sold_Units"]) == 10
    assert int(row["Ship_Units_150d"]) == 30


def test_ads_falls_back_to_ly_when_recent_is_zero():
    """If recent sales are zero but LY window has demand, ADS should not stay zero."""
    rows = []
    # Anchor max_date with another SKU in current window.
    for d in pd.date_range("2025-11-01", periods=5, freq="D"):
        rows.append(
            {
                "Sku": "ANCHOR-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    # Target SKU: sales only in LY-aligned window (not recent).
    for d in pd.date_range("2024-11-01", periods=30, freq="D"):
        rows.append(
            {
                "Sku": "LY-ONLY-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 2,
                "Units_Effective": 2,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["LY-ONLY-SKU-1", "ANCHOR-SKU-1"], "Total_Inventory": [40, 10]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        group_by_parent=False,
    )
    row = po[po["OMS_SKU"] == "LY-ONLY-SKU-1"].iloc[0]
    assert float(row["Recent_ADS"]) == 0.0
    assert float(row["LY_ADS"]) > 0.0
    assert float(row["ADS"]) > 0.0
