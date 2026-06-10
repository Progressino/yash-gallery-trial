"""Audit PO export math — formula consistency and OOS imputation caps."""

import numpy as np
import pandas as pd
import pytest

from backend.services.po_engine import _impute_oos_restock_recent_ads, calculate_po_base


def _expected_po_qty(row: pd.Series, *, target: int = 135, pack: int = 5) -> int:
    inv = float(row.get("Total_Inventory", 0) or 0)
    pipe = float(row.get("PO_Pipeline_Effective", row.get("PO_Pipeline_Total", 0)) or 0)
    ads = float(row.get("ADS", 0) or 0)
    overlay = int(row.get("Return_Overlay_Units", 0) or 0)
    if ads <= 0:
        return 0
    proj = (inv + pipe) / ads
    raw = ads * max(0.0, target - proj)
    gross = int(np.floor(np.ceil(max(raw, 0.0) / pack) * pack))
    return max(0, gross - overlay)


def test_po_export_formula_columns_match_engine_defaults():
    """Gross/PO_Qty must follow balance-days formula for typical rows."""
    days = pd.date_range("2025-11-01", periods=90, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["AUDIT-SKU-M"] * 90,
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * 90,
            "Quantity": [4] * 90,
            "Units_Effective": [4] * 90,
            "Source": ["Myntra"] * 90,
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["AUDIT-SKU-M"], "Total_Inventory": [20]})
    sheet = pd.DataFrame(
        {
            "OMS_SKU": ["AUDIT-SKU-M"],
            "SKU_Sheet_Status": ["Open"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [45.0],
        }
    )
    po = calculate_po_base(
        sales,
        inv,
        period_days=90,
        lead_time=30,
        target_days=135,
        demand_basis="Sold",
        safety_pct=0.0,
        sku_status_df=sheet,
        enforce_two_size_minimum=False,
    )
    row = po.iloc[0]
    assert int(row["PO_Qty"]) == _expected_po_qty(row)
    assert float(row["Days_Left"]) == pytest.approx(round(20 / float(row["ADS"]), 1), abs=0.2)
    pipe_eff = float(row.get("PO_Pipeline_Effective", 0) or 0)
    assert float(row["Projected_Running_Days"]) == pytest.approx(
        round((20 + pipe_eff) / float(row["ADS"]), 1), abs=0.2
    )


def test_oos_restock_impute_capped_by_sibling_ads():
    """Zero-sales OOS row must not exceed sibling Recent_ADS after imputation."""
    po_df = pd.DataFrame(
        {
            "OMS_SKU": ["STYLE-A-L", "STYLE-A-M", "STYLE-A-XL"],
            "Total_Inventory": [0, 0, 10],
            "Eff_Days": [3, 30, 30],
            "Eff_Days_Inventory": [3, 30, 30],
            "Net_Units": [0, 40, 0],
            "Sold_Units": [0, 42, 0],
            "Ship_Units_150d": [17, 50, 5],
            "Recent_ADS": [0.0, 40 / 30, 0.0],
        }
    )
    mask = pd.Series([True, False, False], index=po_df.index)
    out = _impute_oos_restock_recent_ads(po_df, mask, "Sold")
    sibling_cap = 40 / 30
    assert float(out.iloc[0]) <= sibling_cap + 1e-6
    assert float(out.iloc[0]) > 0


def test_audit_user_csv_if_present():
    """Optional regression: validate a local PO export when the file exists."""
    from pathlib import Path

    path = Path("/Users/samraisinghani/Downloads/po_recommendation 2026-06-09.csv")
    if not path.is_file():
        pytest.skip("user export not on disk")
    df = pd.read_csv(path)
    if df.columns[0].startswith("\ufeff"):
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
    hot = df[df["PO_Qty"] > 0]
    hot = hot.copy()
    hot["exp_po"] = hot.apply(_expected_po_qty, axis=1)
    bad = hot[hot["exp_po"] != hot["PO_Qty"]]
    assert bad.empty, f"formula mismatches: {bad[['OMS_SKU','PO_Qty','exp_po']].head().to_dict()}"
