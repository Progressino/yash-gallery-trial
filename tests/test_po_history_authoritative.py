"""PO must use fresh inventory history matrix for stock + Eff_Days when snapshot is stale."""
import pandas as pd
import pytest

from backend.services.daily_inventory_history import overlay_inventory_variant_from_history
from backend.services.po_engine import calculate_po_base


def _sales_sparse_active_span(sku: str, *, end: str = "2026-06-26") -> pd.DataFrame:
    """8 units sold only in the last 14 days of a 30-day window."""
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts - pd.Timedelta(days=13), periods=14, freq="D")
    return pd.DataFrame(
        {
            "Sku": [sku] * len(dates),
            "TxnDate": dates,
            "Transaction Type": ["Shipment"] * len(dates),
            "Quantity": [8 / len(dates)] * len(dates),
            "Units_Effective": [8 / len(dates)] * len(dates),
        }
    )


def _full_stock_history(sku: str, *, days: int = 30, end: str = "2026-06-26") -> pd.DataFrame:
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    return pd.DataFrame(
        {"OMS_SKU": [sku] * days, "Date": dates, "Qty": [10.0] * days}
    )


def test_overlay_replaces_stale_snapshot_from_history():
    hist = _full_stock_history("SKU-A", end="2026-06-26")
    inv = pd.DataFrame({"OMS_SKU": ["SKU-A"], "Total_Inventory": [5.0]})
    out, meta = overlay_inventory_variant_from_history(
        inv,
        hist,
        snapshot_date="2026-06-20",
        reference_date="2026-06-26",
    )
    assert meta["applied"] is True
    assert float(out.loc[0, "Total_Inventory"]) == 10.0


def test_overlay_skipped_when_snapshot_is_current():
    hist = _full_stock_history("SKU-A", end="2026-06-26")
    inv = pd.DataFrame({"OMS_SKU": ["SKU-A"], "Total_Inventory": [10.0]})
    out, meta = overlay_inventory_variant_from_history(
        inv,
        hist,
        snapshot_date="2026-06-26",
        reference_date="2026-06-26",
    )
    assert meta["applied"] is False
    assert meta["reason"] == "snapshot_fresh"
    assert float(out.loc[0, "Total_Inventory"]) == 10.0


def test_authoritative_history_uses_full_in_stock_eff_days():
    """Fresh matrix through today → Eff_Days should follow in-stock days, not sales span."""
    sku = "1001YKBEIGE-5XL"
    sales = _sales_sparse_active_span(sku, end="2026-06-26")
    inv = pd.DataFrame({"OMS_SKU": [sku], "Total_Inventory": [33.0]})
    ih = _full_stock_history(sku, days=30, end="2026-06-26")

    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=45,
        target_days=135,
        planning_date="2026-06-26",
        inventory_history_df=ih,
    )
    row = po.loc[po["OMS_SKU"] == sku].iloc[0]
    assert int(row["Eff_Days_Inventory"]) == 30
    assert int(row["Eff_Days"]) == 30
    assert float(row["ADS"]) == pytest.approx(8 / 30, rel=0.05)


def test_po_csv_fixture_sku_stock_matches_history_overlay():
    """Regression: 1001YKBEIGE-3XL should use matrix on-hand when snapshot missing."""
    sku = "1001YKBEIGE-3XL"
    hist = _full_stock_history(sku, end="2026-06-26")
    hist.loc[hist["Date"] == pd.Timestamp("2026-06-26"), "Qty"] = 62.0
    inv = pd.DataFrame({"OMS_SKU": [sku], "Total_Inventory": [38.0]})
    out, meta = overlay_inventory_variant_from_history(
        inv,
        hist,
        snapshot_date="",
        reference_date="2026-06-26",
    )
    assert meta["applied"] is True
    assert float(out.loc[0, "Total_Inventory"]) == 62.0
