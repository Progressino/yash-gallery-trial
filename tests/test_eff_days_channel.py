"""Eff_Days excludes Qty=0 days and respects inventory_history_channel."""
import pandas as pd

from backend.services.daily_inventory_history import effective_days_from_history


def _frame() -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2026-06-26", "2026-07-25"):
        # OMS stocked every day except the 5th and 12th (explicit zeros).
        oms_qty = 0.0 if d.day in (5, 12) else 10.0
        rows.append(
            {
                "OMS_SKU": "TESTSKU",
                "Date": d,
                "Qty": oms_qty,
                "Channel": "oms",
                "Source": "snapshot",
            }
        )
        # Amazon never in stock for this SKU.
        rows.append(
            {
                "OMS_SKU": "TESTSKU",
                "Date": d,
                "Qty": 0.0,
                "Channel": "amazon",
                "Source": "snapshot",
            }
        )
    return pd.DataFrame(rows)


def test_eff_days_excludes_explicit_zero_days():
    df = _frame()
    start, end = pd.Timestamp("2026-06-26"), pd.Timestamp("2026-07-25")
    combined = effective_days_from_history(df, start, end, channel="combined")
    oms = effective_days_from_history(df, start, end, channel="oms")
    amazon = effective_days_from_history(df, start, end, channel="amazon")
    assert int(combined.iloc[0]["Eff_Days_Inventory"]) == 28
    assert int(oms.iloc[0]["Eff_Days_Inventory"]) == 28
    assert amazon.empty or int(amazon.iloc[0]["Eff_Days_Inventory"]) == 0


def test_eff_days_amazon_only_counts_fba_stock():
    rows = []
    for d in pd.date_range("2026-07-01", "2026-07-10"):
        rows.append(
            {
                "OMS_SKU": "AMZONLY",
                "Date": d,
                "Qty": 0.0,
                "Channel": "oms",
                "Source": "snapshot",
            }
        )
        rows.append(
            {
                "OMS_SKU": "AMZONLY",
                "Date": d,
                "Qty": 5.0 if d.day <= 7 else 0.0,
                "Channel": "amazon",
                "Source": "snapshot",
            }
        )
    df = pd.DataFrame(rows)
    start, end = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-10")
    amazon = effective_days_from_history(df, start, end, channel="amazon")
    oms = effective_days_from_history(df, start, end, channel="oms")
    combined = effective_days_from_history(df, start, end, channel="combined")
    assert int(amazon.iloc[0]["Eff_Days_Inventory"]) == 7
    assert oms.empty or int(oms.iloc[0]["Eff_Days_Inventory"]) == 0
    assert int(combined.iloc[0]["Eff_Days_Inventory"]) == 7


def test_oms_channel_ignores_amazon_only_in_stock_days():
    """OMS Eff_Days must match the OMS matrix — Amazon FBA alone must not inflate."""
    rows = []
    for d in pd.date_range("2026-06-30", "2026-07-29"):
        # OMS in stock first 11 days only.
        oms_qty = 5.0 if d <= pd.Timestamp("2026-07-10") else 0.0
        rows.append(
            {
                "OMS_SKU": "1024YKMUSTARD-4XL",
                "Date": d,
                "Qty": oms_qty,
                "Channel": "oms",
                "Source": "snapshot",
            }
        )
        # Amazon stocked on last 3 days while OMS is OOS.
        amz_qty = 5.0 if d >= pd.Timestamp("2026-07-27") else 0.0
        rows.append(
            {
                "OMS_SKU": "1024YKMUSTARD-4XL",
                "Date": d,
                "Qty": amz_qty,
                "Channel": "amazon",
                "Source": "snapshot",
            }
        )
    df = pd.DataFrame(rows)
    start, end = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-07-29")
    oms = effective_days_from_history(df, start, end, channel="oms")
    combined = effective_days_from_history(df, start, end, channel="combined")
    assert int(oms.iloc[0]["Eff_Days_Inventory"]) == 11
    assert int(combined.iloc[0]["Eff_Days_Inventory"]) == 14
