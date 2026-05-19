"""PO calculate must trim multi-year daily inventory baselines."""
import pandas as pd

from backend.services.daily_inventory_history import trim_inventory_history_for_po
from backend.services.po_engine import calculate_po_base


def test_trim_inventory_history_for_po_keeps_ads_window_only():
    dates = pd.date_range("2025-01-01", periods=400, freq="D")
    rows = []
    for d in dates:
        rows.append({"OMS_SKU": "SKU-A", "Date": d, "Qty": 5})
    ih = pd.DataFrame(rows)
    end = pd.Timestamp("2025-05-18")
    start = end - pd.Timedelta(days=29)
    out = trim_inventory_history_for_po(ih, start, end, po_skus={"SKU-A"})
    assert len(out) == 30
    assert out["Date"].min() >= start
    assert out["Date"].max() <= end


def test_po_calculate_with_million_row_history_trimmed():
    """Simulate a wide baseline; PO must complete using window trim, not full 3M rows."""
    sales_dates = pd.date_range("2026-04-01", periods=30, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["BIG-SKU"] * len(sales_dates),
            "TxnDate": sales_dates,
            "Transaction Type": ["Shipment"] * len(sales_dates),
            "Quantity": [2] * len(sales_dates),
            "Units_Effective": [2] * len(sales_dates),
            "Source": ["Amazon"] * len(sales_dates),
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["BIG-SKU"], "Total_Inventory": [100]})

    # 1 SKU × 400 days — stand-in for multi-year sheet (trim to 30 for PO)
    hist_dates = pd.date_range("2025-08-01", periods=400, freq="D")
    ih = pd.DataFrame(
        {
            "OMS_SKU": ["BIG-SKU"] * len(hist_dates),
            "Date": hist_dates,
            "Qty": [10] * len(hist_dates),
        }
    )
    assert len(ih) == 400

    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=45,
        target_days=90,
        safety_pct=0.0,
        planning_date="2026-05-18",
        inventory_history_df=ih,
    )
    assert not po.empty
    row = po.iloc[0]
    assert int(row.get("Eff_Days_Inventory", 0)) > 0
