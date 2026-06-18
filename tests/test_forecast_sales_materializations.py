"""Materialized SKU sales rollups for PO fast path."""
from __future__ import annotations

import pandas as pd

from backend.db.forecast_sales_materializations import (
    daily_to_engine_sales_df,
    load_po_sales_df,
    refresh_from_sales_df,
    sales_df_to_daily,
)


def _sample_sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A", "SKU-B", "SKU-B"],
            "TxnDate": pd.to_datetime(
                ["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-03"]
            ),
            "Transaction Type": ["Shipment", "Refund", "Shipment", "Shipment"],
            "Quantity": [10, 2, 5, 3],
        }
    )


def test_sales_df_to_daily_aggregates_same_day():
    daily, through = sales_df_to_daily(_sample_sales_df())
    assert through is not None
    a = daily[daily["oms_sku"] == "SKU-A"].iloc[0]
    assert a["sold_units"] == 10
    assert a["return_units"] == 2
    assert a["net_units"] == 8


def test_daily_to_engine_sales_df_smaller_than_line_level():
    daily, _ = sales_df_to_daily(_sample_sales_df())
    engine = daily_to_engine_sales_df(daily)
    assert len(engine) <= len(_sample_sales_df())
    assert "Shipment" in engine["Transaction Type"].values
    assert "Refund" in engine["Transaction Type"].values


def test_refresh_and_load_po_sales_in_memory(monkeypatch):
    monkeypatch.setenv("FORECAST_SKU_ROLLUPS", "1")
    monkeypatch.setenv("FORECAST_OPS_NORMALIZED", "0")
    sales = _sample_sales_df()
    stats = refresh_from_sales_df(sales)
    assert stats["daily_rows"] >= 2

    class _Sess:
        sales_df = sales
        sales_data_revision = 1

    out = load_po_sales_df(
        _Sess(),
        period_days=30,
        planning_date="2026-06-03",
        use_seasonality=False,
        use_ly_fallback=False,
    )
    assert out is not None
    assert not out.empty
    assert len(out) < 20
