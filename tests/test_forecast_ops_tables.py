import pandas as pd

from backend.db import forecast_ops_tables as tbl


def test_inventory_row_from_series():
    row = pd.Series(
        {
            "OMS_SKU": "sku-a",
            "OMS_Inventory": 10,
            "Amazon_Inventory": 3,
            "Total_Inventory": 13,
            "Buffer_Stock": 1,
        }
    )
    out = tbl._inventory_row_from_series(row)
    assert out["oms_sku"] == "SKU-A"
    assert out["oms_inventory"] == 10
    assert out["amazon_inventory"] == 3
    assert out["total_inventory"] == 13


def test_sales_rows_from_mtr_shape():
    df = pd.DataFrame(
        {
            "SKU": ["A"],
            "Date": pd.to_datetime(["2026-06-01"]),
            "Transaction_Type": ["Shipment"],
            "Quantity": [5],
        }
    )
    rows = list(tbl._sales_rows_from_dataframe(df, "amazon"))
    assert len(rows) == 1
    assert rows[0][1] == "A"
    assert rows[0][3] == 5.0


def test_platform_bulk_key_map():
    assert tbl._PLATFORM_BULK_KEYS["mtr_df"] == "amazon"
    assert tbl._PLATFORM_BULK_KEYS["sales_df"] == "unified"


def test_load_warm_cache_mtr_column_rename(monkeypatch):
    amazon_df = pd.DataFrame(
        {
            "Sku": ["A"],
            "TxnDate": pd.to_datetime(["2026-06-01"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
        }
    )

    monkeypatch.setattr(tbl, "normalized_tables_enabled", lambda: True)
    monkeypatch.setattr(tbl, "load_sku_mapping", lambda: {})
    monkeypatch.setattr(tbl, "load_inventory_dataframe", lambda: None)
    monkeypatch.setattr(tbl, "load_platform_sales_dataframe", lambda plat: amazon_df if plat == "amazon" else None)

    out = tbl.load_warm_cache_tables()
    assert out is not None
    assert "mtr_df" in out
    assert "SKU" in out["mtr_df"].columns
    assert "Date" in out["mtr_df"].columns
    assert "Transaction_Type" in out["mtr_df"].columns
