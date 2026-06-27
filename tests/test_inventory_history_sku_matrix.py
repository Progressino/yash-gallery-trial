"""Inventory history must surface uploaded on-hand for canonical SKUs (e.g. 8XL)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.services.daily_inventory_history import (
    drop_zero_derived_rows,
    effective_days_from_history,
    extend_history_with_sales,
    inventory_history_wide_matrix,
    parse_daily_inventory_history_upload,
)
from backend.services.daily_inventory_upload_run import execute_daily_inventory_upload
from backend.session import AppSession

_FIXTURE = Path("/Users/samraisinghani/Downloads/Daily Inventory History 1-May To 25-Jun-26 1.xlsx")
_SKU = "1024YKMUSTARD-8XL"


@pytest.fixture
def mapping():
    import json

    p = Path(__file__).resolve().parents[1] / "backend" / "data" / "yash_sku_mapping_master.json"
    return json.loads(p.read_text())


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="user fixture xlsx not present")
def test_upload_fixture_has_stock_for_8xl(mapping):
    raw = _FIXTURE.read_bytes()
    sess = AppSession()
    sess.sku_mapping = mapping
    out = execute_daily_inventory_upload(sess, raw, _FIXTURE.name)
    assert out["ok"] is True
    sub = sess.daily_inventory_history_df[
        sess.daily_inventory_history_df["OMS_SKU"].astype(str) == _SKU
    ]
    assert len(sub) >= 20
    assert int((sub["Qty"] > 0).sum()) >= 5

    wide = inventory_history_wide_matrix(sess.daily_inventory_history_df, q=_SKU, days=30)
    assert wide["loaded"] is True
    assert len(wide["rows"]) == 1
    pos = sum(1 for q in wide["rows"][0]["qtys"] if q > 0)
    assert pos >= 5

    end = pd.Timestamp(sub["Date"].max()).normalize()
    start = end - pd.Timedelta(days=29)
    eff = effective_days_from_history(sess.daily_inventory_history_df, start, end)
    by_sku = {r["OMS_SKU"]: int(r["Eff_Days_Inventory"]) for _, r in eff.iterrows()}
    assert by_sku.get(_SKU, 0) >= 5


def test_extend_does_not_materialize_zero_rows_for_oos_skus():
    baseline = pd.DataFrame(
        {
            "OMS_SKU": ["IN-STOCK", "OOS-SKU"],
            "Date": [pd.Timestamp("2026-05-10"), pd.Timestamp("2026-05-10")],
            "Qty": [10.0, 0.0],
            "Source": ["uploaded", "uploaded"],
        }
    )
    sales = pd.DataFrame(
        [
            {
                "Sku": "OTHER",
                "TxnDate": pd.Timestamp("2026-05-11"),
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
            }
        ]
    )
    out = extend_history_with_sales(baseline, sales, cap_date=pd.Timestamp("2026-05-11"))
    oos = out[(out["OMS_SKU"] == "OOS-SKU") & (out["Date"] == pd.Timestamp("2026-05-11"))]
    assert oos.empty
    instock = out[(out["OMS_SKU"] == "IN-STOCK") & (out["Date"] == pd.Timestamp("2026-05-11"))]
    assert float(instock.iloc[0]["Qty"]) == 10.0


def test_drop_zero_derived_rows():
    df = pd.DataFrame(
        {
            "OMS_SKU": ["A", "A", "B"],
            "Date": pd.to_datetime(["2026-05-10", "2026-05-11", "2026-05-11"]),
            "Qty": [5.0, 0.0, 0.0],
            "Source": ["uploaded", "derived", "derived"],
        }
    )
    out = drop_zero_derived_rows(df)
    assert len(out) == 2
    assert set(out["OMS_SKU"]) == {"A"}


def test_parse_uses_canonical_key_for_pl_alias(mapping):
    buf = __import__("io").BytesIO()
    # minimal wide sheet: one PL alias row
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Item SkuCode", "Item", "2026-06-01", "2026-06-02"])
    ws.append(["", "", pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")])
    ws.append(["1024PLYKMUSTARD-8XL", "1024YKMUSTARD", 3, 4])
    wb.save(buf)
    buf.seek(0)
    df = parse_daily_inventory_history_upload(buf, "t.xlsx", sku_mapping=mapping)
    assert set(df["OMS_SKU"].astype(str)) == {_SKU}
    assert int(df["Qty"].sum()) == 7
