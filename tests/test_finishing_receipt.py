"""Finishing Dept receipt import tests."""
from pathlib import Path

import pandas as pd

from backend.services.finishing_receipt import (
    apply_finishing_receipt_import,
    merge_finishing_into_existing_po,
    parse_finishing_receipt_workbook,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "finishing_receipt_sample.xls"


def test_parse_finishing_receipt_sample():
    raw = _FIXTURE.read_bytes()
    df, report = parse_finishing_receipt_workbook(raw, _FIXTURE.name)
    assert not df.empty
    assert report["rows_read"] >= 70
    assert report["skus"] >= 5
    assert report["balance_units"] == 1549
    assert report["issued_units"] == 1549
    assert report["received_units"] == 0
    assert "2493-2627" in report["issue_numbers"]
    assert "5012YKPINK-3XL" in set(df["OMS_SKU"].astype(str))


def test_merge_finishing_updates_balance_to_dispatch():
    existing = pd.DataFrame(
        {
            "OMS_SKU": ["5012YKPINK-3XL", "OTHER-SKU"],
            "PO_Qty_Ordered": [100, 50],
            "Pending_Cutting": [10, 5],
            "Balance_to_Dispatch": [90, 40],
            "PO_Pipeline_Total": [100, 45],
        }
    )
    finishing = pd.DataFrame(
        {
            "OMS_SKU": ["5012YKPINK-3XL"],
            "Finishing_Issued": [60],
            "Finishing_Received": [20],
            "Finishing_Balance": [40],
            "Finishing_Issue_No": ["2493-2627"],
            "Finishing_Iss_Date": ["2026-06-12"],
            "Finishing_JO_No": ["97-2627"],
            "Finishing_JO_Date": ["2026-04-17"],
            "Finishing_Status": ["Non-Clear"],
        }
    )
    merged, stats = merge_finishing_into_existing_po(existing, finishing)
    row = merged.loc[merged["OMS_SKU"] == "5012YKPINK-3XL"].iloc[0]
    assert int(row["Balance_to_Dispatch"]) == 40
    assert int(row["Finishing_Balance"]) == 40
    assert int(row["Finishing_Received"]) == 20
    assert row["Finishing_Issue_No"] == "2493-2627"
    assert int(row["PO_Pipeline_Total"]) == 50  # 10 pending + 40 finishing balance
    assert stats["updated_skus"] == 1
    assert int(merged.loc[merged["OMS_SKU"] == "OTHER-SKU", "Balance_to_Dispatch"].iloc[0]) == 40


def test_apply_finishing_receipt_bumps_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))

    class Sess:
        existing_po_df = pd.DataFrame()
        existing_po_generation = 2
        sku_mapping = {}
        _quarterly_cache = {}

    raw = _FIXTURE.read_bytes()
    finishing_df, report = parse_finishing_receipt_workbook(raw, _FIXTURE.name)
    sess = Sess()
    out = apply_finishing_receipt_import(sess, finishing_df, report, filename=_FIXTURE.name)
    assert out["ok"] is True
    assert sess.existing_po_generation == 3
    assert not sess.existing_po_df.empty
    assert int(sess.existing_po_df["Finishing_Balance"].sum()) == 1549
