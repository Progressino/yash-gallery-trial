"""Finishing Dept receipt import tests."""
from pathlib import Path

import pandas as pd
import pytest

from backend.services.finishing_receipt import (
    apply_finishing_receipt_import,
    merge_finishing_into_existing_po,
    parse_finishing_receipt_workbook,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "finishing_receipt_sample.xls"


class _FinishingSess:
    def __init__(self):
        self.existing_po_df = pd.DataFrame()
        self.existing_po_generation = 0
        self.sku_mapping = {}
        self.finishing_receipt_filename = ""
        self._quarterly_cache = {}


@pytest.fixture
def finishing_sess(monkeypatch, tmp_path):
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    return _FinishingSess()


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


def test_apply_finishing_receipt_bumps_generation(finishing_sess):
    raw = _FIXTURE.read_bytes()
    finishing_df, report = parse_finishing_receipt_workbook(raw, _FIXTURE.name)
    finishing_sess.existing_po_generation = 2
    out = apply_finishing_receipt_import(finishing_sess, finishing_df, report, filename=_FIXTURE.name)
    assert out["ok"] is True
    assert finishing_sess.existing_po_generation == 3
    assert not finishing_sess.existing_po_df.empty
    assert int(finishing_sess.existing_po_df["Finishing_Balance"].sum()) == 1549


def test_parse_dedupes_duplicate_design_size_rows():
    raw = _FIXTURE.read_bytes()
    df, report = parse_finishing_receipt_workbook(raw, _FIXTURE.name)
    assert not df["OMS_SKU"].duplicated().any()
    assert report["skus"] == len(df)


def test_merge_dedupes_existing_po_duplicate_skus():
    existing = pd.DataFrame(
        {
            "OMS_SKU": ["5012YKPINK-3XL", "5012YKPINK-3XL", "OTHER-SKU"],
            "PO_Qty_Ordered": [100, 50, 20],
            "Pending_Cutting": [10, 5, 2],
            "Balance_to_Dispatch": [90, 40, 18],
            "PO_Pipeline_Total": [100, 45, 20],
        }
    )
    finishing = pd.DataFrame(
        {
            "OMS_SKU": ["5012YKPINK-3XL"],
            "Finishing_Issued": [60],
            "Finishing_Received": [0],
            "Finishing_Balance": [40],
            "Finishing_Issue_No": ["2493-2627"],
            "Finishing_Iss_Date": ["2026-06-12"],
            "Finishing_JO_No": ["97-2627"],
            "Finishing_JO_Date": ["2026-04-17"],
            "Finishing_Status": ["Non-Clear"],
        }
    )
    merged, stats = merge_finishing_into_existing_po(existing, finishing)
    assert len(merged) == 2
    assert not merged["OMS_SKU"].duplicated().any()
    assert stats["updated_skus"] == 1
    assert stats["added_skus"] == 0
    row = merged.loc[merged["OMS_SKU"] == "5012YKPINK-3XL"].iloc[0]
    assert int(row["Balance_to_Dispatch"]) == 40


def test_reupload_same_file_no_duplicate_rows(finishing_sess):
    raw = _FIXTURE.read_bytes()
    finishing_df, report = parse_finishing_receipt_workbook(raw, _FIXTURE.name)
    expected_skus = int(report["skus"])

    out1 = apply_finishing_receipt_import(finishing_sess, finishing_df, report, filename=_FIXTURE.name)
    rows_after_first = len(finishing_sess.existing_po_df)
    balance_after_first = int(finishing_sess.existing_po_df["Finishing_Balance"].sum())
    pipeline_after_first = int(finishing_sess.existing_po_df["PO_Pipeline_Total"].sum())

    assert out1["ok"] is True
    assert rows_after_first == expected_skus
    assert finishing_sess.existing_po_df["OMS_SKU"].duplicated().sum() == 0
    assert int(out1["parse_report"]["added_skus"]) == expected_skus
    assert int(out1["parse_report"]["updated_skus"]) == 0
    assert out1["parse_report"]["replaced_previous"] is False

    out2 = apply_finishing_receipt_import(finishing_sess, finishing_df, report, filename=_FIXTURE.name)

    assert out2["ok"] is True
    assert len(finishing_sess.existing_po_df) == rows_after_first
    assert int(finishing_sess.existing_po_df["Finishing_Balance"].sum()) == balance_after_first
    assert int(finishing_sess.existing_po_df["PO_Pipeline_Total"].sum()) == pipeline_after_first
    assert finishing_sess.existing_po_df["OMS_SKU"].duplicated().sum() == 0
    assert int(out2["parse_report"]["added_skus"]) == 0
    assert int(out2["parse_report"]["updated_skus"]) == expected_skus
    assert out2["parse_report"]["replaced_previous"] is True
    assert "no duplicate rows" in out2["message"].lower()


def test_reupload_after_existing_po_base_no_duplicate_rows(finishing_sess):
    finishing_sess.existing_po_df = pd.DataFrame(
        {
            "OMS_SKU": ["BASE-SKU", "5012YKPINK-3XL"],
            "PO_Qty_Ordered": [200, 100],
            "Pending_Cutting": [20, 10],
            "Balance_to_Dispatch": [180, 90],
            "PO_Pipeline_Total": [200, 100],
        }
    )
    raw = _FIXTURE.read_bytes()
    finishing_df, report = parse_finishing_receipt_workbook(raw, _FIXTURE.name)

    out1 = apply_finishing_receipt_import(finishing_sess, finishing_df, report, filename=_FIXTURE.name)
    rows_after_first = len(finishing_sess.existing_po_df)
    assert rows_after_first == int(report["skus"]) + 1  # BASE-SKU + finishing SKUs

    out2 = apply_finishing_receipt_import(finishing_sess, finishing_df, report, filename=_FIXTURE.name)
    assert len(finishing_sess.existing_po_df) == rows_after_first
    assert int(out2["parse_report"]["added_skus"]) == 0
    assert finishing_sess.existing_po_df["OMS_SKU"].duplicated().sum() == 0
    assert "BASE-SKU" in set(finishing_sess.existing_po_df["OMS_SKU"].astype(str))
