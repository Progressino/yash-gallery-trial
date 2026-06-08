"""Tests for existing PO sheet ingestion."""

from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from backend.services.existing_po import parse_existing_po

_FIXTURE_PO = Path(__file__).resolve().parent / "fixtures" / "Po_4-Jun-26.xlsx"


def test_parse_existing_po_csv_minimal():
    csv = "OMS SKU,Total Balance\nABC-RED-L,5\n"
    out = parse_existing_po(csv.encode("utf-8"), "po.csv")
    assert len(out) == 1
    assert out["OMS_SKU"].iloc[0] == "ABC-RED-L"
    assert int(out["PO_Pipeline_Total"].iloc[0]) == 5


def test_parse_existing_po_excel_title_row_before_header():
    """Workbooks with a title line above the real header must still parse."""
    wb = Workbook()
    ws = wb.active
    ws.append(["PO status — April"])
    ws.append(["OMS SKU", "Total Balance", "Note"])
    ws.append(["SKU99-XS", "3", ""])
    buf = BytesIO()
    wb.save(buf)
    out = parse_existing_po(buf.getvalue(), "po.xlsx")
    assert len(out) == 1
    assert out["OMS_SKU"].iloc[0] == "SKU99-XS"
    assert int(out["PO_Pipeline_Total"].iloc[0]) == 3


def test_parse_existing_po_headerless_yash_export():
    """Operator exports without a header row (first row is 1917YKBLUE-3XL …)."""
    wb = Workbook()
    ws = wb.active
    for row in [
        ["1917YKBLUE-3XL", "New SKU", "1917YKBLUE", "Monika", 0, 130, 130, 0],
        ["1917YKBLUE-4XL", "New SKU", "1917YKBLUE", "Monika", 0, 170, 170, 0],
    ]:
        ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    out = parse_existing_po(buf.getvalue(), "existing_po.xlsx")
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-3XL", "PO_Pipeline_Total"].iloc[0]) == 130
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-4XL", "PO_Pipeline_Total"].iloc[0]) == 170


def test_parse_existing_po_generic_numeric_columns():
    df = pd.DataFrame(
        {
            "SKU": ["1917YKBLUE-3XL", "1917YKBLUE-L"],
            "Status": ["New SKU", "New SKU"],
            "Style": ["1917YKBLUE", "1917YKBLUE"],
            "Name": ["Monika", "Monika"],
            "Col5": [0, 0],
            "Col6": [130, 120],
            "Col7": [130, 120],
            "Col8": [0, 0],
        }
    )
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Data", index=False)
    out = parse_existing_po(buf.getvalue(), "po.xlsx")
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-3XL", "PO_Pipeline_Total"].iloc[0]) == 130


def test_expand_bundled_po_skus_splits_size_ranges():
    from backend.services.existing_po import expand_bundled_po_skus

    ep = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-XXL-3XL"],
            "PO_Pipeline_Total": [194],
            "Pending_Cutting": [190],
            "Balance_to_Dispatch": [4],
        }
    )
    out = expand_bundled_po_skus(ep)
    assert set(out["OMS_SKU"]) == {"1917YKBLUE-XXL", "1917YKBLUE-3XL"}
    assert int(out["PO_Pipeline_Total"].sum()) == 194


def test_expand_bundled_po_skus_handles_unicode_dashes():
    from backend.services.existing_po import expand_bundled_po_skus

    ep = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-4XL–5XL"],  # en dash
            "PO_Pipeline_Total": [200],
        }
    )
    out = expand_bundled_po_skus(ep)
    assert set(out["OMS_SKU"]) == {"1917YKBLUE-4XL", "1917YKBLUE-5XL"}
    assert int(out["PO_Pipeline_Total"].sum()) == 200


def test_prepare_existing_po_expands_bundled_only_parent_on_mixed_sheet():
    """1917 bundled-only rows expand even when another parent has per-size lines."""
    from backend.services.existing_po import existing_po_merge_key, prepare_existing_po_for_merge

    ep = pd.DataFrame(
        {
            "OMS_SKU": [
                "1361YKBLUE-L",
                "1361YKBLUE-XL",
                "1917YKBLUE-L-XL",
                "1917YKBLUE-4XL-5XL",
            ],
            "PO_Pipeline_Total": [100, 120, 324, 274],
            "Pending_Cutting": [90, 110, 320, 270],
            "Balance_to_Dispatch": [10, 10, 4, 4],
        }
    )
    out = prepare_existing_po_for_merge(ep, existing_po_merge_key)
    assert "1917YKBLUE-L" in set(out["OMS_SKU"])
    assert "1917YKBLUE-XL" in set(out["OMS_SKU"])
    assert "1917YKBLUE-4XL" in set(out["OMS_SKU"])
    assert "1917YKBLUE-5XL" in set(out["OMS_SKU"])
    assert "1917YKBLUE-L-XL" not in set(out["OMS_SKU"])
    assert int(out.loc[out["OMS_SKU"] == "1361YKBLUE-L", "PO_Pipeline_Total"].iloc[0]) == 100


def test_prepare_existing_po_keeps_bundled_when_per_size_children_exist():
    from backend.services.existing_po import existing_po_merge_key, prepare_existing_po_for_merge

    ep = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-L", "1917YKBLUE-XL", "1917YKBLUE-L-XL"],
            "PO_Pipeline_Total": [120, 100, 4],
            "Pending_Cutting": [110, 90, 4],
            "Balance_to_Dispatch": [10, 10, 0],
        }
    )
    out = prepare_existing_po_for_merge(ep, existing_po_merge_key)
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-L-XL", "PO_Pipeline_Total"].iloc[0]) == 4
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-L", "PO_Pipeline_Total"].iloc[0]) == 120


def test_parse_existing_po_vendor_article_column():
    df = pd.DataFrame(
        {
            "Vendor Article Number": ["V1-A", "V1-B"],
            "Open Qty": [2, 4],
        }
    )
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Data", index=False)
    out = parse_existing_po(buf.getvalue(), "po.xlsx")
    assert len(out) == 2
    assert set(out["OMS_SKU"]) == {"V1-A", "V1-B"}


def test_parse_po_4_jun_26_fixture():
    """Regression: operator Po 4-Jun-26 export — pending cutting + balance columns."""
    assert _FIXTURE_PO.is_file(), "fixture missing"
    out = parse_existing_po(_FIXTURE_PO.read_bytes(), _FIXTURE_PO.name)
    assert len(out) > 10_000
    assert "Pending_Cutting" in out.columns
    assert "Balance_to_Dispatch" in out.columns
    row = out.loc[out["OMS_SKU"] == "1003YKMUSTARD-3XL"].iloc[0]
    assert int(row["Pending_Cutting"]) == 40
    assert int(row["Balance_to_Dispatch"]) == 3
    assert int(row["PO_Pipeline_Total"]) == 43
    yk = out.loc[out["OMS_SKU"] == "1917YKBLUE-3XL"].iloc[0]
    assert int(yk["Pending_Cutting"]) == 0
    assert int(yk["Balance_to_Dispatch"]) == 130


def test_collapse_duplicate_trailing_size_suffix():
    from backend.services.helpers import collapse_duplicate_trailing_size_suffix
    from backend.services.existing_po import is_bundled_size_range_sku, _normalize_sku_text

    assert collapse_duplicate_trailing_size_suffix("1361YKBLUE-L-L") == "1361YKBLUE-L"
    assert collapse_duplicate_trailing_size_suffix("1361YKBLUE-XL-XL") == "1361YKBLUE-XL"
    assert collapse_duplicate_trailing_size_suffix("1361YKBLUE-XXL-XXL") == "1361YKBLUE-XXL"
    assert collapse_duplicate_trailing_size_suffix("1361YKBLUE-S-M") == "1361YKBLUE-S-M"
    assert collapse_duplicate_trailing_size_suffix("1917YKBLUE-L-XL") == "1917YKBLUE-L-XL"
    assert not is_bundled_size_range_sku("1361YKBLUE-L-L")
    assert is_bundled_size_range_sku("1917YKBLUE-L-XL")
    assert _normalize_sku_text("1361YKBLUE-M-M") == "1361YKBLUE-M"


def test_po_engine_normalizes_l_l_inventory_sku():
    from backend.services.po_engine import calculate_po_base
    import pandas as pd

    days = pd.date_range("2026-05-01", periods=10, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["1361YKBLUE-L"] * 10,
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * 10,
            "Quantity": [1] * 10,
            "Units_Effective": [1] * 10,
            "Source": ["Amazon"] * 10,
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["1361YKBLUE-L-L"], "Total_Inventory": [12]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=45,
        target_days=135,
        demand_basis="Sold",
        safety_pct=0.0,
    )
    assert len(po) == 1
    assert po.iloc[0]["OMS_SKU"] == "1361YKBLUE-L"


def test_existing_po_aggregated_bundled_only_detected():
    from backend.services.existing_po import existing_po_looks_aggregated_bundled_only

    bundled = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-4XL-5XL", "1917YKBLUE-L-XL"],
            "PO_Pipeline_Total": [320, 220],
        }
    )
    assert existing_po_looks_aggregated_bundled_only(bundled)
    full = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-4XL", "1917YKBLUE-5XL", "1917YKBLUE-4XL-5XL"],
            "PO_Pipeline_Total": [170, 150, 4],
        }
    )
    assert not existing_po_looks_aggregated_bundled_only(full)


def test_existing_po_needs_recalc_tracks_generation():
    from backend.session import AppSession
    from backend.services.existing_po import existing_po_needs_recalc, session_has_fresh_existing_po

    sess = AppSession()
    assert not session_has_fresh_existing_po(sess)
    assert not existing_po_needs_recalc(sess)

    sess.existing_po_df = pd.DataFrame({"OMS_SKU": ["A"], "PO_Pipeline_Total": [1]})
    sess.existing_po_generation = 1
    assert session_has_fresh_existing_po(sess)
    assert existing_po_needs_recalc(sess)

    sess.po_calculate_existing_po_generation = 1
    assert not existing_po_needs_recalc(sess)

    sess.existing_po_generation = 2
    assert existing_po_needs_recalc(sess)


def test_seed_existing_po_warm_cache_from_disk():
    import tempfile
    from backend.services.existing_po import persist_existing_po_to_disk, seed_existing_po_warm_cache_from_disk
    from backend.session import AppSession
    import backend.main as main

    full = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-3XL", "1917YKBLUE-4XL"],
            "PO_Pipeline_Total": [130, 170],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        import os

        os.environ["WARM_CACHE_DIR"] = tmp
        main._warm_cache = {}
        sess = AppSession()
        sess.existing_po_df = full
        sess.existing_po_generation = 4
        sess.existing_po_filename = "Po 4-Jun-26.xlsx"
        assert persist_existing_po_to_disk(sess)
        assert seed_existing_po_warm_cache_from_disk()
        assert len(main._warm_cache["existing_po_df"]) == 2
        assert int(main._warm_cache["existing_po_session_meta"]["existing_po_generation"]) == 4


def test_session_should_keep_existing_po_rejects_aggregated():
    from backend.services.existing_po import session_should_keep_existing_po
    from backend.session import AppSession

    sess = AppSession()
    sess.existing_po_df = pd.DataFrame(
        {"OMS_SKU": ["1917YKBLUE-L-XL"], "PO_Pipeline_Total": [120]}
    )
    sess.existing_po_generation = 1
    assert not session_should_keep_existing_po(sess)


def test_restore_existing_po_from_disk_replaces_partial_session():
    import json
    import tempfile
    from backend.services.existing_po import (
        persist_existing_po_to_disk,
        restore_existing_po_from_disk,
    )
    from backend.session import AppSession

    full = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-3XL", "1917YKBLUE-4XL", "1917YKBLUE-4XL-5XL"],
            "PO_Pipeline_Total": [130, 170, 4],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        import os

        os.environ["WARM_CACHE_DIR"] = tmp
        sess = AppSession()
        sess.existing_po_df = full
        sess.existing_po_generation = 2
        sess.existing_po_uploaded_at = "2026-06-04T12:00:00Z"
        sess.existing_po_filename = "Po 4-Jun-26.xlsx"
        assert persist_existing_po_to_disk(sess)

        partial = AppSession()
        partial.existing_po_df = full.iloc[[2]].copy()
        partial.existing_po_generation = 1
        assert restore_existing_po_from_disk(partial)
        assert len(partial.existing_po_df) == 3
        assert int(partial.existing_po_generation) == 2


def test_cache_apply_preserves_fresh_existing_po():
    from backend.routers.cache import _apply_loaded_into_session
    from backend.session import AppSession

    sess = AppSession()
    sess.existing_po_df = pd.DataFrame({"OMS_SKU": ["KEEP-ME"], "PO_Pipeline_Total": [9]})
    sess.existing_po_generation = 3
    sess.existing_po_uploaded_at = "2026-06-04T12:00:00Z"

    loaded = {
        "existing_po_df": pd.DataFrame({"OMS_SKU": ["STALE"], "PO_Pipeline_Total": [1]}),
        "mtr_df": pd.DataFrame({"Sku": ["X"], "TxnDate": ["2026-01-01"]}),
    }
    _apply_loaded_into_session(sess, loaded)
    assert sess.existing_po_df["OMS_SKU"].iloc[0] == "KEEP-ME"
    assert not sess.mtr_df.empty
