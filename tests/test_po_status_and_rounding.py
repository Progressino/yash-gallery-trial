"""PO engine — excluded statuses and pack rounding."""

import pandas as pd
import pytest

from backend.services.po_engine import calculate_po_base, round_po_pack
from backend.session import AppSession


def test_round_po_pack_fives_and_tens():
    assert round_po_pack(0) == 0
    assert round_po_pack(4) == 5
    assert round_po_pack(9) == 10
    assert round_po_pack(14) == 20
    assert round_po_pack(23) == 30
    assert round_po_pack(68) == 70


def _minimal_sales():
    rows = []
    for d in pd.date_range("2025-11-01", periods=30, freq="D"):
        rows.append(
            {
                "Sku": "TEST-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 3,
                "Units_Effective": 3,
                "Source": "Amazon",
            }
        )
    return pd.DataFrame(rows)


def test_doubt_and_sales_after_closed_skus_get_zero_po():
    rows = []
    for sku in ("DOUBT-1", "SAC-1"):
        for d in pd.date_range("2025-11-01", periods=30, freq="D"):
            rows.append(
                {
                    "Sku": sku,
                    "TxnDate": d,
                    "Transaction Type": "Shipment",
                    "Quantity": 3,
                    "Units_Effective": 3,
                    "Source": "Amazon",
                }
            )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["DOUBT-1", "SAC-1"], "Total_Inventory": [5, 5]})
    status = pd.DataFrame(
        {
            "OMS_SKU": ["DOUBT-1", "SAC-1"],
            "SKU_Sheet_Status": ["Doubt", "Sales After Closed"],
            "Lead_Time_From_Sheet": [45, 45],
            "SKU_Sheet_Closed": [False, False],
        }
    )
    po = calculate_po_base(
        sales,
        inv,
        period_days=30,
        lead_time=7,
        target_days=180,
        safety_pct=0.0,
        sku_status_df=status,
    )
    for sku in ("DOUBT-1", "SAC-1"):
        row = po.loc[po["OMS_SKU"] == sku].iloc[0]
        assert int(row["PO_Qty"]) == 0
        reason = str(row["PO_Block_Reason"]).lower()
        assert "doubt" in reason or "sales after closed" in reason or "closed" in reason


def test_return_overlay_subtract_reapplies_pack_rounding():
    sales = _minimal_sales()
    inv = pd.DataFrame({"OMS_SKU": ["TEST-SKU-1"], "Total_Inventory": [10]})
    overlay = pd.DataFrame({"OMS_SKU": ["TEST-SKU-1"], "Return_Units": [3]})
    po = calculate_po_base(
        sales,
        inv,
        period_days=30,
        lead_time=7,
        target_days=180,
        safety_pct=0.0,
        po_return_overlay_df=overlay,
    )
    row = po.iloc[0]
    assert int(row["PO_Qty"]) % 5 == 0 or int(row["PO_Qty"]) % 10 == 0
    ads = float(row["ADS"])
    if ads > 0 and int(row["PO_Qty"]) > 0:
        cover = float(row["Post_PO_Cover_Days_Capped"])
        assert cover >= 150.0


def test_hydrate_ledger_drops_deleted_day_from_stale_session(tmp_path, monkeypatch):
    from backend.db import po_raised_db
    from backend.services.po_raise_import import apply_ledger_import, hydrate_session_ledger_from_db

    db_path = str(tmp_path / "raised.db")
    monkeypatch.setattr(po_raised_db, "DB_PATH", db_path)
    po_raised_db.init_db()

    sess = AppSession()
    apply_ledger_import(sess, {"SKU-A": 100}, pd.Timestamp("2026-05-16"), replace_day=True)

    stale = AppSession()
    stale.po_raise_ledger_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "Raised_Qty": [100, 50],
            "Raised_Date": pd.to_datetime(["2026-05-16", "2026-05-16"]),
        }
    )
    po_raised_db.delete_raises_for_date("2026-05-16")
    po_raised_db.suppress_raise_date("2026-05-16")

    hydrate_session_ledger_from_db(stale, "2026-05-18", lookback_days=30, authoritative=True)
    assert stale.po_raise_ledger_df.empty or int(
        pd.to_numeric(stale.po_raise_ledger_df["Raised_Qty"], errors="coerce").fillna(0).sum()
    ) == 0


def test_coverage_hydrate_keeps_session_ledger_when_db_empty(tmp_path, monkeypatch):
    """Merge-mode hydrate must not wipe a session ledger that is not yet in SQLite."""
    from backend.services.po_raise_import import hydrate_session_ledger_from_db

    db_path = str(tmp_path / "ledger_merge.db")
    monkeypatch.setattr("backend.db.po_raised_db.DB_PATH", db_path)
    from backend.db import po_raised_db

    po_raised_db.init_db()

    sess = AppSession()
    sess.po_raise_ledger_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Raised_Qty": [25],
            "Raised_Date": pd.to_datetime(["2026-05-16"]),
        }
    )
    hydrate_session_ledger_from_db(sess, "2026-06-15", lookback_days=30, authoritative=False)
    assert len(sess.po_raise_ledger_df) == 1
    assert int(sess.po_raise_ledger_df.iloc[0]["Raised_Qty"]) == 25
