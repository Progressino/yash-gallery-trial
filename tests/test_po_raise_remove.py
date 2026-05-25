"""PO raise ledger removal (session + SQLite)."""
import pandas as pd

from backend.db import po_raised_db
from backend.services.po_raise_remove import (
    clear_raise_ledger_all,
    remove_raise_ledger_day,
    remove_raise_ledger_skus,
)
from backend.session import AppSession


def test_remove_raise_ledger_day_session_and_db(tmp_path, monkeypatch):
    db = tmp_path / "po_raised_test.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()

    sess = AppSession()
    sess.po_raise_ledger_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "Raised_Qty": [10, 5],
            "Raised_Date": pd.to_datetime(["2026-05-20", "2026-05-21"]),
        }
    )
    po_raised_db.replace_raises_for_date(
        "2026-05-20",
        [{"oms_sku": "SKU-A", "qty": 10}],
    )
    po_raised_db.replace_raises_for_date(
        "2026-05-21",
        [{"oms_sku": "SKU-B", "qty": 5}],
    )

    out = remove_raise_ledger_day(sess, "2026-05-20")
    assert out["ok"] is True
    assert out["removed"] >= 1
    assert len(sess.po_raise_ledger_df) == 1
    assert str(sess.po_raise_ledger_df.iloc[0]["OMS_SKU"]) == "SKU-B"
    rows = po_raised_db.list_raises(start_date="2026-05-20", end_date="2026-05-21")
    assert all(r["raised_date"] != "2026-05-20" for r in rows)


def test_remove_raise_ledger_skus_admin_path(tmp_path, monkeypatch):
    db = tmp_path / "po_raised_skus.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()

    sess = AppSession()
    sess.po_raise_ledger_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "Raised_Qty": [10, 5],
            "Raised_Date": pd.to_datetime(["2026-05-21", "2026-05-21"]),
        }
    )
    po_raised_db.replace_raises_for_date(
        "2026-05-21",
        [{"oms_sku": "SKU-A", "qty": 10}, {"oms_sku": "SKU-B", "qty": 5}],
    )

    out = remove_raise_ledger_skus(sess, "2026-05-21", ["SKU-A"])
    assert out["ok"] is True
    assert len(sess.po_raise_ledger_df) == 1
    assert str(sess.po_raise_ledger_df.iloc[0]["OMS_SKU"]) == "SKU-B"


def test_clear_raise_ledger_all(tmp_path, monkeypatch):
    db = tmp_path / "po_raised_clear.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()
    po_raised_db.record_raises([{"oms_sku": "X", "qty": 1}])

    sess = AppSession()
    sess.po_raise_ledger_df = pd.DataFrame(
        {"OMS_SKU": ["X"], "Raised_Qty": [1], "Raised_Date": pd.to_datetime(["2026-05-21"])}
    )
    out = clear_raise_ledger_all(sess)
    assert out["ok"] is True
    assert sess.po_raise_ledger_df.empty
    assert po_raised_db.list_raises(limit=10) == []
