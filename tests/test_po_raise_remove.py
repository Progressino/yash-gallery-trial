"""PO raise ledger removal (session + SQLite)."""
from pathlib import Path

import pandas as pd

from backend.db import po_raised_db
from backend.services.po_raise_remove import (
    clear_raise_ledger_all,
    remove_raise_ledger_day,
    remove_raise_ledger_skus,
)
from backend.session import AppSession


def test_invalidate_po_calculate_clears_stale_table():
    from backend.services.po_raise_remove import invalidate_po_calculate_result

    sess = AppSession()
    sess.po_calculate_status = "done"
    sess.po_calculate_result = {"ok": True, "total_rows": 10}
    sess.po_calculate_result_df = pd.DataFrame({"OMS_SKU": ["A"], "PO_Qty": [5]})
    invalidate_po_calculate_result(sess)
    assert sess.po_calculate_status == "idle"
    assert sess.po_calculate_result == {}
    assert sess.po_calculate_result_df.empty


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
    assert out.get("recalculate_required") is True
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


def test_remove_raise_ledger_day_blocks_auto_import(tmp_path, monkeypatch):
    import backend.services.po_raise_archive as arch
    from backend.services.po_raise_archive import save_archive, try_auto_import_recent_ledgers

    archive_root = tmp_path / "arch"
    archive_root.mkdir()
    monkeypatch.setattr(arch, "_ARCHIVE_DIR", str(archive_root))
    arch._resolved_archive_root = None

    db = tmp_path / "po_raised_sup.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()

    fixture = Path(__file__).resolve().parent / "fixtures" / "po_recommendation_16-5-26.csv"
    if not fixture.is_file():
        import pytest

        pytest.skip("fixture missing")

    day = "2026-05-22"
    save_archive(
        "sess-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        pd.Timestamp(day),
        fixture.read_bytes(),
    )

    sess = AppSession()
    out = remove_raise_ledger_day(sess, day, session_id="sess-bbbb-cccc-dddd-eeee-ffffffffff")
    assert out["ok"] is True
    assert out.get("suppressed") is True
    assert po_raised_db.is_raise_date_suppressed(day)

    fresh = AppSession()
    auto = try_auto_import_recent_ledgers(
        fresh,
        "sess-cccc-dddd-eeee-ffff-gggggggggggg",
        "2026-05-25",
        lookback_days=14,
    )
    assert auto is None or day not in (auto.get("imported_days") or [])
    if not fresh.po_raise_ledger_df.empty:
        rd = pd.to_datetime(fresh.po_raise_ledger_df["Raised_Date"], errors="coerce").dt.normalize()
        assert day not in {str(pd.Timestamp(d).date()) for d in rd.dropna().unique()}


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
