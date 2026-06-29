"""PO calculate lead time from the most recent raise day."""
from backend.db import po_raised_db
from backend.services.po_raise_lead_time import (
    DEFAULT_PO_LEAD_TIME_DAYS,
    apply_lead_time_from_last_raise,
    lead_time_for_raise_record,
    persist_raise_day_lead_time,
)
from backend.session import AppSession


def test_apply_lead_time_from_last_raise_uses_day_meta(tmp_path, monkeypatch):
    db = tmp_path / "po_lt.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()

    persist_raise_day_lead_time("2026-06-27", 45, source="test")

    sess = AppSession()
    body = {"planning_date": "2026-06-29", "lead_time": 60, "raise_ledger_lookback_days": 14}
    out, meta = apply_lead_time_from_last_raise(sess, body)

    assert out["lead_time"] == 45
    assert meta["lead_time_applied"] == 45
    assert meta["lead_time_source"] == "last_raise"
    assert meta["lead_time_raise_date"] == "2026-06-27"
    assert sess.po_calculate_lead_time == 45


def test_apply_lead_time_from_last_raise_respects_ignore_flag(tmp_path, monkeypatch):
    db = tmp_path / "po_lt_ignore.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()
    persist_raise_day_lead_time("2026-06-27", 45, source="test")

    sess = AppSession()
    body = {
        "planning_date": "2026-06-29",
        "lead_time": 60,
        "ignore_raise_lead_time": True,
    }
    out, meta = apply_lead_time_from_last_raise(sess, body)

    assert out["lead_time"] == 60
    assert meta == {}


def test_apply_lead_time_defaults_when_no_raises(tmp_path, monkeypatch):
    db = tmp_path / "po_lt_default.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()

    sess = AppSession()
    body = {"planning_date": "2026-06-29", "raise_ledger_lookback_days": 14}
    out, meta = apply_lead_time_from_last_raise(sess, body)

    assert out["lead_time"] == DEFAULT_PO_LEAD_TIME_DAYS
    assert meta["lead_time_source"] == "default"


def test_lead_time_for_raise_record_prefers_body(tmp_path, monkeypatch):
    db = tmp_path / "po_lt_raise.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()
    persist_raise_day_lead_time("2026-06-27", 45, source="test")

    sess = AppSession()
    sess.po_calculate_lead_time = 30
    assert lead_time_for_raise_record(sess, {"lead_time": 52}) == 52


def test_remove_raise_ledger_day_deletes_meta(tmp_path, monkeypatch):
    from backend.services.po_raise_remove import remove_raise_ledger_day

    db = tmp_path / "po_lt_remove.db"
    monkeypatch.setattr(po_raised_db, "DB_PATH", str(db))
    po_raised_db.init_db()

    day = "2026-06-27"
    persist_raise_day_lead_time(day, 45, source="test")
    po_raised_db.replace_raises_for_date(day, [{"oms_sku": "SKU-A", "qty": 10}])

    sess = AppSession()
    sess.po_raise_ledger_df = __import__("pandas").DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Raised_Qty": [10],
            "Raised_Date": __import__("pandas").to_datetime([day]),
        }
    )

    out = remove_raise_ledger_day(sess, day)
    assert out["ok"] is True
    assert po_raised_db.get_raise_day_meta(day) is None
