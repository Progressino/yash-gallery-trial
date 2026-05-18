"""PO raise ledger must survive new sessions and weekend gaps."""
from pathlib import Path

import pandas as pd
import pytest

from backend.services.po_raise_archive import (
    parse_raise_date_from_filename,
    save_archive,
    try_auto_import_recent_ledgers,
)
from backend.services.po_raise_import import (
    apply_ledger_import,
    hydrate_session_ledger_from_db,
    parse_ledger_upload_bytes,
    sync_ledger_to_durable_db,
)
from backend.session import AppSession

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "po_recommendation_16-5-26.csv"


@pytest.fixture
def archive_root(tmp_path, monkeypatch):
    import backend.services.po_raise_archive as arch

    monkeypatch.setattr(arch, "_ARCHIVE_DIR", str(tmp_path))
    arch._resolved_archive_root = None
    return tmp_path


def test_parse_raise_date_from_filename():
    assert parse_raise_date_from_filename("po_recommendation 16-5-26.csv") == pd.Timestamp("2026-05-16")


def test_global_archive_imported_for_new_session(archive_root):
    """Saturday export must apply Monday for a different session id."""
    raw = _FIXTURE.read_bytes()
    sat = pd.Timestamp("2026-05-16")
    save_archive("old-session-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", sat, raw)

    sess = AppSession()
    out = try_auto_import_recent_ledgers(
        sess,
        "new-session-bbbb-cccc-dddd-eeeeeeeeffff",
        "2026-05-18",
        lookback_days=14,
    )
    assert out and out.get("ok")
    assert "2026-05-16" in (out.get("imported_days") or [])
    units = int(sess.po_raise_ledger_df["Raised_Qty"].sum())
    assert units >= 10_000
    assert int((sess.po_raise_ledger_df["Raised_Qty"] > 0).sum()) >= 200


def test_saturday_csv_builds_confirmed_raise_pipeline():
    """Saturday export (~10k units) must appear in PO_Confirmed_Raise_Pipeline."""
    from backend.services.po_raise_ledger import aggregate_raise_ledger_for_po

    if not _FIXTURE.is_file():
        pytest.skip("fixture missing")

    raw = _FIXTURE.read_bytes()
    accum, err = parse_ledger_upload_bytes(raw, "po_recommendation 16-5-26.csv")
    assert err is None
    total = sum(accum.values())
    assert total >= 10_000

    sess = AppSession()
    apply_ledger_import(sess, accum, pd.Timestamp("2026-05-16"), replace_day=True)
    lag = aggregate_raise_ledger_for_po(
        sess.po_raise_ledger_df,
        None,
        pd.Timestamp("2026-05-18"),
        lookback_days=14,
    )
    assert int(lag["PO_Confirmed_Raise_Pipeline"].sum()) == total
    assert int(lag["PO_Confirmed_Raise_Pipeline"].sum()) >= 10_000


def test_hydrate_session_ledger_from_sqlite(tmp_path, monkeypatch):
    import backend.db.po_raised_db as db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "po_raised.db"))
    db.init_db()

    sess = AppSession()
    apply_ledger_import(
        sess,
        {"SKU-A": 100, "SKU-B": 50},
        pd.Timestamp("2026-05-16"),
        replace_day=True,
    )
    sync_ledger_to_durable_db(sess, pd.Timestamp("2026-05-16"))

    fresh = AppSession()
    ok = hydrate_session_ledger_from_db(fresh, "2026-05-18", lookback_days=14)
    assert ok is True
    assert int(fresh.po_raise_ledger_df["Raised_Qty"].sum()) == 150
