"""POST /data/restore-full — warm + disk + Tier-3 + GitHub gap fill."""
import pandas as pd
import pytest

from backend.main import force_restore_session_from_server_cache
from backend.session import AppSession, wipe_app_session


@pytest.fixture
def warm_with_snapdeal(monkeypatch):
    import backend.main as main

    main._warm_cache = {
        "mtr_df": pd.DataFrame({"OMS_SKU": ["A"], "Qty": [1]}),
        "sales_df": pd.DataFrame(
            {
                "TxnDate": pd.to_datetime(["2026-01-01"]),
                "OMS_SKU": ["A"],
                "Qty": [1],
                "Channel": ["MTR"],
            }
        ),
        "sku_mapping": {"A": "A"},
        "myntra_df": pd.DataFrame(),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(
            {
                "TxnDate": pd.to_datetime(["2026-02-01"]),
                "OMS_SKU": ["SD1"],
                "Qty": [2],
            }
        ),
    }
    main._warm_cache_generation = 3
    yield
    main._warm_cache = {}
    main._warm_cache_generation = 0


def test_restore_full_fills_snapdeal_from_warm(client, session_for_client, warm_with_snapdeal):
    _, sess = session_for_client
    wipe_app_session(sess)
    sess.snapdeal_df = pd.DataFrame()
    assert sess.snapdeal_df.empty

    r = client.post("/api/data/restore-full?sync=1")
    assert r.status_code == 200
    body = r.json()
    assert body["snapdeal"] is True
    assert body["snapdeal_rows"] >= 1
    assert "snapdeal" not in (body.get("missing_platforms") or [])
    assert not sess.snapdeal_df.empty


def test_restore_full_force_tier3_when_platform_empty(monkeypatch, client, session_for_client):
    """Blocking restore merges Tier-3 when session frame is empty but SQLite has rows."""
    import backend.main as main
    from backend.services import daily_store

    _, sess = session_for_client
    wipe_app_session(sess)
    snap_df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2026-03-01"]),
            "OMS_SKU": ["T3-SD"],
            "Qty": [5],
        }
    )

    def _fake_load(platform, months=None, dedup=False, max_files=None):
        if platform == "snapdeal":
            return snap_df.copy()
        return pd.DataFrame()

    monkeypatch.setattr(daily_store, "load_platform_data", _fake_load)
    monkeypatch.setattr(
        daily_store,
        "get_summary",
        lambda: {"snapdeal": {"file_count": 1, "max_date": "2026-03-01"}},
    )
    main._warm_cache = {}
    main._warm_cache_generation = 0

    r = client.post("/api/data/restore-full?sync=1")
    assert r.status_code == 200
    body = r.json()
    assert body["snapdeal"] is True
    assert body["snapdeal_rows"] >= 1
