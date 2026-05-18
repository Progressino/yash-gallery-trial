"""Empty sessions must load server warm cache even when pause_auto_data_restore is set."""
import pandas as pd
import pytest

from backend.main import (
    force_restore_session_from_server_cache,
    session_needs_operational_data,
)
from backend.session import AppSession, wipe_app_session


@pytest.fixture
def warm_cache_sample(monkeypatch):
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
        "snapdeal_df": pd.DataFrame(),
    }
    main._warm_cache_generation = 2
    yield
    main._warm_cache = {}
    main._warm_cache_generation = 0


def test_session_needs_operational_data_empty():
    sess = AppSession()
    assert session_needs_operational_data(sess) is True


def test_force_restore_ignores_pause(warm_cache_sample):
    sess = AppSession()
    wipe_app_session(sess)
    assert sess.pause_auto_data_restore is True
    assert session_needs_operational_data(sess)

    ok = force_restore_session_from_server_cache(sess, 2)
    assert ok is True
    assert not sess.mtr_df.empty
    assert not sess.sales_df.empty
    assert sess.pause_auto_data_restore is False


def test_coverage_fills_empty_session_with_pause(client, session_for_client, warm_cache_sample):
    _, sess = session_for_client
    wipe_app_session(sess)
    assert sess.pause_auto_data_restore is True

    r = client.get("/api/data/coverage")
    assert r.status_code == 200
    body = r.json()
    assert body["mtr"] is True
    assert body["sales"] is True
    assert body["mtr_rows"] >= 1
