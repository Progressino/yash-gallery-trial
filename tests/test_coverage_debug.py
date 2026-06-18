"""Tests for coverage debug payload."""
from __future__ import annotations

import pandas as pd

from backend.services.coverage_debug import build_coverage_debug
from backend.session import AppSession


def test_build_coverage_debug_shape(monkeypatch):
    sales = pd.DataFrame({"TxnDate": ["2026-01-01"] * 5})
    inv = pd.DataFrame({"OMS_SKU": ["A"], "Qty": [1]})
    wc = {"sales_df": sales, "inventory_df_variant": inv}

    import backend.main as main

    monkeypatch.setattr(main, "_warm_cache", wc, raising=False)
    monkeypatch.setattr(main, "_warm_cache_generation", 2, raising=False)
    monkeypatch.setattr(main, "_warm_cache_loaded_at", None, raising=False)

    sess = AppSession()
    sess.sales_df = sales
    sess.inventory_df_variant = inv

    out = build_coverage_debug(sess)
    assert "source" in out
    assert "session" in out
    assert "warm_cache" in out
    assert "platforms" in out
    assert out["session"]["sales_rows"] == 5
    assert out["warm_cache"]["loaded"] is True
    assert isinstance(out["platforms"]["amazon"], bool)
