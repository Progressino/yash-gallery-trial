"""Tests for process-wide shared frame accessors."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from backend.services import shared_frames as sf
from backend.session import AppSession


@pytest.fixture(autouse=True)
def _reset_shared_frames_env(monkeypatch):
    monkeypatch.delenv("SESSION_SHARED_FRAMES", raising=False)


def test_shared_frames_enabled_from_env(monkeypatch):
    monkeypatch.setenv("SESSION_SHARED_FRAMES", "1")
    assert sf.shared_frames_enabled() is True
    monkeypatch.setenv("SESSION_SHARED_FRAMES", "0")
    assert sf.shared_frames_enabled() is False


def test_attach_shared_frames_uses_warm_cache_refs(monkeypatch):
    monkeypatch.setenv("SESSION_SHARED_FRAMES", "1")
    sales = pd.DataFrame({"TxnDate": ["2026-01-01"], "Sku": ["A"]})
    inv = pd.DataFrame({"OMS_SKU": ["A"], "Qty": [1]})
    wc = {"sales_df": sales, "inventory_df_variant": inv, "sku_mapping": {"A": "A"}}

    import backend.main as main

    monkeypatch.setattr(main, "_warm_cache", wc, raising=False)
    sess = AppSession()
    sf.attach_shared_frames(sess, warm_cache_generation=3)

    assert sess._shared_frames is True
    assert sess.sales_df is sales
    assert sess.inventory_df_variant is inv
    assert sf.session_sales_df(sess) is sales
    assert sf.frame_row_count("sales_df", sess) == 1


def test_should_skip_session_copy_large_keys(monkeypatch):
    monkeypatch.setenv("SESSION_SHARED_FRAMES", "1")
    assert sf.should_skip_session_copy("sales_df") is True
    assert sf.should_skip_session_copy("sku_mapping") is False
