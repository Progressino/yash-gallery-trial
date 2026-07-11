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


def test_platform_frame_for_window_falls_back_to_disk_when_mem_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    mem = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-01"]),
            "SKU": ["X"],
            "OMS_SKU": ["X"],
            "TxnType": ["Shipment"],
            "Quantity": [1.0],
        }
    )
    disk = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-02-01", "2025-02-02"]),
            "SKU": ["A", "B"],
            "OMS_SKU": ["A", "B"],
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [1.0, 1.0],
        }
    )
    disk.to_parquet(tmp_path / "meesho_df.parquet", index=False)
    monkeypatch.setattr(sf, "warm_frame", lambda attr, sess=None: mem if attr == "meesho_df" else pd.DataFrame())

    out = sf.platform_frame_for_window(
        "meesho_df", None, start_date="2025-01-01", end_date="2025-03-31"
    )
    assert len(out) == 2
    out_mem = sf.platform_frame_for_window(
        "meesho_df", None, start_date="2026-04-01", end_date="2026-07-01"
    )
    assert len(out_mem) == 1


def test_platform_frame_prefers_disk_when_mem_meesho_mostly_blank(tmp_path, monkeypatch):
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    mem = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-02-01", "2025-02-02"]),
            "SKU": ["", ""],
            "OMS_SKU": ["", ""],
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [1.0, 1.0],
        }
    )
    disk = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-02-01", "2025-02-02"]),
            "SKU": ["1057YKBLUE-5XL", "165YK251MUSTRAD-XL"],
            "OMS_SKU": ["1057YKBLUE-5XL", "165YK251MUSTRAD-XL"],
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [1.0, 1.0],
        }
    )
    disk.to_parquet(tmp_path / "meesho_df.parquet", index=False)
    monkeypatch.setattr(sf, "warm_frame", lambda attr, sess=None: mem if attr == "meesho_df" else pd.DataFrame())

    out = sf.platform_frame_for_window(
        "meesho_df", None, start_date="2025-01-01", end_date="2025-03-31"
    )
    assert len(out) == 2
    assert out["OMS_SKU"].tolist() == ["1057YKBLUE-5XL", "165YK251MUSTRAD-XL"]
