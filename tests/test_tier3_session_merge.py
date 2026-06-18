"""Tests for Tier-3 session merge (PO + Intelligence parity)."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.services import tier3_session_merge as t3
from backend.session import AppSession


def test_tier3_token_mismatch_detects_new_upload(monkeypatch):
    sess = AppSession()
    sess._tier3_sync_token_applied = {"amazon": "1:10:old"}
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token",
        lambda: {"amazon": "2:20:new"},
    )
    assert t3.tier3_token_mismatch(sess) is True
    assert t3.platforms_with_tier3_token_mismatch(sess) == ["amazon"]


def test_build_parity_report_warns_when_tier3_empty_but_session_has_sales(monkeypatch):
    sess = AppSession()
    sess.sales_df = pd.DataFrame(
        {
            "TxnDate": ["2026-06-01"],
            "Quantity": [1],
            "Transaction Type": ["Shipment"],
            "Sku": ["A"],
            "Source": ["Amazon"],
        }
    )
    monkeypatch.setattr("backend.services.daily_store.get_summary", lambda: {})
    monkeypatch.setattr("backend.services.daily_store.get_tier3_sync_token", lambda: {})
    monkeypatch.setattr(t3, "tier3_token_mismatch", lambda _s: False)
    monkeypatch.setattr(t3, "session_platform_shorter_than_tier3", lambda _s: False)

    report = t3.build_parity_report(sess, planning_date="2026-06-18")
    assert report["tier3_file_count"] == 0
    assert any("No Tier-3" in w for w in report["warnings"])
    assert report["ok"] is False


def test_ensure_tier3_merged_for_po_merges_and_rebuilds_sales(monkeypatch):
    sess = AppSession()
    sess.sku_mapping = {"SKU1": "SKU1"}
    sess.mtr_df = pd.DataFrame()
    daily = pd.DataFrame(
        {
            "Date": ["2026-06-10"],
            "SKU": ["SKU1"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [5],
        }
    )

    monkeypatch.setattr(
        "backend.services.daily_store.get_summary",
        lambda: {"amazon": {"file_count": 1}},
    )
    monkeypatch.setattr(
        "backend.services.daily_store.platforms_with_uploads_in_range",
        lambda _s, _e: ["amazon"],
    )
    monkeypatch.setattr(
        "backend.services.daily_store.load_platform_data_for_report_range",
        lambda plat, _s, _e, dedup=False: daily.copy() if plat == "amazon" else pd.DataFrame(),
    )
    monkeypatch.setattr(
        "backend.services.daily_store.merge_platform_data",
        lambda cur, df, plat: pd.concat([cur, df], ignore_index=True) if not df.empty else cur,
    )
    monkeypatch.setattr(t3, "tier3_token_mismatch", lambda _s: True)
    monkeypatch.setattr(t3, "session_platform_shorter_than_tier3", lambda _s: False)
    monkeypatch.setattr(t3, "platforms_with_tier3_token_mismatch", lambda _s: ["amazon"])

    built = pd.DataFrame({"TxnDate": ["2026-06-10"], "Quantity": [5]})

    def _fake_build_sales_df(**_kw):
        return built

    monkeypatch.setattr("backend.services.sales.build_sales_df", _fake_build_sales_df)
    monkeypatch.setattr(
        "backend.services.po_return_import.aggregate_return_overlay_for_use",
        lambda _ov: None,
    )

    out = t3.ensure_tier3_merged_for_po(
        sess,
        planning_date="2026-06-18",
        period_days=30,
    )
    assert out["merged"] is True
    assert "amazon" in out["platforms"]
    assert len(sess.sales_df) == 1


def test_po_fingerprint_includes_tier3_token(monkeypatch):
    from backend.services.po_shared_cache import build_data_fingerprint

    sess = AppSession()
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token",
        lambda: {"amazon": "3:300:x"},
    )
    fp = build_data_fingerprint(sess, {"planning_date": "2026-06-18", "period_days": 30})
    assert fp["tier3_sync_token"] == {"amazon": "3:300:x"}
