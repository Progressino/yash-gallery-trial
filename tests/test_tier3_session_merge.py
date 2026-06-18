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


def test_build_po_ads_platform_sales_overlays_tier3(monkeypatch):
    sess = AppSession()
    sess.sku_mapping = {"SKU1": "SKU1"}
    sess.mtr_df = pd.DataFrame({
        "Date": ["2025-01-01"],
        "SKU": ["SKU1"],
        "Transaction_Type": ["Shipment"],
        "Quantity": [99],
    })
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
        lambda plat, _s, _e, dedup=True, columns_only=False: daily.copy()
        if plat == "amazon"
        else pd.DataFrame(),
    )
    merge_calls: list[tuple] = []

    def _merge(cur, df, plat):
        merge_calls.append((len(cur), len(df), plat))
        return pd.concat([cur, df], ignore_index=True) if not df.empty else cur

    monkeypatch.setattr("backend.services.daily_store.merge_platform_data", _merge)

    built = pd.DataFrame({"TxnDate": ["2026-06-10"], "Quantity": [5], "OMS_SKU": ["SKU1"]})

    def _fake_build_platform_sales_df(_sess, *, frame_overrides=None):
        assert frame_overrides is not None
        assert "mtr_df" in frame_overrides
        return built

    monkeypatch.setattr(
        "backend.services.po_calculate_run._build_platform_sales_df",
        _fake_build_platform_sales_df,
    )

    out = t3.build_po_ads_platform_sales(
        sess,
        planning_date="2026-06-18",
        period_days=30,
    )
    assert len(out) == 1
    # Tier-3 authoritative: merge only pre-tier3 bulk (1 old row), not full session on top of tier3.
    assert merge_calls and merge_calls[0][0] == 1


def test_build_po_ads_platform_sales_fast_path_uses_session_sales(monkeypatch):
    """When sales_df is T-1 fresh, skip Tier-3 rebuild entirely."""
    sess = AppSession()
    days = pd.date_range("2025-02-01", "2026-06-17", freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "TxnDate": days,
            "Quantity": [1] * len(days),
            "Transaction Type": ["Shipment"] * len(days),
            "Sku": ["SKU1"] * len(days),
            "Source": ["Amazon"] * len(days),
            "Units_Effective": [1] * len(days),
        }
    )

    def _should_not_load(*_a, **_kw):
        raise AssertionError("Tier-3 load should not run on fast path")

    monkeypatch.setattr(
        "backend.services.daily_store.load_platform_data_for_report_range",
        _should_not_load,
    )
    out = t3.build_po_ads_platform_sales(
        sess,
        planning_date="2026-06-18",
        period_days=30,
        use_seasonality=True,
        use_ly_fallback=True,
    )
    assert not out.empty
    assert out["TxnDate"].max() <= pd.Timestamp("2026-06-18")


def test_build_po_ads_platform_sales_incremental_gap(monkeypatch):
    sess = AppSession()
    sess.sku_mapping = {"SKU1": "SKU1"}
    days = pd.date_range("2025-02-01", "2026-06-14", freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "TxnDate": days,
            "Quantity": [1] * len(days),
            "Transaction Type": ["Shipment"] * len(days),
            "Sku": ["SKU1"] * len(days),
            "Source": ["Amazon"] * len(days),
            "Units_Effective": [1] * len(days),
        }
    )
    gap_sales = pd.DataFrame(
        {
            "TxnDate": ["2026-06-15", "2026-06-16", "2026-06-17"],
            "Quantity": [2, 2, 2],
            "Transaction Type": ["Shipment"] * 3,
            "Sku": ["SKU1"] * 3,
            "Source": ["Amazon"] * 3,
            "Units_Effective": [2] * 3,
        }
    )
    monkeypatch.setattr(
        "backend.services.daily_store.platforms_with_uploads_in_range",
        lambda _s, _e: ["amazon"],
    )
    monkeypatch.setattr(
        "backend.services.daily_store.load_platform_data_for_report_range",
        lambda plat, _s, _e, dedup=True, columns_only=False: pd.DataFrame(
            {"Date": ["2026-06-15"], "SKU": ["SKU1"], "Transaction_Type": ["Shipment"], "Quantity": [2]}
        )
        if plat == "amazon"
        else pd.DataFrame(),
    )
    monkeypatch.setattr(
        "backend.services.po_calculate_run._build_platform_sales_df",
        lambda _s, *, frame_overrides=None: gap_sales,
    )

    out = t3.build_po_ads_platform_sales(
        sess,
        planning_date="2026-06-18",
        period_days=30,
        use_seasonality=True,
        use_ly_fallback=True,
    )
    assert pd.Timestamp(out["TxnDate"].max()).date() == pd.Timestamp("2026-06-17").date()
    assert int(out.loc[out["TxnDate"] >= "2026-06-15", "Quantity"].sum()) >= 6


def test_po_fingerprint_includes_tier3_token(monkeypatch):
    from backend.services.po_shared_cache import build_data_fingerprint

    sess = AppSession()
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token",
        lambda: {"amazon": "3:300:x"},
    )
    fp = build_data_fingerprint(sess, {"planning_date": "2026-06-18", "period_days": 30})
    assert fp["tier3_sync_token"] == {"amazon": "3:300:x"}
