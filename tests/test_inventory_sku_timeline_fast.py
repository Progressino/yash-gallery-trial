"""Single-SKU Inv History timeline must stay fast and SKU-scoped."""
from __future__ import annotations

import time

import pandas as pd

from backend.services.daily_inventory_history import inventory_history_sku_timeline


def _history(n_skus: int = 800, days: int = 30) -> pd.DataFrame:
    end = pd.Timestamp("2026-07-23")
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    rows = []
    for i in range(n_skus):
        sku = f"SKU{i:04d}-M" if i else "1072YKBLACK-4XL"
        for d in dates:
            rows.append(
                {
                    "OMS_SKU": sku,
                    "Date": d,
                    "Qty": float((i % 7) + 1),
                    "Source": "snapshot",
                    "Channel": "oms",
                }
            )
    return pd.DataFrame(rows)


def test_sku_timeline_returns_window_and_eff_days():
    df = _history(n_skus=200, days=30)
    t0 = time.perf_counter()
    out = inventory_history_sku_timeline(df, "1072YKBLACK-4XL", window_days=30, channel="combined")
    elapsed = time.perf_counter() - t0
    assert out["ok"] is True
    assert out["loaded"] is True
    assert out["window_end"] == "2026-07-23"
    assert len(out["rows"]) == 30
    assert out["in_stock_days"] == 30
    assert out["rows"][-1]["qty"] == 1.0
    # Must not densify the full catalog (old path was ~20s on prod).
    assert elapsed < 2.0, f"sku timeline too slow: {elapsed:.2f}s"


def test_sku_timeline_unknown_sku_empty():
    df = _history(n_skus=50, days=10)
    out = inventory_history_sku_timeline(df, "NOSUCH-SKU", window_days=10)
    assert out["ok"] is True
    assert out["rows"] == []
    assert out["in_stock_days"] == 0
