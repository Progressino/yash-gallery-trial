"""Daily inventory history must never be downgraded on disk/warm-cache by stale PO sessions."""
import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_history_is_newer_than,
    inventory_history_view_end_date,
    overlay_inventory_variant_from_history,
)


def _hist(sku: str, end: str, days: int = 30) -> pd.DataFrame:
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    return pd.DataFrame({"OMS_SKU": [sku] * days, "Date": dates, "Qty": [5.0] * days})


def test_inventory_history_is_newer_than_prefers_later_max_date():
    old = _hist("A", "2026-05-30", 3)
    new = _hist("A", "2026-06-26", 30)
    assert inventory_history_is_newer_than(new, old)
    assert not inventory_history_is_newer_than(old, new)


def test_view_end_date_anchors_on_matrix_when_behind_today():
    df = _hist("A", "2026-05-30", 3)
    assert inventory_history_view_end_date(df) == "2026-05-30"


def test_overlay_updates_total_inventory_only():
    hist = _hist("SKU-A", "2026-06-26", 5)
    inv = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "OMS_Inventory": [10.0], "Total_Inventory": [25.0]},
    )
    out, meta = overlay_inventory_variant_from_history(
        inv,
        hist,
        snapshot_date="",
        reference_date="2026-06-26",
    )
    assert meta["applied"] is True
    assert float(out.loc[0, "Total_Inventory"]) == 5.0
    assert float(out.loc[0, "OMS_Inventory"]) == 10.0
