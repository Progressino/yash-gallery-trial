"""Inv History view end must not invent days past the last snapshot."""
import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_history_view_end_date,
    restore_inventory_history_from_best_disk_backups,
)


def _hist(dates: list[str], source: str = "snapshot") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"] * len(dates),
            "Date": pd.to_datetime(dates),
            "Qty": [10.0] * len(dates),
            "Source": [source] * len(dates),
            "Channel": ["oms"] * len(dates),
        }
    )


def test_view_end_defaults_to_last_snapshot_not_today():
    df = _hist(["2026-07-20", "2026-07-21", "2026-07-23"])
    # derived-only day after last snapshot must not become the default end
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "OMS_SKU": ["SKU-A"],
                    "Date": [pd.Timestamp("2026-07-24")],
                    "Qty": [10.0],
                    "Source": ["derived"],
                    "Channel": ["oms"],
                }
            ),
        ],
        ignore_index=True,
    )
    assert inventory_history_view_end_date(df, None) == "2026-07-23"


def test_view_end_clamps_explicit_future_to_last_snapshot():
    df = _hist(["2026-07-20", "2026-07-23"])
    assert inventory_history_view_end_date(df, "2026-07-25") == "2026-07-23"


def test_restore_does_not_extend_past_current_max(monkeypatch, tmp_path):
    current = _hist(["2026-07-20", "2026-07-23"])
    older = _hist(["2026-07-20", "2026-07-23", "2026-07-24"])  # phantom tail

    def fake_candidates():
        p = tmp_path / "bak.parquet"
        older.to_parquet(p, index=False)
        return [p]

    monkeypatch.setattr(
        "backend.services.daily_inventory_history.iter_inventory_history_parquet_candidates",
        fake_candidates,
    )
    monkeypatch.setattr(
        "backend.services.daily_inventory_history.merge_inventory_history_candidates",
        lambda _paths: older.copy(),
    )
    out = restore_inventory_history_from_best_disk_backups(current)
    # Must not resurrect Jul 24 past current max
    if out is not None and not out.empty:
        assert pd.to_datetime(out["Date"]).max() <= pd.Timestamp("2026-07-23")
