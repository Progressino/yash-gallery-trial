"""Regression: Tier-3 daily-auto must merge SQLite reload into session, not replace."""
import pandas as pd

from backend.services.daily_store import merge_platform_data


def test_merge_preserves_bulk_when_adding_smaller_store_slice():
    """If session already holds bulk history, merging a small Tier-3 slice must not shrink rows."""
    # Snapdeal path: avoids Myntra-specific parent/linekey shadow rules that collapse test grids.
    big = pd.DataFrame(
        {
            "OrderId": [f"o{i}" for i in range(500)],
            "OMS_SKU": [f"S{i % 3}" for i in range(500)],
            "Date": [pd.Timestamp("2024-06-01")] * 500,
            "TxnType": ["Shipment"] * 500,
            "Quantity": [1] * 500,
            "RawStatus": [f"st{i}" for i in range(500)],
        }
    )
    small = pd.DataFrame(
        {
            "OrderId": ["new1", "new2"],
            "OMS_SKU": ["S9", "S9"],
            "Date": [pd.Timestamp("2026-05-20"), pd.Timestamp("2026-05-20")],
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [1, 1],
            "RawStatus": ["a", "b"],
        }
    )
    out = merge_platform_data(big, small, "snapdeal")
    assert len(out) == 502
