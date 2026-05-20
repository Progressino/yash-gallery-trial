"""Regression: Tier-3 daily-auto must merge SQLite reload into session, not replace."""
import pandas as pd
import pytest

from backend.services.daily_store import merge_platform_data


def _tier3_line_df(platform: str, n: int, start: int = 0) -> pd.DataFrame:
    """Minimal line-level exports: unique LineKeys so Myntra/Meesho/Flipkart dedup stays stable."""
    idx = range(start, start + n)
    base = {
        "OrderId": [f"o{i}" for i in idx],
        "LineKey": [f"L{i}" for i in idx],
        "OMS_SKU": [f"S{i % 7}" for i in idx],
        "Date": [pd.Timestamp("2024-06-01")] * n,
        "TxnType": ["Shipment"] * n,
        "Quantity": [1] * n,
        "RawStatus": [f"st{i}" for i in idx],
    }
    if platform == "myntra":
        base["ParentOrderId"] = [f"P{i}" for i in idx]
    return pd.DataFrame(base)


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


@pytest.mark.parametrize("platform", ["flipkart", "meesho", "myntra"])
def test_merge_preserves_bulk_flipkart_meesho_myntra(platform: str):
    big = _tier3_line_df(platform, 120)
    small = _tier3_line_df(platform, 3, start=9000)
    out = merge_platform_data(big, small, platform)
    assert len(out) == 123


def test_restore_daily_merges_sqlite_into_nonempty_flipkart(monkeypatch):
    """_restore_daily_if_needed must merge Tier-3 SQLite, not skip when session already has rows."""
    import sys
    import types

    from backend.session import AppSession

    saved_main = sys.modules.get("backend.main")
    fake_main = types.ModuleType("backend.main")
    fake_main.session_needs_operational_data = lambda _s: False
    fake_main.force_restore_session_from_server_cache = lambda *a, **k: None
    fake_main._warm_cache_generation = 0
    fake_main._warm_cache = {}
    sys.modules["backend.main"] = fake_main
    try:
        # Import after stubbing main so ``import backend.main`` does not run FastAPI startup.
        from backend.routers import data as data_router

        existing = _tier3_line_df("flipkart", 40)

        def fake_load(platform, months=None, dedup=False, max_files=None):
            if platform != "flipkart":
                return pd.DataFrame()
            return _tier3_line_df("flipkart", 2, start=8000)

        monkeypatch.setattr("backend.services.daily_store.load_platform_data", fake_load)

        sess = AppSession()
        sess.flipkart_df = existing.copy()
        sess.daily_restored = False

        data_router._restore_daily_if_needed(sess)

        assert sess.daily_restored is True
        assert len(sess.flipkart_df) == 42
    finally:
        if saved_main is not None:
            sys.modules["backend.main"] = saved_main
        else:
            sys.modules.pop("backend.main", None)
