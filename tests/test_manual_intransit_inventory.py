"""Manual in-transit overlay must merge into inventory on restore; FBA from RAR is opt-in."""
import pandas as pd
import pytest

from backend.services.inventory import (
    inventory_marketplace_breakdown,
    parse_fba_intransit_from_rar,
    strip_fba_intransit_unless_enabled,
)
from backend.services.manual_intransit_sheet import ensure_manual_intransit_overlay_applied
from backend.session import AppSession


def test_strip_fba_when_rar_parsing_disabled(monkeypatch):
    monkeypatch.setenv("INVENTORY_PARSE_FBA_FROM_RAR", "0")
    df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Amazon_Inventory": [10],
            "FBA_InTransit": [50],
            "Total_Inventory": [60],
        }
    )
    out = strip_fba_intransit_unless_enabled(df)
    assert int(out["FBA_InTransit"].iloc[0]) == 0
    assert int(out["Total_Inventory"].iloc[0]) == 10


def test_marketplace_breakdown_hides_fba_when_disabled(monkeypatch):
    monkeypatch.setenv("INVENTORY_PARSE_FBA_FROM_RAR", "0")
    df = pd.DataFrame({"OMS_SKU": ["X"], "FBA_InTransit": [100], "Total_Inventory": [100]})
    rows = inventory_marketplace_breakdown(df, {})
    fba = next(r for r in rows if r["key"] == "FBA_InTransit")
    assert fba["included"] is False
    assert fba["units"] == 0


def test_manual_overlay_merges_on_restore(monkeypatch, tmp_path):
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    import backend.main as m

    monkeypatch.setattr(m, "_DISK_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m, "bootstrap_warm_cache_if_empty", lambda: True)

    inv = pd.DataFrame({"OMS_SKU": ["SKU-A"], "Amazon_Inventory": [5], "Total_Inventory": [5]})
    overlay = pd.DataFrame(
        {"OMS_SKU": ["SKU-A", "SKU-B"], "Manual_InTransit": [10, 3], "Not_In_Inventory_Qty": [2, 7]}
    )
    m._warm_cache = {
        "inventory_df_variant": inv,
        "manual_intransit_overlay_df": overlay,
    }
    sess = AppSession()
    sess.inventory_df_variant = inv.copy()
    sess.manual_intransit_overlay_df = overlay.copy()
    assert ensure_manual_intransit_overlay_applied(sess)
    assert "Manual_InTransit" in sess.inventory_df_variant.columns
    assert int(sess.inventory_df_variant.loc[sess.inventory_df_variant["OMS_SKU"] == "SKU-A", "Manual_InTransit"].iloc[0]) == 10
    rows = inventory_marketplace_breakdown(sess.inventory_df_variant, {})
    manual = next(r for r in rows if r["key"] == "Manual_InTransit")
    assert manual["included"] is True
    assert manual["units"] == 13


def test_parse_fba_default_off():
    assert parse_fba_intransit_from_rar() is False
