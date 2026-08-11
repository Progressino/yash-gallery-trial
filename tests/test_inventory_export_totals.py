"""Inventory Daily: UI totals == export totals; Aug-11 style recon smoke."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pytest

from backend.services.inventory import (
    inventory_column_totals,
    inventory_export_csv_bytes,
    inventory_totals_match_frame,
    inventory_variant_for_api,
    load_inventory_consolidated,
)


def _load_aug11_if_present() -> tuple[pd.DataFrame, dict] | None:
    base = Path(__file__).resolve().parents[1] / "_inv_aug11_inv" / "Inventory 11-Aug-26"
    if not base.is_dir():
        return None
    oms, fk, myn, amz = [], [], [], []
    for f in sorted(base.iterdir()):
        if not f.is_file():
            continue
        raw = f.read_bytes()
        name = f.name.lower()
        if "oms" in name or "combo" in name:
            oms.append(raw)
        elif "current inventory" in name:
            fk.append(raw)
        elif "seller_inventory" in name:
            myn.append(raw)
        else:
            amz.append(raw)
    if not oms:
        return None
    df, dbg = load_inventory_consolidated(oms, fk, myn, amz, {}, return_debug=True)
    return inventory_variant_for_api(df), dbg


def test_export_totals_row_matches_ui_totals():
    df = pd.DataFrame(
        {
            "OMS_SKU": ["A-1", "B-2", "C-3"],
            "OMS_Inventory": [10, 20, 0],
            "Amazon_Inventory": [5, 0, 7],
            "Marketplace_Total": [5, 0, 7],
            "Total_Inventory": [15, 20, 7],
            "Buffer_Stock": [1, 2, 0],
        }
    )
    totals = inventory_column_totals(df)
    raw, name = inventory_export_csv_bytes(df, totals=totals, snapshot_label="11-Aug-2026")
    assert name.endswith(".csv")
    text = raw.decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "OMS_SKU"
    assert rows[1][0] == "__TOTALS__"
    # OMS total column
    oms_idx = rows[0].index("OMS_Inventory")
    assert int(rows[1][oms_idx]) == 30
    total_idx = rows[0].index("Total_Inventory")
    assert int(rows[1][total_idx]) == 42


def test_stale_totals_detection():
    df = pd.DataFrame(
        {
            "OMS_SKU": ["X"],
            "OMS_Inventory": [100],
            "Total_Inventory": [100],
        }
    )
    good = inventory_column_totals(df)
    assert inventory_totals_match_frame(good, df)
    bad = {**good, "OMS_Inventory": 201315}
    assert not inventory_totals_match_frame(bad, df)


def test_aug11_oms_total_and_examples():
    packed = _load_aug11_if_present()
    if packed is None:
        pytest.skip("Aug 11 inventory extract not in workspace")
    df, _dbg = packed
    totals = inventory_column_totals(df)
    assert totals["OMS_Inventory"] == 201315
    # Raw OMS CSV 0 for these SKUs — marketplace may still have FBA stock
    for sku in ("AK-228BLACK-3XL", "AK-228BLACK-5XL"):
        row = df[df["OMS_SKU"].astype(str).str.upper() == sku]
        assert not row.empty, sku
        assert float(row.iloc[0]["OMS_Inventory"]) == 0.0
    # Export OMS total == UI OMS total
    raw, _ = inventory_export_csv_bytes(df, totals=totals)
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    oms_idx = rows[0].index("OMS_Inventory")
    assert int(rows[1][oms_idx]) == totals["OMS_Inventory"]
    # Combined business rule for daily: Total = OMS + Marketplace (sum, not max)
    assert totals["Total_Inventory"] == totals["OMS_Inventory"] + totals["Marketplace_Total"]


def test_export_job_module_importable():
    from backend.services.inventory_history_export_jobs import (
        get_matrix_export_job,
        start_matrix_export_job,
    )

    # Tiny synthetic history
    hist = pd.DataFrame(
        {
            "OMS_SKU": ["SKU1", "SKU1"],
            "Date": pd.to_datetime(["2026-08-10", "2026-08-11"]),
            "Qty": [10.0, 12.0],
            "Source": ["snapshot", "snapshot"],
            "Channel": ["oms", "oms"],
        }
    )
    jid = start_matrix_export_job(history_df=hist, q="", days=5, channel="oms")
    assert jid
    st = get_matrix_export_job(jid)
    assert st is not None
    # Worker is async — at least queued/running/ready/error present
    assert st.get("status") in ("queued", "running", "ready", "error")
