"""Regression: daily Inventory RAR auto-detects OMS + marketplace CSVs."""
from __future__ import annotations

from pathlib import Path

import pytest

RAR_CANDIDATES = [
    Path("/Users/samraisinghani/Downloads/Inventory 16-Jul-26.rar"),
    Path(__file__).resolve().parents[1] / "_inv_rar_extract" / "Inventory 16-Jul-26.rar",
]


def _rar_bytes() -> bytes | None:
    for p in RAR_CANDIDATES:
        if p.is_file():
            return p.read_bytes()
    return None


@pytest.fixture(scope="module")
def inventory_rar():
    raw = _rar_bytes()
    if raw is None:
        pytest.skip("Inventory 16-Jul-26.rar not available locally")
    return raw


def test_inventory_rar_routes_to_snapshot(inventory_rar):
    from backend.services.upload_file_sniff import (
        classify_upload_document,
        partition_files_by_upload_target,
    )

    sniff = classify_upload_document(inventory_rar, "Inventory 16-Jul-26.rar")
    assert sniff["category"] == "snapshot_inventory"

    # Even if dropped on Daily sales, auto-route to snapshot inventory.
    buckets, notes = partition_files_by_upload_target(
        [("Inventory 16-Jul-26.rar", inventory_rar)],
        "daily_sales",
    )
    assert len(buckets.get("snapshot_inventory") or []) == 1
    assert notes


def test_inventory_rar_extracts_all_marketplaces(inventory_rar):
    from backend.services.inventory import _extract_all_from_rar, load_inventory_consolidated

    extracted, manifest = _extract_all_from_rar(inventory_rar)
    loaded = [m for m in manifest if m.get("status") == "loaded"]
    cats = {m.get("category") for m in loaded}
    assert "oms" in cats
    assert "amazon" in cats
    assert "flipkart" in cats
    assert "myntra" in cats
    assert extracted["oms_csvs"]
    assert extracted["amz_csvs"]
    assert extracted["flipkart_csvs"]
    assert extracted["myntra_csvs"]

    df, debug = load_inventory_consolidated(
        None, None, None, inventory_rar, {}, return_debug=True
    )
    assert not df.empty
    assert float(df["OMS_Inventory"].sum()) > 100_000
    assert float(df["Amazon_Inventory"].sum()) > 10_000
    assert debug.get("amz_disclaimer", {}).get("latest_report_date")
