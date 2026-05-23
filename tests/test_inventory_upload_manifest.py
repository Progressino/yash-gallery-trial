"""Snapshot inventory RAR manifest and upload result payload."""

from pathlib import Path

import pytest

from backend.services.inventory import _extract_all_from_rar, load_inventory_consolidated


@pytest.mark.skipif(
    not Path("/Users/samraisinghani/Downloads/Inventory 23-May-26.rar").is_file(),
    reason="sample inventory RAR not on disk",
)
def test_inventory_rar_manifest_lists_all_inner_files():
    raw = Path("/Users/samraisinghani/Downloads/Inventory 23-May-26.rar").read_bytes()
    extracted, manifest = _extract_all_from_rar(raw)
    assert len(manifest) >= 10
    loaded = [m for m in manifest if m["status"] == "loaded"]
    assert len(loaded) >= 10
    assert len(extracted["oms_csvs"]) >= 1
    assert len(extracted["flipkart_csvs"]) >= 1
    assert len(extracted["myntra_csvs"]) >= 1
    assert len(extracted["fba_tsvs"]) >= 1


def test_build_inventory_upload_payload():
    from backend.routers import upload as upload_router

    debug = {
        "oms": "100 SKUs",
        "flipkart": "80 SKUs",
        "rar_manifest": [
            {"filename": "OMS.csv", "category": "oms", "status": "loaded"},
            {"filename": "bad.txt", "category": "other", "status": "skipped", "reason": "Not CSV"},
        ],
    }
    import pandas as pd

    df = pd.DataFrame({"OMS_SKU": ["A"], "Total_Inventory": [1]})
    payload = upload_router._build_inventory_upload_payload(
        df_variant=df,
        debug=debug,
        detected=["RAR archive (bundle.rar)"],
        file_parts=[("bundle.rar", b"x")],
    )
    assert payload["ok"] is True
    assert payload["rows"] == 1
    assert payload["saved_files"] == 1
    assert payload["skipped_files"] == 1
    assert len(payload["file_results"]) == 2
    assert "OMS" in payload["message"]
