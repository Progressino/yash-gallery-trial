"""Snapshot inventory RAR manifest and upload result payload."""

from pathlib import Path

import pandas as pd
import pytest

from backend.services.inventory import _extract_all_from_rar, load_inventory_consolidated


def _sample_inventory_rar() -> Path | None:
    for name in ("Inventory 25-May-26.rar", "Inventory 23-May-26.rar"):
        p = Path("/Users/samraisinghani/Downloads") / name
        if p.is_file():
            return p
    return None


@pytest.mark.skipif(_sample_inventory_rar() is None, reason="sample inventory RAR not on disk")
def test_inventory_rar_manifest_lists_all_inner_files():
    raw = _sample_inventory_rar().read_bytes()
    extracted, manifest = _extract_all_from_rar(raw)
    assert len(manifest) >= 10
    loaded = [m for m in manifest if m["status"] == "loaded"]
    assert len(loaded) >= 10
    assert len(extracted["oms_csvs"]) >= 1
    assert len(extracted["flipkart_csvs"]) >= 1
    assert len(extracted["fba_tsvs"]) >= 1


@pytest.mark.skipif(_sample_inventory_rar() is None, reason="sample inventory RAR not on disk")
def test_inventory_rar_loads_flipkart_and_myntra_columns():
    raw = _sample_inventory_rar().read_bytes()
    df, debug = load_inventory_consolidated(None, None, None, raw, {}, return_debug=True)
    assert "Flipkart_Inventory" in df.columns
    assert int(df["Flipkart_Inventory"].sum()) > 0
    assert debug.get("flipkart", "").startswith("0 SKUs") is False
    # 25-May-26 bundle has Flipkart warehouse + PPMP files; Myntra PPMP may be absent
    if int(df.get("Myntra_Other_Inventory", pd.Series(dtype=float)).sum() or 0) > 0:
        assert "Myntra_Other_Inventory" in df.columns


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
