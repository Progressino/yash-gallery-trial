"""Bundled SKU map + helper lookup behaviour."""

import pytest

from backend.services.helpers import canonical_pl_sku_key, map_to_oms_sku
from backend.services.sku_mapping import (
    bundled_sku_mapping_json_path,
    clear_bundled_sku_mapping_cache,
    load_bundled_sku_mapping,
    parse_sku_mapping,
)


def test_map_to_oms_pl_fallback():
    m = {"1001YKBEIGE-3XL": "OMS-1"}
    assert map_to_oms_sku("1001PLYKBEIGE-3XL", m) == "OMS-1"
    assert map_to_oms_sku("1001YKBEIGE-3XL", m) == "OMS-1"


def test_canonical_pl_sku_key():
    assert canonical_pl_sku_key("1023PLYKBLUE-L") == "1023YKBLUE-L"


def test_parse_myntra_style_and_yrn(tmp_path):
    import pandas as pd

    p = tmp_path / "t.xlsx"
    df = pd.DataFrame(
        {
            "DATE": ["2025-01-01"],
            "YRN NUMBER": ["YR1"],
            "MYNTRA SKU CODE": ["1001PLYKBEIGE-3XL"],
            "STYLE ID": ["99"],
            "OMS SKU CODE": ["1001YKBEIGE-3XL"],
            "BRAND": ["X"],
        }
    )
    df.to_excel(p, sheet_name="MYNTRA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m["1001PLYKBEIGE-3XL"] == "1001YKBEIGE-3XL"
    assert m["99"] == "1001YKBEIGE-3XL"
    assert m["YR1"] == "1001YKBEIGE-3XL"


def test_bundled_json_loads(monkeypatch):
    if not bundled_sku_mapping_json_path().is_file():
        pytest.skip("bundled yash_sku_mapping_master.json not present")
    monkeypatch.delenv("SKIP_BUNDLED_SKU_MAPPING", raising=False)
    clear_bundled_sku_mapping_cache()
    b = load_bundled_sku_mapping()
    assert len(b) > 50_000
    assert "1001PLYKBEIGE-3XL" in b or "1001YKBEIGE-3XL" in b
