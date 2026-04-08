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


def test_yrn_full_string_registers_numeric_suffix_for_ppmp(tmp_path):
    """YRN NUMBER like YARYKASS100672680 must map sales token 100672680 to OMS."""
    import pandas as pd

    p = tmp_path / "t.xlsx"
    df = pd.DataFrame(
        {
            "DATE": ["2025-01-01"],
            "YRN NUMBER": ["YARYKASS100672680"],
            "MYNTRA SKU CODE": [""],
            "STYLE ID": ["31228609"],
            "OMS SKU CODE": ["1001YK-TAIL-L"],
            "BRAND": ["X"],
        }
    )
    df.to_excel(p, sheet_name="MYNTRA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m["YARYKASS100672680"] == "1001YK-TAIL-L"
    assert m["100672680"] == "1001YK-TAIL-L"
    assert m["31228609"] == "1001YK-TAIL-L"


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


def test_meesho_pushpa_prefers_meesho_sku_column(tmp_path):
    """MEESHO PUSHPA sheet: seller key is the Meesho SKU column → OMS."""
    import pandas as pd

    p = tmp_path / "m.xlsx"
    df = pd.DataFrame(
        {
            "DATE": ["2025-01-01"],
            "STYLE ID": [""],
            "MEESHO SKU": ["1158YKGREEN-XL"],
            "OMS SKU": ["1001YKGREEN-XL"],
            "BRAND": ["X"],
        }
    )
    df.to_excel(p, sheet_name="MEESHO PUSHPA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m["1158YKGREEN-XL"] == "1001YKGREEN-XL"


def test_replace_sku_column_registers_meesho_sku_too(tmp_path):
    """Replace SKU is primary column; Meesho SKU on same row also maps to OMS."""
    import pandas as pd

    p = tmp_path / "m.xlsx"
    df = pd.DataFrame(
        {
            "Replace SKU": ["1158YKGREEN-XL"],
            "MEESHO SKU": ["ALT-MEESHO-KEY"],
            "OMS SKU": ["1001YKGREEN-XL"],
        }
    )
    df.to_excel(p, sheet_name="MEESHO PUSHPA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m["1158YKGREEN-XL"] == "1001YKGREEN-XL"
    assert m["ALT-MEESHO-KEY"] == "1001YKGREEN-XL"


def test_replace_sku_sheet_tab_two_column_fallback(tmp_path):
    import pandas as pd

    p = tmp_path / "m.xlsx"
    df = pd.DataFrame(
        {
            "Listing Sku": ["1158YKGREEN-XL"],
            "OMS VALUE": ["1001YKGREEN-XL"],
        }
    )
    df.to_excel(p, sheet_name="Replace SKU", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m["1158YKGREEN-XL"] == "1001YKGREEN-XL"


def test_yrn_maps_when_primary_oms_empty_secondary_oms_filled(tmp_path):
    """Last OMS column may be empty; another OMS-named column has the value."""
    import pandas as pd

    p = tmp_path / "m.xlsx"
    df = pd.DataFrame(
        {
            "YRN": ["100672680"],
            "OMS SKU": [""],
            "OMS SKU CODE": ["1001YK-RED-L"],
        }
    )
    df.to_excel(p, sheet_name="MYNTRA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m.get("100672680") == "1001YK-RED-L"


def test_yrn_filled_when_oms_merged_down_in_excel(tmp_path):
    """Merged OMS: second row OMS NaN inherits first row OMS for YRN registration."""
    import pandas as pd

    p = tmp_path / "m.xlsx"
    df = pd.DataFrame(
        {
            "YRN": ["100672680", "100672681"],
            "OMS SKU": ["1001YK-A", float("nan")],
        }
    )
    df.to_excel(p, sheet_name="MYNTRA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m.get("100672680") == "1001YK-A"
    assert m.get("100672681") == "1001YK-A"


def test_myntra_row_registers_replace_yrn_and_myntra_sku_code(tmp_path):
    import pandas as pd

    p = tmp_path / "m.xlsx"
    df = pd.DataFrame(
        {
            "Replace SKU": ["88022920"],
            "YRN": ["YRN-9"],
            "MYNTRA SKU CODE": ["99001122"],
            "OMS SKU": ["1001YK-RED-L"],
        }
    )
    df.to_excel(p, sheet_name="MYNTRA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m["88022920"] == "1001YK-RED-L"
    assert m["YRN-9"] == "1001YK-RED-L"
    assert m["99001122"] == "1001YK-RED-L"


def test_meesho_sheet_space_in_sku_maps_hyphen_key(tmp_path):
    """Excel may use space before size; orders use hyphen — register both for Meesho sheets."""
    import pandas as pd

    p = tmp_path / "m.xlsx"
    df = pd.DataFrame(
        {
            "MEESHO SKU": ["1158YKGREEN XL"],
            "OMS SKU": ["1001YKGREEN-XL"],
        }
    )
    df.to_excel(p, sheet_name="MEESHO PUSHPA", index=False)
    m = parse_sku_mapping(p.read_bytes())
    assert m["1158YKGREEN XL"] == "1001YKGREEN-XL"
    assert m["1158YKGREEN-XL"] == "1001YKGREEN-XL"


def test_bundled_json_loads(monkeypatch):
    if not bundled_sku_mapping_json_path().is_file():
        pytest.skip("bundled yash_sku_mapping_master.json not present")
    monkeypatch.delenv("SKIP_BUNDLED_SKU_MAPPING", raising=False)
    clear_bundled_sku_mapping_cache()
    b = load_bundled_sku_mapping()
    assert len(b) > 50_000
    assert "1001PLYKBEIGE-3XL" in b or "1001YKBEIGE-3XL" in b
