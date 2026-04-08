"""Myntra YRN-first OMS resolution."""

from backend.services.myntra import resolve_myntra_row_keys_to_oms, resolve_myntra_row_to_oms


def test_yrn_takes_priority_over_sku():
    m = {
        "YARYKRTA1": "OMS-FROM-YRN",
        "1001PLYKBEIGE-3XL": "OMS-FROM-SKU",
    }
    assert resolve_myntra_row_to_oms(m, "YARYKRTA1", "1001PLYKBEIGE-3XL", None) == "OMS-FROM-YRN"


def test_falls_back_to_sku_when_yrn_unmapped():
    m = {"1001PLYKBEIGE-3XL": "OMS-ONLY-SKU"}
    assert resolve_myntra_row_to_oms(m, "UNKNOWN-YRN", "1001PLYKBEIGE-3XL", None) == "OMS-ONLY-SKU"


def test_style_id_used_when_others_missing():
    m = {"13001168": "OMS-STYLE"}
    assert resolve_myntra_row_to_oms(m, "", "", "13001168") == "OMS-STYLE"


def test_pipe_in_cell_tries_each_segment():
    m = {"9609": "OMS-A", "1589": "OMS-B"}
    assert resolve_myntra_row_keys_to_oms(m, ["9609|1589"]) == "OMS-A"


def test_scientific_style_id_matches_mapping_key():
    from backend.services.helpers import map_to_oms_sku

    m = {"102483512": "OMS-ZZ"}
    assert map_to_oms_sku("1.02483512E+8", m) == "OMS-ZZ"
    assert map_to_oms_sku(102483512, m) == "OMS-ZZ"
