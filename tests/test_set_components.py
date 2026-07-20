"""Set BOM / Cutting split / Finishing set-match unit helpers."""

from __future__ import annotations


def test_set_component_helpers():
    from backend.services.set_components import (
        component_sku,
        compute_complete_sets,
        normalize_component_code,
        parse_component_sku,
        style_key_for_set_bom,
    )

    assert normalize_component_code("Top") == "TOP"
    assert component_sku("1001-XS", "Pant") == "1001-XS-PANT"
    assert parse_component_sku("1001-XS-TOP") == ("1001-XS", "TOP")
    assert parse_component_sku("1001-XS") == (None, None)

    result = compute_complete_sets(
        [
            {"component_code": "TOP", "qty_per_set": 1, "available_qty": 100},
            {"component_code": "PANT", "qty_per_set": 1, "available_qty": 99},
            {"component_code": "DUPATTA", "qty_per_set": 1, "available_qty": 100},
        ]
    )
    assert result["complete_sets"] == 99
    by_code = {c["component_code"]: c for c in result["components"]}
    assert by_code["TOP"]["extra_qty"] == 1
    assert by_code["DUPATTA"]["extra_qty"] == 1
    assert by_code["PANT"]["extra_qty"] == 0
    assert style_key_for_set_bom("1001YKBEIGE-XS")


def test_resolve_cutting_main_sku_from_lines_only():
    from backend.services.component_bom import resolve_cutting_main_sku

    assert resolve_cutting_main_sku({"lines": [{"sku": "1001-XS", "planned_qty": 10}]}) == "1001-XS"
    assert resolve_cutting_main_sku({"sku": "1001-XS", "lines": []}) == "1001-XS"
    assert resolve_cutting_main_sku({"sku": "1001-XS-TOP"}) == "1001-XS"
