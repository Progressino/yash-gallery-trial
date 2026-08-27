"""Combo inventory CSV must not double-count SKUs already on the OMS sheet."""

from backend.services.inventory import load_inventory_consolidated


def _oms_csv(rows: list[tuple[str, int]]) -> bytes:
    header = "Item SkuCode,Inventory,Buffer Stock\n"
    body = "".join(f"{sku},{qty},0\n" for sku, qty in rows)
    return (header + body).encode("utf-8")


def _combo_csv(rows: list[tuple[str, int]]) -> bytes:
    header = "Combo SKU Code,Combo Qty Stock\n"
    body = "".join(f"{sku},{qty}\n" for sku, qty in rows)
    return (header + body).encode("utf-8")


def test_combo_only_skus_added_to_oms_total():
    oms = _oms_csv([("1001YKBEIGE-M", 100), ("1001YKBEIGE-L", 50)])
    combo = _combo_csv([("PACK-SET-1", 11), ("PACK-SET-2", 9)])
    df, debug = load_inventory_consolidated(
        oms_bytes=[oms, combo],
        fk_bytes=None,
        myntra_bytes=None,
        amz_bytes=None,
        mapping={},
        return_debug=True,
    )
    by_sku = df.set_index("OMS_SKU")["OMS_Inventory"].to_dict()
    assert int(by_sku["1001YKBEIGE-M"]) == 100
    assert int(by_sku["PACK-SET-1"]) == 11
    assert int(by_sku["PACK-SET-2"]) == 9
    assert int(df["OMS_Inventory"].sum()) == 170


def test_combo_overlap_with_oms_keeps_oms_qty():
    """Same SKU on OMS + Combo CSV must not sum (OMS wins)."""
    oms = _oms_csv([("1916YKRED-S-M", 70), ("OTHER-SKU", 10)])
    combo = _combo_csv([("1916YKRED-S-M", 70), ("PACK-ONLY", 5)])
    df, debug = load_inventory_consolidated(
        oms_bytes=[oms, combo],
        fk_bytes=None,
        myntra_bytes=None,
        amz_bytes=None,
        mapping={},
        return_debug=True,
    )
    by_sku = df.set_index("OMS_SKU")["OMS_Inventory"].to_dict()
    assert int(by_sku["1916YKRED-S-M"]) == 70  # not 140
    assert int(by_sku["PACK-ONLY"]) == 5
    assert int(by_sku["OTHER-SKU"]) == 10
    assert debug.get("combo_oms_overlap_skipped") == 1
