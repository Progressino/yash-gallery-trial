"""OMS Amazon Other Warehouse must not replace FC ledger totals."""

from backend.services.inventory import load_inventory_consolidated


def _oms_with_amazon_col() -> bytes:
    return (
        b"Item SkuCode,Inventory,Buffer Stock,Amazon Other Warehouse\n"
        b"SKU-A,100,10,5\n"
        b"SKU-B,50,0,0\n"
    )


def _amz_ledger() -> bytes:
    return (
        b"Date,MSKU,Disposition,Location,Ending Warehouse Balance\n"
        b"2026-07-15,SKU-A,SELLABLE,BLR7,20\n"
        b"2026-07-15,SKU-B,SELLABLE,BLR7,8\n"
    )


def test_oms_amazon_column_plus_ledger_takes_per_sku_max():
    mapping = {"SKU-A": "SKU-A", "SKU-B": "SKU-B"}
    df, dbg = load_inventory_consolidated(
        _oms_with_amazon_col(),
        None,
        None,
        [_amz_ledger()],
        mapping,
        return_debug=True,
    )
    by_sku = df.set_index("OMS_SKU")["Amazon_Inventory"].to_dict()
    assert int(by_sku["SKU-A"]) == 20
    assert int(by_sku["SKU-B"]) == 8
    assert int(df["Amazon_Inventory"].sum()) == 28
    assert "Amazon_Inventory" not in (dbg.get("oms_provides_marketplace") or [])
