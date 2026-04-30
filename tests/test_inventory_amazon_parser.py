"""Regression tests for Amazon inventory parsing behavior."""

from backend.services.inventory import _parse_amz_csv


def _amz_csv(rows: list[tuple[str, str, str, int]]) -> bytes:
    """
    rows = (date, msku, location, ending_balance)
    Disposition is fixed to SELLABLE for this helper.
    """
    header = "Date,MSKU,Disposition,Location,Ending Warehouse Balance\n"
    body = "".join(
        f"{date},{msku},SELLABLE,{location},{bal}\n"
        for date, msku, location, bal in rows
    )
    return (header + body).encode("utf-8")


def test_parse_amz_csv_uses_latest_report_day_only():
    csv_bytes = _amz_csv(
        [
            ("2026-04-26", "1001YKBEIGE-M", "BLR7", 10),
            ("2026-04-27", "1001YKBEIGE-M", "BLR7", 15),
            ("2026-04-27", "1001YKBEIGE-L", "MAA4", 8),
        ]
    )
    out = _parse_amz_csv(csv_bytes, mapping={})
    totals = out.set_index("OMS_SKU")["Amazon_Inventory"].to_dict()
    assert int(totals["1001YKBEIGE-M"]) == 15
    assert int(totals["1001YKBEIGE-L"]) == 8
    assert int(out["Amazon_Inventory"].sum()) == 23


def test_parse_amz_csv_excludes_znne_location():
    csv_bytes = _amz_csv(
        [
            ("2026-04-27", "1001YKBEIGE-M", "ZNNE", 50),
            ("2026-04-27", "1001YKBEIGE-M", "BLR7", 7),
        ]
    )
    out = _parse_amz_csv(csv_bytes, mapping={})
    assert int(out["Amazon_Inventory"].sum()) == 7

