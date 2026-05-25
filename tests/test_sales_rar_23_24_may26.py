"""Regression: operator daily-sales RAR (May 23–24 2026) must parse all four marketplaces."""

from pathlib import Path

import pandas as pd
import pytest

from backend.routers.upload import _detect_platform, _extract_rar_files
from backend.services.daily_store import merge_platform_data
from backend.services.sales import (
    _filter_by_reporting_days,
    build_sales_df,
    txn_reporting_naive_ist,
)
from backend.session import AppSession


def _sample_sales_rar() -> Path | None:
    for name in (
        "Sales 23-24-5-26.rar",
        "Sales 23-24-5-26.RAR",
    ):
        p = Path("/Users/samraisinghani/Downloads") / name
        if p.is_file():
            return p
    return None


@pytest.mark.skipif(_sample_sales_rar() is None, reason="Sales 23-24-5-26.rar not on disk")
def test_sales_rar_extracts_twelve_inner_files():
    members = _extract_rar_files(_sample_sales_rar().read_bytes())
    names = {Path(n).name for n, _ in members}
    assert len(names) == 12
    assert any("Meesho" in n for n in names)
    assert any("Flipkart" in n for n in names)
    assert any("Myntra" in n for n in names)
    assert any("Amazon" in n for n in names)


@pytest.mark.skipif(_sample_sales_rar() is None, reason="Sales 23-24-5-26.rar not on disk")
def test_sales_rar_may_23_24_shipment_totals():
    """After daily-auto-style merge + build_sales_df, Intelligence window should be ~3.9k gross units."""
    sess = AppSession()
    sess.sku_mapping = {}
    for fname, raw in _extract_rar_files(_sample_sales_rar().read_bytes()):
        plat = _detect_platform(fname, raw)
        base = Path(fname).name
        if plat in ("amazon_b2c", "amazon_b2b"):
            from backend.services.mtr import parse_mtr_csv

            df, _ = parse_mtr_csv(raw, base)
            sess.mtr_df = merge_platform_data(sess.mtr_df, df, "amazon")
        elif plat == "myntra":
            from backend.services.myntra import _parse_myntra_csv

            df, _ = _parse_myntra_csv(raw, base, sess.sku_mapping)
            sess.myntra_df = merge_platform_data(sess.myntra_df, df, "myntra")
        elif plat == "meesho_csv":
            from backend.services.meesho import parse_meesho_csv

            df, _ = parse_meesho_csv(raw)
            sess.meesho_df = merge_platform_data(sess.meesho_df, df, "meesho")
        elif plat == "flipkart":
            from backend.services.flipkart import _parse_flipkart_xlsx

            df = _parse_flipkart_xlsx(raw, base, sess.sku_mapping)
            sess.flipkart_df = merge_platform_data(sess.flipkart_df, df, "flipkart")

    sales = build_sales_df(
        sess.mtr_df, sess.myntra_df, sess.meesho_df, sess.flipkart_df, sess.sku_mapping
    )
    sales = sales.assign(TxnDate=txn_reporting_naive_ist(sales["TxnDate"]))
    win = _filter_by_reporting_days(sales, "TxnDate", "2026-05-23", "2026-05-24")
    ship = win[win["Transaction Type"] == "Shipment"]

    amz = int(ship.loc[ship["Source"] == "Amazon", "Quantity"].sum())
    myn = int(ship.loc[ship["Source"] == "Myntra", "Quantity"].sum())
    mee = int(ship.loc[ship["Source"] == "Meesho", "Quantity"].sum())
    fk = int(ship.loc[ship["Source"] == "Flipkart", "Quantity"].sum())
    total = int(ship["Quantity"].sum())

    assert amz >= 1900, amz
    assert myn >= 900, myn
    assert mee >= 500, mee
    assert fk >= 350, fk
    assert total >= 3500, total
