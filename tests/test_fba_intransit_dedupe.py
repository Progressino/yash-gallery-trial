"""FBA in-transit TSV aggregation: no double-count from duplicate exports."""

import pandas as pd

from backend.services.inventory import _aggregate_fba_intransit_tsvs


def _minimal_fba_tsv(shipment_id: str, rows: list[tuple[str, str, int]]) -> bytes:
    """rows: (merchant_sku, fnsku, shipped)"""
    header_meta = (
        f"Shipment ID\t{shipment_id}\n"
        "Name\tTest\n"
        "Plan ID\t\n"
        "Ship To\tX\n"
        "Total SKUs\t0\n"
        "Total Units\t0\n"
        "\n"
        "Merchant SKU\tTitle\tASIN\tFNSKU\texternal-id\tCondition\t"
        "Who will prep?\tPrep Type\tWho will label?\tShipped\n"
    )
    body = "".join(
        f"{sku}\tT\tB0TEST\t{fn}\t--\tNew\tNo\t--\tMerchant\t{qty}\n"
        for sku, fn, qty in rows
    )
    return (header_meta + body).encode("utf-8")


def test_identical_fba_tsvs_deduped_by_hash():
    tsv = _minimal_fba_tsv("FBA15LN6M80T", [("1001YKTEST-M", "X001", 10), ("1001YKTEST-L", "X002", 5)])
    mapping = {"1001YKTEST-M": "1001YKTEST-M", "1001YKTEST-L": "1001YKTEST-L"}
    df, dbg = _aggregate_fba_intransit_tsvs([tsv, tsv], mapping)
    assert dbg["fba_tsv_identical_dropped"] == 1
    assert df["FBA_InTransit"].sum() == 15


def test_same_shipment_overlapping_lines_use_max_not_sum():
    """Two partial exports for one shipment: same MSKU×FNSKU must not be summed."""
    a = _minimal_fba_tsv("FBASHIP1", [("2002YK-A", "F1", 3)])
    b = _minimal_fba_tsv("FBASHIP1", [("2002YK-A", "F1", 3), ("2002YK-B", "F2", 7)])
    mapping = {"2002YK-A": "2002YK-A", "2002YK-B": "2002YK-B"}
    df, _dbg = _aggregate_fba_intransit_tsvs([a, b], mapping)
    totals = df.set_index("OMS_SKU")["FBA_InTransit"]
    assert int(totals["2002YK-A"]) == 3
    assert int(totals["2002YK-B"]) == 7
    assert df["FBA_InTransit"].sum() == 10


def test_different_shipments_still_summed_per_oms_sku():
    t1 = _minimal_fba_tsv("FBA_A", [("3003YK-Z", "Z1", 4)])
    t2 = _minimal_fba_tsv("FBA_B", [("3003YK-Z", "Z2", 6)])
    mapping = {"3003YK-Z": "3003YK-Z"}
    df, _dbg = _aggregate_fba_intransit_tsvs([t1, t2], mapping)
    assert len(df) == 1
    assert int(df.iloc[0]["FBA_InTransit"]) == 10
