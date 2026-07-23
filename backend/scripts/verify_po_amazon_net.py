#!/usr/bin/env python3
"""Verify PO Amazon quarterly net matches sales_df; check dedup."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.services.mtr import dedup_amazon_mtr_dataframe
from backend.services.po_quarterly_fast import _accumulate_shipment_frame
from backend.services.sales import _mtr_to_sales_df
from backend.services.sku_mapping import load_sku_mapping_from_disk


def _po_q1(chunk: pd.DataFrame, sku_key: str, sku_mapping: dict | None) -> int:
    start = pd.Timestamp("2024-06-01")
    end = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label = {(2025, 4): "Jan-Mar 2025"}
    qs: dict = defaultdict(int)
    _accumulate_shipment_frame(
        chunk,
        "amazon",
        sku_mapping,
        strip_pl=True,
        canonical_oms=False,
        group_by_parent=False,
        start_ts=start,
        end_ts=end,
        cutoff_90=today - pd.Timedelta(days=90),
        cutoff_30=today - pd.Timedelta(days=30),
        q_label_map=q_label,
        quarter_sums=qs,
        units_90=defaultdict(int),
        units_30=defaultdict(int),
        days_30=defaultdict(set),
    )
    return int(qs[(sku_key, "Jan-Mar 2025")])


def main() -> None:
    mtr = pd.read_parquet("/data/warm_cache/mtr_df.parquet")
    print("MTR rows", len(mtr))
    ded = dedup_amazon_mtr_dataframe(mtr.copy())
    print("After dedup", len(ded), "removed", len(mtr) - len(ded))

    keys = ded[["Order_Id", "Invoice_Number", "SKU", "Transaction_Type", "Quantity"]].astype(str).agg("|".join, axis=1)
    dup_n = int(keys.duplicated().sum())
    print("Duplicate business keys after dedup (same order+invoice+sku+txn+qty):", dup_n)
    if dup_n:
        raise SystemExit(1)

    sku = "1379YKGREEN-XXL"
    sub = ded[ded["SKU"].astype(str).str.upper() == sku].copy()
    sub["_rep"] = pd.to_datetime(sub.get("Reporting_Date", sub.get("Date")), errors="coerce")
    mo = sub[(sub["_rep"].dt.year == 2025) & (sub["_rep"].dt.month.isin([1, 2, 3]))]

    sku_mapping = load_sku_mapping_from_disk() or {}
    po_q1 = _po_q1(mo, sku, sku_mapping)
    sales_net = int(_mtr_to_sales_df(mo, {})["Units_Effective"].sum())
    print("1379 Q1 2025 PO quarterly", po_q1, "sales net", sales_net, "match", po_q1 == sales_net)

    out = _mtr_to_sales_df(mo, {})
    out["_m"] = out["TxnDate"].dt.to_period("M").astype(str)
    for m in ["2025-01", "2025-02", "2025-03"]:
        g = out[out["_m"] == m]
        ship = int(g.loc[g["Transaction Type"] == "Shipment", "Quantity"].sum())
        ret = int(g.loc[g["Transaction Type"] == "Refund", "Quantity"].sum())
        free = int(g.loc[g["Transaction Type"] == "FreeReplacement", "Quantity"].sum())
        net = int(g["Units_Effective"].sum())
        print(m, "ship", ship, "ret", ret, "free", free, "net", net)

    ded["_rep"] = pd.to_datetime(ded.get("Reporting_Date", ded.get("Date")), errors="coerce")
    q1 = ded[(ded["_rep"].dt.year == 2025) & (ded["_rep"].dt.month.isin([1, 2, 3]))]
    vol = (
        q1[q1["Transaction_Type"] == "Shipment"]
        .groupby("SKU")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nSpot-check Q1 2025 top 10 SKUs:")
    mismatch = 0
    for s in vol.index:
        chunk = q1[q1["SKU"] == s]
        sk = str(s).strip().upper()
        pq = _po_q1(chunk, sk, sku_mapping)
        sn = int(_mtr_to_sales_df(chunk, {})["Units_Effective"].sum())
        ok = pq == sn
        if not ok:
            mismatch += 1
        print(sk, "PO", pq, "sales", sn, "OK" if ok else "MISMATCH")
    print("Mismatches", mismatch)
    if mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
