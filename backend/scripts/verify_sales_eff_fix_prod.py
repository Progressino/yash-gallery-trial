#!/usr/bin/env python3
"""Verify quarterly combo-fan fix + OMS Eff_Days against user Excel lists."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> int:
    from backend.services.po_quarterly_fast import (
        _load_unified_sales_df,
        calculate_quarterly_from_tier3_streaming,
    )
    from backend.services.po_quarterly_warmup import quarterly_cache_key, quarterly_report_window
    from backend.services.po_quarterly_cache import (
        invalidate_shared_quarterly,
        store_shared_quarterly,
    )
    from backend.services.daily_inventory_history import effective_days_from_history
    from backend.session import AppSession
    from backend.services.sku_mapping import restore_sku_mapping_to_session
    import backend.main as m

    sales = _load_unified_sales_df()
    print("sales_cols_has_fan", "_Combo_Fan" in sales.columns, "rows", len(sales), flush=True)
    assert "_Combo_Fan" in sales.columns, "MISSING _Combo_Fan after fix"

    invalidate_shared_quarterly()
    sess = AppSession()
    restore_sku_mapping_to_session(sess)
    start, end = quarterly_report_window(8)
    print("rebuilding Sold quarterly...", start, end, flush=True)
    sold = calculate_quarterly_from_tier3_streaming(
        sess.sku_mapping or {}, start, end, n_quarters=8, demand_basis="Sold"
    )
    print("sold rows", len(sold), flush=True)
    store_shared_quarterly(
        quarterly_cache_key(False, 8, "Sold"),
        {
            "loaded": True,
            "columns": list(sold.columns),
            "rows": sold.fillna(0).to_dict(orient="records"),
        },
    )
    print("rebuilding Net quarterly...", flush=True)
    net = calculate_quarterly_from_tier3_streaming(
        sess.sku_mapping or {}, start, end, n_quarters=8, demand_basis="Net"
    )
    store_shared_quarterly(
        quarterly_cache_key(False, 8, "Net"),
        {
            "loaded": True,
            "columns": list(net.columns),
            "rows": net.fillna(0).to_dict(orient="records"),
        },
    )
    print("net rows", len(net), flush=True)

    df140 = pd.read_excel("/tmp/140_sku.xlsx")
    df140["Expected"] = df140["Unnamed: 48"]
    sold_m = sold.set_index("OMS_SKU")["Jan-Mar 2025"].to_dict()
    net_m = net.set_index("OMS_SKU")["Jan-Mar 2025"].to_dict()
    rows = []
    for _, r in df140.iterrows():
        sku = str(r["OMS_SKU"]).upper()
        rows.append(
            {
                "OMS_SKU": sku,
                "old_po": r["Jan-Mar 2025"],
                "expected": r["Expected"],
                "new_sold": sold_m.get(sku),
                "new_net": net_m.get(sku),
            }
        )
    cmp = pd.DataFrame(rows)
    cmp["sold_match"] = cmp["new_sold"] == cmp["expected"]
    cmp["sold_err"] = (cmp["new_sold"] - cmp["expected"]).abs()
    cmp["old_err"] = (cmp["old_po"] - cmp["expected"]).abs()
    print(
        "140 sold exact",
        int(cmp["sold_match"].sum()),
        "/",
        len(cmp),
        "mae",
        round(float(cmp["sold_err"].mean()), 2),
        "old_mae",
        round(float(cmp["old_err"].mean()), 2),
        flush=True,
    )
    print(
        "140 net mae",
        round(float((cmp["new_net"] - cmp["expected"]).abs().mean()), 2),
        flush=True,
    )
    print(
        "improved",
        int((cmp["sold_err"] < cmp["old_err"]).sum()),
        "worse",
        int((cmp["sold_err"] > cmp["old_err"]).sum()),
        "same",
        int((cmp["sold_err"] == cmp["old_err"]).sum()),
        flush=True,
    )
    sample = [
        "1037YKBLUE-3XL",
        "1037YKBLUE-M",
        "1592YKBLUE-XL",
        "1057YKBLUE-3XL",
        "1007YKBLACK-6XL",
    ]
    print(cmp[cmp["OMS_SKU"].isin(sample)].to_string(index=False), flush=True)

    hist = pd.read_parquet(Path(m._DISK_CACHE_DIR) / "daily_inventory_history_df.parquet")
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.normalize()
    end_d = pd.Timestamp("2026-07-29")
    start_d = end_d - pd.Timedelta(days=29)
    eff = effective_days_from_history(hist, start_d, end_d, channel="oms").set_index("OMS_SKU")[
        "Eff_Days_Inventory"
    ]
    df413 = pd.read_excel("/tmp/413_eff.xlsx")
    df413["oms_eff"] = df413["OMS_SKU"].map(eff)
    df413["match"] = df413["oms_eff"] == df413["FSAD"]
    print(
        "413 oms match FSAD",
        int(df413["match"].sum()),
        "/",
        len(df413),
        "miss",
        int((~df413["match"]).sum()),
        flush=True,
    )
    miss = df413.loc[~df413["match"], ["OMS_SKU", "Eff_Days", "FSAD", "oms_eff"]]
    if not miss.empty:
        print(miss.head(10).to_string(index=False), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
