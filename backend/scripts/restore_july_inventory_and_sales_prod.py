#!/usr/bin/env python3
"""Restore Jul inventory history (auth days) + rebuild sales_df for missing FY quarters.

Prod Jul 30 snapshot append left only Jul 1 + Jul 30 as uploaded; mid-Jul became
sales-derived (CARRIED). Rebuild from bak-yr-scrub + OMS matrix CSV, keep Jul 30
snapshot, then rebuild unified sales from platform frames (Amazon MTR has Oct-Dec 2024).
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")
_CACHE = Path("/data/warm_cache")
_BAK = _CACHE / "daily_inventory_history_df.parquet.bak-yr-scrub-20260728190127"
_MATRIX = Path("/tmp/inventory-matrix-oms-2026-07-30.csv")
_HIST = _CACHE / "daily_inventory_history_df.parquet"


def main() -> int:
    import pandas as pd

    from backend.services.daily_inventory_history import (
        clear_inventory_channel_view_cache,
        daily_inventory_history_meta_bundle,
        filter_inventory_history_channel,
        inventory_history_wide_matrix,
        merge_inventory_history,
        merge_inventory_history_preserving_channels,
        non_uploaded_inventory_dates,
        parse_daily_inventory_history_upload,
        persist_daily_inventory_history_meta,
        persist_inventory_history_authoritative,
        snapshot_dates_from_history,
    )
    from backend.services.helpers import _coerce_df_for_parquet
    from backend.services.po_quarterly_cache import invalidate_shared_quarterly
    from backend.services.po_shared_cache import invalidate_all_shared_caches
    from backend.services.sales import build_sales_df
    from backend.services.sku_mapping import load_sku_mapping_from_disk
    from backend.session import AppSession
    import backend.main as _main

    if not _BAK.is_file():
        print("FAIL missing bak", _BAK, flush=True)
        return 1
    if not _MATRIX.is_file():
        print("FAIL missing matrix", _MATRIX, flush=True)
        return 1

    stamp = datetime.now(_IST).strftime("%Y%m%d%H%M%S")
    bak_out = _HIST.with_name(f"{_HIST.name}.bak-before-july-restore-{stamp}")
    if _HIST.is_file():
        shutil.copy2(_HIST, bak_out)
        print("backup", bak_out, flush=True)

    current = pd.read_parquet(_HIST) if _HIST.is_file() else pd.DataFrame()
    base = pd.read_parquet(_BAK)
    print("base", len(base), "current", len(current), flush=True)

    # Prefer bak auth days; overlay current Jul 30 snapshot rows (all channels).
    merged = merge_inventory_history(base, current)
    if not current.empty:
        cur = current.copy()
        cur["Date"] = pd.to_datetime(cur["Date"], errors="coerce").dt.normalize()
        jul30 = cur[cur["Date"] == pd.Timestamp("2026-07-30")]
        if not jul30.empty:
            merged = merge_inventory_history(merged, jul30)

    # Re-apply OMS matrix CSV as authoritative snapshot for Jul 1–30.
    raw = _MATRIX.read_bytes()
    mapping = load_sku_mapping_from_disk()
    import io as _io

    matrix_tall = parse_daily_inventory_history_upload(
        _io.BytesIO(raw), "inventory-matrix-oms-2026-07-30.csv", mapping
    )
    if matrix_tall is None or matrix_tall.empty:
        print("FAIL matrix parse empty", flush=True)
        return 2
    matrix_tall = matrix_tall.copy()
    matrix_tall["Source"] = "snapshot"
    matrix_tall["Channel"] = "oms"
    print("matrix_tall", len(matrix_tall), flush=True)
    merged = merge_inventory_history_preserving_channels(merged, matrix_tall)

    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.normalize()
    # Keep last 45 days ending Jul 30 (covers full Jul window).
    end = pd.Timestamp("2026-07-30")
    start = end - pd.Timedelta(days=44)
    merged = merged[(merged["Date"] >= start) & (merged["Date"] <= end)].copy()

    sess = AppSession()
    sess.daily_inventory_history_df = merged
    sess.daily_inventory_history_filename = "inventory-matrix-oms-2026-07-30.csv"
    sess.daily_inventory_history_wide_end_date = "2026-07-30"
    sess.daily_inventory_history_matrix_max_date = "2026-07-30"
    sess.daily_inventory_history_uploaded_at = datetime.now(_IST).isoformat()
    sess.daily_inventory_history_snapshot_dates = snapshot_dates_from_history(merged)
    sess.inventory_snapshot_date = "2026-07-30"

    persist_inventory_history_authoritative(sess)
    _coerce_df_for_parquet(merged).to_parquet(_HIST, index=False)
    meta = daily_inventory_history_meta_bundle(sess)
    (_CACHE / "daily_inventory_history_meta.json").write_text(
        json.dumps(meta, default=str, indent=2), encoding="utf-8"
    )
    persist_daily_inventory_history_meta(sess)
    clear_inventory_channel_view_cache()

    # Verify auth days
    df = pd.read_parquet(_HIST)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    jul = df[(df["Date"] >= "2026-07-01") & (df["Date"] <= "2026-07-30")]
    auth = jul[jul["Source"].astype(str).str.lower().isin(["snapshot", "uploaded"])]
    auth_days = sorted(auth["Date"].dt.strftime("%Y-%m-%d").unique())
    dates = [pd.Timestamp(f"2026-07-{d:02d}") for d in range(1, 31)]
    gaps = non_uploaded_inventory_dates(df, dates)
    print("auth_days", len(auth_days), auth_days[:5], "...", auth_days[-5:], flush=True)
    print("gap_dates", gaps, flush=True)
    wide = inventory_history_wide_matrix(df, days=30, end_date="2026-07-30", channel="oms")
    print(
        "oms_totals_tail",
        list(zip((wide.get("dates") or [])[-5:], (wide.get("date_totals") or [])[-5:])),
        flush=True,
    )
    if len(auth_days) < 20:
        print("FAIL too few auth days", flush=True)
        return 3

    # Rebuild sales_df so Oct-Dec 2024 (Amazon MTR) lands in quarterly columns.
    def _load(name: str) -> pd.DataFrame:
        p = _CACHE / f"{name}.parquet"
        return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

    mtr = _load("mtr_df")
    meesho = _load("meesho_df")
    myntra = _load("myntra_df")
    flipkart = _load("flipkart_df")
    snapdeal = _load("snapdeal_df")
    print(
        "platform rows mtr/meesho/myntra/fk/sd",
        len(mtr),
        len(meesho),
        len(myntra),
        len(flipkart),
        len(snapdeal),
        flush=True,
    )
    sales = build_sales_df(
        mtr_df=mtr,
        meesho_df=meesho,
        myntra_df=myntra,
        flipkart_df=flipkart,
        snapdeal_df=snapdeal,
        sku_mapping=mapping,
    )
    sales_path = _CACHE / "sales_df.parquet"
    sales_bak = sales_path.with_name(f"sales_df.parquet.bak-before-july-restore-{stamp}")
    if sales_path.is_file():
        shutil.copy2(sales_path, sales_bak)
    sales.to_parquet(sales_path, index=False)
    print("sales_rows", len(sales), flush=True)
    sdt = pd.to_datetime(sales["TxnDate"], errors="coerce")
    octdec = int(((sdt >= "2024-10-01") & (sdt <= "2024-12-31")).sum())
    print("oct_dec_2024_rows", octdec, flush=True)

    # Refresh warm cache pointers.
    if not _main._warm_cache:
        _main._warm_cache = {}
    _main._warm_cache["daily_inventory_history_df"] = df.copy()
    _main._warm_cache["sales_df"] = sales
    _main._warm_cache[_main._DAILY_INV_META_WARM_KEY] = meta
    invalidate_all_shared_caches()
    invalidate_shared_quarterly()
    try:
        from backend.services.po_quarterly_warmup import schedule_shared_quarterly_prewarm

        schedule_shared_quarterly_prewarm()
    except Exception:
        pass

    if octdec < 1000:
        print("WARN oct-dec still thin", flush=True)
    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
