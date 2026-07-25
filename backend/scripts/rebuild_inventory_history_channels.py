#!/usr/bin/env python3
"""Rebuild prod Inv History with explicit OMS and Amazon channels."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_CACHE = Path("/data/warm_cache")
_ANCHORS = [
    ("2026-07-06", Path("/tmp/Inventory_6_Jul_26.rar")),
    ("2026-07-16", Path("/tmp/Inventory-16-Jul-26-1.rar")),
    ("2026-07-20", Path("/tmp/inv_20jul26.rar")),
    ("2026-07-21", Path("/tmp/inv_21jul26.rar")),
    ("2026-07-22", Path("/tmp/Inventory_22-Jul-26.rar")),
]


def _parse_rar(path: Path) -> pd.DataFrame:
    from backend.services.inventory import load_inventory_consolidated
    from backend.services.sku_mapping import load_sku_mapping_from_disk

    df, debug = load_inventory_consolidated(
        None,
        None,
        None,
        path.read_bytes(),
        load_sku_mapping_from_disk() or {},
        return_debug=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"empty RAR parse: {path}: {debug}")
    return df


def _snapshot(variant: pd.DataFrame, day: str, channel: str, col: str) -> pd.DataFrame:
    work = variant.copy()
    work["OMS_SKU"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    work = work[work["OMS_SKU"].str.len() > 0]
    return pd.DataFrame({
        "OMS_SKU": work["OMS_SKU"].values,
        "Date": pd.Timestamp(day),
        "Qty": pd.to_numeric(work[col], errors="coerce").fillna(0.0).values,
        "Source": "snapshot",
        "Channel": channel,
    })


def _derive_oms(anchor: pd.DataFrame, sales: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    from backend.services.daily_inventory_history import extend_history_with_sales

    if pd.Timestamp(anchor["Date"].iloc[0]) >= end:
        return anchor.iloc[0:0].copy()
    derived = extend_history_with_sales(anchor, sales_df=sales, cap_date=end)
    derived = derived[pd.to_datetime(derived["Date"]) > pd.Timestamp(anchor["Date"].iloc[0])].copy()
    derived["Channel"] = "oms"
    return derived


def _carry_amazon(anchor: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(anchor["Date"].iloc[0]) + pd.Timedelta(days=1)
    if start > end:
        return anchor.iloc[0:0].copy()
    parts: list[pd.DataFrame] = []
    for day in pd.date_range(start, end, freq="D"):
        part = anchor.copy()
        part["Date"] = day
        part["Source"] = "derived"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else anchor.iloc[0:0].copy()


def main() -> int:
    from backend.services.daily_inventory_history import (
        _coalesce_history_rows,
        daily_inventory_history_meta_bundle,
        inventory_history_wide_matrix,
    )
    from backend.services.helpers import _coerce_df_for_parquet
    from backend.session import AppSession

    hist_path = _CACHE / "daily_inventory_history_df.parquet"
    if not hist_path.is_file():
        raise FileNotFoundError(hist_path)
    # Prefer a known-good OMS blank matrix as the pre-anchor base when the live
    # file has already been collapsed by spike-repair (~130k blank channels).
    seed = hist_path
    candidates = sorted(
        _CACHE.glob("daily_inventory_history_df.parquet.bak-prerar-rebuild-*"),
        reverse=True,
    )
    for cand in candidates:
        try:
            probe = pd.read_parquet(cand)
            probe["Date"] = pd.to_datetime(probe["Date"], errors="coerce").dt.normalize()
            last = probe[probe["Date"] == probe["Date"].max()]
            if float(last["Qty"].sum()) >= 170_000:
                seed = cand
                print("seed_base", cand.name, "last_qty", int(last["Qty"].sum()), flush=True)
                break
        except Exception as exc:
            print("seed_skip", cand.name, exc, flush=True)
    current = pd.read_parquet(seed)
    current["Date"] = pd.to_datetime(current["Date"], errors="coerce").dt.normalize()
    current = current.dropna(subset=["Date"]).copy()
    if "Channel" not in current:
        current["Channel"] = ""
    if "Source" not in current:
        current["Source"] = "uploaded"

    backup = hist_path.with_suffix(
        hist_path.suffix + f".bak-before-channel-split-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    )
    backup.write_bytes(hist_path.read_bytes())
    print("backup", backup.name, flush=True)

    sales_path = _CACHE / "sales_df.parquet"
    sales = pd.read_parquet(sales_path) if sales_path.is_file() else pd.DataFrame()
    parsed: list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []
    for day, path in _ANCHORS:
        if not path.is_file():
            raise FileNotFoundError(path)
        variant = _parse_rar(path)
        oms = _snapshot(variant, day, "oms", "OMS_Inventory")
        amazon = _snapshot(variant, day, "amazon", "Amazon_Inventory")
        parsed.append((pd.Timestamp(day), oms, amazon))
        print(
            day,
            "OMS", int(oms["Qty"].sum()),
            "Amazon", int(amazon["Qty"].sum()),
            "Combined", int(
                pd.concat(
                    [
                        oms.set_index("OMS_SKU")["Qty"],
                        amazon.set_index("OMS_SKU")["Qty"],
                    ],
                    axis=1,
                ).max(axis=1).sum()
            ),
            flush=True,
        )

    # Current live census is authoritative for its metadata date.
    inv_meta = json.loads((_CACHE / "inventory_session_meta.json").read_text())
    live_day = pd.Timestamp(str(inv_meta["inventory_snapshot_date"])[:10])
    live_variant = pd.read_parquet(_CACHE / "inventory_df_variant.parquet")
    parsed.append((
        live_day,
        _snapshot(live_variant, str(live_day.date()), "oms", "OMS_Inventory"),
        _snapshot(live_variant, str(live_day.date()), "amazon", "Amazon_Inventory"),
    ))
    # A duplicate anchor can occur if today's RAR is also listed; newest/live wins.
    by_day = {day: (oms, amazon) for day, oms, amazon in parsed}
    parsed = [(day, *by_day[day]) for day in sorted(by_day)]

    first_day = parsed[0][0]
    base = current[current["Date"] < first_day].copy()
    parts: list[pd.DataFrame] = [base]
    for index, (day, oms, amazon) in enumerate(parsed):
        parts.extend([oms, amazon])
        if index + 1 >= len(parsed):
            continue
        gap_end = parsed[index + 1][0] - pd.Timedelta(days=1)
        if gap_end > day:
            parts.append(_derive_oms(oms, sales, gap_end))
            parts.append(_carry_amazon(amazon, gap_end))

    rebuilt = _coalesce_history_rows(
        pd.concat(parts, ignore_index=True),
        drop_zero_derived=False,
    )
    uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sess = AppSession()
    sess.daily_inventory_history_df = rebuilt
    sess.daily_inventory_history_filename = "Daily Inventory History channel rebuild"
    sess.daily_inventory_history_wide_end_date = str(live_day.date())
    sess.daily_inventory_history_snapshot_dates = [str(day.date()) for day, _, _ in parsed]
    sess.daily_inventory_history_matrix_max_date = str(live_day.date())
    sess.daily_inventory_history_uploaded_at = uploaded_at

    _coerce_df_for_parquet(rebuilt).to_parquet(hist_path, index=False)
    meta = daily_inventory_history_meta_bundle(sess)
    meta["daily_inventory_history_uploaded_at"] = uploaded_at
    (_CACHE / "daily_inventory_history_meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    good = hist_path.with_suffix(
        hist_path.suffix + f".bak-good-channel-split-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    )
    good.write_bytes(hist_path.read_bytes())
    print("good_backup", good.name, flush=True)

    for channel in ("combined", "oms", "amazon"):
        wide = inventory_history_wide_matrix(
            rebuilt, days=30, end_date=str(live_day.date()), channel=channel
        )
        tail = [
            (d, int(q))
            for d, q in zip(wide["dates"][-6:], wide["date_totals"][-6:])
        ]
        print(channel, "split", wide["channel_split_available"], tail, flush=True)
        if channel != "amazon" and not any(q > 100_000 for _, q in tail):
            raise RuntimeError(f"{channel} total unexpectedly low: {tail}")
        if channel == "amazon" and not any(q > 20_000 for _, q in tail):
            raise RuntimeError(f"Amazon total unexpectedly low: {tail}")
    print("RESULT: PASS rows", len(rebuilt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
