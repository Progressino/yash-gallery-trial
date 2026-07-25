#!/usr/bin/env python3
"""Rebuild Inv History OMS columns from dated inventory RAR snapshots (prod repair).

Parses each RAR for OMS_Inventory only (does not replace live Jul-24 inventory),
merges those days as authoritative snapshots, sales-rolls the gaps, then
re-appends the current warm-cache variant as the latest snapshot.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

_CACHE = Path("/data/warm_cache")

# (snapshot date ISO, rar path on container)
_RAR_SNAPSHOTS: list[tuple[str, str]] = [
    ("2026-07-06", "/tmp/Inventory_6_Jul_26.rar"),
    ("2026-07-16", "/tmp/Inventory-16-Jul-26-1.rar"),
    ("2026-07-20", "/tmp/inv_20jul26.rar"),
    ("2026-07-21", "/tmp/inv_21jul26.rar"),
    ("2026-07-22", "/tmp/Inventory_22-Jul-26.rar"),
]


def _oms_snapshot_frame(variant: "pd.DataFrame", snap_date: str) -> "pd.DataFrame":
    import pandas as pd

    from backend.services.daily_inventory_history import _variant_snapshot_qty_series

    work = variant.copy()
    work["OMS_SKU"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    qty = _variant_snapshot_qty_series(work)
    if qty is None:
        raise RuntimeError(f"no OMS qty series for {snap_date}")
    work["Qty"] = qty.values
    work = work[work["OMS_SKU"].str.len() > 0]
    # Keep explicit zeros. An OMS RAR is a full census, so a SKU sitting at 0
    # must be recorded as 0 — dropping the row makes it look "not sampled" and
    # the matrix/Eff_Days forward-fill then carries stale stock across the day
    # (SKUs that sold out kept claiming in-stock days they never had).
    snap_ts = pd.Timestamp(snap_date).normalize()
    return pd.DataFrame(
        {
            "OMS_SKU": work["OMS_SKU"].values,
            "Date": snap_ts,
            "Qty": work["Qty"].astype(float).values,
            "Source": "snapshot",
            "Channel": "",
        }
    )


def _parse_rar(path: Path):
    from backend.services.inventory import load_inventory_consolidated
    from backend.services.sku_mapping import load_sku_mapping_from_disk

    mapping = load_sku_mapping_from_disk() or {}
    raw = path.read_bytes()
    df, dbg = load_inventory_consolidated(None, None, None, raw, mapping, return_debug=True)
    if df is None or getattr(df, "empty", True):
        raise RuntimeError(f"empty parse: {path} dbg={dbg}")
    return df


def main() -> int:
    import pandas as pd

    from backend.services.daily_inventory_history import (
        daily_inventory_history_meta_bundle,
        extend_history_with_sales,
        inventory_history_wide_matrix,
        last_authoritative_history_date,
        merge_inventory_history,
        persist_daily_inventory_history_meta,
        refresh_inventory_history_rollforward,
    )
    from backend.services.helpers import _coerce_df_for_parquet
    from backend.session import AppSession

    hist_path = _CACHE / "daily_inventory_history_df.parquet"
    if not hist_path.is_file():
        print("FAIL: missing history", flush=True)
        return 1

    bak = hist_path.with_suffix(
        hist_path.suffix + f".bak-prerar-rebuild-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    )
    bak.write_bytes(hist_path.read_bytes())
    print("backup", bak.name, flush=True)

    hist = pd.read_parquet(hist_path)
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["Date"]).copy()
    if "Channel" not in hist.columns:
        hist["Channel"] = ""
    if "Source" not in hist.columns:
        hist["Source"] = "uploaded"

    # Drop everything from first RAR snapshot day onward — rebuild from RAR + current.
    first_rar = pd.Timestamp(_RAR_SNAPSHOTS[0][0]).normalize()
    last_rar = pd.Timestamp(_RAR_SNAPSHOTS[-1][0]).normalize()
    # Preserve authoritative snapshot columns newer than the last RAR (e.g. the
    # Jul-24 OMS snapshot whose RAR is no longer on the server).
    hist_src = hist["Source"].astype(str).str.strip().str.lower()
    later_snaps = hist.loc[(hist["Date"] > last_rar) & (hist_src == "snapshot")].copy()
    if not later_snaps.empty:
        print(
            "preserving later snapshots:",
            {
                str(d.date()): int(q)
                for d, q in later_snaps.groupby("Date")["Qty"].sum().items()
            },
            flush=True,
        )
    base = hist.loc[hist["Date"] < first_rar].copy()
    print(
        "base keep through",
        (first_rar - pd.Timedelta(days=1)).date(),
        "rows",
        len(base),
        "day_total",
        int(base.loc[base["Date"] == base["Date"].max(), "Qty"].sum()) if not base.empty else 0,
        flush=True,
    )

    snap_frames: list[pd.DataFrame] = []
    for day, rar in _RAR_SNAPSHOTS:
        path = Path(rar)
        if not path.is_file():
            print("SKIP missing", rar, flush=True)
            continue
        variant = _parse_rar(path)
        part = _oms_snapshot_frame(variant, day)
        oms_sum = float(part["Qty"].sum())
        zeros = int((pd.to_numeric(part["Qty"], errors="coerce").fillna(0.0) <= 0).sum())
        print(
            f"RAR {day} OMS_skus={len(part)} (zero-stock rows kept: {zeros}) "
            f"OMS_total={int(oms_sum)}",
            flush=True,
        )
        if oms_sum < 50_000:
            print("FAIL: OMS total too low for", day, flush=True)
            return 1
        snap_frames.append(part)

    if not snap_frames:
        print("FAIL: no RAR snapshots", flush=True)
        return 1

    merged = base
    for part in snap_frames:
        day = pd.Timestamp(part["Date"].iloc[0]).normalize()
        # Drop any leftover rows on this calendar day, then insert snapshot.
        if not merged.empty:
            merged = merged.loc[merged["Date"] != day].copy()
        # Sales-roll gap from last authoritative day up to day-1.
        auth = last_authoritative_history_date(merged, before=day)
        fill_cap = day - pd.Timedelta(days=1)
        sales = None
        sales_path = _CACHE / "sales_df.parquet"
        if sales_path.is_file():
            sales = pd.read_parquet(sales_path)
        if auth is not None and fill_cap > auth:
            src = merged["Source"].astype(str).str.strip().str.lower()
            keep = (merged["Date"] <= auth) | (
                (src == "snapshot") & (merged["Date"] < day)
            )
            clip = merged.loc[keep].copy()
            clip_src = clip["Source"].astype(str).str.strip().str.lower()
            clip = clip.loc[~((clip["Date"] > auth) & (clip_src == "derived"))].copy()
            filled = extend_history_with_sales(clip, sales_df=sales, cap_date=fill_cap)
            other = merged[
                (merged["Date"] > auth)
                & (merged["Date"] < day)
                & (src == "snapshot")
            ]
            merged = merge_inventory_history(filled, other)
            print(f"  filled gap {auth.date()} → {fill_cap.date()}", flush=True)
        merged = merge_inventory_history(merged, part)

    # Attach current live variant as Jul-24 (or meta snapshot date) without
    # re-parsing an older RAR over live stock.
    if not later_snaps.empty:
        merged = merge_inventory_history(merged, later_snaps)

    rar_days_present = sorted(
        {str(pd.Timestamp(part["Date"].iloc[0]).date()) for part in snap_frames}
    )
    later_days = sorted(
        {str(pd.Timestamp(d).date()) for d in later_snaps["Date"].unique()}
    ) if not later_snaps.empty else []
    sess = AppSession()
    sess.daily_inventory_history_df = merged
    sess.daily_inventory_history_filename = "Daily Inventory History RAR-rebuild.xlsx"
    # wide_end = last RAR day so prune keeps sales-filled continuum between RARs.
    sess.daily_inventory_history_wide_end_date = rar_days_present[-1]
    sess.daily_inventory_history_snapshot_dates = rar_days_present + later_days
    sess.daily_inventory_history_uploaded_at = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    inv_meta_path = _CACHE / "inventory_session_meta.json"
    snap_live = "2026-07-24"
    if inv_meta_path.is_file():
        inv_meta = json.loads(inv_meta_path.read_text(encoding="utf-8"))
        snap_live = str(inv_meta.get("inventory_snapshot_date") or snap_live)[:10]
    sess.inventory_snapshot_date = snap_live
    if (_CACHE / "inventory_df_variant.parquet").is_file():
        sess.inventory_df_variant = pd.read_parquet(_CACHE / "inventory_df_variant.parquet")

    sales = None
    if (_CACHE / "sales_df.parquet").is_file():
        sales = pd.read_parquet(_CACHE / "sales_df.parquet")

    result = refresh_inventory_history_rollforward(
        sess, include_snapshot=True, sales_df=sales
    )
    print("REFRESH", json.dumps(result, default=str), flush=True)
    if not result.get("ok"):
        return 1

    _coerce_df_for_parquet(sess.daily_inventory_history_df).to_parquet(
        hist_path, index=False
    )
    persist_daily_inventory_history_meta(sess)
    meta = daily_inventory_history_meta_bundle(sess)
    # Ensure uploaded_at is fresh for warm-cache prefer-newest guards.
    meta["daily_inventory_history_uploaded_at"] = sess.daily_inventory_history_uploaded_at
    (_CACHE / "daily_inventory_history_meta.json").write_text(
        json.dumps(meta, default=str, indent=2), encoding="utf-8"
    )

    try:
        import backend.main as main_mod

        if not main_mod._warm_cache:
            main_mod._warm_cache = {}
        main_mod._warm_cache["daily_inventory_history_df"] = (
            sess.daily_inventory_history_df.copy()
        )
        main_mod._warm_cache[main_mod._DAILY_INV_META_WARM_KEY] = meta
    except Exception as exc:
        print("warm update skipped", exc, flush=True)

    df = sess.daily_inventory_history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    g = df.groupby("Date")["Qty"].sum().sort_index()
    print("=== day totals (Jul) ===", flush=True)
    for d, q in g.items():
        if d >= pd.Timestamp("2026-07-08"):
            print(str(d.date()), int(q), flush=True)

    wide = inventory_history_wide_matrix(df, days=30, channel="oms")
    totals = list(zip(wide.get("dates", []), wide.get("date_totals", [])))
    print("OMS last10", totals[-10:], flush=True)
    spike = any(t > 500_000 for _, t in totals)
    jul22 = dict(totals).get("2026-07-22")
    jul24 = dict(totals).get("2026-07-24")
    if spike:
        print("FAIL: spike >500k", flush=True)
        return 1
    if jul22 is not None and jul22 < 150_000:
        print("FAIL: Jul22 still undercounted", jul22, flush=True)
        return 1
    if jul24 is not None and jul22 is not None and jul24 > jul22 * 1.35:
        print(
            "WARN: Jul24 still cliffs vs Jul22",
            int(jul22),
            "→",
            int(jul24),
            flush=True,
        )
    # Spot-check MUSTRAD cliff shrinks
    for sku in ("165YK251MUSTRAD-XXL", "165YK251MUSTRAD-M"):
        sub = df[df["OMS_SKU"].astype(str).str.upper() == sku].sort_values("Date")
        print(
            sku,
            sub[sub["Date"] >= "2026-07-16"][["Date", "Qty", "Source"]]
            .tail(10)
            .to_string(index=False),
            flush=True,
        )
    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    # silence unused import lint for re in editors
    _ = re
    sys.exit(main())
