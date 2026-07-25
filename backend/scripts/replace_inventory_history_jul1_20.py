#!/usr/bin/env python3
"""Replace Jul 1–20 2026 inventory history from daily RAR snapshots (OMS + Amazon).

Deletes existing rows in [2026-07-01, 2026-07-20], inserts channel-split snapshots
from each RAR (keeping explicit Qty=0), sales-rolls OMS gaps and carries Amazon
for missing census days (5, 7, 12), then persists warm-cache history + meta and
invalidates shared PO caches.

Usage (inside progressino-backend container):
  python -m backend.scripts.replace_inventory_history_jul1_20 \\
      /tmp/inv_jul1_20/Inventory\\ data\\ 1\\ to\\ 20\\ Jul
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_CACHE = Path("/data/warm_cache")
_WINDOW_START = pd.Timestamp("2026-07-01")
_WINDOW_END = pd.Timestamp("2026-07-20")

# Filenames inside the extracted outer RAR → snapshot date
_RAR_DATE_RE = re.compile(
    r"Inventory[\s_-]*(\d{1,2})[-_]?(?:July?|Jul|7)[-_]?(\d{2,4})",
    re.IGNORECASE,
)


def _day_from_name(name: str) -> str | None:
    m = _RAR_DATE_RE.search(name)
    if not m:
        return None
    day = int(m.group(1))
    year_raw = m.group(2)
    year = int(year_raw) if len(year_raw) == 4 else 2000 + int(year_raw)
    if year < 2000:
        year += 2000
    return f"{year:04d}-07-{day:02d}"


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
    if col not in work.columns:
        raise RuntimeError(f"{day}: missing column {col} in {list(work.columns)}")
    return pd.DataFrame(
        {
            "OMS_SKU": work["OMS_SKU"].values,
            "Date": pd.Timestamp(day),
            "Qty": pd.to_numeric(work[col], errors="coerce").fillna(0.0).values,
            "Source": "snapshot",
            "Channel": channel,
        }
    )


def _derive_oms(anchor: pd.DataFrame, sales: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    from backend.services.daily_inventory_history import extend_history_with_sales

    if anchor.empty or pd.Timestamp(anchor["Date"].iloc[0]) >= end:
        return anchor.iloc[0:0].copy()
    derived = extend_history_with_sales(anchor, sales_df=sales, cap_date=end)
    derived = derived[pd.to_datetime(derived["Date"]) > pd.Timestamp(anchor["Date"].iloc[0])].copy()
    derived["Channel"] = "oms"
    return derived


def _carry_amazon(anchor: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(anchor["Date"].iloc[0]) + pd.Timedelta(days=1)
    if start > end or anchor.empty:
        return anchor.iloc[0:0].copy()
    parts: list[pd.DataFrame] = []
    for day in pd.date_range(start, end, freq="D"):
        part = anchor.copy()
        part["Date"] = day
        part["Source"] = "derived"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else anchor.iloc[0:0].copy()


def _discover_rars(folder: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for path in sorted(folder.rglob("*.rar")):
        day = _day_from_name(path.name)
        if not day:
            print("skip_unparsed_name", path.name, flush=True)
            continue
        ts = pd.Timestamp(day)
        if ts < _WINDOW_START or ts > _WINDOW_END:
            print("skip_out_of_window", day, path.name, flush=True)
            continue
        found[day] = path
    return [(d, found[d]) for d in sorted(found)]


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: replace_inventory_history_jul1_20.py /path/to/extracted/daily/rars",
            file=sys.stderr,
        )
        return 2

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"FAIL: not a directory: {folder}", file=sys.stderr)
        return 1

    from backend.services.daily_inventory_history import (
        _coalesce_history_rows,
        daily_inventory_history_meta_bundle,
        filter_inventory_history_channel,
        inventory_history_wide_matrix,
    )
    from backend.services.helpers import _coerce_df_for_parquet
    from backend.services.po_shared_cache import invalidate_all_shared_caches
    from backend.session import AppSession

    hist_path = _CACHE / "daily_inventory_history_df.parquet"
    if not hist_path.is_file():
        print("FAIL: missing history", hist_path, flush=True)
        return 1

    rars = _discover_rars(folder)
    if not rars:
        print("FAIL: no dated RARs found in", folder, flush=True)
        return 1
    print("rars", len(rars), [d for d, _ in rars], flush=True)

    backup = hist_path.with_suffix(
        hist_path.suffix + f".bak-before-jul1-20-replace-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    )
    backup.write_bytes(hist_path.read_bytes())
    print("backup", backup.name, flush=True)

    current = pd.read_parquet(hist_path)
    current["Date"] = pd.to_datetime(current["Date"], errors="coerce").dt.normalize()
    current = current.dropna(subset=["Date"]).copy()
    if "Channel" not in current.columns:
        current["Channel"] = ""
    if "Source" not in current.columns:
        current["Source"] = "uploaded"

    before = current[current["Date"] < _WINDOW_START].copy()
    after = current[current["Date"] > _WINDOW_END].copy()
    deleted = int(((current["Date"] >= _WINDOW_START) & (current["Date"] <= _WINDOW_END)).sum())
    print(
        "keep_before",
        len(before),
        "delete_window_rows",
        deleted,
        "keep_after",
        len(after),
        flush=True,
    )

    sales_path = _CACHE / "sales_df.parquet"
    sales = pd.read_parquet(sales_path) if sales_path.is_file() else pd.DataFrame()

    parsed: list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []
    for day, path in rars:
        variant = _parse_rar(path)
        oms = _snapshot(variant, day, "oms", "OMS_Inventory")
        amazon = _snapshot(variant, day, "amazon", "Amazon_Inventory")
        combined = (
            pd.concat(
                [oms.set_index("OMS_SKU")["Qty"], amazon.set_index("OMS_SKU")["Qty"]],
                axis=1,
            )
            .max(axis=1)
            .sum()
        )
        print(
            day,
            path.name,
            "OMS",
            int(oms["Qty"].sum()),
            "Amazon",
            int(amazon["Qty"].sum()),
            "Combined",
            int(combined),
            "zeros_oms",
            int((oms["Qty"] <= 0).sum()),
            flush=True,
        )
        parsed.append((pd.Timestamp(day), oms, amazon))

    parts: list[pd.DataFrame] = [before]
    for index, (day, oms, amazon) in enumerate(parsed):
        parts.extend([oms, amazon])
        next_day = (
            parsed[index + 1][0]
            if index + 1 < len(parsed)
            else (_WINDOW_END + pd.Timedelta(days=1))
        )
        gap_end = next_day - pd.Timedelta(days=1)
        if gap_end > day and gap_end <= _WINDOW_END:
            parts.append(_derive_oms(oms, sales, gap_end))
            parts.append(_carry_amazon(amazon, gap_end))

    parts.append(after)
    rebuilt = _coalesce_history_rows(
        pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True),
        drop_zero_derived=False,
    )

    uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    live_max = pd.Timestamp(rebuilt["Date"].max())
    sess = AppSession()
    sess.daily_inventory_history_df = rebuilt
    sess.daily_inventory_history_filename = "Daily Inventory History Jul 1-20 replace"
    sess.daily_inventory_history_wide_end_date = str(live_max.date())
    sess.daily_inventory_history_snapshot_dates = [str(day.date()) for day, _, _ in parsed]
    sess.daily_inventory_history_matrix_max_date = str(live_max.date())
    sess.daily_inventory_history_uploaded_at = uploaded_at

    _coerce_df_for_parquet(rebuilt).to_parquet(hist_path, index=False)
    meta = daily_inventory_history_meta_bundle(sess)
    meta["daily_inventory_history_uploaded_at"] = uploaded_at
    (_CACHE / "daily_inventory_history_meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    good = hist_path.with_suffix(
        hist_path.suffix + f".bak-good-jul1-20-replace-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    )
    good.write_bytes(hist_path.read_bytes())
    print("good_backup", good.name, flush=True)

    # Verify window day totals
    for day in pd.date_range(_WINDOW_START, _WINDOW_END, freq="D"):
        day_s = str(day.date())
        for channel in ("combined", "oms", "amazon"):
            ch = filter_inventory_history_channel(rebuilt, channel)
            sub = ch[pd.to_datetime(ch["Date"]).dt.normalize() == day]
            print(
                "verify",
                day_s,
                channel,
                "skus",
                int(sub["OMS_SKU"].nunique()) if not sub.empty else 0,
                "qty",
                int(pd.to_numeric(sub["Qty"], errors="coerce").fillna(0).sum()) if not sub.empty else 0,
                flush=True,
            )

    for channel in ("combined", "oms", "amazon"):
        wide = inventory_history_wide_matrix(
            rebuilt, days=20, end_date="2026-07-20", channel=channel
        )
        totals = wide.get("date_totals") or []
        dates = wide.get("dates") or []
        print(
            "matrix20",
            channel,
            "dates",
            len(dates),
            "last_total",
            int(totals[-1]) if totals else 0,
            flush=True,
        )

    try:
        invalidate_all_shared_caches()
        print("shared_po_cache_invalidated", flush=True)
    except Exception as exc:
        print("shared_cache_invalidate_warn", exc, flush=True)

    # Hot-reload sidecar into running backend process if available
    try:
        import backend.main as _main

        _main.bootstrap_warm_cache_if_empty()
        # Prefer explicit disk reload of inventory history sidecar
        if hasattr(_main, "sync_daily_inventory_history_sidecar"):
            _main.sync_daily_inventory_history_sidecar(sess)
            print("sidecar_synced", flush=True)
    except Exception as exc:
        print("sidecar_sync_warn", exc, flush=True)

    print(
        "DONE",
        "rows",
        len(rebuilt),
        "min",
        str(rebuilt["Date"].min().date()),
        "max",
        str(rebuilt["Date"].max().date()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
