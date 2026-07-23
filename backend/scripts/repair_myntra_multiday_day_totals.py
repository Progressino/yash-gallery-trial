#!/usr/bin/env python3
"""
Repair Myntra multi-day Seller Orders blobs whose created-on buckets disagree with
ops daily File totals.

When single-day CSVs are missing, multi-day exports often pile units onto the
window end date. Re-bucket rows by OrderId order so day totals match the ops
File row (SKU timing is approximate; re-upload single-day CSVs for exactness).
"""
from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

from backend.services.daily_store import (  # noqa: E402
    _df_to_parquet,
    _extract_date_range,
    _get_conn,
    clear_tier3_range_cache,
)
from backend.services.myntra import (  # noqa: E402
    _myntra_seller_report_window,
    _myntra_seller_single_report_day,
)

# Ops "File" day totals (Myntra Daily Sales sheet).
_FILE_DAY_TOTALS = {
    "2026-07-04": 638,
    "2026-07-07": 510,
    "2026-07-10": 602,
    "2026-07-15": 385,
    "2026-07-17": 437,
    "2026-07-18": 483,
    "2026-07-19": 530,
}

# Brand-accurate split for 17–19 (YG account 36841 vs Other Brand 19498).
_FILE_DAY_BY_ACCOUNT = {
    ("2026-07-17", "yg"): 109,
    ("2026-07-18", "yg"): 146,
    ("2026-07-19", "yg"): 157,
    ("2026-07-17", "other"): 328,
    ("2026-07-18", "other"): 337,
    ("2026-07-19", "other"): 373,
}


def _account_key(filename: str) -> str | None:
    s = filename.lower()
    if "36841" in s or (" yg" in f" {s}" and "other" not in s):
        return "yg"
    if "19498" in s or "other brand" in s:
        return "other"
    return None


def _scale_targets(targets: list[int], n_rows: int) -> list[int]:
    total = sum(targets)
    if total <= 0 or n_rows <= 0:
        return [0] * len(targets)
    if total == n_rows:
        return list(targets)
    scaled = [max(0, int(round(t * n_rows / total))) for t in targets]
    drift = n_rows - sum(scaled)
    if scaled:
        idx = max(range(len(scaled)), key=lambda i: scaled[i])
        scaled[idx] = max(0, scaled[idx] + drift)
    return scaled


def _rebucket(df: pd.DataFrame, day_list: list[pd.Timestamp], targets: list[int]) -> pd.DataFrame:
    d = df.copy()
    if "OrderId" in d.columns:
        oid = pd.to_numeric(d["OrderId"], errors="coerce")
        d = d.assign(_oid=oid).sort_values(["_oid", "OMS_SKU"], kind="mergesort")
    else:
        d = d.sort_index(kind="mergesort")
    d = d.reset_index(drop=True)
    out_parts = []
    i = 0
    for day, n in zip(day_list, targets):
        chunk = d.iloc[i : i + n].copy()
        if not chunk.empty:
            chunk["Date"] = pd.Timestamp(day)
            out_parts.append(chunk)
        i += n
    if i < len(d):
        rest = d.iloc[i:].copy()
        rest["Date"] = pd.Timestamp(day_list[-1])
        out_parts.append(rest)
    out = pd.concat(out_parts, ignore_index=True) if out_parts else d
    return out.drop(columns=["_oid"], errors="ignore")


def _window_file_targets(day_list: list[pd.Timestamp], soft_total: int) -> list[int] | None:
    days = [str(d.date()) for d in day_list]
    known = {d: _FILE_DAY_TOTALS[d] for d in days if d in _FILE_DAY_TOTALS}
    if not known:
        return None
    if len(known) == len(days):
        return _scale_targets([known[d] for d in days], soft_total)
    if len(known) == len(days) - 1:
        missing = next(d for d in days if d not in known)
        # Conserve soft window total: unknown day = soft - known file days
        # (scaled later per file). Use File known + residual soft.
        residual = max(0, soft_total - sum(known.values()))
        known[missing] = residual
        return [known[d] for d in days]
    return None


def main() -> int:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, filename, data_parquet FROM daily_uploads WHERE platform='myntra'"
    ).fetchall()

    # Drop exact basename duplicates (keep highest id).
    by_base: dict[str, list[int]] = defaultdict(list)
    for row_id, filename, _blob in rows:
        base = Path(str(filename).replace("\\", "/")).name
        by_base[base].append(row_id)
    deleted = 0
    for base, ids in by_base.items():
        if len(ids) < 2:
            continue
        keep = max(ids)
        for rid in ids:
            if rid == keep:
                continue
            conn.execute("DELETE FROM daily_uploads WHERE id=?", (rid,))
            deleted += 1
            print("deleted_dup", rid, base[:70])

    junk = conn.execute(
        "SELECT id, filename FROM daily_uploads WHERE platform='myntra' "
        "AND lower(filename) LIKE '%order_info%'"
    ).fetchall()
    for rid, fn in junk:
        conn.execute("DELETE FROM daily_uploads WHERE id=?", (rid,))
        deleted += 1
        print("deleted_junk", rid, Path(fn).name[:70])

    rows = conn.execute(
        "SELECT id, filename, data_parquet FROM daily_uploads WHERE platform='myntra'"
    ).fetchall()

    # Soft totals per multi-day window (seller files only).
    window_soft: dict[tuple[str, str], int] = defaultdict(int)
    window_files: dict[tuple[str, str], list[tuple[int, str, pd.DataFrame]]] = defaultdict(list)
    for row_id, filename, blob in rows:
        if _myntra_seller_single_report_day(filename) is not None:
            continue
        win = _myntra_seller_report_window(filename)
        if win is None:
            continue
        w0, w1 = win
        key = (str(w0.date()), str(w1.date()))
        df = pd.read_parquet(io.BytesIO(blob))
        if df.empty:
            continue
        window_soft[key] += len(df)
        window_files[key].append((row_id, filename, df))

    updated = 0
    for key, files in window_files.items():
        w0, w1 = key
        day_list = list(pd.date_range(w0, w1, freq="D"))
        if len(day_list) < 2:
            continue
        soft_total = window_soft[key]
        shared_targets = _window_file_targets(day_list, soft_total)

        for row_id, filename, df in files:
            days = [str(d.date()) for d in day_list]
            acct = _account_key(filename)
            if acct and all((day, acct) in _FILE_DAY_BY_ACCOUNT for day in days):
                targets = _scale_targets(
                    [_FILE_DAY_BY_ACCOUNT[(day, acct)] for day in days], len(df)
                )
            elif shared_targets is not None:
                targets = _scale_targets(shared_targets, len(df))
            else:
                continue

            before = (
                pd.to_datetime(df["Date"], errors="coerce")
                .dt.normalize()
                .value_counts()
                .sort_index()
            )
            fixed = _rebucket(df, day_list, targets)
            after = (
                pd.to_datetime(fixed["Date"], errors="coerce")
                .dt.normalize()
                .value_counts()
                .sort_index()
            )
            if before.equals(after):
                continue
            date_from, date_to = _extract_date_range(fixed)
            conn.execute(
                "UPDATE daily_uploads SET data_parquet=?, rows=?, date_from=?, date_to=? WHERE id=?",
                (_df_to_parquet(fixed), len(fixed), date_from, date_to, row_id),
            )
            updated += 1
            print(
                "rebucket",
                Path(filename).name[:64],
                "targets",
                targets,
                "before",
                {str(k.date()): int(v) for k, v in before.items()},
                "after",
                {str(k.date()): int(v) for k, v in after.items()},
            )

    conn.commit()
    conn.close()
    clear_tier3_range_cache()
    print("done updated", updated, "deleted", deleted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
