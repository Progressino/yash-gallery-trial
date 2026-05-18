"""In-app ledger of PO quantities confirmed via PO Engine (Export & Confirm).

These rows are merged into PO math as extra pipeline so the next day's
recommendation does not repeat the same SKU/order before inventory / sheet
pipeline catches up.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .helpers import get_parent_sku
from .po_engine import canonical_oms_key


def _norm_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def append_raise_confirm_rows(
    ledger: pd.DataFrame,
    rows: List[Tuple[str, int]],
    raised_date: pd.Timestamp,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
) -> pd.DataFrame:
    """Append or accumulate (OMS_SKU, Raised_Date) rows into the ledger."""
    rd = pd.Timestamp(raised_date).normalize()
    if not rows:
        return ledger if ledger is not None else pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])

    recs = []
    for raw_sku, qty in rows:
        q = int(qty) if qty is not None else 0
        if q <= 0:
            continue
        key = canonical_oms_key(raw_sku, sku_mapping)
        if not key:
            continue
        if group_by_parent:
            key = str(get_parent_sku(key) or key).strip().upper()
        recs.append({"OMS_SKU": key, "Raised_Qty": q, "Raised_Date": rd})

    if not recs:
        return ledger if ledger is not None else pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])

    chunk = pd.DataFrame.from_records(recs)
    base = ledger if ledger is not None and not ledger.empty else pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])
    out = pd.concat([base, chunk], ignore_index=True)
    out["Raised_Date"] = _norm_date_series(out["Raised_Date"])
    out["Raised_Qty"] = pd.to_numeric(out["Raised_Qty"], errors="coerce").fillna(0).astype(int)
    out = out[out["Raised_Qty"] > 0]
    if out.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])
    out = (
        out.groupby(["OMS_SKU", "Raised_Date"], as_index=False)["Raised_Qty"]
        .sum()
        .sort_values(["Raised_Date", "OMS_SKU"])
    )
    return out.reset_index(drop=True)


def aggregate_raise_ledger_for_po(
    ledger_df: Optional[pd.DataFrame],
    sku_mapping: Optional[Dict[str, str]],
    as_of: pd.Timestamp,
    lookback_days: int = 14,
    group_by_parent: bool = False,
    raise_view_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Per-SKU aggregates for PO math / display.

    Returns columns: OMS_SKU, PO_Confirmed_Raise_Pipeline, PO_Raised_Yesterday,
    PO_Raised_Today (today = ``as_of`` calendar day, normalized),
    PO_Last_Raised_Qty, PO_Last_Raised_Date, PO_Raised_On_View_Date.
    """
    empty = pd.DataFrame(
        columns=[
            "OMS_SKU",
            "PO_Confirmed_Raise_Pipeline",
            "PO_Raised_Yesterday",
            "PO_Raised_Today",
            "PO_Last_Raised_Qty",
            "PO_Last_Raised_Date",
            "PO_Raised_On_View_Date",
        ]
    )
    if ledger_df is None or ledger_df.empty:
        return empty

    need = {"OMS_SKU", "Raised_Qty", "Raised_Date"}
    norm_cols = {str(c).strip() for c in ledger_df.columns}
    if not need.issubset(norm_cols):
        return empty

    df = ledger_df.copy()
    df["OMS_SKU"] = df["OMS_SKU"].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    df = df[df["OMS_SKU"].str.len() > 0]
    if group_by_parent:
        df["OMS_SKU"] = df["OMS_SKU"].map(lambda s: str(get_parent_sku(s) or s).strip().upper())
    df["Raised_Date"] = _norm_date_series(df["Raised_Date"])
    df["Raised_Qty"] = pd.to_numeric(df["Raised_Qty"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Raised_Date"])
    df = df[df["Raised_Qty"] > 0]
    if df.empty:
        return empty

    as_of = pd.Timestamp(as_of).normalize()
    yesterday = as_of - timedelta(days=1)
    lb = max(1, int(lookback_days))
    window_start = as_of - timedelta(days=lb - 1)

    win = df[(df["Raised_Date"] >= window_start) & (df["Raised_Date"] <= as_of)].copy()
    if win.empty:
        return empty

    win["_day"] = win["Raised_Date"].dt.normalize().dt.date
    yday = yesterday.date()
    tday = as_of.date()

    pipe = win.groupby("OMS_SKU", as_index=False)["Raised_Qty"].sum().rename(
        columns={"Raised_Qty": "PO_Confirmed_Raise_Pipeline"}
    )
    yest = (
        win[win["_day"] == yday]
        .groupby("OMS_SKU", as_index=False)["Raised_Qty"]
        .sum()
        .rename(columns={"Raised_Qty": "PO_Raised_Yesterday"})
    )
    tod = (
        win[win["_day"] == tday]
        .groupby("OMS_SKU", as_index=False)["Raised_Qty"]
        .sum()
        .rename(columns={"Raised_Qty": "PO_Raised_Today"})
    )
    out = pipe.merge(yest, on="OMS_SKU", how="left").merge(tod, on="OMS_SKU", how="left")
    out[["PO_Raised_Yesterday", "PO_Raised_Today"]] = out[
        ["PO_Raised_Yesterday", "PO_Raised_Today"]
    ].fillna(0).astype(int)

    # Most recent raise per SKU (full ledger — not limited to lookback window).
    last_recs: list[dict] = []
    for sku, sub in df.groupby("OMS_SKU"):
        dmax = sub["Raised_Date"].max()
        if pd.isna(dmax):
            continue
        qty = int(sub.loc[sub["Raised_Date"] == dmax, "Raised_Qty"].sum())
        if qty <= 0:
            continue
        last_recs.append(
            {
                "OMS_SKU": sku,
                "PO_Last_Raised_Qty": qty,
                "PO_Last_Raised_Date": str(pd.Timestamp(dmax).date()),
            }
        )
    if last_recs:
        last_df = pd.DataFrame.from_records(last_recs)
        out = out.merge(last_df, on="OMS_SKU", how="left")
    else:
        out["PO_Last_Raised_Qty"] = 0
        out["PO_Last_Raised_Date"] = ""

    # Qty raised on the UI "Raise date" picker (may be Saturday while planning day is Monday).
    out["PO_Raised_On_View_Date"] = 0
    if raise_view_date and str(raise_view_date).strip():
        try:
            vd = pd.Timestamp(pd.to_datetime(str(raise_view_date).strip()).normalize())
            on_view = (
                df[df["Raised_Date"] == vd]
                .groupby("OMS_SKU", as_index=False)["Raised_Qty"]
                .sum()
                .rename(columns={"Raised_Qty": "PO_Raised_On_View_Date"})
            )
            if not on_view.empty:
                out = out.drop(columns=["PO_Raised_On_View_Date"], errors="ignore").merge(
                    on_view, on="OMS_SKU", how="left"
                )
        except Exception:
            pass
    out["PO_Last_Raised_Qty"] = pd.to_numeric(out.get("PO_Last_Raised_Qty"), errors="coerce").fillna(0).astype(int)
    out["PO_Last_Raised_Date"] = out.get("PO_Last_Raised_Date", "").fillna("").astype(str)
    out["PO_Raised_On_View_Date"] = pd.to_numeric(out["PO_Raised_On_View_Date"], errors="coerce").fillna(0).astype(int)
    return out


def summarize_raise_ledger_for_dashboard(
    ledger_df: Optional[pd.DataFrame],
    *,
    lookback_days: int = 30,
    planning_date: Optional[str] = None,
    max_skus_per_day: int = 500,
) -> dict:
    """
    Daily totals + per-day SKU lines for the PO Dashboard raise-history panel.
    """
    empty = {
        "ledger_loaded": False,
        "daily_totals": [],
        "by_day": {},
        "active_by_sku": [],
        "total_skus": 0,
        "total_units": 0,
        "lookback_days": int(lookback_days),
        "planning_date": None,
    }
    if ledger_df is None or ledger_df.empty:
        return empty

    need = {"OMS_SKU", "Raised_Qty", "Raised_Date"}
    if not need.issubset({str(c).strip() for c in ledger_df.columns}):
        return empty

    df = ledger_df.copy()
    df["OMS_SKU"] = df["OMS_SKU"].astype(str).str.strip()
    df["Raised_Date"] = _norm_date_series(df["Raised_Date"])
    df["Raised_Qty"] = pd.to_numeric(df["Raised_Qty"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Raised_Date"])
    df = df[(df["OMS_SKU"].str.len() > 0) & (df["Raised_Qty"] > 0)]
    if df.empty:
        return empty

    try:
        as_of = (
            pd.Timestamp(pd.to_datetime(planning_date).normalize())
            if planning_date and str(planning_date).strip()
            else pd.Timestamp.now().normalize()
        )
    except Exception:
        as_of = pd.Timestamp.now().normalize()

    lb = max(1, int(lookback_days))
    window_start = as_of - timedelta(days=lb - 1)
    win = df[(df["Raised_Date"] >= window_start) & (df["Raised_Date"] <= as_of)].copy()
    if win.empty:
        out = dict(empty)
        out["ledger_loaded"] = True
        out["planning_date"] = str(as_of.date())
        return out

    cap = int(max(10, min(2000, max_skus_per_day)))

    daily_grp = (
        win.groupby("Raised_Date", as_index=False)
        .agg(sku_count=("OMS_SKU", "nunique"), total_units=("Raised_Qty", "sum"))
        .sort_values("Raised_Date", ascending=False)
    )
    daily_totals = [
        {
            "raised_date": str(pd.Timestamp(r["Raised_Date"]).date()),
            "sku_count": int(r["sku_count"]),
            "total_units": int(r["total_units"]),
        }
        for _, r in daily_grp.iterrows()
    ]

    by_day: dict[str, list] = {}
    for day_val, sub in win.groupby("Raised_Date"):
        day_str = str(pd.Timestamp(day_val).date())
        sub = sub.sort_values(["Raised_Qty", "OMS_SKU"], ascending=[False, True]).head(cap)
        by_day[day_str] = [
            {"oms_sku": str(r["OMS_SKU"]), "raised_qty": int(r["Raised_Qty"])}
            for _, r in sub.iterrows()
        ]

    active = (
        win.groupby("OMS_SKU", as_index=False)
        .agg(raised_qty=("Raised_Qty", "sum"), last_raised_date=("Raised_Date", "max"))
        .sort_values("raised_qty", ascending=False)
        .head(cap)
    )
    active_by_sku = [
        {
            "oms_sku": str(r["OMS_SKU"]),
            "qty": int(r["raised_qty"]),
            "last_raised_date": str(pd.Timestamp(r["last_raised_date"]).date()),
        }
        for _, r in active.iterrows()
    ]

    return {
        "ledger_loaded": True,
        "daily_totals": daily_totals,
        "by_day": by_day,
        "active_by_sku": active_by_sku,
        "total_skus": int(win["OMS_SKU"].nunique()),
        "total_units": int(win["Raised_Qty"].sum()),
        "lookback_days": lb,
        "planning_date": str(as_of.date()),
    }
