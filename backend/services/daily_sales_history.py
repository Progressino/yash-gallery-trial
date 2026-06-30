"""Daily sales history matrix — verify uploaded shipment/return activity per SKU-day."""
from __future__ import annotations

import os
from zoneinfo import ZoneInfo

import pandas as pd

_IST = ZoneInfo("Asia/Kolkata")
_DEFAULT_VIEW_DAYS = int(os.environ.get("DAILY_SALES_VIEW_DAYS", "30"))
_CORE_PLATFORMS = ("amazon", "flipkart", "meesho", "myntra")


def today_ist_timestamp() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_IST)


def _normalize_sales_tall(sales_df: pd.DataFrame | None) -> pd.DataFrame:
    if sales_df is None or getattr(sales_df, "empty", True):
        return pd.DataFrame(columns=["OMS_SKU", "Date", "Units", "Source", "TxnType"])
    s = sales_df.copy()
    sku_col = "Sku" if "Sku" in s.columns else "OMS_SKU"
    date_col = "TxnDate" if "TxnDate" in s.columns else "Date"
    eff_col = "Units_Effective" if "Units_Effective" in s.columns else "Quantity"
    txn_col = "Transaction Type" if "Transaction Type" in s.columns else "TxnType"
    if sku_col not in s.columns or date_col not in s.columns or eff_col not in s.columns:
        return pd.DataFrame(columns=["OMS_SKU", "Date", "Units", "Source", "TxnType"])
    out = pd.DataFrame(
        {
            "OMS_SKU": s[sku_col].astype(str).str.strip().str.upper(),
            "Date": pd.to_datetime(s[date_col], errors="coerce").dt.normalize(),
            "Units": pd.to_numeric(s[eff_col], errors="coerce").fillna(0.0),
            "Source": s["Source"].astype(str) if "Source" in s.columns else "",
            "TxnType": s[txn_col].astype(str) if txn_col in s.columns else "",
        }
    )
    out = out.dropna(subset=["Date"])
    out = out[out["OMS_SKU"].str.len() > 0]
    return out.reset_index(drop=True)


def sales_history_view_end_date(sales_df: pd.DataFrame | None, end_date: str | None = None) -> pd.Timestamp:
    if end_date:
        try:
            return pd.Timestamp(end_date).normalize()
        except Exception:
            pass
    tall = _normalize_sales_tall(sales_df)
    if tall.empty:
        return today_ist_timestamp().normalize()
    mx = tall["Date"].max()
    return pd.Timestamp(mx).normalize() if pd.notna(mx) else today_ist_timestamp().normalize()


def filter_sales_history_window(
    sales_df: pd.DataFrame | None,
    *,
    days: int | None = None,
    end_date: str | None = None,
    platform: str | None = None,
) -> pd.DataFrame:
    tall = _normalize_sales_tall(sales_df)
    if tall.empty:
        return tall
    span = int(days if days is not None else _DEFAULT_VIEW_DAYS)
    end = sales_history_view_end_date(sales_df, end_date)
    start = end - pd.Timedelta(days=max(0, span - 1))
    mask = (tall["Date"] >= start) & (tall["Date"] <= end)
    sub = tall.loc[mask].copy()
    plat = (platform or "").strip()
    if plat and plat.lower() not in ("all", "combined", ""):
        needle = plat.strip().lower()
        sub = sub[sub["Source"].astype(str).str.strip().str.lower() == needle]
    return sub.reset_index(drop=True)


def sales_platforms_available(sales_df: pd.DataFrame | None) -> list[str]:
    tall = _normalize_sales_tall(sales_df)
    if tall.empty:
        return []
    return sorted({str(x) for x in tall["Source"].astype(str).str.strip().unique() if str(x).strip()})


def sales_history_upload_coverage(
    *,
    days: int | None = None,
    end_date: str | None = None,
    sales_df: pd.DataFrame | None = None,
) -> dict:
    """Per-day Tier-3 upload gaps for core marketplaces in the view window."""
    from .daily_store import get_upload_report_day_coverage

    span = int(days if days is not None else _DEFAULT_VIEW_DAYS)
    end = sales_history_view_end_date(sales_df, end_date)
    start = end - pd.Timedelta(days=max(0, span - 1))
    coverage = get_upload_report_day_coverage()
    gaps: list[dict] = []
    for d in pd.date_range(start, end, freq="D"):
        iso = str(pd.Timestamp(d).date())
        present: list[str] = []
        missing: list[str] = []
        for plat in _CORE_PLATFORMS:
            days_set = coverage.get(plat) or set()
            if iso in days_set:
                present.append(plat)
            else:
                missing.append(plat)
        if missing:
            gaps.append(
                {
                    "date": iso,
                    "missing_platforms": missing,
                    "present_platforms": present,
                }
            )
    return {
        "core_platforms": list(_CORE_PLATFORMS),
        "coverage_gaps": gaps,
    }


def sales_history_summary(
    sales_df: pd.DataFrame | None,
    *,
    days: int | None = None,
    end_date: str | None = None,
    platform: str | None = None,
) -> dict:
    view = filter_sales_history_window(
        sales_df, days=days, end_date=end_date, platform=platform
    )
    if view.empty:
        coverage = sales_history_upload_coverage(
            days=days, end_date=end_date, sales_df=sales_df
        )
        return {
            "loaded": False,
            "rows": 0,
            "skus": 0,
            "days": 0,
            "min_date": "",
            "max_date": "",
            "platforms": sales_platforms_available(sales_df),
            **coverage,
        }
    daily = view.groupby("Date", as_index=False).agg(
        units=("Units", "sum"),
        skus=("OMS_SKU", "nunique"),
        txns=("Units", "count"),
    )
    coverage = sales_history_upload_coverage(
        days=days, end_date=end_date, sales_df=sales_df
    )
    return {
        "loaded": True,
        "rows": int(len(view)),
        "skus": int(view["OMS_SKU"].nunique()),
        "days": int(daily.shape[0]),
        "min_date": str(view["Date"].min().date()),
        "max_date": str(view["Date"].max().date()),
        "window_days": int(days if days is not None else _DEFAULT_VIEW_DAYS),
        "window_end": str(sales_history_view_end_date(sales_df, end_date).date()),
        "platforms": sales_platforms_available(sales_df),
        "total_units": float(view["Units"].sum()),
        **coverage,
    }


def sales_history_wide_matrix(
    sales_df: pd.DataFrame | None,
    *,
    q: str = "",
    limit: int = 150,
    offset: int = 0,
    days: int | None = None,
    end_date: str | None = None,
    platform: str | None = None,
) -> dict:
    """Pivot net daily units (Units_Effective) to SKU rows × date columns."""
    platform = (platform or "all").strip()
    empty = {
        "loaded": False,
        "dates": [],
        "date_totals": [],
        "rows": [],
        "total": 0,
        "limit": int(limit),
        "offset": int(offset),
        "window_days": int(days if days is not None else _DEFAULT_VIEW_DAYS),
        "window_end": str(end_date or today_ist_timestamp().date()),
        "platform": platform,
        "platforms": sales_platforms_available(sales_df),
    }
    view = filter_sales_history_window(
        sales_df, days=days, end_date=end_date, platform=platform
    )
    if view.empty:
        return empty

    span = int(days if days is not None else _DEFAULT_VIEW_DAYS)
    end = sales_history_view_end_date(sales_df, end_date)
    start = end - pd.Timedelta(days=max(0, span - 1))
    dates_sorted = list(pd.date_range(start, end, freq="D"))
    date_strs = [str(pd.Timestamp(d).date()) for d in dates_sorted]

    needle = (q or "").strip().upper()
    if needle:
        view = view[view["OMS_SKU"].str.upper().str.contains(needle, na=False)]
    if view.empty:
        return {**empty, "loaded": True, "dates": date_strs, "date_totals": [0.0] * len(date_strs)}

    daily = (
        view.groupby(["OMS_SKU", "Date"], as_index=False)["Units"]
        .sum()
        .sort_values(["OMS_SKU", "Date"])
    )
    totals_by_day = daily.groupby("Date", as_index=False)["Units"].sum()
    totals_map = {
        pd.Timestamp(r["Date"]).normalize(): float(r["Units"])
        for _, r in totals_by_day.iterrows()
    }
    date_totals = [float(totals_map.get(pd.Timestamp(d).normalize(), 0.0)) for d in dates_sorted]

    sku_rank = (
        daily.groupby("OMS_SKU", as_index=False)["Units"]
        .max()
        .sort_values(["Units", "OMS_SKU"], ascending=[False, True])["OMS_SKU"]
        .astype(str)
        .tolist()
    )
    total = int(len(sku_rank))
    start_i = max(0, int(offset))
    end_i = start_i + max(1, int(limit))
    page_skus = sku_rank[start_i:end_i]
    if not page_skus:
        return {
            **empty,
            "loaded": True,
            "dates": date_strs,
            "date_totals": date_totals,
            "total": total,
        }

    page = daily[daily["OMS_SKU"].isin(page_skus)]
    pivot = page.pivot(index="OMS_SKU", columns="Date", values="Units")
    pivot = pivot.reindex(index=page_skus, columns=dates_sorted).fillna(0.0)

    rows = [
        {
            "sku": str(sku),
            "units": [float(row.get(d, 0.0) or 0.0) for d in dates_sorted],
        }
        for sku, row in pivot.iterrows()
    ]
    coverage = (
        sales_history_upload_coverage(days=days, end_date=end_date, sales_df=sales_df)
        if offset == 0
        else {"core_platforms": list(_CORE_PLATFORMS), "coverage_gaps": []}
    )
    return {
        "loaded": True,
        "dates": date_strs,
        "date_totals": date_totals,
        "rows": rows,
        "total": total,
        "limit": int(limit),
        "offset": start_i,
        "window_days": span,
        "window_end": date_strs[-1] if date_strs else str(end.date()),
        "platform": platform,
        "platforms": sales_platforms_available(sales_df),
        **coverage,
    }


def sales_history_for_sku(
    sales_df: pd.DataFrame | None,
    sku: str,
    *,
    window_days: int = 30,
    end_date: str | None = None,
    platform: str | None = None,
    sku_mapping: dict | None = None,
) -> dict:
    from .po_engine import canonical_oms_key

    tall = filter_sales_history_window(
        sales_df,
        days=window_days,
        end_date=end_date,
        platform=platform,
    )
    target = canonical_oms_key(sku, sku_mapping)
    sub = tall[tall["OMS_SKU"] == target].copy()
    if sub.empty:
        return {
            "loaded": True,
            "sku": sku,
            "rows": [],
            "net_units": 0.0,
            "window_days": int(window_days),
        }
    by_day = (
        sub.groupby("Date", as_index=False)
        .agg(units=("Units", "sum"), txns=("Units", "count"))
        .sort_values("Date")
    )
    rows = [
        {
            "date": str(pd.Timestamp(r["Date"]).date()),
            "units": float(r["units"]),
            "txns": int(r["txns"]),
        }
        for _, r in by_day.iterrows()
    ]
    return {
        "loaded": True,
        "sku": sku,
        "rows": rows,
        "net_units": float(by_day["units"].sum()),
        "window_days": int(window_days),
        "window_start": rows[0]["date"] if rows else "",
        "window_end": rows[-1]["date"] if rows else "",
    }
