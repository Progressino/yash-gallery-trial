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


def _as_naive_day(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Calendar day as tz-naive midnight — safe to compare with datetime64[ns] columns."""
    t = pd.Timestamp(ts)
    if getattr(t, "tzinfo", None) is not None:
        t = t.tz_convert(_IST)
    return pd.Timestamp(t.date())


def _is_smoke_test_sku(sku: str) -> bool:
    """True for synthetic rows from scripts/smoke_uploads_production.py."""
    u = (sku or "").strip().upper()
    return u.startswith("SMOKE-") or u.startswith("SMOKE_")


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
    # Combo listings are fanned out to component SKUs for PO demand, with the
    # listing row retained. Sales History must match the uploaded files, so the
    # synthetic component copies (_Combo_Fan=True) are excluded here.
    if "_Combo_Fan" in s.columns:
        fan = s["_Combo_Fan"].fillna(False).astype(bool)
        if fan.any():
            s = s.loc[~fan]
    dates = pd.to_datetime(s[date_col], errors="coerce")
    # Drop timezone so window filters never mix aware Timestamp with naive columns.
    try:
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_convert(_IST).dt.tz_localize(None)
    except (TypeError, AttributeError, ValueError):
        pass
    out = pd.DataFrame(
        {
            "OMS_SKU": s[sku_col].astype(str).str.strip().str.upper(),
            "Date": dates.dt.normalize(),
            "Units": pd.to_numeric(s[eff_col], errors="coerce").fillna(0.0),
            "Source": s["Source"].astype(str) if "Source" in s.columns else "",
            "TxnType": s[txn_col].astype(str) if txn_col in s.columns else "",
        }
    )
    out = out.dropna(subset=["Date"])
    out = out[out["OMS_SKU"].str.len() > 0]
    # Never surface synthetic smoke-test SKUs in Sales History.
    out = out[~out["OMS_SKU"].map(_is_smoke_test_sku)]
    return out.reset_index(drop=True)


def sales_history_view_end_date(sales_df: pd.DataFrame | None, end_date: str | None = None) -> pd.Timestamp:
    if end_date:
        try:
            return _as_naive_day(end_date)
        except Exception:
            pass
    # Sales for "today" are uploaded tomorrow — default the matrix to yesterday (IST).
    return _as_naive_day(today_ist_timestamp()) - pd.Timedelta(days=1)


def sales_history_window_bounds(
    *,
    days: int | None = None,
    end_date: str | None = None,
    start_date: str | None = None,
    sales_df: pd.DataFrame | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = _as_naive_day(sales_history_view_end_date(sales_df, end_date))
    start_raw = (start_date or "").strip()[:10]
    if len(start_raw) == 10:
        start = _as_naive_day(start_raw)
        if start > end:
            start, end = end, start
        return start, end
    span = int(days if days is not None else _DEFAULT_VIEW_DAYS)
    start = end - pd.Timedelta(days=max(0, span - 1))
    return start, end


def filter_sales_history_window(
    sales_df: pd.DataFrame | None,
    *,
    days: int | None = None,
    end_date: str | None = None,
    start_date: str | None = None,
    platform: str | None = None,
) -> pd.DataFrame:
    tall = _normalize_sales_tall(sales_df)
    if tall.empty:
        return tall
    start, end = sales_history_window_bounds(
        days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
    )
    mask = (tall["Date"] >= start) & (tall["Date"] <= end)
    sub = tall.loc[mask].copy()
    plat = (platform or "").strip()
    if plat and plat.lower() not in ("all", "combined", ""):
        needle = plat.strip().lower()
        sub = sub[sub["Source"].astype(str).str.strip().str.lower() == needle]
    return sub.reset_index(drop=True)


_CORE_PLATFORM_LABELS = {
    "amazon": "Amazon",
    "flipkart": "Flipkart",
    "meesho": "Meesho",
    "myntra": "Myntra",
    "snapdeal": "Snapdeal",
}


def _source_label_ok(label: str) -> bool:
    s = (label or "").strip().lower()
    return bool(s) and s not in ("nan", "none", "null", "nat")


def sales_platforms_available(sales_df: pd.DataFrame | None) -> list[str]:
    tall = _normalize_sales_tall(sales_df)
    if tall.empty:
        return []
    out: set[str] = set()
    for x in tall["Source"].astype(str).str.strip().unique():
        if _source_label_ok(str(x)):
            out.add(str(x).strip())
    return sorted(out, key=lambda s: s.lower())


def sales_history_platform_filters(
    sales_df: pd.DataFrame | None,
    *,
    days: int | None = None,
    end_date: str | None = None,
    start_date: str | None = None,
) -> list[str]:
    """Platforms for UI tabs: Tier-3 uploads in window plus any sources in sales rows."""
    from .daily_store import get_upload_report_day_coverage

    start, end = sales_history_window_bounds(
        days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
    )
    coverage = get_upload_report_day_coverage()
    out: set[str] = set(sales_platforms_available(sales_df))
    for pk, label in _CORE_PLATFORM_LABELS.items():
        days_set = coverage.get(pk) or set()
        for d in pd.date_range(start, end, freq="D"):
            if str(pd.Timestamp(d).date()) in days_set:
                out.add(label)
                break
    return sorted(out, key=lambda s: s.lower())


def sales_history_data_quality_checks(
    sales_df: pd.DataFrame | None,
    *,
    days: int | None = None,
    end_date: str | None = None,
    start_date: str | None = None,
    max_days: int = 3,
) -> list[dict]:
    """
    Automatic spot checks: for recent days with Amazon Tier-3 uploads, compare
    matrix net units (Sales History build) to a fresh Tier-3 re-read.
    """
    from .daily_store import get_upload_report_day_coverage, load_platform_data_for_report_range
    from .sales import _compute_platform_metrics

    if sales_df is None or getattr(sales_df, "empty", True):
        return []

    start, end = sales_history_window_bounds(
        days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
    )
    coverage = get_upload_report_day_coverage()
    amazon_days = sorted(coverage.get("amazon") or set())
    in_window: list[str] = []
    for iso in amazon_days:
        try:
            d = pd.Timestamp(iso)
        except Exception:
            continue
        if start <= d <= end:
            in_window.append(iso)
    if not in_window:
        return []

    checks: list[dict] = []
    for iso in in_window[-max(1, int(max_days)) :]:
        df = load_platform_data_for_report_range("amazon", iso, iso, dedup=True)
        if df is None or df.empty:
            continue
        metrics = _compute_platform_metrics(
            df,
            "Amazon",
            "SKU",
            "Transaction_Type",
            start_date=iso,
            end_date=iso,
            headline_only=True,
        )
        expected = float(metrics.get("net_units") or 0)
        view = filter_sales_history_window(
            sales_df,
            days=days,
            end_date=end_date,
            start_date=start_date,
            platform="Amazon",
        )
        day = pd.Timestamp(iso).normalize()
        actual = float(view.loc[view["Date"] == day, "Units"].sum())
        delta = round(actual - expected, 2)
        checks.append(
            {
                "date": iso,
                "platform": "Amazon",
                "check": "tier3_vs_sales_history",
                "ok": abs(delta) < 0.5,
                "expected_net_units": expected,
                "matrix_net_units": actual,
                "delta": delta,
            }
        )
    return checks


def sales_history_upload_coverage(
    *,
    days: int | None = None,
    end_date: str | None = None,
    start_date: str | None = None,
    sales_df: pd.DataFrame | None = None,
) -> dict:
    """Per-day Tier-3 upload gaps for core marketplaces in the view window."""
    from .daily_store import get_upload_report_day_coverage

    start, end = sales_history_window_bounds(
        days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
    )
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
    start_date: str | None = None,
    platform: str | None = None,
) -> dict:
    view = filter_sales_history_window(
        sales_df, days=days, end_date=end_date, start_date=start_date, platform=platform
    )
    if view.empty:
        coverage = sales_history_upload_coverage(
            days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
        )
        return {
            "loaded": False,
            "rows": 0,
            "skus": 0,
            "days": 0,
            "min_date": "",
            "max_date": "",
            "platforms": sales_history_platform_filters(
            sales_df, days=days, end_date=end_date, start_date=start_date
        ),
            **coverage,
        }
    daily = view.groupby("Date", as_index=False).agg(
        units=("Units", "sum"),
        skus=("OMS_SKU", "nunique"),
        txns=("Units", "count"),
    )
    coverage = sales_history_upload_coverage(
        days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
    )
    auto_checks = sales_history_data_quality_checks(
        sales_df, days=days, end_date=end_date, start_date=start_date
    )
    base = {
        "loaded": True,
        "rows": int(len(view)),
        "skus": int(view["OMS_SKU"].nunique()),
        "days": int(daily.shape[0]),
        "min_date": str(view["Date"].min().date()),
        "max_date": str(view["Date"].max().date()),
        "window_days": int(days if days is not None else _DEFAULT_VIEW_DAYS),
        "window_end": str(sales_history_view_end_date(sales_df, end_date).date()),
        "platforms": sales_history_platform_filters(
            sales_df, days=days, end_date=end_date, start_date=start_date
        ),
        "total_units": float(view["Units"].sum()),
        "auto_checks": auto_checks,
        "auto_checks_ok": all(c.get("ok") for c in auto_checks) if auto_checks else True,
        **coverage,
    }
    return base


def sales_history_wide_matrix(
    sales_df: pd.DataFrame | None,
    *,
    q: str = "",
    limit: int = 150,
    offset: int = 0,
    days: int | None = None,
    end_date: str | None = None,
    start_date: str | None = None,
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
        "platforms": sales_history_platform_filters(
            sales_df, days=days, end_date=end_date, start_date=start_date
        ),
    }
    view = filter_sales_history_window(
        sales_df, days=days, end_date=end_date, start_date=start_date, platform=platform
    )
    if view.empty:
        return empty

    start, end = sales_history_window_bounds(
        days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
    )
    span = int((end - start).days) + 1
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

    # Rank by total units in the window (not peak day — peak ranked smoke test SKUs first).
    sku_rank = (
        daily.groupby("OMS_SKU", as_index=False)["Units"]
        .sum()
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
        sales_history_upload_coverage(
            days=days, end_date=end_date, start_date=start_date, sales_df=sales_df
        )
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
        "platforms": sales_history_platform_filters(
            sales_df, days=days, end_date=end_date, start_date=start_date
        ),
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


def build_sales_history_sales_df(
    sess,
    *,
    days: int | None = None,
    end_date: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """
    Build unified sales for the Sales History window from Tier-3 uploads (deduped),
    not the session warm-cache sales_df (which can double-count re-uploaded days).
    """
    import pandas as pd

    from .daily_store import load_platform_data_for_report_range
    from .po_calculate_run import _build_platform_sales_df
    from .sales import _dedup_sales_linekey_rows, _downcast_sales

    start, end = sales_history_window_bounds(
        days=days, end_date=end_date, start_date=start_date, sales_df=None
    )
    s0, s1 = str(start.date()), str(end.date())
    specs = (
        ("amazon", "mtr_df"),
        ("myntra", "myntra_df"),
        ("meesho", "meesho_df"),
        ("flipkart", "flipkart_df"),
        ("snapdeal", "snapdeal_df"),
    )
    overrides: dict[str, pd.DataFrame] = {}
    for pk, attr in specs:
        df = load_platform_data_for_report_range(pk, s0, s1, dedup=True)
        if df is not None and not df.empty:
            overrides[attr] = df
    if not overrides:
        return pd.DataFrame()
    out = _build_platform_sales_df(sess, frame_overrides=overrides)
    if out is None or out.empty:
        return pd.DataFrame()
    if "Source" in out.columns:
        src = out["Source"].astype(str).str.strip().str.lower()
        miss = out["Source"].isna() | src.isin(("", "nan", "none"))
        if miss.any() and "mtr_df" in overrides:
            out.loc[miss, "Source"] = "Amazon"
    # Sales History mirrors the uploaded files: zero-amount Amazon shipments
    # (reclassified to FreeReplacement with Units_Effective 0 for PO math)
    # are physical shipments in the daily files — count their units here.
    if {"Transaction Type", "Units_Effective", "Quantity"}.issubset(out.columns):
        fr = out["Transaction Type"].astype(str).str.strip().eq("FreeReplacement")
        if fr.any():
            out.loc[fr, "Units_Effective"] = pd.to_numeric(
                out.loc[fr, "Quantity"], errors="coerce"
            ).fillna(0)
    out = _dedup_sales_linekey_rows(out)
    return _downcast_sales(out)
