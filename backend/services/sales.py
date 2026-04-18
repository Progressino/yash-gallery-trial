"""
Sales aggregation — build + dedup logic extracted from app.py.
"""
import gc
import logging
import re
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("erp.sales")

# Dashboard / DSR filters use India reporting calendar (seller exports are IST or UTC+05:30).
_REPORTING_TZ = ZoneInfo("Asia/Kolkata")


def txn_reporting_naive_ist(series: pd.Series) -> pd.Series:
    """
    Normalize marketplace timestamps to **naive wall clock in Asia/Kolkata**.

    - tz-aware values (UTC, +05:30, etc.) → convert to IST then drop tz info.
    - tz-naive values → left as-is (already local wall time in typical exports).

    Avoid ``tz_convert(None)`` on mixed series — that folds to **UTC** and shifts calendar days,
    which makes single-day dashboard filters wrong for Myntra/Flipkart while Amazon (often
    date-only) still looks fine.
    """
    t = pd.to_datetime(series, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        return t.dt.tz_convert(_REPORTING_TZ).dt.tz_localize(None)
    return t


def _filter_by_reporting_days(
    df: pd.DataFrame,
    date_col: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """Inclusive calendar-day window on IST-normalized dates."""
    if df.empty or date_col not in df.columns:
        return df
    t = txn_reporting_naive_ist(df[date_col])
    day = t.dt.normalize()
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= day >= pd.Timestamp(start_date).normalize()
    if end_date:
        mask &= day <= pd.Timestamp(end_date).normalize()
    return df.loc[mask]

from .helpers import (
    _downcast_sales,
    canonical_pl_sku_key,
    clean_line_id_series,
    clean_sku,
    map_to_oms_sku,
    mapping_lookup_sets,
    sku_recognized_in_master,
)
from .myntra import myntra_to_sales_rows
from .meesho import meesho_to_sales_rows
from .flipkart import flipkart_to_sales_rows
from .snapdeal import snapdeal_to_sales_rows

# Strip "PL" infix in Amazon seller SKUs (mirrors inventory._PL_RE)
_PL_RE = re.compile(r'^(\d+)PL(YK)', re.I)


def _is_aggregate_sales_sku(token: str) -> bool:
    """Bucket / ledger rows that must not be pushed through marketplace→OMS mapping."""
    s = str(token).strip().lower()
    return (
        s == "meesho_total"
        or "_total" in s
        or (s.startswith("total") and len(s) <= 24)
        or (s.endswith("total") and len(s) <= 24)
    )


def _apply_unified_oms_skus(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    One canonical pass: map_to_oms_sku (PL + key fallbacks) then canonical_sales_sku (PL strip).
    Ensures every marketplace uses the same final tokens as the SKU master / inventory.
    """
    if df.empty or "Sku" not in df.columns:
        return df
    m = mapping or {}

    def _one(raw) -> str:
        if pd.isna(raw):
            return ""
        c = str(raw).strip()
        if not c or c.upper() in ("NAN", "NONE"):
            return ""
        if _is_aggregate_sales_sku(c):
            return c
        if m:
            return canonical_sales_sku(map_to_oms_sku(c, m))
        return canonical_sales_sku(c)

    out = df.copy()
    out["Sku"] = out["Sku"].apply(_one)
    return out


def list_sku_mapping_gaps(sales_df: pd.DataFrame, mapping: Dict[str, str], *, limit: int = 80) -> List[str]:
    """
    SKUs present in unified sales that do not appear as a mapping key (incl. PL-alias of keys)
    or as a mapped OMS value — i.e. likely missing / typo'd lines on the master sheet.
    """
    if sales_df.empty or not mapping or "Sku" not in sales_df.columns:
        return []
    key_set, val_set, num_embed = mapping_lookup_sets(mapping)
    bad: List[str] = []
    seen: set[str] = set()
    for raw in sales_df["Sku"].dropna().unique():
        c = clean_sku(raw)
        if not c or _is_aggregate_sales_sku(c):
            continue
        if sku_recognized_in_master(
            str(raw), mapping, key_set=key_set, val_set=val_set, numeric_embed=num_embed
        ):
            continue
        if c not in seen:
            seen.add(c)
            bad.append(c)
    bad.sort()
    return bad[:limit]


def _resolve_mtr_sku(sku, mapping: Dict[str, str]) -> str:
    raw = str(sku).strip().upper()
    stripped = _PL_RE.sub(r"\1\2", raw)
    return mapping.get(stripped, mapping.get(raw, stripped))


def canonical_sales_sku(sku) -> str:
    """
    Normalise seller / OMS tokens so Amazon PL listings match OMS (1023PLYK → 1023YK).
    Apply to every platform row in build_sales_df before deduplication.
    """
    if pd.isna(sku):
        return ""
    t = str(sku).strip().upper()
    if t in ("", "NAN", "NONE"):
        return t
    return _PL_RE.sub(r"\1\2", t)


def canonical_sales_sku_series(skus: pd.Series) -> pd.Series:
    """Vectorised PL strip for deep-dive / bulk filters (matches ``canonical_sales_sku``)."""
    s = skus.fillna("").astype(str).str.strip().str.upper()
    s = s.mask(s.isin(["", "NAN", "NONE"]), "")
    return s.str.replace(_PL_RE, r"\1\2", regex=True)


def _mtr_to_sales_df(
    mtr_df: pd.DataFrame,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
) -> pd.DataFrame:
    """Convert MTR DataFrame to sales rows format."""
    from .helpers import get_parent_sku

    if mtr_df.empty:
        return pd.DataFrame()

    m = mtr_df[["Date", "SKU", "Transaction_Type", "Quantity"]].copy()
    m = m.rename(columns={
        "Date":             "TxnDate",
        "SKU":              "Sku",
        "Transaction_Type": "Transaction Type",
    })
    m["TxnDate"]  = pd.to_datetime(m["TxnDate"], errors="coerce")
    m["Quantity"] = pd.to_numeric(m["Quantity"], errors="coerce").fillna(0)
    m = m.dropna(subset=["TxnDate"])
    m["Sku"] = m["Sku"].apply(lambda x: _resolve_mtr_sku(x, sku_mapping or {}))
    m["Sku"] = m["Sku"].map(canonical_sales_sku)

    # Line-level keys for build_sales_df dedup (Amazon MTR exposes Order_Id / Invoice_Number).
    idx = m.index
    if "Order_Id" in mtr_df.columns:
        oid = mtr_df.loc[idx, "Order_Id"].astype(str).str.strip()
        oid = oid.mask(oid.str.lower().isin(["", "nan", "none"]), np.nan)
    else:
        oid = pd.Series(np.nan, index=idx)
    if "Invoice_Number" in mtr_df.columns:
        inv = mtr_df.loc[idx, "Invoice_Number"].astype(str).str.strip()
        inv = inv.mask(inv.str.lower().isin(["", "nan", "none"]), np.nan)
    else:
        inv = pd.Series(np.nan, index=idx)
    need_syn = oid.isna() & inv.notna()
    syn_key = (
        "AMZINV:" + inv.astype(str) + ":" + m["Sku"].astype(str) + ":" + m["Transaction Type"].astype(str)
    )
    m["OrderId"] = oid
    m.loc[need_syn, "OrderId"] = syn_key.loc[need_syn]

    if group_by_parent:
        m["Sku"] = m["Sku"].apply(get_parent_sku)

    # Cancel rows pair with Shipment/Refund events — net must reflect seller/MACO final units.
    m["Units_Effective"] = np.where(
        m["Transaction Type"] == "Refund",  -m["Quantity"],
        np.where(m["Transaction Type"] == "Cancel", -m["Quantity"], m["Quantity"])
    )
    m["LineKey"] = ""
    return m[
        ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "OrderId", "LineKey"]
    ]


def _drop_amazon_unkeyed_shadows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amazon rows without OrderId (e.g. some FBA extracts) often duplicate rows that
    already exist with the same shipment keyed by OrderId. Those land in separate
    dedup buckets in build_sales_df and were both kept — inflating units.

    For each (Sku, calendar day, Transaction Type, quantity) fingerprint that has
    exactly one keyed Amazon row, drop unkeyed Amazon rows matching that fingerprint.
    If multiple keyed rows share the fingerprint (distinct orders, same sku/qty/day),
    keep unkeyed rows — dropping them under-counted vs seller reports.
    """
    if df.empty or "Source" not in df.columns or "OrderId" not in df.columns:
        return df

    is_amz = df["Source"].astype(str) == "Amazon"
    if not is_amz.any():
        return df

    amz = df.loc[is_amz].copy()
    rest = df.loc[~is_amz].copy()

    amz["_day"] = pd.to_datetime(amz["TxnDate"], errors="coerce").dt.normalize()
    amz["_qtyk"] = pd.to_numeric(amz["Quantity"], errors="coerce").fillna(0).round().astype("int64")

    _oid_str = amz["OrderId"].astype(str).str.strip()
    has_oid = amz["OrderId"].notna() & ~_oid_str.str.lower().isin(["", "nan", "none"])

    keyed = amz.loc[has_oid]
    unkeyed = amz.loc[~has_oid]
    if keyed.empty or unkeyed.empty:
        amz = amz.drop(columns=["_day", "_qtyk"], errors="ignore")
        return pd.concat([rest, amz], ignore_index=True)

    key_cols = ["Sku", "_day", "Transaction Type", "_qtyk"]
    kc = keyed.groupby(key_cols).size()
    single = kc[kc == 1].reset_index()[key_cols]
    if single.empty:
        amz = amz.drop(columns=["_day", "_qtyk"], errors="ignore")
        return pd.concat([rest, amz], ignore_index=True)

    unkeyed_merge = unkeyed.merge(single, on=key_cols, how="left", indicator=True)
    kept_unkeyed = unkeyed_merge.loc[unkeyed_merge["_merge"] == "left_only"].drop(columns=["_merge"])

    amz_out = pd.concat([keyed, kept_unkeyed], ignore_index=True)
    amz_out = amz_out.drop(columns=["_day", "_qtyk"], errors="ignore")
    return pd.concat([rest, amz_out], ignore_index=True)


def _dedup_sales_linekey_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    After OMS mapping, collapse duplicate marketplace lines that share the same LineKey,
    SKU, txn type, and reporting calendar day (upload + cache paths sometimes keep twins).
    """
    if df.empty or "LineKey" not in df.columns:
        return df
    lk = clean_line_id_series(df["LineKey"])
    use = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
    if not use.any():
        return df
    sub = df.loc[use].copy()
    rest = df.loc[~use]
    sub["LineKey"] = lk.loc[sub.index]
    sub["_day"] = txn_reporting_naive_ist(sub["TxnDate"]).dt.normalize()
    sub = sub.drop_duplicates(
        subset=["Source", "LineKey", "Sku", "Transaction Type", "_day"],
        keep="last",
    )
    sub = sub.drop(columns=["_day"], errors="ignore")
    return pd.concat([rest, sub], ignore_index=True)


def build_sales_df(
    mtr_df: pd.DataFrame,
    myntra_df: pd.DataFrame,
    meesho_df: pd.DataFrame,
    flipkart_df: pd.DataFrame,
    sku_mapping: Dict[str, str],
    snapdeal_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Concatenate all platform DataFrames into a unified sales_df and deduplicate.
    Mirrors the 'Load All Data' block in app.py.
    """
    sales_parts: List[pd.DataFrame] = []

    if not meesho_df.empty:
        from .meesho import refresh_meesho_dataframe_oms_inplace

        refresh_meesho_dataframe_oms_inplace(meesho_df, sku_mapping or None)
        sales_parts.append(_downcast_sales(meesho_to_sales_rows(meesho_df, sku_mapping=sku_mapping or None)))
    if not myntra_df.empty:
        sales_parts.append(_downcast_sales(myntra_to_sales_rows(myntra_df)))
    if not flipkart_df.empty:
        sales_parts.append(_downcast_sales(flipkart_to_sales_rows(flipkart_df)))
    if snapdeal_df is not None and not snapdeal_df.empty:
        sales_parts.append(_downcast_sales(snapdeal_to_sales_rows(snapdeal_df)))
    # Amazon: always merge when MTR rows exist. Empty sku_mapping is OK — _resolve_mtr_sku
    # falls back to PL-stripped seller SKU (same as other channels without a map).
    if not mtr_df.empty:
        _mtr_sales = _mtr_to_sales_df(mtr_df, sku_mapping or {})
        if not _mtr_sales.empty:
            _mtr_sales["Source"] = "Amazon"
            sales_parts.append(_downcast_sales(_mtr_sales))
        del _mtr_sales
        gc.collect()

    if not sales_parts:
        return pd.DataFrame()

    combined_sales = pd.concat([d for d in sales_parts if not d.empty], ignore_index=True)
    del sales_parts
    gc.collect()

    # Single canonical resolution for all channels (same rules as SKU master).
    combined_sales = _apply_unified_oms_skus(combined_sales, sku_mapping or {})
    combined_sales = _dedup_sales_linekey_rows(combined_sales)

    if "LineKey" not in combined_sales.columns:
        combined_sales["LineKey"] = ""

    lk_series = clean_line_id_series(combined_sales["LineKey"])
    has_lk = lk_series.ne("") & ~lk_series.str.lower().isin(["nan", "none"])

    # Rows with a marketplace line id: dedupe on LineKey (Flipkart often repeats Order ID
    # across invoice lines — OrderId-only dedupe was dropping real units).
    _with_lk = combined_sales.loc[has_lk].drop_duplicates(
        subset=["Source", "LineKey", "Sku", "Transaction Type"],
        keep="last",
    )

    _oid_str = combined_sales["OrderId"].astype(str).str.strip()
    _oid_valid = combined_sales["OrderId"].notna() & ~_oid_str.str.lower().isin(["", "nan", "none"])

    rest = combined_sales.loc[~has_lk]
    _with_oid = rest.loc[_oid_valid].drop_duplicates(
        subset=["OrderId", "Sku", "Source", "Transaction Type"],
        keep="last",
    )
    _without_oid = rest.loc[~_oid_valid].drop_duplicates(
        subset=["Sku", "TxnDate", "Source", "Transaction Type", "Quantity"],
        keep="last",
    )
    del combined_sales, _oid_valid, _oid_str, has_lk, lk_series, rest
    gc.collect()

    result = pd.concat([_with_lk, _with_oid, _without_oid], ignore_index=True)
    del _with_lk, _with_oid, _without_oid
    gc.collect()

    result = _drop_amazon_unkeyed_shadows(result)
    gc.collect()

    # Exact duplicate lines (same channel line key) — catches merge/upload bugs.
    _cols = [
        c for c in ("Sku", "TxnDate", "Transaction Type", "Quantity", "Source", "OrderId")
        if c in result.columns
    ]
    if _cols and not result.empty:
        result = result.drop_duplicates(subset=_cols, keep="last")
    gc.collect()

    return _downcast_sales(result)


def get_sales_summary(
    sales_df: pd.DataFrame,
    months: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Return KPI summary for the dashboard.

    If start_date/end_date (ISO strings) are provided they take priority over months.
    months=0 means all-time.
    """
    if sales_df.empty:
        return {"total_units": 0, "total_returns": 0, "net_units": 0, "return_rate": 0.0}

    df = sales_df.copy()
    df["TxnDate"] = txn_reporting_naive_ist(df["TxnDate"])
    df = df.dropna(subset=["TxnDate"])

    if start_date or end_date:
        df = _filter_by_reporting_days(df, "TxnDate", start_date, end_date)
    elif months > 0:
        cutoff = df["TxnDate"].max() - pd.DateOffset(months=months)
        df = df[df["TxnDate"] >= cutoff]

    shipped  = df[df["Transaction Type"] == "Shipment"]["Quantity"].sum()
    returned = df[df["Transaction Type"] == "Refund"]["Quantity"].sum()
    net      = df["Units_Effective"].sum()
    rate     = float(returned / shipped * 100) if shipped > 0 else 0.0

    return {
        "total_units":  int(shipped),
        "total_returns": int(returned),
        "net_units":    int(net),
        "return_rate":  round(rate, 1),
    }


def filter_sales_for_export(
    sales_df: pd.DataFrame,
    months: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply the same date window as get_sales_summary; optionally restrict Source(s)."""
    if sales_df.empty:
        return sales_df
    df = sales_df.copy()
    df["TxnDate"] = txn_reporting_naive_ist(df["TxnDate"])
    df = df.dropna(subset=["TxnDate"])

    if start_date or end_date:
        df = _filter_by_reporting_days(df, "TxnDate", start_date, end_date)
    elif months > 0:
        cutoff = df["TxnDate"].max() - pd.DateOffset(months=months)
        df = df[df["TxnDate"] >= cutoff]

    if sources:
        want = {str(s).strip() for s in sources if str(s).strip()}
        if want and "Source" in df.columns:
            df = df[df["Source"].astype(str).str.strip().isin(want)]

    return df


def get_sales_by_source(sales_df: pd.DataFrame) -> List[dict]:
    """Returns pie chart data: [{source, units}]."""
    if sales_df.empty:
        return []
    df = sales_df[sales_df["Transaction Type"] == "Shipment"].copy()
    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str)
    grp = df.groupby("Source")["Quantity"].sum().reset_index()
    grp.columns = ["source", "units"]
    return grp.sort_values("units", ascending=False).to_dict("records")


DSR_PLATFORM_ORDER = ("Amazon", "Meesho", "Myntra", "Flipkart", "Snapdeal", "Others")
DSR_MAIN_SOURCES = frozenset({"Amazon", "Meesho", "Myntra", "Flipkart", "Snapdeal"})


def _fmt_dsr_display_date(iso_date: str) -> str:
    """e.g. 2026-04-09 → 9-Apr-26"""
    try:
        ts = pd.Timestamp(iso_date)
        return f"{ts.day}-{ts.strftime('%b')}-{str(ts.year)[-2:]}"
    except Exception:
        return iso_date


def get_daily_dsr_report(sales_df: pd.DataFrame, report_date: str) -> dict:
    """
    Single-day DSR-style breakdown: primary marketplaces plus an Others bucket.
    Rows under each platform use ``DSR_Segment`` when present (e.g. Flipkart Brand,
    Snapdeal Company); otherwise one combined ``All`` row. Others uses each distinct
    ``Source`` as the segment label.
    """
    empty = {
        "date":            report_date or "",
        "display_date":    _fmt_dsr_display_date(report_date) if report_date else "",
        "sections":        [],
        "subtotal":        {"sales": 0, "returns": 0},
    }
    if sales_df.empty or not (report_date and str(report_date).strip()):
        return empty

    try:
        day = pd.Timestamp(report_date).normalize()
    except Exception:
        return empty

    d = sales_df.copy()
    d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
    d = d.dropna(subset=["TxnDate"])
    d = d[d["TxnDate"].dt.normalize() == day]
    if d.empty:
        return {
            **empty,
            "date":         str(day.date()),
            "display_date": _fmt_dsr_display_date(str(day.date())),
        }

    d["_qty"] = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)
    txn = d["Transaction Type"].astype(str).str.strip()
    d["_ship"] = txn == "Shipment"
    d["_ref"] = txn == "Refund"

    src = d["Source"].astype(str).str.strip()
    d["_bucket"] = src.where(src.isin(DSR_MAIN_SOURCES), "Others")

    seg = pd.Series("All", index=d.index, dtype=str)
    if "DSR_Segment" in d.columns:
        seg = d["DSR_Segment"].fillna("").astype(str).str.strip()
        seg = seg.mask(seg.str.len() == 0, "All")
    elif "Company" in d.columns:
        seg = d["Company"].fillna("").astype(str).str.strip()
        seg = seg.mask(seg.str.len() == 0, "All")
    d["_seg"] = seg
    others_m = d["_bucket"] == "Others"
    if others_m.any():
        d.loc[others_m, "_seg"] = src.loc[others_m]

    sections: List[dict] = []
    sub_sales = 0
    sub_ret = 0

    for plat in DSR_PLATFORM_ORDER:
        sub = d[d["_bucket"] == plat]
        if sub.empty:
            continue
        row_list: List[dict] = []
        for seg_name in sorted(
            sub["_seg"].unique(),
            key=lambda x: (str(x).lower() == "all", str(x).lower()),
        ):
            g = sub[sub["_seg"] == seg_name]
            sales_n = int(g.loc[g["_ship"], "_qty"].sum())
            ret_n = int(g.loc[g["_ref"], "_qty"].sum())
            row_list.append({"segment": str(seg_name), "sales": sales_n, "returns": ret_n})
        sec_sales = sum(r["sales"] for r in row_list)
        sec_ret = sum(r["returns"] for r in row_list)
        sub_sales += sec_sales
        sub_ret += sec_ret
        sections.append({
            "platform":        plat,
            "rows":            row_list,
            "section_sales":   sec_sales,
            "section_returns": sec_ret,
        })

    return {
        "date":         str(day.date()),
        "display_date": _fmt_dsr_display_date(str(day.date())),
        "sections":     sections,
        "subtotal":     {"sales": sub_sales, "returns": sub_ret},
    }


def daily_dsr_report_to_csv_rows(report: dict) -> List[List[object]]:
    """Flatten a ``get_daily_dsr_report`` dict into CSV rows (header + body)."""
    rows: List[List[object]] = [
        ["Report type", "Daily DSR"],
        ["Date (ISO)", report.get("date", "")],
        ["Display date", report.get("display_date", "")],
        ["", "", "", ""],
        ["Marketplace", "Segment", "Sales (units)", "Returns (units)"],
    ]
    for sec in report.get("sections") or []:
        plat = sec.get("platform", "")
        for r in sec.get("rows") or []:
            rows.append(
                [plat, r.get("segment", ""), r.get("sales", 0), r.get("returns", 0)]
            )
        rows.append(
            [
                plat,
                f"— {plat} total —",
                sec.get("section_sales", 0),
                sec.get("section_returns", 0),
            ]
        )
    st = report.get("subtotal") or {}
    rows.append(["Subtotal (all platforms)", "", st.get("sales", 0), st.get("returns", 0)])
    return rows


def _series_canonical_dsr_brand(seg: pd.Series) -> pd.Series:
    """
    Map DSR segment / company labels to ``YG`` or ``Akiko`` for head-to-head reporting.
    Rows that are not clearly one of these brands become NA (excluded from comparison).
    """
    s = seg.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=seg.index, dtype=object)
    m_akiko = s.str.contains("akiko", na=False)
    m_yg = (
        (s == "yg")
        | s.str.contains("yash gallery", na=False)
        | s.str.contains(r"\byg\b", regex=True, na=False)
    )
    conflict = m_yg & m_akiko
    m_yg = m_yg & ~conflict
    m_akiko = m_akiko & ~conflict
    out = out.mask(m_yg, "YG")
    out = out.mask(m_akiko, "Akiko")
    return out


def get_dsr_brand_monthly_comparison(
    sales_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Month-over-month shipment units for **YG** vs **Akiko** based on ``DSR_Segment``
    (or ``Company`` fallback), same labeling rules as the daily DSR table.
    """
    empty: dict = {
        "rows":   [],
        "totals": {"YG": 0, "Akiko": 0},
        "note":   "",
    }
    if sales_df.empty or "TxnDate" not in sales_df.columns:
        return empty

    d = sales_df.copy()
    d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
    d = d.dropna(subset=["TxnDate"])
    if start_date or end_date:
        d = _filter_by_reporting_days(d, "TxnDate", start_date, end_date)

    d = d[d["Transaction Type"].astype(str).str.strip() == "Shipment"]
    if d.empty:
        return {**empty, "note": "No shipment rows in the selected range."}

    seg = pd.Series("All", index=d.index, dtype=str)
    if "DSR_Segment" in d.columns:
        seg = d["DSR_Segment"].fillna("").astype(str).str.strip()
        seg = seg.mask(seg.str.len() == 0, "All")
    elif "Company" in d.columns:
        seg = d["Company"].fillna("").astype(str).str.strip()
        seg = seg.mask(seg.str.len() == 0, "All")
    d["_brand"] = _series_canonical_dsr_brand(seg)
    d = d[d["_brand"].notna()]
    if d.empty:
        return {
            **empty,
            "note": (
                "No rows tagged as YG or Akiko. Segments come from Flipkart Brand, "
                "Snapdeal Company, or a DSR_Segment column on unified sales."
            ),
        }

    d["_month"] = d["TxnDate"].dt.to_period("M").astype(str)
    qty = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)
    pivot = (
        d.assign(_q=qty)
        .groupby(["_month", "_brand"], observed=True)["_q"]
        .sum()
        .unstack(fill_value=0)
    )
    for col in ("YG", "Akiko"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.reindex(columns=["YG", "Akiko"], fill_value=0)

    out_rows: List[dict] = []
    for month in sorted(pivot.index.astype(str)):
        yg = int(pivot.loc[month, "YG"])
        ak = int(pivot.loc[month, "Akiko"])
        if yg > ak:
            leader, delta = "YG", yg - ak
        elif ak > yg:
            leader, delta = "Akiko", ak - yg
        else:
            leader, delta = "Tie", 0
        try:
            ts = pd.Timestamp(month + "-01")
            month_display = ts.strftime("%b %Y")
        except Exception:
            month_display = month
        out_rows.append({
            "month":         month,
            "month_display": month_display,
            "YG":            yg,
            "Akiko":         ak,
            "leader":        leader,
            "delta":         delta,
        })

    totals = {
        "YG":    int(pivot["YG"].sum()),
        "Akiko": int(pivot["Akiko"].sum()),
    }
    return {"rows": out_rows, "totals": totals, "note": ""}


def dsr_brand_monthly_to_csv_rows(result: dict) -> List[List[object]]:
    """Flatten ``get_dsr_brand_monthly_comparison`` output for CSV download."""
    rows: List[List[object]] = [
        ["Report", "YG vs Akiko — monthly shipment units"],
        ["Filter note", "Respects Intelligence dashboard start/end dates when provided."],
    ]
    if result.get("note"):
        rows.append(["Status", result["note"]])
    rows.extend([[], ["Month (label)", "Month (ISO)", "YG", "Akiko", "Leader", "Margin (units)"]])
    for r in result.get("rows") or []:
        margin = "" if r.get("leader") == "Tie" else r.get("delta", 0)
        rows.append([
            r.get("month_display", ""),
            r.get("month", ""),
            r.get("YG", 0),
            r.get("Akiko", 0),
            r.get("leader", ""),
            margin,
        ])
    t = result.get("totals") or {}
    rows.append([])
    rows.append([
        "Range total",
        "",
        t.get("YG", 0),
        t.get("Akiko", 0),
        (
            "YG ahead"
            if t.get("YG", 0) > t.get("Akiko", 0)
            else "Akiko ahead"
            if t.get("Akiko", 0) > t.get("YG", 0)
            else "Tie"
        ),
        abs(int(t.get("YG", 0)) - int(t.get("Akiko", 0))),
    ])
    return rows


def get_top_skus(
    sales_df: pd.DataFrame,
    limit: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    basis: str = "gross",
) -> List[dict]:
    """Top SKUs by gross shipment units (``basis=gross``) or ``Units_Effective`` sum (``basis=net``)."""
    if sales_df.empty or "Sku" not in sales_df.columns:
        return []
    b = (basis or "gross").strip().lower()
    if b not in ("gross", "net"):
        b = "gross"

    df = sales_df.copy()
    if start_date or end_date:
        df["TxnDate"] = txn_reporting_naive_ist(df["TxnDate"])
        df = df.dropna(subset=["TxnDate"])
        df = _filter_by_reporting_days(df, "TxnDate", start_date, end_date)

    _sku_lower = df["Sku"].astype(str).str.lower()
    df = df[~(_sku_lower.str.contains("_total") | _sku_lower.str.endswith("total") | _sku_lower.str.startswith("total"))]

    if b == "net" and "Units_Effective" in df.columns:
        eff = pd.to_numeric(df["Units_Effective"], errors="coerce").fillna(0)
        grp = df.assign(_e=eff).groupby("Sku")["_e"].sum().reset_index()
        grp.columns = ["sku", "units"]
        grp["units"] = grp["units"].astype(int)
        return grp.sort_values("units", ascending=False).head(limit).to_dict("records")

    df = df[df["Transaction Type"] == "Shipment"].copy()
    if df.empty:
        return []
    grp = df.groupby("Sku")["Quantity"].sum().reset_index()
    grp.columns = ["sku", "units"]
    return grp.sort_values("units", ascending=False).head(limit).to_dict("records")


# ── AI Dashboard helpers ───────────────────────────────────────────────────────

def _compute_platform_metrics(
    df: pd.DataFrame,
    platform_name: str,
    sku_col: str,
    txn_col: str,
    ship_val: str = "Shipment",
    refund_val: str = "Refund",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Shared computation for a single platform DataFrame."""
    stub = {
        "platform": platform_name, "loaded": False,
        "total_units": 0, "total_returns": 0, "net_units": 0,
        "return_rate": 0.0,
        "top_sku": "", "trend_direction": "flat", "trend_direction_net": "flat",
        "monthly": [], "by_state": [],
    }
    if df.empty:
        return stub

    try:
        d = df.copy()
        d["_Date"] = txn_reporting_naive_ist(
            pd.to_datetime(d.get("Date", d.get("_Date")), errors="coerce")
        )
        d = d.dropna(subset=["_Date"])
        if d.empty:
            return stub

        if start_date or end_date:
            d = _filter_by_reporting_days(d, "_Date", start_date, end_date)
        if d.empty:
            # Platform IS loaded but has no data in this date window — show as loaded with 0
            stub["loaded"] = True
            return stub

        shipped_mask  = d[txn_col].astype(str).str.strip() == ship_val
        refund_mask   = d[txn_col].astype(str).str.strip() == refund_val
        qty           = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)

        total_units   = int(qty[shipped_mask].sum())
        total_returns = int(qty[refund_mask].sum())
        return_rate   = round(total_returns / total_units * 100, 1) if total_units > 0 else 0.0

        # Top SKU
        top_grp = d[shipped_mask].copy()
        top_grp["_qty"] = qty[shipped_mask]
        if sku_col in top_grp.columns and not top_grp.empty:
            top_sku = top_grp.groupby(sku_col)["_qty"].sum().idxmax()
        else:
            top_sku = ""

        # Monthly (last 6 months)
        d["_Month"] = d["_Date"].dt.to_period("M").astype(str)
        monthly_grp = (
            d.groupby(["_Month", txn_col])["Quantity"]
            .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum())
            .reset_index()
            .pivot_table(index="_Month", columns=txn_col, values="Quantity", fill_value=0)
            .reset_index()
        )
        monthly_grp.columns.name = None
        monthly_grp = monthly_grp.rename(columns={
            ship_val:  "shipments",
            refund_val: "refunds",
        })
        if "shipments" not in monthly_grp.columns:
            monthly_grp["shipments"] = 0
        if "refunds" not in monthly_grp.columns:
            monthly_grp["refunds"] = 0
        monthly_grp = monthly_grp.sort_values("_Month").tail(6)
        monthly_grp = monthly_grp.rename(columns={"_Month": "month"})
        monthly_grp["net"] = (
            monthly_grp["shipments"].astype(int) - monthly_grp["refunds"].astype(int)
        )
        keep_cols = [c for c in ["month", "shipments", "refunds", "net"] if c in monthly_grp.columns]
        monthly = monthly_grp[keep_cols].to_dict("records")

        # Trend direction (last month vs 3 months ago)
        trend_direction = "flat"
        ships = monthly_grp["shipments"].tolist() if "shipments" in monthly_grp.columns else []
        if len(ships) >= 3:
            last, three_ago = ships[-1], ships[-3]
            if three_ago > 0:
                change = (last - three_ago) / three_ago
                if change > 0.10:
                    trend_direction = "up"
                elif change < -0.10:
                    trend_direction = "down"

        trend_direction_net = "flat"
        nets = monthly_grp["net"].tolist() if "net" in monthly_grp.columns else []
        if len(nets) >= 3:
            last_n, three_ago_n = nets[-1], nets[-3]
            if three_ago_n != 0:
                change_n = (last_n - three_ago_n) / abs(three_ago_n)
            else:
                change_n = 0.0 if last_n == 0 else (1.0 if last_n > 0 else -1.0)
            if change_n > 0.10:
                trend_direction_net = "up"
            elif change_n < -0.10:
                trend_direction_net = "down"

        net_units = total_units - total_returns

        # By state
        by_state = []
        if "State" in d.columns:
            ship_st = (
                d[shipped_mask].copy()
                .assign(_qty=qty[shipped_mask].values)
                .groupby("State")["_qty"].sum()
            )
            ret_st = (
                d[refund_mask].copy()
                .assign(_qty=qty[refund_mask].values)
                .groupby("State")["_qty"].sum()
            )
            for stname in ship_st.index.union(ret_st.index):
                u = int(ship_st.get(stname, 0) or 0)
                r = int(ret_st.get(stname, 0) or 0)
                by_state.append({"state": stname, "units": u, "net_units": u - r})
            by_state.sort(key=lambda x: x["units"], reverse=True)

        return {
            "platform": platform_name, "loaded": True,
            "total_units": total_units, "total_returns": total_returns,
            "net_units": net_units,
            "return_rate": return_rate, "top_sku": str(top_sku),
            "trend_direction": trend_direction,
            "trend_direction_net": trend_direction_net,
            "monthly": monthly, "by_state": by_state,
        }
    except Exception as _exc:
        log.warning("_compute_platform_metrics failed for %s: %s", platform_name, _exc, exc_info=True)
        return stub


def _unified_platform_stub(platform_name: str, loaded: bool) -> dict:
    return {
        "platform": platform_name,
        "loaded": loaded,
        "total_units": 0,
        "total_returns": 0,
        "net_units": 0,
        "return_rate": 0.0,
        "top_sku": "",
        "trend_direction": "flat",
        "trend_direction_net": "flat",
        "monthly": [],
        "by_state": [],
    }


def _unified_platform_summary_one(
    s: pd.DataFrame,
    platform_name: str,
    raw_df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> dict:
    """
    One platform card from rows already filtered to that ``Source`` and reporting dates.
    ``raw_df`` gates ``loaded`` and supplies by_state when unified rows omit State.
    """
    loaded = not raw_df.empty
    if s.empty:
        out = _unified_platform_stub(platform_name, loaded)
        return out

    txn = s["Transaction Type"].astype(str).str.strip()
    qty = pd.to_numeric(s["Quantity"], errors="coerce").fillna(0)
    shipped_mask = txn == "Shipment"
    refund_mask = txn == "Refund"

    total_units = int(qty[shipped_mask].sum())
    total_returns = int(qty[refund_mask].sum())
    return_rate = round(total_returns / total_units * 100, 1) if total_units > 0 else 0.0

    eff = pd.to_numeric(s["Units_Effective"], errors="coerce").fillna(0) if "Units_Effective" in s.columns else None
    net_units = int(eff.sum()) if eff is not None else total_units - total_returns

    top_sku = ""
    if shipped_mask.any() and "Sku" in s.columns:
        _tg = s.loc[shipped_mask].copy()
        _tg["_qty"] = qty[shipped_mask]
        if not _tg.empty:
            top_sku = str(_tg.groupby("Sku")["_qty"].sum().idxmax())

    s["_Month"] = s["TxnDate"].dt.to_period("M").astype(str)
    monthly_grp = (
        s.assign(_q=qty)
        .groupby(["_Month", "Transaction Type"])["_q"]
        .sum()
        .reset_index()
        .pivot_table(index="_Month", columns="Transaction Type", values="_q", fill_value=0)
        .reset_index()
    )
    monthly_grp.columns.name = None
    monthly_grp = monthly_grp.rename(columns={"Shipment": "shipments", "Refund": "refunds"})
    if "shipments" not in monthly_grp.columns:
        monthly_grp["shipments"] = 0
    if "refunds" not in monthly_grp.columns:
        monthly_grp["refunds"] = 0
    monthly_grp = monthly_grp.sort_values("_Month").tail(6)
    monthly_grp = monthly_grp.rename(columns={"_Month": "month"})
    monthly_grp["net"] = (
        monthly_grp["shipments"].astype(int) - monthly_grp["refunds"].astype(int)
    )
    keep_cols = [c for c in ["month", "shipments", "refunds", "net"] if c in monthly_grp.columns]
    monthly = monthly_grp[keep_cols].to_dict("records")

    trend_direction = "flat"
    ships = monthly_grp["shipments"].tolist() if "shipments" in monthly_grp.columns else []
    if len(ships) >= 3:
        last, three_ago = ships[-1], ships[-3]
        if three_ago > 0:
            change = (last - three_ago) / three_ago
            if change > 0.10:
                trend_direction = "up"
            elif change < -0.10:
                trend_direction = "down"

    trend_direction_net = "flat"
    nets = monthly_grp["net"].tolist() if "net" in monthly_grp.columns else []
    if len(nets) >= 3:
        last_n, three_ago_n = nets[-1], nets[-3]
        if three_ago_n != 0:
            change_n = (last_n - three_ago_n) / abs(three_ago_n)
        else:
            change_n = 0.0 if last_n == 0 else (1.0 if last_n > 0 else -1.0)
        if change_n > 0.10:
            trend_direction_net = "up"
        elif change_n < -0.10:
            trend_direction_net = "down"

    by_state: List[dict] = []
    if "State" in s.columns:
        gross_by = (
            s.loc[shipped_mask]
            .assign(_qty=qty[shipped_mask])
            .groupby("State")["_qty"]
            .sum()
        )
        if eff is not None:
            net_by = s.assign(_eff=eff).groupby("State")["_eff"].sum()
        else:
            ret_by = (
                s.loc[refund_mask]
                .assign(_qty=qty[refund_mask])
                .groupby("State")["_qty"]
                .sum()
            )
            net_by = gross_by.subtract(ret_by, fill_value=0)
        for stname in gross_by.index.union(net_by.index):
            g = int(gross_by.get(stname, 0) or 0)
            n = int(net_by.get(stname, 0) or 0)
            by_state.append({"state": stname, "units": g, "net_units": n})
        by_state.sort(key=lambda x: x["units"], reverse=True)
    elif loaded and not raw_df.empty:
        # Unified rows usually omit State; keep heatmap from raw upload (same date window).
        kwargs = dict(start_date=start_date, end_date=end_date)
        if platform_name == "Amazon":
            mtr = raw_df.copy()
            if not mtr.empty and "Date" in mtr.columns:
                mtr["_Date"] = mtr["Date"]
            raw_stub = _compute_platform_metrics(
                mtr, platform_name, "SKU", "Transaction_Type", **kwargs
            )
        else:
            raw_stub = _compute_platform_metrics(
                raw_df, platform_name, "OMS_SKU", "TxnType", **kwargs
            )
        by_state = raw_stub.get("by_state") or []

    return {
        "platform": platform_name,
        "loaded": loaded,
        "total_units": total_units,
        "total_returns": total_returns,
        "net_units": net_units,
        "return_rate": return_rate,
        "top_sku": top_sku,
        "trend_direction": trend_direction,
        "trend_direction_net": trend_direction_net,
        "monthly": monthly,
        "by_state": by_state,
    }


def _platform_summaries_from_unified_bulk(
    sales_df: pd.DataFrame,
    mtr_df: pd.DataFrame,
    myntra_df: pd.DataFrame,
    meesho_df: pd.DataFrame,
    flipkart_df: pd.DataFrame,
    snapdeal_df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[dict]:
    """
    Build all five platform cards with one ``TxnDate`` normalisation and one groupby
    on ``Source`` (avoids five full scans of ``sales_df`` on the Intelligence dashboard).
    """
    order = [
        ("Amazon", mtr_df),
        ("Myntra", myntra_df),
        ("Meesho", meesho_df),
        ("Flipkart", flipkart_df),
        ("Snapdeal", snapdeal_df),
    ]
    if sales_df.empty or "Source" not in sales_df.columns:
        return [_unified_platform_stub(name, not raw.empty) for name, raw in order]

    prep = sales_df.copy()
    prep["TxnDate"] = txn_reporting_naive_ist(prep["TxnDate"])
    prep = prep.dropna(subset=["TxnDate"])
    if start_date or end_date:
        prep = _filter_by_reporting_days(prep, "TxnDate", start_date, end_date)
    if prep.empty:
        return [_unified_platform_stub(name, not raw.empty) for name, raw in order]

    src_key = prep["Source"].astype(str).str.strip()
    parts: dict[str, pd.DataFrame] = {}
    for k, sub in prep.groupby(src_key, sort=False):
        label = str(k).strip()
        if label and label.lower() != "nan":
            parts[label] = sub

    return [
        _unified_platform_summary_one(
            parts.get(name, pd.DataFrame()),
            name,
            raw,
            start_date,
            end_date,
        )
        for name, raw in order
    ]


def get_platform_summary(
    mtr_df: pd.DataFrame,
    myntra_df: pd.DataFrame,
    meesho_df: pd.DataFrame,
    flipkart_df: pd.DataFrame,
    snapdeal_df: Optional[pd.DataFrame] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sales_df: Optional[pd.DataFrame] = None,
) -> List[dict]:
    """Returns 5 platform summary dicts (always 5, even for unloaded platforms)."""
    _snapdeal = snapdeal_df if snapdeal_df is not None else pd.DataFrame()

    if sales_df is not None and not sales_df.empty:
        return _platform_summaries_from_unified_bulk(
            sales_df,
            mtr_df,
            myntra_df,
            meesho_df,
            flipkart_df,
            _snapdeal,
            start_date,
            end_date,
        )

    # Legacy: raw platform frames only (differs from unified export when dedup/mapping changes rows)
    mtr = mtr_df.copy() if not mtr_df.empty else mtr_df
    if not mtr.empty and "Date" in mtr.columns:
        mtr["_Date"] = mtr["Date"]

    kwargs = dict(start_date=start_date, end_date=end_date)
    return [
        _compute_platform_metrics(mtr,        "Amazon",   "SKU",     "Transaction_Type", **kwargs),
        _compute_platform_metrics(myntra_df,   "Myntra",   "OMS_SKU", "TxnType",          **kwargs),
        _compute_platform_metrics(meesho_df,   "Meesho",   "OMS_SKU", "TxnType",          **kwargs),
        _compute_platform_metrics(flipkart_df, "Flipkart", "OMS_SKU", "TxnType",          **kwargs),
        _compute_platform_metrics(_snapdeal,   "Snapdeal", "OMS_SKU", "TxnType",          **kwargs),
    ]


def get_anomalies(
    mtr_df: pd.DataFrame,
    myntra_df: pd.DataFrame,
    meesho_df: pd.DataFrame,
    flipkart_df: pd.DataFrame,
    snapdeal_df: Optional[pd.DataFrame] = None,
    inventory_df: pd.DataFrame = None,
    sales_df: pd.DataFrame = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    """Runs 5 anomaly rules. Returns list sorted critical → warning → info."""
    if inventory_df is None:
        inventory_df = pd.DataFrame()
    if sales_df is None:
        sales_df = pd.DataFrame()

    SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
    alerts: List[dict] = []

    _snapdeal = snapdeal_df if snapdeal_df is not None else pd.DataFrame()
    platform_dfs = [
        ("Amazon",   mtr_df,        "Transaction_Type", "Shipment", "Refund", "Date"),
        ("Myntra",   myntra_df,      "TxnType",          "Shipment", "Refund", "Date"),
        ("Meesho",   meesho_df,      "TxnType",          "Shipment", "Refund", "Date"),
        ("Flipkart", flipkart_df,    "TxnType",          "Shipment", "Refund", "Date"),
        ("Snapdeal", _snapdeal,      "TxnType",          "Shipment", "Refund", "Date"),
    ]

    use_unified_return_spike = (
        not sales_df.empty
        and all(
            c in sales_df.columns
            for c in ("Source", "TxnDate", "Transaction Type", "Quantity")
        )
    )

    if use_unified_return_spike:
        try:
            s = sales_df.copy()
            s["TxnDate"] = txn_reporting_naive_ist(s["TxnDate"])
            s = s.dropna(subset=["TxnDate"])
            if start_date or end_date:
                s = _filter_by_reporting_days(s, "TxnDate", start_date, end_date)
            for name in ("Amazon", "Myntra", "Meesho", "Flipkart", "Snapdeal"):
                sub = s[s["Source"].astype(str).str.strip() == name]
                if sub.empty:
                    continue
                txn = sub["Transaction Type"].astype(str).str.strip()
                qty = pd.to_numeric(sub["Quantity"], errors="coerce").fillna(0)
                shipped = float(qty[txn == "Shipment"].sum())
                returned = float(qty[txn == "Refund"].sum())
                rate = returned / shipped * 100 if shipped > 0 else 0.0
                if rate > 30:
                    alerts.append({
                        "type": "return_spike", "severity": "warning",
                        "platform": name,
                        "message": (
                            f"{name} return rate is {rate:.1f}% (threshold: 30%) — "
                            "investigate product quality or sizing"
                        ),
                        "sku": None,
                    })
        except Exception:
            log.exception("anomaly return_spike (unified sales)")

    for name, df, txn_col, ship_val, refund_val, date_col in platform_dfs:
        if df.empty:
            continue

        if not use_unified_return_spike:
            try:
                d = df.copy()
                qty = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)
                shipped = qty[d[txn_col].astype(str).str.strip() == ship_val].sum()
                returned = qty[d[txn_col].astype(str).str.strip() == refund_val].sum()

                # Rule 1: Return spike (raw platform frames — only when no unified sales yet)
                rate = returned / shipped * 100 if shipped > 0 else 0
                if rate > 30:
                    alerts.append({
                        "type": "return_spike", "severity": "warning",
                        "platform": name,
                        "message": (
                            f"{name} return rate is {rate:.1f}% (threshold: 30%) — "
                            "investigate product quality or sizing"
                        ),
                        "sku": None,
                    })
            except Exception:
                pass

        try:
            # Rule 2: Zero sales in last 30 days
            d2 = df.copy()
            d2["_date"] = pd.to_datetime(d2[date_col], errors="coerce")
            d2 = d2.dropna(subset=["_date"])
            if not d2.empty:
                max_date = d2["_date"].max()
                cutoff   = max_date - pd.Timedelta(days=30)
                recent   = d2[d2["_date"] >= cutoff]
                recent_shipped = recent[recent[txn_col].astype(str).str.strip() == ship_val]
                if len(recent_shipped) == 0:
                    alerts.append({
                        "type": "zero_sales", "severity": "warning",
                        "platform": name,
                        "message": f"{name} has no shipments in the last 30 days — check listing status",
                        "sku": None,
                    })
        except Exception:
            pass

        try:
            # Rule 4: Sales drop >50% month-over-month
            d3 = df.copy()
            d3["_date"] = pd.to_datetime(d3[date_col], errors="coerce")
            d3 = d3.dropna(subset=["_date"])
            d3["_month"] = d3["_date"].dt.to_period("M")
            qty3 = pd.to_numeric(d3["Quantity"], errors="coerce").fillna(0)
            monthly_shipped = (
                d3[d3[txn_col].astype(str).str.strip() == ship_val]
                .assign(_qty=qty3[d3[txn_col].astype(str).str.strip() == ship_val].values)
                .groupby("_month")["_qty"].sum()
                .sort_index()
            )
            if len(monthly_shipped) >= 2:
                last_m  = monthly_shipped.iloc[-1]
                prev_m  = monthly_shipped.iloc[-2]
                if prev_m > 0 and last_m < prev_m * 0.5:
                    pct = int((1 - last_m / prev_m) * 100)
                    alerts.append({
                        "type": "sales_drop", "severity": "warning",
                        "platform": name,
                        "message": (
                            f"{name} sales dropped {pct}% month-over-month "
                            f"({monthly_shipped.index[-2]}: {int(prev_m):,} → "
                            f"{monthly_shipped.index[-1]}: {int(last_m):,} units)"
                        ),
                        "sku": None,
                    })
        except Exception:
            pass

    # Rule 3: Stockout (cap at 5)
    try:
        if not inventory_df.empty and not sales_df.empty:
            inv = inventory_df.copy()
            inv_col = "Total_Inventory" if "Total_Inventory" in inv.columns else inv.columns[1]
            inv["_inv"] = pd.to_numeric(inv[inv_col], errors="coerce").fillna(0)
            zero_inv = inv[inv["_inv"] <= 0]["OMS_SKU"].tolist()

            if zero_inv and "Sku" in sales_df.columns:
                s = sales_df.copy()
                s["TxnDate"] = pd.to_datetime(s["TxnDate"], errors="coerce")
                max_d    = s["TxnDate"].max()
                cutoff90 = max_d - pd.Timedelta(days=90)
                recent_s = s[(s["TxnDate"] >= cutoff90) & (s["Transaction Type"] == "Shipment")]
                ads_map  = (
                    recent_s.groupby("Sku")["Quantity"]
                    .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum() / 90)
                    .to_dict()
                )
                stockouts = [
                    sku for sku in zero_inv
                    if ads_map.get(sku, 0) > 0
                ][:5]
                for sku in stockouts:
                    ads = ads_map[sku]
                    alerts.append({
                        "type": "stockout", "severity": "critical",
                        "platform": "All",
                        "message": (
                            f"SKU {sku} has 0 inventory but avg daily sales of "
                            f"{ads:.1f} units — restock urgently"
                        ),
                        "sku": sku,
                    })
    except Exception:
        pass

    # Rule 5: Single platform loaded
    try:
        loaded_count = sum(1 for _, df, *_ in platform_dfs if not df.empty)
        if loaded_count == 1:
            loaded_name = next(name for name, df, *_ in platform_dfs if not df.empty)
            others = [n for n, df, *_ in platform_dfs if df.empty]
            alerts.append({
                "type": "single_platform", "severity": "info",
                "platform": "All",
                "message": (
                    f"Only {loaded_name} is loaded. Upload "
                    f"{', '.join(others)} data for cross-platform insights"
                ),
                "sku": None,
            })
    except Exception:
        pass

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 9))
    return alerts
