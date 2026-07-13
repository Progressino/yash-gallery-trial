"""
Multi-platform sales match export for PO quarterly reconciliation.

Builds SKU × Platform × Quarter pivots from unified ``sales_df`` using the same
Sold(Gross) / Net signing and combo listing-only attribution as quarterly history,
so analysts can see *which marketplace* drives File vs App gaps.
"""
from __future__ import annotations

import io
import logging
from typing import Dict, Optional

import pandas as pd

from .po_engine import get_indian_fy_quarter, get_parent_sku, quarter_col_name
from .po_quarterly_fast import (
    _NET_TXN_TYPES,
    _ordered_q_cols,
    _qty_signs_for_demand_basis,
    _quarter_seq,
    normalize_quarterly_demand_basis,
)

logger = logging.getLogger(__name__)

# Display order for marketplace columns (matches operator mental model / File).
PLATFORM_DISPLAY_ORDER = (
    "Amazon",
    "Flipkart",
    "Myntra",
    "Meesho",
    "Snapdeal",
    "Other",
)

_SOURCE_TO_DISPLAY = (
    ("amazon", "Amazon"),
    ("flipkart", "Flipkart"),
    ("myntra", "Myntra"),
    ("meesho", "Meesho"),
    ("snapdeal", "Snapdeal"),
)


def normalize_platform_display(src: object) -> str:
    s = str(src or "").strip().lower()
    if not s or s in ("nan", "none", "nat", ""):
        return "Other"
    for needle, label in _SOURCE_TO_DISPLAY:
        if needle in s:
            return label
    if s in ("return_sheet", "return sheet"):
        return "Other"
    return "Other"


def _quarter_label_map(n_quarters: int) -> dict[tuple[int, int], str]:
    return {(fy, qn): quarter_col_name(fy, qn) for fy, qn in _quarter_seq(n_quarters)}


def _window_bounds(n_quarters: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive start (first day of oldest quarter) → end of today."""
    seq = _quarter_seq(n_quarters)
    fy0, q0 = seq[0]
    # Match quarter_col_name: Q1–Q3 calendar year is FY-1; Q4 is FY.
    cal_year = fy0 - 1 if q0 in (1, 2, 3) else fy0
    month = {1: 4, 2: 7, 3: 10, 4: 1}[q0]
    start = pd.Timestamp(year=cal_year, month=month, day=1)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return start, end


def build_platform_match_frames(
    sales_df: pd.DataFrame,
    sku_mapping: Optional[Dict[str, str]] = None,
    *,
    n_quarters: int = 8,
    demand_basis: str = "Sold",
    group_by_parent: bool = False,
    combo_sku_map: Optional[Dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Return analysis-ready frames:

    - ``long``: OMS_SKU, Quarter, Platform, Units
    - ``wide_total``: OMS_SKU + one column per platform + Total (all quarters)
    - ``wide_by_quarter``: OMS_SKU + ``{Platform}|{Quarter}`` columns + per-quarter Total
    - ``summary``: Quarter × Platform totals
    - ``notes``: methodology rows
    """
    basis = normalize_quarterly_demand_basis(demand_basis)
    n_quarters = max(1, min(int(n_quarters or 8), 16))
    q_cols = _ordered_q_cols(n_quarters)
    q_label_map = _quarter_label_map(n_quarters)
    start_ts, end_ts = _window_bounds(n_quarters)

    empty_long = pd.DataFrame(columns=["OMS_SKU", "Quarter", "Platform", "Units"])
    if sales_df is None or getattr(sales_df, "empty", True):
        return _empty_bundle(q_cols, basis, n_quarters, group_by_parent, empty_long)

    work = sales_df.copy()
    # Drop combo-fan component copies — keep listing identity (matches quarterly File).
    if "_Combo_Fan" in work.columns:
        fan = work["_Combo_Fan"].fillna(False).astype(bool)
        if fan.any():
            work = work.loc[~fan].copy()

    colmap = {c.lower(): c for c in work.columns}
    sku_c = colmap.get("sku") or colmap.get("oms_sku")
    date_c = colmap.get("txndate") or colmap.get("date")
    qty_c = colmap.get("quantity") or colmap.get("qty")
    txn_c = colmap.get("transaction type") or colmap.get("txntype")
    src_c = colmap.get("source") or colmap.get("_source")
    if not all([sku_c, date_c, qty_c, txn_c]):
        logger.warning("platform_match: sales_df missing required columns")
        return _empty_bundle(q_cols, basis, n_quarters, group_by_parent, empty_long)

    slim = pd.DataFrame(
        {
            "SKU": work[sku_c].astype(str).str.strip(),
            "Date": pd.to_datetime(work[date_c], errors="coerce"),
            "Qty": pd.to_numeric(work[qty_c], errors="coerce").fillna(0),
            "TxnType": work[txn_c].astype(str).str.strip(),
            "Source": work[src_c].astype(str) if src_c else "",
        }
    )
    txn_lower = slim["TxnType"].str.lower()
    slim = slim[txn_lower.isin(_NET_TXN_TYPES)].copy()
    if slim.empty:
        return _empty_bundle(q_cols, basis, n_quarters, group_by_parent, empty_long)

    txn_lower = slim["TxnType"].str.lower()
    is_amz = slim["Source"].astype(str).str.strip().str.lower().str.contains("amazon", na=False)
    slim["Qty"] = _qty_signs_for_demand_basis(
        txn_lower, slim["Qty"], demand_basis=basis, is_amazon=is_amz
    )
    slim = slim.dropna(subset=["Date"])
    slim = slim[(slim["Date"] >= start_ts) & (slim["Date"] <= end_ts)]
    slim = slim[slim["Qty"] != 0]
    slim = slim[slim["SKU"].astype(str).str.len() > 0]
    if slim.empty:
        return _empty_bundle(q_cols, basis, n_quarters, group_by_parent, empty_long)

    from .combo_sku_map import explode_sku_qty_dataframe, resolve_active_combo_sku_map

    combo = resolve_active_combo_sku_map(combo_sku_map) if combo_sku_map is not None else resolve_active_combo_sku_map()
    # Preserve Source through explode (qty fan stays on same listing row for listing-only).
    slim = explode_sku_qty_dataframe(
        slim,
        sku_col="SKU",
        qty_col="Qty",
        sku_mapping=sku_mapping,
        combo_map=combo,
        strip_pl=False,
        retain_combo_listings=False,
        attribute_combo_to_listing_only=True,
    )
    slim = slim[slim["SKU"].astype(str).str.len() > 0]
    if group_by_parent:
        slim["SKU"] = slim["SKU"].map(lambda s: get_parent_sku(s))

    slim["Platform"] = slim["Source"].map(normalize_platform_display)
    fy_q = slim["Date"].map(lambda d: get_indian_fy_quarter(pd.Timestamp(d)))
    slim["Quarter"] = [q_label_map.get((fy, qn), "") for fy, qn in fy_q]
    slim = slim[slim["Quarter"].isin(q_cols)]
    if slim.empty:
        return _empty_bundle(q_cols, basis, n_quarters, group_by_parent, empty_long)

    long = (
        slim.groupby(["SKU", "Quarter", "Platform"], as_index=False)["Qty"]
        .sum()
        .rename(columns={"SKU": "OMS_SKU", "Qty": "Units"})
    )
    long["Units"] = long["Units"].round(0).astype(int)
    long = long.sort_values(["OMS_SKU", "Quarter", "Platform"]).reset_index(drop=True)

    # Wide totals across the full window
    tot = (
        long.groupby(["OMS_SKU", "Platform"], as_index=False)["Units"]
        .sum()
        .pivot(index="OMS_SKU", columns="Platform", values="Units")
        .reindex(columns=list(PLATFORM_DISPLAY_ORDER))
        .fillna(0)
        .astype(int)
    )
    tot["Total"] = tot.sum(axis=1)
    wide_total = tot.reset_index().sort_values("Total", ascending=False).reset_index(drop=True)

    # Wide by quarter: Platform|Quarter columns (analyst can filter one quarter in Excel)
    by_q = (
        long.assign(Col=long["Platform"] + "|" + long["Quarter"])
        .pivot_table(index="OMS_SKU", columns="Col", values="Units", aggfunc="sum", fill_value=0)
        .astype(int)
    )
    ordered_cols: list[str] = []
    for q in q_cols:
        for plat in PLATFORM_DISPLAY_ORDER:
            c = f"{plat}|{q}"
            if c not in by_q.columns:
                by_q[c] = 0
            ordered_cols.append(c)
        tot_c = f"Total|{q}"
        plat_cols = [f"{p}|{q}" for p in PLATFORM_DISPLAY_ORDER]
        by_q[tot_c] = by_q[plat_cols].sum(axis=1)
        ordered_cols.append(tot_c)
    wide_by_quarter = by_q[ordered_cols].reset_index()

    summary = (
        long.groupby(["Quarter", "Platform"], as_index=False)["Units"]
        .sum()
        .pivot(index="Quarter", columns="Platform", values="Units")
        .reindex(index=q_cols, columns=list(PLATFORM_DISPLAY_ORDER))
        .fillna(0)
        .astype(int)
    )
    summary["Total"] = summary.sum(axis=1)
    summary = summary.reset_index()

    notes = pd.DataFrame(
        [
            {"Key": "demand_basis", "Value": "Sold (Gross)" if basis == "sold" else "Net"},
            {"Key": "n_quarters", "Value": str(n_quarters)},
            {"Key": "group_by_parent", "Value": str(bool(group_by_parent))},
            {"Key": "window_start", "Value": str(start_ts.date())},
            {"Key": "window_end", "Value": str(end_ts.date())},
            {
                "Key": "method",
                "Value": (
                    "Unified sales_df by Source; Sold=shipments only; "
                    "Net=ship−returns/cancels (Amazon cancel rules); "
                    "combo listings stay on listing SKU (no component fan)."
                ),
            },
            {
                "Key": "how_to_use",
                "Value": (
                    "Filter SKU_Platform_Quarter to one Quarter (e.g. Jan-Mar 2025). "
                    "Compare platform columns to File/Deepdive by marketplace. "
                    "SKU_Platform_Total is the sum across all quarters in the window. "
                    "Quarter_Platform_Summary shows marketplace mix drift over time."
                ),
            },
            {
                "Key": "platforms",
                "Value": ", ".join(PLATFORM_DISPLAY_ORDER),
            },
            {
                "Key": "sku_rows_long",
                "Value": str(len(long)),
            },
            {
                "Key": "distinct_skus",
                "Value": str(long["OMS_SKU"].nunique()),
            },
        ]
    )

    return {
        "long": long,
        "wide_total": wide_total,
        "wide_by_quarter": wide_by_quarter,
        "summary": summary,
        "notes": notes,
    }


def _empty_bundle(
    q_cols: list[str],
    basis: str,
    n_quarters: int,
    group_by_parent: bool,
    empty_long: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    plats = list(PLATFORM_DISPLAY_ORDER)
    wide_total = pd.DataFrame(columns=["OMS_SKU", *plats, "Total"])
    cols = ["OMS_SKU"]
    for q in q_cols:
        for p in plats:
            cols.append(f"{p}|{q}")
        cols.append(f"Total|{q}")
    wide_by_quarter = pd.DataFrame(columns=cols)
    summary = pd.DataFrame(columns=["Quarter", *plats, "Total"])
    notes = pd.DataFrame(
        [
            {"Key": "demand_basis", "Value": "Sold (Gross)" if basis == "sold" else "Net"},
            {"Key": "n_quarters", "Value": str(n_quarters)},
            {"Key": "group_by_parent", "Value": str(bool(group_by_parent))},
            {"Key": "status", "Value": "empty — no sales rows in window"},
        ]
    )
    return {
        "long": empty_long,
        "wide_total": wide_total,
        "wide_by_quarter": wide_by_quarter,
        "summary": summary,
        "notes": notes,
    }


def platform_match_to_xlsx_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    """Write multi-sheet workbook for Excel analysis."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        frames["notes"].to_excel(writer, sheet_name="Readme", index=False)
        frames["summary"].to_excel(writer, sheet_name="Quarter_Platform_Summary", index=False)
        frames["wide_total"].to_excel(writer, sheet_name="SKU_Platform_Total", index=False)
        frames["long"].to_excel(writer, sheet_name="SKU_Platform_Quarter", index=False)
        # Wide-by-quarter can be very wide — still useful; cap rows for Excel comfort
        wq = frames["wide_by_quarter"]
        if len(wq) > 100_000:
            wq = wq.head(100_000)
        wq.to_excel(writer, sheet_name="SKU_x_Platform_x_Quarter", index=False)
    return buf.getvalue()


def platform_match_to_csv_zip_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    """Fallback when xlsx is unavailable — zip of CSVs."""
    import zipfile

    buf = io.BytesIO()
    names = {
        "notes": "00_readme.csv",
        "summary": "01_quarter_platform_summary.csv",
        "wide_total": "02_sku_platform_total.csv",
        "long": "03_sku_platform_quarter.csv",
        "wide_by_quarter": "04_sku_x_platform_x_quarter.csv",
    }
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, fname in names.items():
            df = frames.get(key)
            if df is None:
                continue
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            zf.writestr(fname, csv_buf.getvalue())
    return buf.getvalue()


def build_platform_match_export_bytes(
    sales_df: pd.DataFrame,
    sku_mapping: Optional[Dict[str, str]] = None,
    *,
    n_quarters: int = 8,
    demand_basis: str = "Sold",
    group_by_parent: bool = False,
    fmt: str = "xlsx",
) -> tuple[bytes, str, str]:
    """
    Returns (body, media_type, filename).
    ``fmt``: ``xlsx`` (default) or ``zip``.
    """
    frames = build_platform_match_frames(
        sales_df,
        sku_mapping,
        n_quarters=n_quarters,
        demand_basis=demand_basis,
        group_by_parent=group_by_parent,
    )
    basis = normalize_quarterly_demand_basis(demand_basis)
    basis_tag = "sold" if basis == "sold" else "net"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fmt_l = str(fmt or "xlsx").strip().lower()
    if fmt_l in ("zip", "csv", "csvzip"):
        body = platform_match_to_csv_zip_bytes(frames)
        return (
            body,
            "application/zip",
            f"platform_sales_match_{basis_tag}_{n_quarters}q_{today}.zip",
        )
    try:
        body = platform_match_to_xlsx_bytes(frames)
    except Exception:
        logger.exception("xlsx platform match failed — falling back to zip")
        body = platform_match_to_csv_zip_bytes(frames)
        return (
            body,
            "application/zip",
            f"platform_sales_match_{basis_tag}_{n_quarters}q_{today}.zip",
        )
    return (
        body,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        f"platform_sales_match_{basis_tag}_{n_quarters}q_{today}.xlsx",
    )
