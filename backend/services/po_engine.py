"""
PO Engine — extracted 1-for-1 from app.py.
calculate_quarterly_history + calculate_po_base.
"""
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

import re

from .helpers import (
    map_to_oms_sku,
    get_parent_sku,
    clean_sku,
    normalize_id_token_for_mapping,
    collapse_duplicate_trailing_size_suffix,
)
from .myntra import myntra_to_sales_rows

# Strip "PL" infix in Amazon seller SKUs (e.g. 1001PLYKBEIGE-3XL → 1001YKBEIGE-3XL)
# Must match the same pattern used in inventory.py _resolve_amz_sku.
_PL_RE = re.compile(r'^(\d+)PL(YK)', re.I)
_SIZE_SUFFIX_RE = re.compile(r"(?:-|_)?(XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|5XL|6XL)$", re.I)


def round_po_pack(qty: float) -> int:
    """Round PO quantity up to pack size — 5 for small lots, 10 when qty >= 10."""
    q = max(0.0, float(qty or 0))
    if q <= 0:
        return 0
    pack = 10.0 if q >= 10.0 else 5.0
    return int(np.floor(np.ceil(q / pack) * pack))


def _lead_time_gate_uses_entered(lead_time: int) -> bool:
    return int(lead_time or 0) > 0


def _lead_time_column_fallback(lead_time: int) -> int:
    """Default ``Lead_Time_Days`` when sheet merge has no value (45d when gate uses sheet)."""
    entered = int(lead_time or 0)
    return entered if entered > 0 else 45


def _lead_time_release_gate_days(po_df: pd.DataFrame, lead_time: int) -> np.ndarray:
    """Per-row lead days for PO release gate: entered param when >0, else sheet ``Lead_Time_Days``."""
    if _lead_time_gate_uses_entered(lead_time):
        return np.full(len(po_df), float(int(lead_time)))
    sheet_lt = pd.to_numeric(
        po_df.get("Lead_Time_Days", pd.Series(_lead_time_column_fallback(lead_time), index=po_df.index)),
        errors="coerce",
    ).fillna(_lead_time_column_fallback(lead_time)).clip(lower=1, upper=730)
    return sheet_lt.to_numpy(dtype=float)


def _lead_time_gate_block_message(uses_entered: bool) -> str:
    if uses_entered:
        return "Projected cover exceeds entered lead time — no PO until cover is within lead days"
    return "Projected cover exceeds sheet lead time — no PO until cover is within lead days"


def canonical_oms_key(raw, sku_mapping: Optional[Dict[str, str]] = None) -> str:
    """Module-level twin of the nested ``_canonical_oms_key`` used in ``calculate_po_base``.

    Exposed so other endpoints (e.g. the per-SKU inventory-history drill-down)
    can produce the exact same canonical key the engine uses to join everything.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    t = normalize_id_token_for_mapping(str(raw).strip())
    t = clean_sku(t or raw)
    if not t:
        t = str(raw).strip().upper()
    return collapse_duplicate_trailing_size_suffix(
        _strip_pl(str(t).strip(), sku_mapping or {})
    )


_YKN_TYPO_RE = re.compile(r"YKN", re.I)


def _ykn_typo_canonical(sku: str) -> str:
    """1059YKNMUSTARD → 1059YKMUSTARD (common marketplace listing typo)."""
    return _YKN_TYPO_RE.sub("YK", str(sku).strip().upper(), count=1)


def _strip_pl(sku: str, mapping: Dict[str, str]) -> str:
    """Map an Amazon seller SKU to OMS SKU, stripping PL infix if needed."""
    raw = str(sku).strip().upper()
    stripped = _PL_RE.sub(r"\1\2", raw)
    if stripped in mapping:
        return mapping[stripped]
    if raw in mapping:
        return mapping[raw]
    fixed = _ykn_typo_canonical(stripped)
    if fixed != stripped:
        return mapping.get(fixed, mapping.get(stripped, fixed))
    return stripped


def get_indian_fy_quarter(date: pd.Timestamp) -> tuple:
    m = date.month
    y = date.year
    if m >= 4:
        fy = y + 1
        q  = 1 if m <= 6 else 2 if m <= 9 else 3
    else:
        fy = y
        q  = 4
    return fy, q


# ASCII hyphen only — en/em dashes break in Excel/CSV when UTF-8 is misread as Latin-1.
_Q_LABELS = {1: "Apr-Jun", 2: "Jul-Sep", 3: "Oct-Dec", 4: "Jan-Mar"}


def quarter_col_name(fy: int, q: int) -> str:
    cal_year = fy - 1 if q in (1, 2, 3) else fy
    return f"{_Q_LABELS[q]} {cal_year}"


def _platform_shipment_history_part(
    df: pd.DataFrame,
    sku_mapping: Optional[Dict[str, str]],
    *,
    strip_pl: bool = False,
    canonical_oms: bool = False,
) -> pd.DataFrame:
    """Normalize one platform frame to [SKU, Date, Qty, TxnType] shipment rows."""
    if df is None or df.empty:
        return pd.DataFrame()
    sku_col = next((c for c in df.columns if c in ["OMS_SKU", "SKU", "Sku"]), None)
    date_col = next((c for c in df.columns if c in ["Date", "TxnDate"]), None)
    qty_col = next((c for c in df.columns if c in ["Quantity", "Qty"]), None)
    txn_col = next(
        (c for c in df.columns if c in ["Transaction_Type", "Transaction Type", "TxnType"]),
        None,
    )
    if not sku_col or not date_col or not qty_col:
        return pd.DataFrame()
    tmp = df[[sku_col, date_col, qty_col]].copy()
    tmp.columns = ["SKU", "Date", "Qty"]
    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
    tmp["Qty"] = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
    tmp["TxnType"] = df[txn_col].values if txn_col else "Shipment"
    if strip_pl:
        if sku_mapping:
            tmp["SKU"] = tmp["SKU"].apply(lambda x: _strip_pl(x, sku_mapping))
        else:
            tmp["SKU"] = tmp["SKU"].apply(
                lambda x: _PL_RE.sub(r"\1\2", str(x).strip().upper())
            )
        if sku_mapping:
            _u = tmp["SKU"].unique()
            _c = {s: canonical_oms_key(s, sku_mapping) for s in _u}
            tmp["SKU"] = tmp["SKU"].map(_c)
    elif canonical_oms and sku_mapping:
        _u = tmp["SKU"].unique()
        _c = {s: canonical_oms_key(s, sku_mapping) for s in _u}
        tmp["SKU"] = tmp["SKU"].map(_c)
    return tmp.dropna(subset=["Date"])


def _sales_shipment_history_part(sales_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df is None or sales_df.empty or "Sku" not in sales_df.columns:
        return pd.DataFrame()
    tmp = sales_df[["Sku", "TxnDate", "Quantity", "Transaction Type"]].copy()
    tmp.columns = ["SKU", "Date", "Qty", "TxnType"]
    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
    tmp["Qty"] = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
    return tmp.dropna(subset=["Date"])


def _collect_platform_shipment_history_parts(
    mtr_df: Optional[pd.DataFrame],
    myntra_df: Optional[pd.DataFrame],
    meesho_df: Optional[pd.DataFrame],
    flipkart_df: Optional[pd.DataFrame],
    snapdeal_df: Optional[pd.DataFrame],
    sku_mapping: Optional[Dict[str, str]],
) -> list[pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for raw, strip_pl, canon in (
        (mtr_df, True, False),
        (myntra_df, False, True),
        (meesho_df, False, True),
        (flipkart_df, False, True),
        (snapdeal_df, False, True),
    ):
        chunk = _platform_shipment_history_part(
            raw, sku_mapping, strip_pl=strip_pl, canonical_oms=canon
        )
        if not chunk.empty:
            parts.append(chunk)
    return parts


def _merge_sales_and_platform_history_parts(
    sales_part: pd.DataFrame,
    plat_parts: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """
    Platform bulk/Tier-3 frames are the source of truth for deep history.
    Unified ``sales_df`` often spans the same calendar range but omits SKU-days;
    only append sales rows whose (SKU, day) keys are not already on the platform side.
    """
    if not plat_parts:
        return [sales_part] if sales_part is not None and not sales_part.empty else []
    plat_hist = pd.concat(plat_parts, ignore_index=True)
    if sales_part is None or sales_part.empty:
        return [plat_hist]
    plat_hist = plat_hist.copy()
    plat_hist["_day"] = pd.to_datetime(plat_hist["Date"], errors="coerce").dt.normalize()

    sales_part = sales_part.copy()
    sales_part["_day"] = pd.to_datetime(sales_part["Date"], errors="coerce").dt.normalize()
    sales_part["_skukey"] = sales_part["SKU"].astype(str).str.strip()

    plat_keys = (
        plat_hist[["SKU", "_day"]]
        .assign(_skukey=lambda d: d["SKU"].astype(str).str.strip())
        [["_skukey", "_day"]]
        .drop_duplicates()
    )
    plat_keys["_in_plat"] = True

    # Vectorized hash-join membership check (replaces a per-row .apply, which was
    # the dominant cost for large multi-year sales frames).
    merged = sales_part.merge(plat_keys, on=["_skukey", "_day"], how="left")
    extra_sales = sales_part[merged["_in_plat"].isna().to_numpy()]

    drop_cols = ["_day", "_skukey"]
    plat_out = plat_hist.drop(columns=["_day"], errors="ignore")
    if extra_sales.empty:
        return [plat_out]
    return [plat_out, extra_sales.drop(columns=drop_cols, errors="ignore")]


def _mtr_to_sales_df_local(mtr_df, sku_mapping, group_by_parent=False):
    if mtr_df.empty:
        return pd.DataFrame()
    m = mtr_df[["Date", "SKU", "Transaction_Type", "Quantity"]].copy()
    m = m.rename(columns={"Date": "TxnDate", "SKU": "Sku", "Transaction_Type": "Transaction Type"})
    m["TxnDate"]  = pd.to_datetime(m["TxnDate"], errors="coerce")
    m["Quantity"] = pd.to_numeric(m["Quantity"], errors="coerce").fillna(0)
    m = m.dropna(subset=["TxnDate"])
    m["Sku"] = m["Sku"].apply(lambda x: _strip_pl(x, sku_mapping))
    if group_by_parent:
        m["Sku"] = m["Sku"].apply(get_parent_sku)
    m["Units_Effective"] = np.where(
        m["Transaction Type"] == "Refund", -m["Quantity"],
        np.where(m["Transaction Type"] == "Cancel", -m["Quantity"], m["Quantity"])
    )
    return m[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective"]]


def calculate_quarterly_history(
    sales_df: pd.DataFrame,
    mtr_df: Optional[pd.DataFrame] = None,
    myntra_df: Optional[pd.DataFrame] = None,
    meesho_df: Optional[pd.DataFrame] = None,
    flipkart_df: Optional[pd.DataFrame] = None,
    snapdeal_df: Optional[pd.DataFrame] = None,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
    n_quarters: int = 8,
    retain_bundled_listing_skus: Optional[set[str]] = None,
) -> pd.DataFrame:
    sales_part = _sales_shipment_history_part(sales_df)
    plat_parts = _collect_platform_shipment_history_parts(
        mtr_df, myntra_df, meesho_df, flipkart_df, snapdeal_df, sku_mapping
    )
    parts = _merge_sales_and_platform_history_parts(sales_part, plat_parts)

    if not parts:
        return pd.DataFrame()

    hist = pd.concat(parts, ignore_index=True)
    _txn = hist["TxnType"].astype(str).str.strip().str.lower()
    hist = hist[_txn.eq("shipment")]
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date"])
    hist["Qty"] = pd.to_numeric(hist["Qty"], errors="coerce").fillna(0)
    hist = hist[hist["Qty"] > 0]
    if hist.empty:
        return pd.DataFrame()

    # Same canonical OMS keys as calculate_po_base (mapping + PL strip + clean_sku).
    _uniq_hist = hist["SKU"].unique()
    _canon_cache = {s: canonical_oms_key(s, sku_mapping) for s in _uniq_hist}
    hist["SKU"] = hist["SKU"].map(_canon_cache).fillna("")
    hist = hist[hist["SKU"].str.len() > 0]
    if hist.empty:
        return pd.DataFrame()

    if group_by_parent:
        _uniq_par = hist["SKU"].unique()
        _par_cache = {s: get_parent_sku(s) for s in _uniq_par}
        hist["SKU"] = hist["SKU"].map(_par_cache)

    # Vectorized FY/Quarter computation (avoids slow row-by-row .apply)
    _month = hist["Date"].dt.month
    _year  = hist["Date"].dt.year
    hist["FY"] = np.where(_month >= 4, _year + 1, _year)
    hist["QN"] = np.select(
        [(_month >= 4) & (_month <= 6),
         (_month >= 7) & (_month <= 9),
         _month >= 10],
        [1, 2, 3],
        default=4,
    )

    today         = pd.Timestamp.today()
    cur_fy, cur_q = get_indian_fy_quarter(today)
    quarter_seq   = []
    fy_i, q_i     = cur_fy, cur_q
    for _ in range(n_quarters):
        quarter_seq.append((fy_i, q_i))
        q_i -= 1
        if q_i == 0:
            q_i = 4
            fy_i -= 1
    quarter_seq = list(reversed(quarter_seq))

    # Build quarter label via a lookup map (avoids row-by-row apply)
    _unique_fq = hist[["FY", "QN"]].drop_duplicates()
    _q_label_map = {
        (int(r.FY), int(r.QN)): quarter_col_name(int(r.FY), int(r.QN))
        for r in _unique_fq.itertuples(index=False)
    }
    hist["col"] = [_q_label_map[(int(fy), int(qn))]
                   for fy, qn in zip(hist["FY"], hist["QN"])]
    grp   = hist.groupby(["SKU", "col"])["Qty"].sum().reset_index()
    pivot = grp.pivot_table(index="SKU", columns="col", values="Qty",
                            aggfunc="sum", fill_value=0).reset_index()
    pivot = pivot.rename(columns={"SKU": "OMS_SKU"})
    pivot.columns.name = None

    ordered_q_cols = []
    for fy_j, q_j in quarter_seq:
        col = quarter_col_name(fy_j, q_j)
        ordered_q_cols.append(col)
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["OMS_SKU"] + ordered_q_cols]

    last4 = ordered_q_cols[-4:]
    pivot["Avg_Monthly"] = (pivot[last4].mean(axis=1) / 3).round(1)

    cutoff_90 = today - timedelta(days=90)
    r90 = hist[hist["Date"] >= cutoff_90].groupby("SKU")["Qty"].sum().reset_index()
    r90.columns = ["OMS_SKU", "Units_90d"]
    pivot = pivot.merge(r90, on="OMS_SKU", how="left").fillna({"Units_90d": 0})
    pivot["ADS"] = (pivot["Units_90d"] / 90).round(3)

    cutoff_30 = today - timedelta(days=30)
    r30 = hist[hist["Date"] >= cutoff_30].groupby("SKU")["Qty"].sum().reset_index()
    r30.columns = ["OMS_SKU", "Units_30d"]
    pivot = pivot.merge(r30, on="OMS_SKU", how="left").fillna({"Units_30d": 0})

    f30 = (
        hist[hist["Date"] >= cutoff_30]
        .assign(_day=lambda d: d["Date"].dt.normalize())
        .groupby("SKU")["_day"].nunique()
        .reset_index()
    )
    f30.columns = ["OMS_SKU", "Freq_30d"]
    pivot = pivot.merge(f30, on="OMS_SKU", how="left").fillna({"Freq_30d": 0})
    pivot["Freq_30d"] = pivot["Freq_30d"].astype(int)

    # Vectorized status (avoids per-row Python call overhead)
    _ads = pivot["ADS"]
    pivot["Status"] = np.select(
        [_ads >= 1.0, _ads >= 0.33, _ads >= 0.10],
        ["Fast Moving", "Moderate", "Slow Selling"],
        default="Not Moving",
    )
    pivot["Units_90d"] = pivot["Units_90d"].astype(int)
    pivot["Units_30d"] = pivot["Units_30d"].astype(int)

    if not group_by_parent:
        from .existing_po import fan_out_bundled_listing_metrics

        q_metric_cols = ordered_q_cols + ["Avg_Monthly", "Units_90d", "Units_30d", "Freq_30d", "ADS"]
        # Always fan out for quarterly history — individual sizes need proportional
        # sales history regardless of whether inventory lists a bundled SKU.
        fan = fan_out_bundled_listing_metrics(
            pivot,
            q_metric_cols,
        )
        if fan is not None and not fan.empty:
            pivot = pivot.merge(fan, on="OMS_SKU", how="left", suffixes=("", "__bund"))
            for c in q_metric_cols:
                base = pd.to_numeric(pivot.get(c), errors="coerce").fillna(0)
                fb = pd.to_numeric(pivot.get(f"{c}__bund"), errors="coerce").fillna(0)
                pivot[c] = np.where(base > 0, base, fb)
            pivot.drop(
                columns=[f"{c}__bund" for c in q_metric_cols],
                inplace=True,
                errors="ignore",
            )
            existing = set(pivot["OMS_SKU"].astype(str))
            missing = fan[~fan["OMS_SKU"].astype(str).isin(existing)]
            if not missing.empty:
                extra = missing.copy()
                for col in pivot.columns:
                    if col not in extra.columns:
                        extra[col] = 0 if col in q_metric_cols else ""
                pivot = pd.concat([pivot, extra[pivot.columns]], ignore_index=True)
            _ads = pivot["ADS"]
            pivot["Status"] = np.select(
                [_ads >= 1.0, _ads >= 0.33, _ads >= 0.10],
                ["Fast Moving", "Moderate", "Slow Selling"],
                default="Not Moving",
            )
            pivot["Units_90d"] = pivot["Units_90d"].astype(int)
            pivot["Units_30d"] = pivot["Units_30d"].astype(int)
            pivot["Freq_30d"] = pivot["Freq_30d"].astype(int)

    return pivot


def _seasonal_adjacent_months_ads(
    sales_df: pd.DataFrame,
    max_date: pd.Timestamp,
    group_by_parent: bool,
    demand_basis: str,
    years_lookback: int = 2,
    min_denominator: int = 7,
    months_forward: int = 2,
) -> pd.DataFrame:
    """
    Daily ADS from the current calendar month plus the next *months_forward* months
    in prior years (default: 3-month window — e.g. June run uses Jun+Jul+Aug history).

    Example: max_date in June 2026 → use Jun+Jul+Aug of 2025 and 2024 per SKU.
    Lifts ADS before a seasonal peak when recent weeks look weak but last year's
    peak months (e.g. July–August) sold strongly.
    """
    if sales_df.empty or "TxnDate" not in sales_df.columns or "Sku" not in sales_df.columns:
        return pd.DataFrame(columns=["OMS_SKU", "Seasonal_Month_ADS"])

    # Avoid full-copy when TxnDate is already datetime64 (called from calculate_po_base).
    if pd.api.types.is_datetime64_any_dtype(sales_df["TxnDate"]):
        work = sales_df  # read-only — only filter/groupby below
    else:
        work = sales_df.copy()
        work["TxnDate"] = pd.to_datetime(work["TxnDate"], errors="coerce")
        work = work.dropna(subset=["TxnDate"])
    if work.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Seasonal_Month_ADS"])

    if group_by_parent:
        work = work.copy()
        _uniq_sea = work["Sku"].unique()
        _par_sea_cache = {s: get_parent_sku(s) for s in _uniq_sea}
        work = work.copy()
        work["Sku"] = work["Sku"].map(_par_sea_cache)

    m0 = int(max_date.month)
    span = max(1, int(months_forward) + 1)
    rate_lists: Dict[str, list] = defaultdict(list)

    for yo in range(1, years_lookback + 1):
        y = max_date.year - yo
        start = pd.Timestamp(year=y, month=m0, day=1)
        end_month = m0 + span - 1
        end_year = y
        while end_month > 12:
            end_month -= 12
            end_year += 1
        end = pd.Timestamp(year=end_year, month=end_month, day=1) + MonthEnd(0)
        days_span = max((end.normalize() - start.normalize()).days + 1, min_denominator)

        chunk = work[(work["TxnDate"] >= start) & (work["TxnDate"] <= end)]
        if chunk.empty:
            continue
        if demand_basis == "Net":
            g = chunk.groupby("Sku")["Units_Effective"].sum().clip(lower=0)
        else:
            _sh = chunk["Transaction Type"].astype(str).str.strip().str.lower().eq("shipment")
            g = chunk.loc[_sh].groupby("Sku")["Quantity"].sum()
        for sku, qty in g.items():
            if float(qty) <= 0:
                continue
            rate_lists[str(sku)].append(float(qty) / float(days_span))

    if not rate_lists:
        return pd.DataFrame(columns=["OMS_SKU", "Seasonal_Month_ADS"])

    rows = [
        {"OMS_SKU": sku, "Seasonal_Month_ADS": round(float(np.mean(rates)), 3)}
        for sku, rates in rate_lists.items()
    ]
    return pd.DataFrame(rows)


def _inv_parent_extends_sheet_style(pk: str, sk: str) -> bool:
    """True when inventory parent (e.g. ``AK-139BROWN``) is the same style row or extends ``AK-139``."""
    pk = str(pk or "").strip()
    sk = str(sk or "").strip()
    if not pk or not sk:
        return False
    if pk == sk:
        return True
    if not pk.startswith(sk):
        return False
    if len(pk) == len(sk):
        return True
    nxt = pk[len(sk)]
    if nxt in "-_":
        return True
    # ``AK-139`` + ``BROWN`` (no extra hyphen before colour token)
    if sk[-1].isdigit() and nxt.isalpha():
        return True
    return False


def _longest_prefix_lead(
    pk: str,
    lead_by_parent: dict,
    sorted_sheet_keys: Optional[list] = None,
) -> float:
    """
    Best lead when inventory parent extends a shorter sheet style key.

    When ``sorted_sheet_keys`` is provided (sheet parent keys sorted by length,
    descending), the first extending match is the longest — same result as the
    original scan but with early exit. Callers should pass this list when
    resolving many SKUs to avoid O(rows × keys) work.
    """
    pk = str(pk or "").strip()
    if not pk or not lead_by_parent:
        return float("nan")
    direct = lead_by_parent.get(pk)
    if direct is not None:
        try:
            dv = float(direct)
        except (TypeError, ValueError):
            dv = float("nan")
        if np.isfinite(dv) and dv > 0:
            return dv
    keys_iter = (
        sorted_sheet_keys
        if sorted_sheet_keys is not None
        else sorted(
            (str(sk).strip() for sk in lead_by_parent.keys() if str(sk).strip()),
            key=len,
            reverse=True,
        )
    )
    for sks in keys_iter:
        if not sks:
            continue
        try:
            ltv = float(lead_by_parent.get(sks, float("nan")))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ltv) or ltv <= 0:
            continue
        if _inv_parent_extends_sheet_style(pk, sks):
            return ltv
    return float("nan")


def _longest_prefix_sheet_meta(
    pk: str,
    meta: dict[str, tuple[str, bool]],
    sorted_sheet_keys: Optional[list] = None,
) -> tuple[str, bool]:
    """
    Best (status, closed) when inventory parent extends a shorter sheet style key,
    mirroring ``_longest_prefix_lead`` so size variants inherit style-level status rows.
    """
    pk = str(pk or "").strip()
    if not pk or not meta:
        return "", False
    direct = meta.get(pk)
    if direct is not None:
        st_d, cl_d = str(direct[0] or ""), bool(direct[1])
        if st_d or cl_d:
            return st_d, cl_d
    keys_iter = (
        sorted_sheet_keys
        if sorted_sheet_keys is not None
        else sorted((str(k).strip() for k in meta if str(k).strip()), key=len, reverse=True)
    )
    for sks in keys_iter:
        if not sks:
            continue
        row = meta.get(sks)
        if not row:
            continue
        st, cl = str(row[0] or ""), bool(row[1])
        if not st and not cl:
            continue
        if _inv_parent_extends_sheet_style(pk, sks):
            return st, cl
    return "", False


def _style_digit_token(oms_sku: str) -> str:
    """Extract style digits token from SKU/parent (e.g. 1657YK..., AK-1394BROWN -> 1657/1394)."""
    if oms_sku is None or (isinstance(oms_sku, float) and pd.isna(oms_sku)):
        return ""
    s = str(oms_sku).strip().upper()
    m = re.search(r"(?:^|[-_])(\d{3,})", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{3,})", s)
    return m2.group(1) if m2 else ""


def _oos_restock_mask(po_df: pd.DataFrame) -> pd.Series:
    """
    Sizes that just went OOS: no ADS-window sales on this key, but inventory history
    shows recent in-stock days and non-bundled siblings in the same style still sell.
    """
    from .existing_po import is_bundled_size_range_sku

    sku = po_df["OMS_SKU"].astype(str)
    style = sku.map(_fallback_parent_key)
    is_bund = sku.map(is_bundled_size_range_sku).fillna(False)
    inv_total = pd.to_numeric(po_df.get("Total_Inventory"), errors="coerce").fillna(0)
    inv_hist = pd.to_numeric(po_df.get("Eff_Days_Inventory"), errors="coerce").fillna(0)
    net_u = pd.to_numeric(po_df.get("Net_Units"), errors="coerce").fillna(0)
    sold_u = pd.to_numeric(po_df.get("Sold_Units"), errors="coerce").fillna(0)
    row_zero = (net_u <= 0) & (sold_u <= 0)
    tmp = pd.DataFrame({"style": style, "bund": is_bund, "net": net_u}, index=po_df.index)
    style_nb_net = (
        tmp.loc[~tmp["bund"] & (tmp["net"] > 0)]
        .groupby("style")["net"]
        .sum()
    )
    style_sells = style.map(style_nb_net).fillna(0) > 0
    return (
        (inv_total <= 0)
        & (inv_hist > 0)
        & row_zero
        & (~is_bund)
        & style_sells
    )


def _impute_oos_restock_recent_ads(
    po_df: pd.DataFrame,
    mask: pd.Series,
    demand_basis: str,
) -> pd.Series:
    """Spread style demand across OOS per-size rows that recently had stock.

    Caps imputed ADS so a size with a very short Eff_Days window (e.g. 3 days
    in-stock) cannot inherit the full style's unit total and outrank siblings
    that actually sold in the ADS window — a common source of 1000+ unit PO
    outliers on zero-stock rows.
    """
    style = po_df["OMS_SKU"].astype(str).map(_fallback_parent_key)
    from .existing_po import is_bundled_size_range_sku

    is_bund = po_df["OMS_SKU"].astype(str).map(is_bundled_size_range_sku).fillna(False)
    net_u = pd.to_numeric(po_df.get("Net_Units"), errors="coerce").fillna(0)
    tmp = pd.DataFrame({"style": style, "bund": is_bund, "net": net_u}, index=po_df.index)
    style_sell_net = (
        tmp.loc[~tmp["bund"] & (tmp["net"] > 0)]
        .groupby("style")["net"]
        .sum()
    )
    ship = pd.to_numeric(po_df.get("Ship_Units_150d"), errors="coerce").fillna(0)
    eff = pd.to_numeric(po_df.get("Eff_Days"), errors="coerce").fillna(0)
    recent_before = pd.to_numeric(po_df.get("Recent_ADS"), errors="coerce").fillna(0.0)
    out = recent_before.copy()
    _min_eff = 14.0
    for st, idx in po_df.loc[mask].groupby(style, sort=False).groups.items():
        sell_total = float(style_sell_net.get(st, 0))
        if sell_total <= 0:
            continue
        style_mask = style == st
        net_style = net_u.loc[style_mask]
        recent_style = recent_before.loc[style_mask]
        selling = net_style > 0
        cap = float(recent_style[selling].max()) if bool(selling.any()) else 0.0
        oos_idx = pd.Index(idx)
        ship_w = ship.loc[oos_idx]
        if float(ship_w.sum()) > 0:
            weights = ship_w / ship_w.sum()
        else:
            weights = pd.Series(1.0 / len(oos_idx), index=oos_idx)
        imputed_units = sell_total * weights
        eff_denom = eff.loc[oos_idx].clip(lower=_min_eff)
        raw_ads = np.where(eff_denom > 0, imputed_units / eff_denom, 0.0)
        if cap > 0:
            raw_ads = np.minimum(raw_ads, cap)
        out.loc[oos_idx] = raw_ads
    return out.round(3)


def _catalog_sku_allowlist(
    inv_skus: set[str],
    ep_prepared: pd.DataFrame | None = None,
) -> set[str]:
    """SKUs whose sales rows are needed for PO math (inventory + pipeline ghosts)."""
    allow: set[str] = set()
    for raw in inv_skus:
        s = str(raw or "").strip()
        if not s:
            continue
        allow.add(s)
        par = str(get_parent_sku(s) or "").strip()
        if par:
            allow.add(par)
        fp = str(_fallback_parent_key(s) or "").strip()
        if fp:
            allow.add(fp)
    if ep_prepared is not None and not ep_prepared.empty and "OMS_SKU" in ep_prepared.columns:
        for raw in ep_prepared["OMS_SKU"].astype(str):
            s = str(raw or "").strip()
            if s:
                allow.add(s)
    return allow


def _sales_sku_in_catalog(sku: object, allow: set[str]) -> bool:
    """Keep sales for catalog SKUs, parents, and bundled listings that fan out to them."""
    from .existing_po import _split_bundled_po_sku, is_bundled_size_range_sku

    s = str(sku or "").strip()
    if not s:
        return False
    if s in allow:
        return True
    par = str(get_parent_sku(s) or "").strip()
    if par and par in allow:
        return True
    fp = str(_fallback_parent_key(s) or "").strip()
    if fp and fp in allow:
        return True
    if is_bundled_size_range_sku(s):
        for kid in _split_bundled_po_sku(s):
            if str(kid).strip() in allow:
                return True
    return False


def _filter_sales_df_to_catalog(df: pd.DataFrame, allow: set[str]) -> pd.DataFrame:
    """Drop historical sales for SKUs outside the active catalog — largest PO calc win."""
    if df.empty or not allow:
        return df
    uniq = df["Sku"].unique()
    keep = {u: _sales_sku_in_catalog(u, allow) for u in uniq}
    mask = df["Sku"].map(keep).fillna(False)
    n_before = len(df)
    out = df.loc[mask]
    if len(out) < n_before:
        import logging

        logging.getLogger(__name__).info(
            "PO calc: sales rows %s → %s after catalog SKU filter",
            f"{n_before:,}",
            f"{len(out):,}",
        )
    return out


def _fallback_parent_key(sku: str) -> str:
    """
    Parent key for sales↔inventory mismatch fallback.
    Handles both hyphenated sizes (``ABC-BLACK-XL``) and non-delimited ones
    (``ABCBLACKXL``) that appear in some marketplace extracts.
    """
    s = str(sku or "").strip().upper()
    if not s:
        return ""
    p = get_parent_sku(s)
    p = str(p or "").strip().upper() if p is not None else s
    if p and p != s:
        return p
    return _SIZE_SUFFIX_RE.sub("", s).strip("-_")


def _size_suffix_label(oms_sku: str) -> str:
    s = str(oms_sku or "").strip()
    if "-" in s:
        return s.rsplit("-", 1)[-1]
    return s


def _apply_multi_size_sibling_po_lift(
    po_df: pd.DataFrame,
    target_cover_days: float,
) -> pd.DataFrame:
    """When ≥2 sizes in a parent already need PO, top up all siblings below target cover.

    Ensures one combined order covers every active size up to post-PO target days instead
    of raising PO for a subset and revisiting the style days later.
    """
    if po_df is None or po_df.empty or "Parent_SKU" not in po_df.columns:
        return po_df

    po_df = po_df.copy()
    try:
        from .sku_status_lead import is_excluded_po_status
    except Exception:  # pragma: no cover
        def is_excluded_po_status(_x: object) -> bool:
            return False

    _par = po_df["Parent_SKU"].astype(str)
    _po_qty = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0.0)
    _ads = pd.to_numeric(po_df["ADS"], errors="coerce").fillna(0.0)
    _proj = pd.to_numeric(po_df["Projected_Running_Days"], errors="coerce").fillna(999.0)
    _n_active = (_po_qty > 0).astype(int).groupby(_par).transform("sum")
    _excl = po_df.get("SKU_Sheet_Status", pd.Series("", index=po_df.index)).astype(str).map(
        is_excluded_po_status
    )
    if not isinstance(_excl, pd.Series):
        _excl = pd.Series(_excl, index=po_df.index)
    _closed = po_df.get("SKU_Sheet_Closed", pd.Series(False, index=po_df.index)).fillna(False)

    _target = float(max(1.0, target_cover_days))
    _lift = (
        (_n_active >= 2)
        & (_po_qty <= 0)
        & (_ads > 0)
        & (_proj < _target)
        & ~_excl.fillna(False)
        & ~_closed.astype(bool)
    )
    if not bool(_lift.any()):
        return po_df

    _lift_idx = po_df.index[_lift]
    _bal = (_target - _proj.loc[_lift_idx]).clip(lower=0.0)
    _rounded = np.vectorize(round_po_pack)((_bal * _ads.loc[_lift_idx]).to_numpy(dtype=float))
    po_df.loc[_lift_idx, "Gross_PO_Qty"] = _rounded
    po_df.loc[_lift_idx, "PO_Qty"] = _rounded

    _proj_lift = _proj.loc[_lift_idx].to_numpy(dtype=float)
    _ads_lift = _ads.loc[_lift_idx].to_numpy(dtype=float)
    po_df.loc[_lift_idx, "Post_PO_Cover_Days_Capped"] = (
        _proj_lift + _rounded / np.maximum(_ads_lift, 1e-9)
    ).round(1)

    _lt_pat = r"Projected cover exceeds (?:entered |sheet |factory )?lead time[^;]*"
    _old_br = po_df.loc[_lift_idx, "PO_Block_Reason"].astype(str)
    po_df.loc[_lift_idx, "PO_Block_Reason"] = (
        _old_br.str.replace(_lt_pat, "", regex=True).str.strip("; ").str.strip()
    )
    if bool((_rounded > 0).any()):
        po_df.loc[_lift_idx[_rounded > 0], "Priority"] = "🟡 HIGH"

    import logging as _log_mod

    _log_mod.getLogger(__name__).info(
        "multi-size sibling PO lift: %d row(s) across %d parent(s)",
        int(_lift.sum()),
        int(_par.loc[_lift_idx].nunique()),
    )
    return po_df


def _apply_sibling_cut_from_pending(po_df: pd.DataFrame, target_cover_days: float) -> pd.DataFrame:
    """When some sizes need a PO and siblings are over-covered with pending cutting, suggest cutting
    from those sizes instead of ordering new fabric (saves grey-fabric cost)."""
    if po_df is None or po_df.empty or "Parent_SKU" not in po_df.columns:
        return po_df

    po_df = po_df.copy()
    for col in ("Cutting_Source", "Cut_From_Siblings", "PO_Cutting_Note"):
        if col not in po_df.columns:
            po_df[col] = ""

    proj = pd.to_numeric(po_df.get("Projected_Running_Days"), errors="coerce").fillna(999.0)
    if "Pending_Cutting" in po_df.columns:
        pending = pd.to_numeric(po_df["Pending_Cutting"], errors="coerce").fillna(0).astype(int)
    else:
        pending = pd.Series(0, index=po_df.index, dtype=int)
    po_qty = pd.to_numeric(po_df.get("PO_Qty"), errors="coerce").fillna(0).astype(int)
    par = po_df["Parent_SKU"].astype(str)
    sku = po_df["OMS_SKU"].astype(str)
    threshold = float(max(1.0, target_cover_days))

    is_donor = (proj > threshold) & (pending > 0)
    needs_po = po_qty > 0

    for parent in par.loc[needs_po].dropna().unique():
        if not str(parent).strip():
            continue
        donor_mask = is_donor & (par == parent)
        need_mask = needs_po & (par == parent)
        if not donor_mask.any():
            continue

        donor_parts: list[str] = []
        for idx in po_df.index[donor_mask]:
            donor_parts.append(
                f"{_size_suffix_label(sku.at[idx])} "
                f"({int(pending.at[idx])} pending cut, {float(proj.at[idx]):.0f}d cover)"
            )
        donor_text = "; ".join(donor_parts)
        note = "ADJUST FROM OTHER SIZES — use sibling pending cutting before ordering new fabric"

        for idx in po_df.index[need_mask]:
            po_df.at[idx, "Cutting_Source"] = "Adjust from siblings (pending cutting)"
            po_df.at[idx, "Cut_From_Siblings"] = donor_text
            po_df.at[idx, "PO_Cutting_Note"] = note

        for idx in po_df.index[donor_mask]:
            po_df.at[idx, "Cutting_Source"] = "Donor — cut for sibling PO"
            hint = (
                f"Cut pending stock for sibling PO — {int(pending.at[idx])} units "
                f"({float(proj.at[idx]):.0f}d cover vs {threshold:.0f}d target)"
            )
            if "Suggest_Close_SKU" in po_df.columns:
                old = str(po_df.at[idx, "Suggest_Close_SKU"] or "").strip()
                po_df.at[idx, "Suggest_Close_SKU"] = hint if not old else f"{old}; {hint}"

    return po_df


def calculate_po_base(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    period_days: int,
    lead_time: int,
    target_days: int,
    demand_basis: str = "Sold",
    min_denominator: int = 7,
    grace_days: int = 0,
    safety_pct: float = 20.0,
    use_seasonality: bool = False,
    seasonal_weight: float = 0.5,
    mtr_df: Optional[pd.DataFrame] = None,
    myntra_df: Optional[pd.DataFrame] = None,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
    existing_po_df: Optional[pd.DataFrame] = None,
    sku_status_df: Optional[pd.DataFrame] = None,
    enforce_two_size_minimum: bool = False,
    # When True (default): for rows with **sheet-resolved** factory lead only, do not
    # recommend a release while projected cover ``(Tot inv + eff. pipeline) / ADS`` is
    # **strictly greater** than ``Lead_Time_Days``. Without a status sheet (or when lead
    # is not sheet-resolved), this rule does not apply. Set False for legacy target-only
    # mode (top up toward target even when cover still exceeds sheet lead).
    enforce_lead_time_release_gate: bool = True,
    inventory_history_df: Optional[pd.DataFrame] = None,
    po_raise_ledger_df: Optional[pd.DataFrame] = None,
    planning_date: Optional[str] = None,
    raise_ledger_lookback_days: int = 14,
    raise_view_date: Optional[str] = None,
    po_return_overlay_df: Optional[pd.DataFrame] = None,
    urgent_all_sizes_days: int = 45,
    use_ly_fallback: bool = True,
    stage_timer: Any = None,
    manual_existing_po_raise_skus: Optional[set[str]] = None,
    manual_existing_po_raise_date: Optional[str] = None,
) -> pd.DataFrame:
    import time as _time

    _engine_start = _time.perf_counter() if stage_timer is not None else None
    if sales_df.empty or inv_df.empty:
        return pd.DataFrame()

    # ── Trim sales_df to the lookback window BEFORE copying ───────────────────
    # Old pattern: df = sales_df.copy() + later df = df[mask].copy() held two
    # full copies in RAM simultaneously (can be 800 MB–1.5 GB for large catalogs).
    # New pattern: compute max_date/hist_cutoff on the original (no copy), make
    # ONE trimmed copy, then the later trim at lines below is a fast no-op.
    _s_plan_early: "pd.Timestamp | None" = None
    if planning_date:
        try:
            _s_plan_early = pd.Timestamp(pd.to_datetime(planning_date).normalize())
        except Exception:
            pass
    _s_txn = (
        sales_df["TxnDate"]
        if pd.api.types.is_datetime64_any_dtype(sales_df["TxnDate"])
        else pd.to_datetime(sales_df["TxnDate"], errors="coerce")
    )
    if hasattr(_s_txn, "dt") and _s_txn.dt.tz is not None:
        _s_txn = _s_txn.dt.tz_localize(None)
    _s_max = _s_txn.max()
    if pd.notna(_s_max):
        _s_today = pd.Timestamp.now().normalize()
        if _s_max > _s_today + timedelta(days=1):
            _s_max = _s_today
        if _s_plan_early is not None:
            _s_max = min(_s_max, _s_plan_early)
        _s_lookback = int(max(30, period_days) + 365 + period_days + 7)
        _s_cut = _s_max - timedelta(days=_s_lookback)
        _s_mask = _s_txn >= _s_cut
        df = sales_df[_s_mask].copy()
        del _s_mask
    else:
        df = sales_df.copy()
    del _s_txn, _s_max, _s_plan_early
    # ── End early trim ────────────────────────────────────────────────────────

    if not pd.api.types.is_datetime64_any_dtype(df["TxnDate"]):
        df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
        df = df.dropna(subset=["TxnDate"])
    if df["TxnDate"].dt.tz is not None:
        df["TxnDate"] = df["TxnDate"].dt.tz_localize(None)

    _map = sku_mapping if sku_mapping is not None else {}

    def _canonical_oms_key(raw) -> str:
        """Align inventory, sales, and SKU-status sheet keys (Excel tokens + clean_sku + PL strip)."""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return ""
        t = normalize_id_token_for_mapping(str(raw).strip())
        t = clean_sku(t or raw)
        if not t:
            t = str(raw).strip().upper()
        return collapse_duplicate_trailing_size_suffix(_strip_pl(str(t).strip(), _map))

    def _merge_metric_with_parent_fallback(
        base_df: pd.DataFrame,
        metric_df: pd.DataFrame,
        metric_cols: list[str],
        sales_key_col: str = "OMS_SKU",
    ) -> pd.DataFrame:
        """
        Merge metrics by exact SKU, then fallback to parent-rollup when exact key is missing.
        Used only in variant mode where inventory can contain parent-like SKUs.
        """
        if metric_df is None or metric_df.empty:
            out = base_df.copy()
            for c in metric_cols:
                if c not in out.columns:
                    out[c] = 0
            return out
        out = base_df.merge(metric_df, on="OMS_SKU", how="left")
        if group_by_parent:
            return out
        exact_keys = set(metric_df[sales_key_col].astype(str))
        if not exact_keys:
            return out
        out_sku = out["OMS_SKU"].astype(str)
        _is_parent_token = out_sku.map(_fallback_parent_key).eq(out_sku)
        # Parent-like inventory tokens (e.g. ``1007YKBLACK``) should represent the
        # whole style family (exact + size variants), not only exact-key rows.
        # Still fallback for non-parent rows when exact key is absent.
        use_parent = _is_parent_token | (~out_sku.isin(exact_keys))
        if not use_parent.any():
            return out
        parent_metric = metric_df.copy()
        parent_metric["_Parent_SKU"] = parent_metric[sales_key_col].map(_fallback_parent_key)
        agg = (
            parent_metric.groupby("_Parent_SKU", as_index=False)[metric_cols]
            .sum()
            .rename(columns={"_Parent_SKU": "OMS_SKU"})
        )
        out = out.merge(agg, on="OMS_SKU", how="left", suffixes=("", "__p"))
        for c in metric_cols:
            pcol = f"{c}__p"
            out[c] = np.where(
                use_parent,
                pd.to_numeric(out.get(pcol), errors="coerce").fillna(0),
                pd.to_numeric(out[c], errors="coerce").fillna(0),
            )
        drop_cols = [f"{c}__p" for c in metric_cols]
        out.drop(columns=drop_cols, inplace=True, errors="ignore")
        return out

    _fan_out_cache: dict[tuple[tuple[str, ...], int], pd.DataFrame] = {}

    def _merge_metric_with_bundled_listing_fallback(
        base_df: pd.DataFrame,
        metric_df: pd.DataFrame,
        metric_cols: list[str],
    ) -> pd.DataFrame:
        """Exact merge + parent fallback + bundled listing (L-XL) share for zero rows.

        Sales and ship metrics are always fanned out to individual sizes from bundled
        listings (e.g. 1917YKBLUE-L-XL → L and XL share).  Pipeline data
        (Gross_PO_Qty / PO_Pipeline_Total) is handled through the existing-PO merge
        path and is never passed here, so there is no double-count risk.
        """
        from .existing_po import fan_out_bundled_listing_metrics

        out = _merge_metric_with_parent_fallback(base_df, metric_df, metric_cols)
        if group_by_parent or metric_df is None or metric_df.empty:
            return out
        # Do NOT restrict by bundled-listing inventory: sales history must always
        # fan out so individual sizes get ADS and quarterly-history data.
        fan_key = (tuple(metric_cols), id(metric_df))
        fan = _fan_out_cache.get(fan_key)
        if fan is None:
            fan = fan_out_bundled_listing_metrics(metric_df, metric_cols)
            _fan_out_cache[fan_key] = fan
        if fan is None or fan.empty:
            return out
        out = out.merge(fan, on="OMS_SKU", how="left", suffixes=("", "__bund"))
        for c in metric_cols:
            base = pd.to_numeric(out.get(c), errors="coerce").fillna(0)
            fb = pd.to_numeric(out.get(f"{c}__bund"), errors="coerce").fillna(0)
            out[c] = np.where(base > 0, base, fb)
        out.drop(columns=[f"{c}__bund" for c in metric_cols], inplace=True, errors="ignore")
        return out

    # ── Unique-SKU cache: normalize once per unique raw value, not per row ──────
    # With 225k rows but only ~7-8k unique SKUs, this is ~28x faster than row-by-row.
    _unique_sales_skus = df["Sku"].unique()
    _sku_canon_cache = {s: _canonical_oms_key(s) for s in _unique_sales_skus}
    df["Sku"] = df["Sku"].map(_sku_canon_cache).fillna("")
    df = df[df["Sku"].str.len() > 0]
    df["_is_ship"] = df["Transaction Type"].astype(str).str.strip().str.lower().eq("shipment")

    # Parent-level PO: ``inventory_df_parent`` uses style parents (e.g. ``1057YKBLUE``) while
    # sales lines stay size-suffixed (``1057YKBLUE-6XL``). Without rolling sales up to the same
    # key, merges leave ``Sold_Units`` / ADS columns near zero for almost every row.
    if group_by_parent:
        _uniq_for_parent = df["Sku"].unique()
        _sku_to_parent: Dict[object, str] = {}
        for _raw_s in _uniq_for_parent:
            if _raw_s is None or (isinstance(_raw_s, float) and pd.isna(_raw_s)):
                continue
            _par = get_parent_sku(_raw_s)
            if _par is None or (isinstance(_par, float) and pd.isna(_par)):
                _tok = str(_raw_s).strip()
            else:
                _tok = str(_par).strip()
            _sku_to_parent[_raw_s] = _tok if _tok else str(_raw_s).strip()
        df["Sku"] = df["Sku"].map(_sku_to_parent).fillna("")
        df = df[df["Sku"].str.len() > 0]

    inv_work = inv_df.copy()
    from .existing_po import existing_po_merge_key, is_bundled_size_range_sku

    _bundled_from_per_size_inv: set[str] = set()
    if "OMS_SKU" in inv_work.columns:
        for raw in inv_work["OMS_SKU"].astype(str):
            raw_key = str(raw).strip().upper()
            if not raw_key:
                continue
            mapped = canonical_oms_key(raw, sku_mapping)
            if (
                mapped
                and mapped != existing_po_merge_key(raw)
                and is_bundled_size_range_sku(mapped)
            ):
                _bundled_from_per_size_inv.add(mapped)
    _ep_prepared = pd.DataFrame()
    _unique_inv_skus = inv_work["OMS_SKU"].unique()
    _inv_canon_cache = {s: _canonical_oms_key(s) for s in _unique_inv_skus}
    inv_work["OMS_SKU"] = inv_work["OMS_SKU"].map(_inv_canon_cache).fillna("")
    inv_work = inv_work[inv_work["OMS_SKU"].str.len() > 0]
    if inv_work["OMS_SKU"].duplicated().any():
        inv_work = inv_work.drop_duplicates(subset=["OMS_SKU"], keep="last")

    # Pipeline-only SKUs (OOS / dropped from inventory upload) must enter po_df before
    # inventory-history Eff_Days — otherwise they are injected later with Eff_Days=0.
    if existing_po_df is not None and not existing_po_df.empty:
        from .existing_po import existing_po_merge_key, prepare_existing_po_for_merge

        _inv_keys = {
            str(existing_po_merge_key(s)).strip().upper()
            for s in inv_work["OMS_SKU"].astype(str)
        }
        _ep_prepared = prepare_existing_po_for_merge(
            existing_po_df,
            existing_po_merge_key,
            inventory_skus=_inv_keys,
        )
        _ep_seed = _ep_prepared
        if not _ep_seed.empty and "PO_Pipeline_Total" in _ep_seed.columns:
            _have_inv = set(inv_work["OMS_SKU"].astype(str).str.strip())
            _ep_active = pd.to_numeric(_ep_seed["PO_Pipeline_Total"], errors="coerce").fillna(0) > 0
            for _ac in ("Pending_Cutting", "Balance_to_Dispatch", "PO_Qty_Ordered"):
                if _ac in _ep_seed.columns:
                    _ep_active |= pd.to_numeric(_ep_seed[_ac], errors="coerce").fillna(0) > 0
            _need = []
            for s in _ep_seed.loc[_ep_active, "OMS_SKU"].astype(str).str.strip().unique():
                if not s or s in _have_inv:
                    continue
                mapped = canonical_oms_key(s, sku_mapping)
                if mapped and mapped in _have_inv:
                    continue
                _need.append(s)
            if _need:
                _pad = pd.DataFrame({"OMS_SKU": _need})
                for _c in inv_work.columns:
                    if _c != "OMS_SKU":
                        _pad[_c] = 0
                inv_work = pd.concat([inv_work, _pad[inv_work.columns]], ignore_index=True)

    _catalog_allow = _catalog_sku_allowlist(
        set(inv_work["OMS_SKU"].astype(str)),
        _ep_prepared if not _ep_prepared.empty else None,
    )
    df = _filter_sales_df_to_catalog(df, _catalog_allow)

    _plan = None
    if planning_date:
        try:
            _plan = pd.Timestamp(pd.to_datetime(planning_date).normalize())
        except Exception:
            _plan = None

    if df.empty:
        max_date = _plan if _plan is not None else pd.Timestamp.now().normalize()
    else:
        max_date = df["TxnDate"].max()
    # Guard against stray future-dated rows (parse quirks / bad source dates).
    # A single outlier can shift the global recent window and zero-out ADS for most SKUs.
    _today = pd.Timestamp.now().normalize()
    _max_allowed = _today + timedelta(days=1)
    if pd.notna(max_date) and max_date > _max_allowed:
        max_date = _today
    # Do not run ADS / eff-days as if sales exist past the last upload (or past the
    # operator's planning day when the browser sends it).
    if _plan is not None and pd.notna(max_date):
        max_date = min(max_date, _plan)
    elif _plan is not None:
        max_date = _plan
    # PO only needs recent and LY windows; trimming old rows avoids multi-year
    # full-table groupbys that make calculation feel stuck for large histories.
    lookback_days = int(max(30, period_days) + 365 + period_days + 7)
    hist_cutoff = max_date - timedelta(days=lookback_days)
    # sales_df was already trimmed to this window at function entry (see early-trim
    # block above). Only copy again if max_date was adjusted further after the early
    # trim (e.g. planning_date capped it differently).
    _df_min = df["TxnDate"].min()
    if pd.notna(_df_min) and _df_min < hist_cutoff:
        df = df[df["TxnDate"] >= hist_cutoff].copy()
    cutoff   = max_date - timedelta(days=period_days)
    recent   = df[df["TxnDate"] >= cutoff]

    sold = recent[recent["_is_ship"]].groupby("Sku")["Quantity"].sum().reset_index()
    sold.columns = ["OMS_SKU", "Sold_Units"]
    _is_ref = recent["Transaction Type"].astype(str).str.strip().str.lower().eq("refund")
    returns = recent[_is_ref].groupby("Sku")["Quantity"].sum().reset_index()
    returns.columns = ["OMS_SKU", "Return_Units"]
    net = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU", "Net_Units"]

    summary = sold.merge(returns, on="OMS_SKU", how="outer").merge(net, on="OMS_SKU", how="outer").fillna(0)
    po_df = _merge_metric_with_bundled_listing_fallback(
        inv_work, summary, ["Sold_Units", "Return_Units", "Net_Units"]
    ).fillna({"Sold_Units": 0, "Return_Units": 0, "Net_Units": 0})

    if po_return_overlay_df is not None and not po_return_overlay_df.empty:
        from .po_return_import import aggregate_return_overlay_for_use

        ov = aggregate_return_overlay_for_use(po_return_overlay_df.copy())
        if ov is None or ov.empty:
            ov = po_return_overlay_df.copy()
        if "OMS_SKU" in ov.columns and "Return_Units" in ov.columns:
            # Date-window the overlay: only count returns that fall within the ADS
            # period window (same cutoff used for sales). Returns from prior months
            # must not cancel out current-period sales and inflate ADS.
            if "Return_Date" in ov.columns:
                try:
                    ov_dates = pd.to_datetime(ov["Return_Date"], errors="coerce")
                    _cutoff_ts = pd.Timestamp(cutoff).normalize()
                    _max_ts = pd.Timestamp(max_date).normalize()
                    _in_window = (
                        ov_dates.notna()
                        & (ov_dates >= _cutoff_ts)
                        & (ov_dates <= _max_ts)
                    )
                    # Rows with no parseable date (legacy undated bundles) are kept
                    # so PO overlay still works for files that don't have per-row dates.
                    _no_date = ov_dates.isna()
                    ov = ov[_in_window | _no_date].copy()
                except Exception:
                    pass  # date filtering failed — keep all rows (safe fallback)

            ov["OMS_SKU"] = ov["OMS_SKU"].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
            ov = ov[ov["OMS_SKU"].str.len() > 0]
            if group_by_parent:
                ov["OMS_SKU"] = ov["OMS_SKU"].map(
                    lambda s: str(get_parent_sku(s) or s).strip().upper()
                )
            ov["Return_Overlay_Units"] = (
                pd.to_numeric(ov["Return_Units"], errors="coerce").fillna(0).astype(int)
            )
            ov = ov.groupby("OMS_SKU", as_index=False)["Return_Overlay_Units"].sum()
            po_df = po_df.merge(ov, on="OMS_SKU", how="left")
            po_df["Return_Overlay_Units"] = (
                pd.to_numeric(po_df["Return_Overlay_Units"], errors="coerce").fillna(0).astype(int)
            )
            po_df["Return_Units"] = (
                pd.to_numeric(po_df["Return_Units"], errors="coerce").fillna(0).astype(int)
                + po_df["Return_Overlay_Units"]
            )
            po_df["Net_Units"] = (
                pd.to_numeric(po_df["Sold_Units"], errors="coerce").fillna(0).astype(int)
                - po_df["Return_Units"]
            ).clip(lower=0)
        else:
            po_df["Return_Overlay_Units"] = 0
    else:
        po_df["Return_Overlay_Units"] = 0

    # Shipment context column for operators (not used in PO math):
    # show broader last ~5 months shipments even when period window is shorter.
    ship_150_cutoff = max_date - timedelta(days=150)
    ship_150 = (
        df[(df["TxnDate"] >= ship_150_cutoff) & (df["_is_ship"])]
        .groupby("Sku")["Quantity"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Quantity": "Ship_Units_150d"})
    )
    po_df = _merge_metric_with_bundled_listing_fallback(
        po_df, ship_150, ["Ship_Units_150d"]
    )
    po_df["Ship_Units_150d"] = pd.to_numeric(po_df["Ship_Units_150d"], errors="coerce").fillna(0).astype(int)

    # Defaults until SKU Status / Lead merge — applied later after pipeline ghost rows are appended,
    # so pipeline-only SKUs still inherit per-sheet lead times.
    po_df["Lead_Time_Days"] = _lead_time_column_fallback(lead_time)
    po_df["PO_Block_Reason"] = ""
    po_df["SKU_Sheet_Status"] = ""
    po_df["SKU_Sheet_Closed"] = False

    # ADS starts from period_days-window Recent_ADS / LY blend, is floored by
    # seasonal same-month+next-month history, and by Flat30_ADS (Req.xlsx FREQ).
    ADS_WINDOW = period_days
    ads_cutoff = max_date - timedelta(days=ADS_WINDOW)
    ads_recent = df[df["TxnDate"] >= ads_cutoff]

    ads_sold = (
        ads_recent[ads_recent["_is_ship"]]
        .groupby("Sku")["Quantity"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Quantity": "ADS_Sold_Units"})
    )
    ads_net = (
        ads_recent.groupby("Sku")["Units_Effective"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Units_Effective": "ADS_Net_Units"})
    )
    # Eff_Days = active demand days only (first→last qualifying txn within the ADS window,
    # inclusive), not calendar days from first sale to max_date. Stops diluting ADS when
    # sales paused mid-window while the clock still ran to ``max_date``.
    if demand_basis == "Sold":
        act = ads_recent[ads_recent["_is_ship"]].copy()
    else:
        act = ads_recent
    if act.empty:
        ads_active_span = pd.DataFrame(columns=["OMS_SKU", "_cal_span_days", "_distinct_days"])
    else:
        ads_active_span = (
            act.groupby("Sku", as_index=False)
            .agg(
                ADS_First_Active=("TxnDate", "min"),
                ADS_Last_Active=("TxnDate", "max"),
                ADS_Distinct_Active=("TxnDate", "nunique"),
            )
            .rename(columns={"Sku": "OMS_SKU"})
        )
        _cal_span = (
            ads_active_span["ADS_Last_Active"] - ads_active_span["ADS_First_Active"]
        ).dt.days + 1
        _distinct = pd.to_numeric(ads_active_span["ADS_Distinct_Active"], errors="coerce").fillna(0)
        ads_active_span["_cal_span_days"] = _cal_span.astype(int)
        ads_active_span["_distinct_days"] = _distinct.astype(int)

    ads_summary = ads_sold.merge(ads_net, on="OMS_SKU", how="outer").fillna(0)
    po_df = _merge_metric_with_bundled_listing_fallback(
        po_df, ads_summary, ["ADS_Sold_Units", "ADS_Net_Units"]
    )
    _span_merge_cols = ["OMS_SKU", "_cal_span_days", "_distinct_days"]
    if not ads_active_span.empty:
        po_df = po_df.merge(ads_active_span[_span_merge_cols], on="OMS_SKU", how="left")
    else:
        po_df["_cal_span_days"] = 0
        po_df["_distinct_days"] = 0
    if not group_by_parent and not ads_recent.empty:
        span_exact_keys = set(ads_active_span["OMS_SKU"].astype(str)) if not ads_active_span.empty else set()
        _p_act = act.copy()
        if not _p_act.empty:
            _p_act["_Parent_SKU"] = _p_act["Sku"].map(_fallback_parent_key)
            p_ads_active_span = (
                _p_act.groupby("_Parent_SKU", as_index=False)
                .agg(
                    P_ADS_First_Active=("TxnDate", "min"),
                    P_ADS_Last_Active=("TxnDate", "max"),
                    P_ADS_Distinct_Active=("TxnDate", "nunique"),
                )
                .rename(columns={"_Parent_SKU": "OMS_SKU"})
            )
            _p_cal_span = (
                p_ads_active_span["P_ADS_Last_Active"] - p_ads_active_span["P_ADS_First_Active"]
            ).dt.days + 1
            _p_distinct = pd.to_numeric(
                p_ads_active_span["P_ADS_Distinct_Active"], errors="coerce"
            ).fillna(0)
            p_ads_active_span["P__cal_span_days"] = _p_cal_span.astype(int)
            p_ads_active_span["P__distinct_days"] = _p_distinct.astype(int)
            po_df = po_df.merge(
                p_ads_active_span[["OMS_SKU", "P__cal_span_days", "P__distinct_days"]],
                on="OMS_SKU",
                how="left",
            )
            out_sku = po_df["OMS_SKU"].astype(str)
            use_parent_days = out_sku.map(_fallback_parent_key).eq(out_sku) | (~out_sku.isin(span_exact_keys))
            po_df["_cal_span_days"] = np.where(
                use_parent_days,
                pd.to_numeric(po_df["P__cal_span_days"], errors="coerce"),
                pd.to_numeric(po_df["_cal_span_days"], errors="coerce"),
            )
            po_df["_distinct_days"] = np.where(
                use_parent_days,
                pd.to_numeric(po_df["P__distinct_days"], errors="coerce"),
                pd.to_numeric(po_df["_distinct_days"], errors="coerce"),
            )
            po_df.drop(columns=["P__cal_span_days", "P__distinct_days"], inplace=True, errors="ignore")
    po_df[["ADS_Sold_Units", "ADS_Net_Units"]] = po_df[["ADS_Sold_Units", "ADS_Net_Units"]].fillna(0)
    # Active span is only meaningful when this row has demand in the ADS window.
    _demand_for_eff = (
        po_df["ADS_Net_Units"].fillna(0)
        if demand_basis == "Net"
        else po_df["ADS_Sold_Units"].fillna(0)
    )
    po_df.loc[_demand_for_eff <= 0, "_cal_span_days"] = np.nan
    po_df.loc[_demand_for_eff <= 0, "_distinct_days"] = np.nan

    # Sold_Units / Net_Units reflect the full ADS window (period_days).
    po_df["Sold_Units"] = po_df["ADS_Sold_Units"].astype(int)
    po_df["Net_Units"]  = po_df["ADS_Net_Units"].clip(lower=0).astype(int)
    if "Return_Overlay_Units" in po_df.columns:
        _rov = pd.to_numeric(po_df["Return_Overlay_Units"], errors="coerce").fillna(0).astype(int)
        if int(_rov.sum()) > 0:
            po_df["ADS_Net_Units"] = (
                pd.to_numeric(po_df["ADS_Net_Units"], errors="coerce").fillna(0).astype(int) - _rov
            ).clip(lower=0)
            po_df["Net_Units"] = po_df["ADS_Net_Units"].astype(int)

    _cal_eff = pd.to_numeric(po_df["_cal_span_days"], errors="coerce").fillna(0)
    _dist_eff = pd.to_numeric(po_df["_distinct_days"], errors="coerce").fillna(0)
    _sold_eff = pd.to_numeric(_demand_for_eff, errors="coerce").fillna(0)
    # ≤5 sold in the ADS window: keep calendar/period dilution — distinct-day collapse
    # would 10× ADS on noise (e.g. 2 units → Eff_Days=1 → PO explosion).
    _low_vol = (_sold_eff > 0) & (_sold_eff <= 5)
    # Only collapse to distinct sale days for genuinely sparse sellers (4032DRSGREEN:
    # 6 units / 26d calendar). Bursty but real demand (e.g. 20 units on 4 days over 40d)
    # must keep the inclusive calendar span — otherwise ADS inflates 10× and PO explodes.
    _sparse_intermittent = (
        (_sold_eff >= 6)
        & (_cal_eff > 0)
        & ((_sold_eff / _cal_eff) < 0.35)
        & ((_dist_eff * 2) < _cal_eff)
    )
    po_df["_eff_days_active"] = np.where(
        _sparse_intermittent,
        _dist_eff.clip(lower=1.0),
        _cal_eff,
    )
    po_df["_sparse_intermittent"] = _sparse_intermittent
    po_df.drop(columns=["_distinct_days"], inplace=True, errors="ignore")
    po_df["Eff_Days"] = (
        pd.to_numeric(po_df["_eff_days_active"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=float(ADS_WINDOW))
    )
    po_df.drop(columns=["_eff_days_active"], inplace=True, errors="ignore")

    # ── Daily-inventory-history override ─────────────────────────────────────
    # Optional Excel: rows = SKU, columns = dates, cell = on-hand units. Counts
    # only the days a SKU actually had stock (>=1 unit) inside the ADS window.
    # That replaces the active-span Eff_Days when available — days the item was
    # OOS shouldn't pad the ADS denominator and lowball ADS.
    if inventory_history_df is not None and not inventory_history_df.empty:
        try:
            import logging

            from .daily_inventory_history import (
                coverage_days_within,
                effective_days_from_history,
                extend_history_with_sales,
                should_skip_inventory_history_extend,
                trim_inventory_history_for_po,
            )

            _inv_log = logging.getLogger(__name__)

            # Anchor at the most recent date we actually have data for.
            inv_window_end = pd.Timestamp(max_date).normalize()
            _ihmax = pd.to_datetime(inventory_history_df["Date"], errors="coerce").max()
            if pd.notna(_ihmax):
                sheet_end = pd.Timestamp(_ihmax).normalize()
                if _plan is not None:
                    sheet_end = min(sheet_end, _plan)
                inv_window_end = max(inv_window_end, sheet_end)
            inv_window_start = inv_window_end - timedelta(days=int(ADS_WINDOW) - 1)

            po_skus = set(po_df["OMS_SKU"].astype(str))
            ih_src = inventory_history_df.copy()
            if "OMS_SKU" in ih_src.columns:
                ih_src["OMS_SKU"] = ih_src["OMS_SKU"].astype(str).map(_canonical_oms_key)
            ih = trim_inventory_history_for_po(
                ih_src,
                inv_window_start,
                inv_window_end,
                po_skus=po_skus,
            )
            if ih.empty:
                po_df["Eff_Days_Inventory"] = 0
                po_df["Inv_Coverage_Days"] = 0
            else:
                ih["OMS_SKU"] = ih["OMS_SKU"].astype(str).map(_canonical_oms_key)
                ih = ih[ih["OMS_SKU"].str.len() > 0]
                if group_by_parent and not ih.empty:
                    ih["OMS_SKU"] = ih["OMS_SKU"].map(get_parent_sku)
                    ih = ih.groupby(["OMS_SKU", "Date"], as_index=False)["Qty"].max()

                _orig_rows = int(len(inventory_history_df))
                _trim_rows = int(len(ih))
                if _orig_rows > _trim_rows * 2:
                    _inv_log.info(
                        "PO inventory history trimmed %s → %s rows (ADS window %s..%s)",
                        f"{_orig_rows:,}",
                        f"{_trim_rows:,}",
                        inv_window_start.date(),
                        inv_window_end.date(),
                    )

                sheet_max = pd.to_datetime(ih["Date"], errors="coerce").max()
                coverage_in_window = coverage_days_within(ih, inv_window_start, inv_window_end)
                skip_extend = should_skip_inventory_history_extend(
                    sheet_max,
                    inv_window_end,
                    coverage_in_window,
                    ads_window=int(ADS_WINDOW),
                )

                if skip_extend:
                    ih_work = ih
                else:
                    ih_work = extend_history_with_sales(
                        ih,
                        sales_df=df,
                        cap_date=inv_window_end,
                    )
                    if ih_work is not None and not ih_work.empty:
                        ih_work = ih_work[["OMS_SKU", "Date", "Qty"]].copy()
                    else:
                        ih_work = ih

                eff_inv = effective_days_from_history(ih_work, inv_window_start, inv_window_end)
                coverage_days = coverage_days_within(ih_work, inv_window_start, inv_window_end)
                if not eff_inv.empty and coverage_days > 0:
                    po_df = po_df.merge(eff_inv, on="OMS_SKU", how="left")
                    inv_days = pd.to_numeric(po_df["Eff_Days_Inventory"], errors="coerce")
                    # Match inventory-history UI: Eff_Days = days with Qty >= 1 in the ADS
                    # window (no sales-span override, no extrapolation over missing snapshots).
                    _has_inv_hist = inv_days.notna()
                    po_df.loc[_has_inv_hist, "Eff_Days"] = (
                        inv_days[_has_inv_hist]
                        .clip(lower=0, upper=float(ADS_WINDOW))
                        .astype(int)
                    )
                    po_df["_has_inv_hist"] = _has_inv_hist
                    po_df["Eff_Days_Inventory"] = inv_days.fillna(0).astype(int)
                    po_df["Inv_Coverage_Days"] = int(coverage_days)
                else:
                    po_df["Eff_Days_Inventory"] = 0
                    po_df["Inv_Coverage_Days"] = int(coverage_days)
        except Exception as _ih_exc:
            import logging

            logging.getLogger(__name__).warning(
                "Inventory-history override skipped: %s", _ih_exc, exc_info=True
            )
            po_df["Eff_Days_Inventory"] = 0
            po_df["Inv_Coverage_Days"] = 0
    else:
        po_df["Eff_Days_Inventory"] = 0
        po_df["Inv_Coverage_Days"] = 0

    if bool(_low_vol.any()) and "_cal_span_days" in po_df.columns:
        _has_inv_hist = po_df.get("_has_inv_hist", pd.Series(False, index=po_df.index)).fillna(False)
        _low_vol_floor = _low_vol & (~_has_inv_hist)
        if bool(_low_vol_floor.any()):
            _cal_floor = pd.to_numeric(po_df["_cal_span_days"], errors="coerce").fillna(0)
            _span_floor = np.where(
                _cal_floor >= float(min_denominator),
                _cal_floor,
                float(ADS_WINDOW),
            )
            _eff_lv = (
                pd.to_numeric(po_df.loc[_low_vol_floor, "Eff_Days"], errors="coerce")
                .fillna(0)
                .to_numpy()
            )
            _floor_lv = pd.Series(_span_floor, index=po_df.index).loc[_low_vol_floor].to_numpy()
            po_df.loc[_low_vol_floor, "Eff_Days"] = np.clip(
                np.maximum(_eff_lv, _floor_lv), 0, float(ADS_WINDOW)
            )
    po_df.drop(columns=["_cal_span_days", "_has_inv_hist"], inplace=True, errors="ignore")

    # SKUs with no ADS-window sales may still show Eff_Days from the 150d ship context
    # when inventory history is missing (common for SKUs not in the daily snapshot file).
    _eff_now = pd.to_numeric(po_df.get("Eff_Days"), errors="coerce").fillna(0)
    _ship150_u = pd.to_numeric(po_df.get("Ship_Units_150d"), errors="coerce").fillna(0)
    _need_ship_span = (_eff_now <= 0) & (_ship150_u > 0)
    if bool(_need_ship_span.any()) and not df.empty:
        ship150_act = df[(df["TxnDate"] >= ship_150_cutoff) & (df["_is_ship"])].copy()
        if not ship150_act.empty:
            span150 = (
                ship150_act.groupby("Sku", as_index=False)
                .agg(_ship_first=("TxnDate", "min"), _ship_last=("TxnDate", "max"))
                .rename(columns={"Sku": "OMS_SKU"})
            )
            span150["_ship_span_days"] = (
                (span150["_ship_last"] - span150["_ship_first"]).dt.days + 1
            ).astype(int)
            po_df = _merge_metric_with_parent_fallback(
                po_df, span150[["OMS_SKU", "_ship_span_days"]], ["_ship_span_days"]
            )
            span_vals = pd.to_numeric(po_df["_ship_span_days"], errors="coerce")
            po_df.loc[_need_ship_span & span_vals.notna() & (span_vals > 0), "Eff_Days"] = (
                span_vals.clip(lower=1.0, upper=float(ADS_WINDOW))
            )
            po_df.drop(columns=["_ship_span_days"], inplace=True, errors="ignore")

    # Eff_Days must never exceed the ADS window (period_days). The 150d shipment-span
    # fallback is only for SKUs missing from inventory history — not calendar days beyond
    # the demand window (e.g. 54d span shown as "54 of 30" in the UI).
    po_df["Eff_Days"] = (
        pd.to_numeric(po_df["Eff_Days"], errors="coerce").fillna(0).clip(lower=0, upper=float(ADS_WINDOW))
    )
    _window_demand = (
        po_df["Net_Units"].fillna(0) if demand_basis == "Net" else po_df["Sold_Units"].fillna(0)
    )
    _zero_demand = pd.to_numeric(_window_demand, errors="coerce").fillna(0) <= 0
    _oos_restock_pre = _oos_restock_mask(po_df)
    po_df.loc[_zero_demand & (~_oos_restock_pre), "Eff_Days"] = 0

    ads_demand = po_df["ADS_Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["ADS_Sold_Units"]
    po_df["Recent_ADS"] = np.where(
        po_df["Eff_Days"] > 0,
        ads_demand / po_df["Eff_Days"],
        0,
    )
    _oos_restock = _oos_restock_mask(po_df)
    if bool(_oos_restock.any()):
        po_df.loc[_oos_restock, "Recent_ADS"] = _impute_oos_restock_recent_ads(
            po_df, _oos_restock, demand_basis
        )

    # Spreadsheet-style FREQ = "1 MONTH SALE" / 30. Two cases from Req.xlsx:
    # (a) Rolling last 30 calendar days / 30 — can be *below* Recent_ADS when units
    #     are front-loaded (e.g. 21/30=0.7 < 21/27=0.778), so it must NOT be the
    #     only floor.
    # (b) Calendar month-to-date shipments / 30 — teams often type MTD as "month"
    #     and still divide by 30; that reproduces FREQ when MTD > rolling/30.
    # ``df["Sku"]`` is already parent tokens when ``group_by_parent`` (see above).
    flat_sales = df
    flat_denom = 30.0
    flat_start_roll = max_date.normalize() - timedelta(days=29)
    win_roll = flat_sales[flat_sales["TxnDate"] >= flat_start_roll]
    month_start = max_date.normalize().replace(day=1)
    win_mtd = flat_sales[
        (flat_sales["TxnDate"] >= month_start) & (flat_sales["TxnDate"] <= max_date)
    ]
    if demand_basis == "Net":
        roll_g = (
            win_roll.groupby("Sku")["Units_Effective"].sum().clip(lower=0).reset_index()
            .rename(columns={"Sku": "OMS_SKU", "Units_Effective": "Roll30_Units"})
        )
        mtd_g = (
            win_mtd.groupby("Sku")["Units_Effective"].sum().clip(lower=0).reset_index()
            .rename(columns={"Sku": "OMS_SKU", "Units_Effective": "MTD_Units"})
        )
    else:
        roll_s = win_roll[win_roll["_is_ship"]]
        mtd_s = win_mtd[win_mtd["_is_ship"]]
        roll_g = roll_s.groupby("Sku")["Quantity"].sum().reset_index().rename(
            columns={"Sku": "OMS_SKU", "Quantity": "Roll30_Units"}
        )
        mtd_g = mtd_s.groupby("Sku")["Quantity"].sum().reset_index().rename(
            columns={"Sku": "OMS_SKU", "Quantity": "MTD_Units"}
        )
    flat_g = roll_g.merge(mtd_g, on="OMS_SKU", how="outer").fillna(0)
    roll_rate = pd.to_numeric(flat_g["Roll30_Units"], errors="coerce").fillna(0) / flat_denom
    mtd_rate = pd.to_numeric(flat_g["MTD_Units"], errors="coerce").fillna(0) / flat_denom
    flat_g["Flat30_ADS"] = np.maximum(roll_rate, mtd_rate).round(3)
    po_df = _merge_metric_with_parent_fallback(
        po_df, flat_g[["OMS_SKU", "Flat30_ADS"]], ["Flat30_ADS"]
    )
    po_df["Flat30_ADS"] = pd.to_numeric(po_df["Flat30_ADS"], errors="coerce").fillna(0.0)

    # ── Last-year same-window ADS (always computed) ──────────────────────────────
    # For the same calendar window one year ago, compute LY_ADS.
    # e.g. today = April 3, 2026 → LY window = April 3, 2025 → July 2, 2025
    # If April 2025 sold more than recent 90 days suggest, the PO reflects that.
    # This is the core seasonal-uplift logic the team uses manually.
    ly_window_start = max_date - timedelta(days=365)
    ly_window_end   = ly_window_start + timedelta(days=ADS_WINDOW)
    ly_window = df[(df["TxnDate"] >= ly_window_start) & (df["TxnDate"] < ly_window_end)]

    ly_sold_grp = (
        ly_window[ly_window["_is_ship"]]
        .groupby("Sku")["Quantity"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Quantity": "LY_Sold_Units"})
    )
    ly_net_grp = (
        ly_window.groupby("Sku")["Units_Effective"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Units_Effective": "LY_Net_Units"})
    )
    ly_summary = ly_sold_grp.merge(ly_net_grp, on="OMS_SKU", how="outer").fillna(0)
    po_df = _merge_metric_with_parent_fallback(
        po_df, ly_summary, ["LY_Sold_Units", "LY_Net_Units"]
    ).fillna(
        {"LY_Sold_Units": 0, "LY_Net_Units": 0}
    )
    ly_demand_col = po_df["LY_Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["LY_Sold_Units"]
    po_df["LY_ADS"] = (ly_demand_col / ADS_WINDOW).round(3)

    # Same calendar month + next month in prior years (e.g. Mar+Apr when run in March).
    seasonal_df = _seasonal_adjacent_months_ads(
        df, max_date, group_by_parent, demand_basis, years_lookback=2, min_denominator=min_denominator
    )
    po_df = _merge_metric_with_parent_fallback(
        po_df, seasonal_df, ["Seasonal_Month_ADS"]
    )
    po_df["Seasonal_Month_ADS"] = pd.to_numeric(po_df["Seasonal_Month_ADS"], errors="coerce").fillna(0.0)

    if use_seasonality:
        # Weighted blend of recent vs rolling LY window, then floor by seasonal month-pair ADS.
        blended = np.where(
            po_df["LY_ADS"] > 0,
            (po_df["Recent_ADS"] * (1 - seasonal_weight)) + (po_df["LY_ADS"] * seasonal_weight),
            po_df["Recent_ADS"],
        )
    else:
        if use_ly_fallback:
            blended = np.maximum(po_df["Recent_ADS"], po_df["LY_ADS"])
        else:
            blended = po_df["Recent_ADS"]

    # Final ADS for PO: recent signal (sold ÷ Eff_Days), optionally lifted by LY,
    # capped for non-sparse bursty SKUs, then floored by seasonal + Flat30 (sheet FREQ).
    recent_ads = pd.to_numeric(po_df["Recent_ADS"], errors="coerce").fillna(0.0)
    ly_ads = pd.to_numeric(po_df["LY_ADS"], errors="coerce").fillna(0.0)
    seasonal_ads = pd.to_numeric(po_df["Seasonal_Month_ADS"], errors="coerce").fillna(0.0)
    flat_ads = pd.to_numeric(po_df["Flat30_ADS"], errors="coerce").fillna(0.0)
    if use_seasonality:
        prim_ads = pd.to_numeric(pd.Series(blended, index=po_df.index), errors="coerce").fillna(0.0)
    else:
        if use_ly_fallback:
            prim_ads = np.maximum(recent_ads, ly_ads)
        else:
            prim_ads = recent_ads
    # Short-span Recent_ADS cannot exceed the period average (sold ÷ period_days).
    # Applies to all sellers — e.g. 6 sold in 30d → max 0.2/day even when Eff_Days=6.
    _sold_cap = (
        pd.to_numeric(po_df["Net_Units"], errors="coerce").fillna(0)
        if demand_basis == "Net"
        else pd.to_numeric(po_df["Sold_Units"], errors="coerce").fillna(0)
    )
    _period_rate = (_sold_cap / float(ADS_WINDOW)).clip(lower=0.0)
    _ceil = np.maximum(flat_ads, _period_rate)
    prim_ads = pd.Series(prim_ads, index=po_df.index, dtype=float)
    _cap_mask = (_sold_cap >= 6) & (prim_ads > _ceil)
    prim_ads = np.where(_cap_mask, np.minimum(prim_ads, _ceil), prim_ads)
    po_df["ADS"] = np.maximum.reduce([prim_ads, seasonal_ads, flat_ads]).round(3)
    po_df.drop(columns=["_sparse_intermittent"], inplace=True, errors="ignore")

    if stage_timer is not None and _engine_start is not None:
        stage_timer.mark("forecast", since=_engine_start)

    # PO formula uses OMS_Inventory (physical warehouse only) when available.
    # Total_Inventory includes marketplace stock (FBA, Myntra shelf, etc.) which is
    # already deployed and shouldn't reduce the warehouse replenishment order —
    # this matches how the team manually calculates PO.
    if "OMS_Inventory" in po_df.columns:
        inv_col = "OMS_Inventory"
    elif "Total_Inventory" in po_df.columns:
        inv_col = "Total_Inventory"
    else:
        inv_col = po_df.columns[1]

    inv_vals = pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0)

    # Gross/Net PO is finalised after pipeline merge below (sheet-style balance-days formula).
    po_df["Gross_PO_Qty"] = 0

    # Pipeline deduction from existing PO sheet
    _breakdown_cols: list[str] = []
    if not _ep_prepared.empty and "PO_Pipeline_Total" in _ep_prepared.columns:
        from .existing_po import (
            bundled_band_redundant_with_child_pipeline,
            bundled_pipeline_children_in_po,
            coalesce_pipeline_columns_on_po_df,
            existing_po_merge_key,
            fan_bundled_demand_to_sheet_children,
            is_bundled_size_range_sku,
            per_size_pipeline_covered_by_bundled_po_row,
            rollup_pipeline_onto_bundled_rows,
            unbundle_inventory_rows_for_existing_po,
            zero_bundled_pipeline_when_children_carry_qty,
            zero_bundled_po_when_sheet_children_present,
        )

        _breakdown_cols = [
            c for c in ["PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch"]
            if c in _ep_prepared.columns
        ]
        _finishing_cols = [
            c
            for c in [
                "Finishing_Issued",
                "Finishing_Received",
                "Finishing_Balance",
                "Finishing_Issue_No",
                "Finishing_Iss_Date",
                "Finishing_Receive_No",
                "Finishing_Receive_Date",
                "Finishing_JO_No",
                "Finishing_JO_Date",
                "Finishing_Status",
            ]
            if c in _ep_prepared.columns
        ]
        _merge_cols = ["OMS_SKU", "PO_Pipeline_Total"] + _breakdown_cols + _finishing_cols
        if not _ep_prepared.empty:
            _ep_merge = _ep_prepared[_merge_cols].drop_duplicates(subset=["OMS_SKU"], keep="last")
            po_df = po_df.drop(columns=["PO_Pipeline_Total"] + _breakdown_cols + _finishing_cols, errors="ignore")
            po_df = pd.merge(po_df, _ep_merge, on="OMS_SKU", how="left")
            po_df = rollup_pipeline_onto_bundled_rows(
                po_df,
                _ep_prepared,
                bundled_from_per_size_inv=_bundled_from_per_size_inv,
            )
            po_df = unbundle_inventory_rows_for_existing_po(
                po_df, _ep_prepared, _breakdown_cols, canonical_fn=existing_po_merge_key
            )
            po_df = coalesce_pipeline_columns_on_po_df(po_df, _ep_prepared)
            po_df = zero_bundled_pipeline_when_children_carry_qty(po_df, _ep_prepared)
            po_df, _fan_touched = fan_bundled_demand_to_sheet_children(
                po_df,
                _ep_prepared,
                _breakdown_cols,
                canonical_fn=existing_po_merge_key,
            )
            # Recompute ADS only on rows touched by bundled→per-size fan-out.
            if _fan_touched:
                _fan_idx = po_df.index.isin(list(_fan_touched))
                _has_eff = pd.to_numeric(po_df["Eff_Days"], errors="coerce").fillna(0) > 0
                _fan_active = _fan_idx & _has_eff
                if _fan_active.any():
                    _ads_demand = (
                        pd.to_numeric(po_df.loc[_fan_active, "Net_Units"], errors="coerce").fillna(0)
                        if demand_basis == "Net"
                        else pd.to_numeric(po_df.loc[_fan_active, "Sold_Units"], errors="coerce").fillna(0)
                    )
                    po_df.loc[_fan_active, "Recent_ADS"] = (
                        _ads_demand
                        / pd.to_numeric(po_df.loc[_fan_active, "Eff_Days"], errors="coerce").clip(lower=1)
                    ).round(3)
                    _recent = pd.to_numeric(po_df["Recent_ADS"], errors="coerce").fillna(0.0)
                    _ly = pd.to_numeric(po_df["LY_ADS"], errors="coerce").fillna(0.0)
                    _seasonal = pd.to_numeric(po_df["Seasonal_Month_ADS"], errors="coerce").fillna(0.0)
                    _flat = pd.to_numeric(po_df["Flat30_ADS"], errors="coerce").fillna(0.0)
                    if use_seasonality:
                        _prim = np.where(
                            _ly > 0,
                            (_recent * (1 - seasonal_weight)) + (_ly * seasonal_weight),
                            _recent,
                        )
                    elif use_ly_fallback:
                        _prim = np.maximum(_recent, _ly)
                    else:
                        _prim = _recent
                    po_df.loc[_fan_active, "ADS"] = (
                        np.maximum.reduce(
                            [
                                pd.Series(_prim, index=po_df.index).loc[_fan_active],
                                _seasonal.loc[_fan_active],
                                _flat.loc[_fan_active],
                            ]
                        )
                    ).round(3)
                _fan_zero = _fan_idx & ~_has_eff
                if _fan_zero.any():
                    po_df.loc[_fan_zero, "Recent_ADS"] = 0.0
                    po_df.loc[_fan_zero, "ADS"] = 0.0
            # Unbundled per-size rows keep pipeline qty only — not parent/bundled Eff_Days.
            # Do not wipe inventory-based active days (e.g. 1361YKBLUE-XL with 8/30 in-stock
            # days) — only clear inherited Eff_Days when there is no demand and no history.
            _no_row_demand = (
                pd.to_numeric(po_df["Net_Units"], errors="coerce").fillna(0) <= 0
            ) & (
                pd.to_numeric(po_df["Sold_Units"], errors="coerce").fillna(0) <= 0
            )
            _no_inv_hist = (
                pd.to_numeric(po_df.get("Eff_Days_Inventory"), errors="coerce").fillna(0) <= 0
            )
            _has_ship_ctx = pd.to_numeric(po_df.get("Ship_Units_150d"), errors="coerce").fillna(0) > 0
            _has_stock = pd.to_numeric(po_df.get("Total_Inventory"), errors="coerce").fillna(0) > 0
            # Keep ship-150 / in-stock Eff_Days set earlier — do not wipe after unbundle.
            po_df.loc[
                _no_row_demand & _no_inv_hist & ~_has_ship_ctx & ~_has_stock,
                "Eff_Days",
            ] = 0
        po_df["PO_Pipeline_Total"] = pd.to_numeric(
            po_df["PO_Pipeline_Total"], errors="coerce"
        ).fillna(0).astype(int)
        for _bc in _breakdown_cols:
            po_df[_bc] = pd.to_numeric(po_df[_bc], errors="coerce").fillna(0).astype(int)
        for _fc in _finishing_cols:
            if _fc in (
                "Finishing_Issue_No",
                "Finishing_Iss_Date",
                "Finishing_Receive_No",
                "Finishing_Receive_Date",
                "Finishing_JO_No",
                "Finishing_JO_Date",
                "Finishing_Status",
            ):
                po_df[_fc] = po_df[_fc].fillna("").astype(str)
            else:
                po_df[_fc] = pd.to_numeric(po_df[_fc], errors="coerce").fillna(0).astype(int)
    else:
        po_df["PO_Pipeline_Total"] = 0

    # Days of stock remaining = current inventory cover only (no pipeline).
    if "Total_Inventory" in po_df.columns:
        inv_days_left = pd.to_numeric(po_df["Total_Inventory"], errors="coerce").fillna(0)
    else:
        inv_days_left = inv_vals
    po_df["Days_Left"] = np.where(
        po_df["ADS"] > 0,
        (inv_days_left / po_df["ADS"]).round(1),
        999.0,
    )

    # ── Inject PO-sheet SKUs missing from inventory ──────────────
    # If a SKU has an active pipeline order but isn't in the inventory file
    # (e.g. out of stock, removed from listing), add it as a ghost row so it
    # still shows up as "🔄 In Pipeline" and isn't invisible to the user.
    if not _ep_prepared.empty:
        _ep_ghost = _ep_prepared
        _po_keys = set(po_df["OMS_SKU"].astype(str).str.strip())
        _pipe_canon = _ep_ghost["OMS_SKU"].astype(str).str.strip()
        _ep_active = pd.to_numeric(_ep_ghost["PO_Pipeline_Total"], errors="coerce").fillna(0) > 0
        for _ac in ("Pending_Cutting", "Balance_to_Dispatch", "PO_Qty_Ordered"):
            if _ac in _ep_ghost.columns:
                _ep_active |= pd.to_numeric(_ep_ghost[_ac], errors="coerce").fillna(0) > 0
        missing_mask = ~_pipe_canon.isin(_po_keys) & _ep_active
        if missing_mask.any():
            for idx in _ep_ghost.index[missing_mask]:
                sk = existing_po_merge_key(_ep_ghost.at[idx, "OMS_SKU"])
                if is_bundled_size_range_sku(sk) and (
                    bundled_pipeline_children_in_po(sk, _ep_ghost, _po_keys)
                    or bundled_band_redundant_with_child_pipeline(sk, po_df, _ep_ghost)
                ):
                    missing_mask.at[idx] = False
                elif per_size_pipeline_covered_by_bundled_po_row(sk, po_df):
                    missing_mask.at[idx] = False
        missing_po = _ep_ghost[missing_mask].copy()
        if not missing_po.empty:
            ghost = pd.DataFrame(
                {"OMS_SKU": [existing_po_merge_key(x) for x in missing_po["OMS_SKU"].values]},
            )
            ghost["Total_Inventory"] = 0
            ghost["Sold_Units"]      = 0
            ghost["Return_Units"]    = 0
            ghost["Net_Units"]       = 0
            ghost["Recent_ADS"]      = 0.0
            ghost["ADS"]             = 0.0
            ghost["LY_ADS"]          = 0.0
            ghost["Seasonal_Month_ADS"] = 0.0
            ghost["Flat30_ADS"]      = 0.0
            ghost["Days_Left"]       = 999.0
            ghost["Gross_PO_Qty"]    = 0
            ghost["Lead_Time_Days"] = _lead_time_column_fallback(lead_time)
            ghost["SKU_Sheet_Status"] = ""
            ghost["SKU_Sheet_Closed"] = False
            ghost["PO_Block_Reason"] = ""
            ghost["Ship_Units_150d"] = 0
            ghost["Suggest_Close_SKU"] = ""
            for c in ["PO_Pipeline_Total"] + _breakdown_cols:
                ghost[c] = missing_po[c].values if c in missing_po.columns else 0
            # Fill any other columns po_df already has
            for c in po_df.columns:
                if c in ghost.columns:
                    continue
                dt = po_df[c].dtype
                if c in ("SKU_Sheet_Closed",):
                    ghost[c] = False
                elif dt == object or str(dt).startswith("string"):
                    ghost[c] = ""
                else:
                    ghost[c] = 0
            po_df = pd.concat([po_df, ghost[po_df.columns]], ignore_index=True)

    # ── SKU Status / Lead sheet (optional, after pipeline ghost rows) ────────────
    # Upload overrides lead days per SKU / parent style. Without this pass, every row
    # keeps the global ``lead_time`` default — operators often mistake that for “missing”.
    _status_catalog_keys: set[str] = set()
    if sku_status_df is not None and not sku_status_df.empty:
        po_df.drop(columns=["SKU_Sheet_Status", "SKU_Sheet_Closed"], inplace=True, errors="ignore")
        m = sku_status_df.copy()
        _uniq_ss = m["OMS_SKU"].unique()
        _ss_canon = {s: _canonical_oms_key(s) for s in _uniq_ss}
        m["OMS_SKU"] = m["OMS_SKU"].map(_ss_canon).fillna("")
        m = m[m["OMS_SKU"].str.len() > 0]
        _status_catalog_keys = set(m["OMS_SKU"].astype(str).str.strip())

        def _max_positive_lead(series: pd.Series) -> float:
            v = pd.to_numeric(series, errors="coerce")
            v = v[v > 0]
            return float(v.max()) if len(v) else float("nan")

        m["_par_key"] = m["OMS_SKU"].map(get_parent_sku)
        lead_by_parent = m.groupby("_par_key")["Lead_Time_From_Sheet"].apply(_max_positive_lead).to_dict()
        _sorted_lead_parent_keys = sorted(
            (str(sk).strip() for sk in lead_by_parent if str(sk).strip()),
            key=len,
            reverse=True,
        )

        try:
            from .sku_status_lead import is_closed_sku_status as _is_closed_st
        except Exception:  # pragma: no cover
            def _is_closed_st(_x: object) -> bool:
                return False

        m["_st_clean"] = m["SKU_Sheet_Status"].astype(str).str.strip()
        m["_closed_row"] = (
            m["SKU_Sheet_Closed"].fillna(False).astype(bool) | m["_st_clean"].map(_is_closed_st)
        )
        status_by_parent: dict[str, str] = {}
        closed_by_parent: dict[str, bool] = {}
        for par, grp in m.groupby("_par_key", sort=False):
            par_s = str(par or "").strip()
            if not par_s:
                continue
            closed_by_parent[par_s] = bool(grp["_closed_row"].any())
            closed_st = grp.loc[grp["_closed_row"] & grp["_st_clean"].str.len().gt(0), "_st_clean"]
            if not closed_st.empty:
                status_by_parent[par_s] = str(closed_st.iloc[0])
            else:
                nonempty = grp.loc[grp["_st_clean"].str.len().gt(0), "_st_clean"]
                status_by_parent[par_s] = str(nonempty.iloc[-1]) if not nonempty.empty else ""

        lead_by_digit_token: dict[str, float] = {}
        m["_lt_pos"] = pd.to_numeric(m["Lead_Time_From_Sheet"], errors="coerce")
        m_lt = m[m["_lt_pos"] > 0]
        if not m_lt.empty:
            tok_frames = [
                m_lt.assign(_tok=m_lt["OMS_SKU"].map(_style_digit_token)),
                m_lt.assign(_tok=m_lt["_par_key"].map(_style_digit_token)),
            ]
            for tf in tok_frames:
                sub = tf[tf["_tok"].astype(str).str.len().gt(0)]
                if sub.empty:
                    continue
                for tok, val in sub.groupby("_tok")["_lt_pos"].max().items():
                    tok_s = str(tok)
                    prev = lead_by_digit_token.get(tok_s, float("nan"))
                    if not np.isfinite(prev) or float(val) > prev:
                        lead_by_digit_token[tok_s] = float(val)

        # Sheet rows whose entire OMS key is a style numeric (e.g. ``1394``) are a direct
        # factory style code — match inventory by the same style digit token as parent/OMS.
        # This is **not** the same as ``lead_by_digit_token`` (which keys off any substring
        # token from keys like ``PREFIX-4002`` and must not satisfy the sheet PO gate alone).
        lead_by_pure_digit_style: dict[str, float] = {}
        status_by_pure_digit_style: dict[str, tuple[str, bool]] = {}
        pure = m[m["OMS_SKU"].astype(str).str.strip().str.fullmatch(r"\d{3,}", na=False)].copy()
        if not pure.empty:
            pure["_oms_c"] = pure["OMS_SKU"].astype(str).str.strip()
            for oms_c, grp in pure.groupby("_oms_c", sort=False):
                st_one = str(grp["_st_clean"].iloc[-1])
                cl_one = bool(grp["_closed_row"].any())
                status_by_pure_digit_style[oms_c] = (st_one, cl_one)
                lt_max = float(pd.to_numeric(grp["Lead_Time_From_Sheet"], errors="coerce").max())
                if np.isfinite(lt_max) and lt_max > 0:
                    lead_by_pure_digit_style[oms_c] = lt_max

        if group_by_parent:
            m = (
                m.groupby("_par_key", as_index=False)
                .agg(
                    Lead_Time_From_Sheet=("Lead_Time_From_Sheet", _max_positive_lead),
                    SKU_Sheet_Status=("SKU_Sheet_Status", "first"),
                    SKU_Sheet_Closed=("SKU_Sheet_Closed", "max"),
                )
                .rename(columns={"_par_key": "OMS_SKU"})
            )
        else:
            m = m.drop(columns=["_par_key"], errors="ignore")

        keep = [c for c in ["OMS_SKU", "SKU_Sheet_Status", "SKU_Sheet_Closed", "Lead_Time_From_Sheet"] if c in m.columns]
        m = m[keep].drop_duplicates(subset=["OMS_SKU"], keep="last")
        sheet_key_meta: dict[str, tuple[str, bool]] = {}
        mk = m.copy()
        mk["_stc"] = mk["SKU_Sheet_Status"].astype(str).str.strip()
        mk["_clc"] = (
            mk["SKU_Sheet_Closed"].fillna(False).astype(bool) | mk["_stc"].map(_is_closed_st)
        )
        for k, stc, clc in zip(
            mk["OMS_SKU"].astype(str).str.strip(),
            mk["_stc"],
            mk["_clc"],
        ):
            if k:
                sheet_key_meta[k] = (str(stc), bool(clc))
        _sorted_sheet_keys_for_status = sorted((x for x in sheet_key_meta if x), key=len, reverse=True)

        po_df = po_df.merge(m, on="OMS_SKU", how="left")
        po_df["SKU_Sheet_Status"] = po_df["SKU_Sheet_Status"].fillna("").astype(str)
        po_df["SKU_Sheet_Closed"] = po_df["SKU_Sheet_Closed"].fillna(False).astype(bool)

        _pk_all = po_df["OMS_SKU"].astype(str).map(get_parent_sku).str.strip()
        _st_empty_m = po_df["SKU_Sheet_Status"].astype(str).str.strip().eq("")
        _fill_st_par = _pk_all.map(status_by_parent).fillna("").astype(str)
        _upd_par = _st_empty_m & _fill_st_par.str.len().gt(0)
        if _upd_par.any():
            po_df.loc[_upd_par, "SKU_Sheet_Status"] = _fill_st_par.loc[_upd_par]
            _cl_par = _pk_all.map(closed_by_parent).fillna(False)
            po_df.loc[_upd_par, "SKU_Sheet_Closed"] = (
                po_df.loc[_upd_par, "SKU_Sheet_Closed"].astype(bool).to_numpy()
                | _cl_par.loc[_upd_par].astype(bool).to_numpy()
            )

        _st_empty_p = po_df["SKU_Sheet_Status"].astype(str).str.strip().eq("")
        if bool(_st_empty_p.any()) and sheet_key_meta:

            def _pfx_meta(o: str) -> tuple[str, bool]:
                pk_here = str(get_parent_sku(o)).strip()
                return _longest_prefix_sheet_meta(pk_here, sheet_key_meta, _sorted_sheet_keys_for_status)

            _pairs = po_df["OMS_SKU"].astype(str).map(_pfx_meta)
            _st_part = _pairs.map(lambda t: t[0] if isinstance(t, tuple) else "")
            _cl_part = _pairs.map(lambda t: bool(t[1]) if isinstance(t, tuple) else False)
            _use_pfx = _st_empty_p & _st_part.astype(str).str.len().gt(0)
            if _use_pfx.any():
                po_df.loc[_use_pfx, "SKU_Sheet_Status"] = _st_part.loc[_use_pfx]
                po_df.loc[_use_pfx, "SKU_Sheet_Closed"] = (
                    po_df.loc[_use_pfx, "SKU_Sheet_Closed"].astype(bool).to_numpy()
                    | _cl_part.loc[_use_pfx].astype(bool).to_numpy()
                )

        _st_empty_d = po_df["SKU_Sheet_Status"].astype(str).str.strip().eq("")
        if bool(_st_empty_d.any()) and status_by_pure_digit_style:
            _tok_d = po_df["OMS_SKU"].astype(str).map(get_parent_sku).map(_style_digit_token)

            def _dig_status(tok: str) -> tuple[str, bool]:
                if not tok:
                    return "", False
                return status_by_pure_digit_style.get(tok, ("", False))

            _dig_pairs = _tok_d.map(_dig_status)
            _dst = _dig_pairs.map(lambda t: t[0] if isinstance(t, tuple) else "")
            _dcl = _dig_pairs.map(lambda t: bool(t[1]) if isinstance(t, tuple) else False)
            _use_dig = _st_empty_d & _dst.astype(str).str.len().gt(0)
            if _use_dig.any():
                po_df.loc[_use_dig, "SKU_Sheet_Status"] = _dst.loc[_use_dig]
                po_df.loc[_use_dig, "SKU_Sheet_Closed"] = (
                    po_df.loc[_use_dig, "SKU_Sheet_Closed"].astype(bool).to_numpy()
                    | _dcl.loc[_use_dig].astype(bool).to_numpy()
                )

        po_df["SKU_Sheet_Closed"] = po_df["SKU_Sheet_Closed"] | po_df["SKU_Sheet_Status"].map(_is_closed_st)
        if "Lead_Time_From_Sheet" in po_df.columns:
            lt_vals = pd.to_numeric(po_df["Lead_Time_From_Sheet"], errors="coerce")
            bad = lt_vals.isna() | (lt_vals <= 0)
            # Always attempt parent / longest-prefix fill when the sheet merge missed —
            # ``if lead_by_parent`` was accidentally skipping fallbacks whenever the dict was empty.
            if bad.any():
                pk = po_df["OMS_SKU"].astype(str).map(get_parent_sku)
                fill_s = pd.to_numeric(pk.map(lead_by_parent), errors="coerce")
                use = bad & (fill_s > 0)
                lt_vals = lt_vals.where(~use, fill_s)
                bad = lt_vals.isna() | (lt_vals <= 0)
                if bad.any():
                    s_pk = pk.fillna("").astype(str).str.strip()
                    uniq_bad = pd.unique(s_pk[bad].to_numpy())
                    pfx_map: dict[str, float] = {}
                    for p in uniq_bad:
                        if not p or p.lower() == "nan":
                            continue
                        pfx_map[p] = _longest_prefix_lead(
                            p, lead_by_parent, _sorted_lead_parent_keys
                        )
                    fill_pfx = pd.to_numeric(s_pk.map(pfx_map), errors="coerce")
                    use2 = bad & (fill_pfx > 0)
                    lt_vals = lt_vals.where(~use2, fill_pfx)
            bad = lt_vals.isna() | (lt_vals <= 0)
            if bad.any() and lead_by_pure_digit_style:
                pk_d = po_df["OMS_SKU"].astype(str).map(get_parent_sku)
                s_pk_d = pk_d.fillna("").astype(str).str.strip()
                s_row_d = po_df["OMS_SKU"].astype(str).str.strip()
                tok_p = s_pk_d.map(_style_digit_token)
                tok_r = s_row_d.map(_style_digit_token)
                fill_pd = pd.to_numeric(tok_p.map(lead_by_pure_digit_style), errors="coerce")
                fill_rd = pd.to_numeric(tok_r.map(lead_by_pure_digit_style), errors="coerce")
                fill_dig_style = pd.concat([fill_pd, fill_rd], axis=1).max(axis=1, skipna=True)
                use_ds = bad & (fill_dig_style > 0)
                lt_vals = lt_vals.where(~use_ds, fill_dig_style)
            bad = lt_vals.isna() | (lt_vals <= 0)
            # Snapshot before digit-token fill: digit borrow can attach an unrelated style's
            # factory lead to this SKU for *display* math, but it must not satisfy the
            # "status sheet supplied a lead for this SKU" PO release gate.
            lt_vals_for_gate = lt_vals.copy()
            if bad.any() and lead_by_digit_token:
                pk2 = po_df["OMS_SKU"].astype(str).map(get_parent_sku)
                dig = pk2.map(_style_digit_token)
                fill2 = pd.to_numeric(dig.map(lead_by_digit_token), errors="coerce")
                use2 = bad & (fill2 > 0)
                lt_vals = lt_vals.where(~use2, fill2)
            # True when we resolved a positive lead from the status sheet via a direct row,
            # parent rollup, longest-prefix match, or a pure numeric style code row — **not**
            # digit-token substring inference alone (e.g. ``PREFIX-4002`` → unrelated SKUs).
            lt_num_gate = pd.to_numeric(lt_vals_for_gate, errors="coerce")
            po_df["Lead_Time_From_Status_Sheet"] = (lt_num_gate > 0) & lt_num_gate.notna()
            lt_num = pd.to_numeric(lt_vals, errors="coerce")
            repl = lt_vals.where(lt_vals > 0)
            po_df["Lead_Time_Days"] = (
                pd.to_numeric(repl, errors="coerce")
                .fillna(po_df["Lead_Time_Days"])
                .clip(lower=1, upper=730)
                .round()
                .astype(int)
            )
            po_df.drop(columns=["Lead_Time_From_Sheet"], inplace=True, errors="ignore")
        else:
            po_df["Lead_Time_From_Status_Sheet"] = False

    po_df["Lead_Time_Days"] = (
        pd.to_numeric(po_df["Lead_Time_Days"], errors="coerce")
        .fillna(_lead_time_column_fallback(lead_time))
        .clip(lower=1, upper=730)
        .astype(int)
    )

    # ── In-app confirmed PO raises (Export & Confirm) ───────────────────────────
    # Merged into effective pipeline so tomorrow's run does not re-recommend the
    # same release for a SKU before sheet pipeline / inventory reflect it.
    _as_of_plan = _plan if _plan is not None else pd.Timestamp.now().normalize()

    lag = pd.DataFrame()
    if po_raise_ledger_df is not None and not po_raise_ledger_df.empty:
        from .po_raise_ledger import aggregate_raise_ledger_for_po

        lag = aggregate_raise_ledger_for_po(
            po_raise_ledger_df,
            _map,
            _as_of_plan,
            int(max(1, raise_ledger_lookback_days)),
            group_by_parent,
            raise_view_date=raise_view_date,
        )
    if lag is not None and not lag.empty:
        lag_merge_cols = [
            c
            for c in lag.columns
            if c != "OMS_SKU"
            and (
                c.startswith("PO_")
                or c == "PO_Last_Raised_Date"
            )
        ]
        po_df = po_df.merge(lag, on="OMS_SKU", how="left")

        # Variant-level PO rows: match ledger on each OMS_SKU (size), not parent totals.
        # Only fall back to a parent-key ledger row when import used the parent SKU
        # (no per-size row in the ledger for that variant).
        if not group_by_parent:

            def _raise_parent_key(raw: str) -> str:
                key = canonical_oms_key(raw, _map)
                return str(get_parent_sku(key) or key).strip().upper()

            def _ledger_row_is_parent_key(raw: str) -> bool:
                key = canonical_oms_key(raw, _map)
                return str(key).strip().upper() == _raise_parent_key(key)

            parent_lag = lag[lag["OMS_SKU"].map(_ledger_row_is_parent_key)].copy()
            if not parent_lag.empty:
                parent_lag["_raise_parent"] = parent_lag["OMS_SKU"].map(_raise_parent_key)
                parent_lag = parent_lag.drop_duplicates(subset=["_raise_parent"], keep="last")
                # If any per-size ledger rows exist for a parent, do not broadcast the parent
                # total onto other sizes (e.g. parent 2100 + 5XL 350 must not give 3XL 2100).
                parents_with_variant_ledger: set[str] = set()
                for raw in lag["OMS_SKU"].astype(str):
                    key = canonical_oms_key(raw, _map)
                    pk = _raise_parent_key(key)
                    if str(key).strip().upper() != pk:
                        parents_with_variant_ledger.add(pk)
                if parents_with_variant_ledger:
                    parent_lag = parent_lag[
                        ~parent_lag["_raise_parent"].isin(parents_with_variant_ledger)
                    ]
                po_df["_raise_parent"] = po_df["OMS_SKU"].map(_raise_parent_key)
                missing = pd.to_numeric(
                    po_df.get("PO_Confirmed_Raise_Pipeline"), errors="coerce"
                ).fillna(0) <= 0
                if missing.any() and not parent_lag.empty:
                    fill = po_df.loc[missing, ["OMS_SKU", "_raise_parent"]].merge(
                        parent_lag.drop(columns=["OMS_SKU"], errors="ignore"),
                        on="_raise_parent",
                        how="left",
                    )
                    # Only pipeline totals may fall back to a parent-key ledger row.
                    # Per-size "last raised" / day columns must never inherit parent totals.
                    _parent_fill_cols = {
                        c
                        for c in lag_merge_cols
                        if c in ("PO_Confirmed_Raise_Pipeline",)
                    }
                    for c in _parent_fill_cols:
                        if c not in fill.columns:
                            continue
                        src = fill[c]
                        po_df.loc[missing, c] = (
                            pd.to_numeric(src, errors="coerce").fillna(0).astype(int).values
                        )
                po_df.drop(columns=["_raise_parent"], inplace=True, errors="ignore")
    for c in (
        "PO_Confirmed_Raise_Pipeline",
        "PO_Raised_Yesterday",
        "PO_Raised_Today",
        "PO_Last_Raised_Qty",
        "PO_Raised_On_View_Date",
    ):
        if c not in po_df.columns:
            po_df[c] = 0
        po_df[c] = pd.to_numeric(po_df[c], errors="coerce").fillna(0).astype(int)
    if "PO_Last_Raised_Date" not in po_df.columns:
        po_df["PO_Last_Raised_Date"] = ""
    else:
        po_df["PO_Last_Raised_Date"] = po_df["PO_Last_Raised_Date"].fillna("").astype(str)

    _pipe_total_s = pd.to_numeric(po_df["PO_Pipeline_Total"], errors="coerce").fillna(0).astype(int)
    _conf_raise_s = pd.to_numeric(po_df["PO_Confirmed_Raise_Pipeline"], errors="coerce").fillna(0).astype(int)
    # Confirmed raises bridge the gap between "confirmed in app" and "reflected in existing-PO
    # upload."  Once the uploaded existing-PO already carries that pipeline (PO_Pipeline_Total > 0),
    # only count the EXCESS confirmed raises to avoid double-counting the same orders.
    # effective = PO_Pipeline_Total + max(0, confirmed_raises − PO_Pipeline_Total)
    #           = max(PO_Pipeline_Total, confirmed_raises)
    po_df["PO_Pipeline_Effective"] = np.maximum(_pipe_total_s.to_numpy(), _conf_raise_s.to_numpy()).astype(int)

    # projected_days_now = (Total_Inventory + effective pipeline) / ADS
    #
    # PO quantity (hybrid lead gate):
    # • Lead gate ON  — release when projected cover (inv + pipeline) is below lead days:
    #   operator-entered lead_time when >0, else per-SKU sheet ``Lead_Time_Days``.
    #   Qty tops up toward post-PO target using full projected days.
    # • Lead gate OFF — balance-days toward target_cover_days (same qty rule; no lead gate).
    target_cover_days = float(max(0, target_days + grace_days))
    _lt_gate_arr = _lead_time_release_gate_days(po_df, lead_time)
    _lt_gate_uses_entered = _lead_time_gate_uses_entered(lead_time)
    if "Total_Inventory" in po_df.columns:
        inv_for_cover = pd.to_numeric(po_df["Total_Inventory"], errors="coerce").fillna(0.0)
    else:
        inv_for_cover = pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0.0)
    ads_num = pd.to_numeric(po_df["ADS"], errors="coerce").fillna(0.0)
    pipe_num = pd.to_numeric(po_df["PO_Pipeline_Effective"], errors="coerce").fillna(0.0)
    projected_days_now = np.where(ads_num > 0, (inv_for_cover + pipe_num) / ads_num, 999.0)
    _ads_arr = ads_num.to_numpy(dtype=float)
    _inv_arr = inv_for_cover.to_numpy(dtype=float)
    _pipe_arr = pipe_num.to_numpy(dtype=float)
    _proj_arr = np.asarray(projected_days_now, dtype=float)

    if enforce_lead_time_release_gate:
        need_po = (_ads_arr > 0) & (_proj_arr < _lt_gate_arr)
        balance_days = np.maximum(target_cover_days - _proj_arr, 0.0)
        shortfall_units = _ads_arr * balance_days
        po_qty_round = np.where(
            need_po,
            np.vectorize(round_po_pack)(shortfall_units),
            0.0,
        ).astype(int)
    else:
        balance_days = target_cover_days - projected_days_now
        raw_po = ads_num * balance_days
        po_qty_round = np.vectorize(round_po_pack)(np.maximum(raw_po, 0.0)).astype(int)

    # Closed / doubt / sales-after-closed SKUs must never receive a PO recommendation.
    try:
        from .sku_status_lead import is_excluded_po_status as _is_excl_st
    except Exception:  # pragma: no cover
        def _is_excl_st(_x: object) -> bool:
            return False

    _st_excl = po_df.get("SKU_Sheet_Status", pd.Series("", index=po_df.index)).astype(str).map(_is_excl_st)
    _cl_excl = po_df.get("SKU_Sheet_Closed", pd.Series(False, index=po_df.index)).fillna(False).astype(bool)
    _excl_mask = (_st_excl | _cl_excl).fillna(False).to_numpy(dtype=bool)
    if _excl_mask.any():
        po_qty_round = np.where(_excl_mask, 0, po_qty_round)

    po_df["Gross_PO_Qty"] = po_qty_round.astype(int)
    po_df["PO_Qty"] = po_df["Gross_PO_Qty"].astype(int)

    if "Return_Overlay_Units" in po_df.columns:
        overlay = pd.to_numeric(po_df["Return_Overlay_Units"], errors="coerce").fillna(0).astype(int)
        net_po = np.maximum(
            pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0).astype(int) - overlay,
            0,
        )
        po_df["PO_Qty"] = np.vectorize(round_po_pack)(net_po).astype(int)
        po_df["Gross_PO_Qty"] = po_df["PO_Qty"]

    # Surface why gated mode left PO at zero when projected cover already meets lead gate.
    _msg_lead_window = _lead_time_gate_block_message(_lt_gate_uses_entered)
    if enforce_lead_time_release_gate:
        _block_lead_win = (_ads_arr > 0) & (_proj_arr >= _lt_gate_arr)
        if bool(np.any(_block_lead_win)):
            _idx_bw = po_df.index[_block_lead_win]
            br_bw = po_df.loc[_idx_bw, "PO_Block_Reason"].astype(str).str.strip()
            po_df.loc[_idx_bw, "PO_Block_Reason"] = np.where(
                br_bw.eq("") | br_bw.eq("nan"),
                _msg_lead_window,
                br_bw + "; " + _msg_lead_window,
            )

    # Two-size minimum rule. Only one size in a parent SKU having demand is
    # almost always a data / sizing-mix problem, not a real demand signal —
    # production won't cut a single size economically, and the operator
    # usually wants to alter the SKU to redistribute the demand. We block
    # the PO and surface an actionable recommendation instead of a raw
    # "single size only" reason.
    _RECOMMEND_MSG = (
        "Only 1 size in parent SKU has demand — recommend altering this SKU "
        "to another size before raising PO"
    )
    if enforce_two_size_minimum:
        _par_key = po_df["OMS_SKU"].apply(get_parent_sku)
        has_g = pd.to_numeric(po_df["Gross_PO_Qty"], errors="coerce").fillna(0) > 0
        n_sizes = has_g.astype(int).groupby(_par_key).transform("sum")
        single_only = has_g & (n_sizes == 1)
        po_df.loc[single_only, "Gross_PO_Qty"] = 0
        po_df.loc[single_only, "PO_Qty"] = 0
        br = po_df.loc[single_only, "PO_Block_Reason"].astype(str).str.strip()
        po_df.loc[single_only, "PO_Block_Reason"] = np.where(
            br.eq("") | br.eq("nan"),
            _RECOMMEND_MSG,
            br + "; " + _RECOMMEND_MSG,
        )

    if enforce_two_size_minimum:
        _par_key_final = po_df["OMS_SKU"].apply(get_parent_sku)
        has_final_po = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0
        n_sizes_final = has_final_po.astype(int).groupby(_par_key_final).transform("sum")
        single_only_final = has_final_po & (n_sizes_final == 1)
        po_df.loc[single_only_final, "PO_Qty"] = 0
        br = po_df.loc[single_only_final, "PO_Block_Reason"].astype(str).str.strip()
        po_df.loc[single_only_final, "PO_Block_Reason"] = np.where(
            br.eq("") | br.eq("nan"),
            _RECOMMEND_MSG,
            br + "; " + _RECOMMEND_MSG,
        )

    if not _ep_prepared.empty and "PO_Pipeline_Total" in _ep_prepared.columns:
        from .existing_po import existing_po_merge_key, zero_bundled_po_when_sheet_children_present

        po_df = zero_bundled_po_when_sheet_children_present(
            po_df,
            _ep_prepared,
            _breakdown_cols if _breakdown_cols else None,
            canonical_fn=existing_po_merge_key,
        )

    # SKU Status sheet rules (after every automated PO_Qty adjustment):
    # • Closed / Doubt / Sales-after-closed must never receive a fresh PO release.
    # • When a status sheet IS loaded, a positive lead must have been resolvable from that
    #   sheet (direct row, parent rollup, longest-prefix, or pure numeric style code row) —
    #   not merely the global default, and not digit-token-only inference from non-numeric
    #   keys (which can borrow an unrelated style).
    from .sku_status_lead import is_excluded_po_status, po_block_reason_for_excluded_status

    # Safety guard: zero out any SKU whose CURRENT coverage already meets or exceeds the
    # target horizon.  The balance-days formula already handles this arithmetically, but
    # pack-size rounding or sibling-lift logic can occasionally leave a non-zero PO_Qty
    # on an already-covered SKU.  This explicit check is the belt-and-suspenders guarantee.
    _proj_cover_now = np.where(
        ads_num > 0, (inv_for_cover + pipe_num) / ads_num, 999.0
    )
    _already_at_target = (_proj_cover_now >= target_cover_days) & (
        pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0
    )
    if _already_at_target.any():
        po_df.loc[_already_at_target, "PO_Qty"] = 0
        po_df.loc[_already_at_target, "Gross_PO_Qty"] = 0
        _msg_covered = f"Current cover ≥ {int(target_cover_days)}d target — no PO needed"
        br = po_df.loc[_already_at_target, "PO_Block_Reason"].astype(str).str.strip()
        po_df.loc[_already_at_target, "PO_Block_Reason"] = np.where(
            br.eq("") | br.eq("nan"), _msg_covered, br + "; " + _msg_covered
        )

    _closed = po_df.get("SKU_Sheet_Closed", pd.Series(False, index=po_df.index)).fillna(False).astype(bool)
    _status_excl = po_df["SKU_Sheet_Status"].astype(str).map(is_excluded_po_status)
    _excluded = (_closed | _status_excl).fillna(False).astype(bool)
    _hot = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0
    _msg_nolead = "No lead time resolved from SKU status sheet for this SKU"
    block_excluded = _excluded & _hot
    if block_excluded.any():
        po_df.loc[block_excluded, "PO_Qty"] = 0
        po_df.loc[block_excluded, "Gross_PO_Qty"] = 0
        br = po_df.loc[block_excluded, "PO_Block_Reason"].astype(str).str.strip()
        reasons = po_df.loc[block_excluded, "SKU_Sheet_Status"].map(po_block_reason_for_excluded_status)
        po_df.loc[block_excluded, "PO_Block_Reason"] = np.where(
            br.eq("") | br.eq("nan"),
            reasons,
            br + "; " + reasons.astype(str),
        )
        po_df.loc[block_excluded, "Lead_Time_Days"] = 0
    if "Lead_Time_From_Status_Sheet" in po_df.columns:
        _sheet_ok = po_df["Lead_Time_From_Status_Sheet"].fillna(False).astype(bool)
        _hot2 = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0
        block_lt = (~_sheet_ok) & _hot2
        # When the loaded status sheet only covers part of the catalog, do not zero PO
        # for SKUs whose parent style never appeared on that sheet — they keep the global
        # lead_time default.  Full-sheet uploads still require a sheet-resolved lead per row.
        if _status_catalog_keys and len(po_df) > 100:
            _sku_k = po_df["OMS_SKU"].astype(str).str.strip()
            _par_k = _sku_k.map(get_parent_sku).astype(str).str.strip()
            _governed = _sku_k.isin(_status_catalog_keys) | _par_k.isin(_status_catalog_keys)
            block_lt = block_lt & _governed
        if block_lt.any():
            po_df.loc[block_lt, "PO_Qty"] = 0
            po_df.loc[block_lt, "Gross_PO_Qty"] = 0
            br = po_df.loc[block_lt, "PO_Block_Reason"].astype(str).str.strip()
            po_df.loc[block_lt, "PO_Block_Reason"] = np.where(
                br.eq("") | br.eq("nan"),
                _msg_nolead,
                br + "; " + _msg_nolead,
            )
            # Do not leave the global default ``lead_time`` in ``Lead_Time_Days`` for rows
            # we just blocked — operators read that column as "authoritative per-SKU lead"
            # and mistake digit/global fallbacks for a real sheet row.
            po_df.loc[block_lt, "Lead_Time_Days"] = 0

    inv_for_metrics = pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0)

    # Stockout flag (after ghost rows — length must match po_df)
    po_df["Stockout_Flag"] = np.where(
        (po_df["ADS"] > 0) & (inv_for_metrics <= 0), "OOS", ""
    )

    # Priority classification (vectorised) — uses per-SKU lead days when present
    lt_arr = pd.to_numeric(po_df["Lead_Time_Days"], errors="coerce").fillna(float(lead_time)).clip(lower=1)
    conditions = [
        (po_df["Days_Left"] < lt_arr) & (po_df["PO_Qty"] > 0),
        (po_df["Days_Left"] < (lt_arr + float(grace_days))) & (po_df["PO_Qty"] > 0),
        po_df["PO_Qty"] > 0,
        po_df["PO_Pipeline_Effective"] > 0,
    ]
    choices = ["🔴 URGENT", "🟡 HIGH", "🟢 MEDIUM", "🔄 In Pipeline"]
    po_df["Priority"] = np.select(conditions, choices, default="⚪ OK")

    # Parent SKU — strip marketplace suffix + size/colour suffix
    po_df["Parent_SKU"] = po_df["OMS_SKU"].apply(get_parent_sku)

    # Cutting ratio for the Cutting Planner:
    # - Net PO share when any variant has PO_Qty > 0 (unchanged).
    # - When no net PO, spread by ADS (or gross share) only if **two or more** sizes have
    #   Gross_PO_Qty > 0 — otherwise a single-size need stays on that size (team can
    #   adjust within one size; no phantom split across siblings).
    parent_po_sum = po_df.groupby("Parent_SKU")["PO_Qty"].transform("sum")
    parent_ads_sum = po_df.groupby("Parent_SKU")["ADS"].transform("sum")
    gross_num2 = pd.to_numeric(po_df["Gross_PO_Qty"], errors="coerce").fillna(0)
    has_gross_req = gross_num2 > 0
    _par = po_df["Parent_SKU"]
    n_gross_sizes = has_gross_req.astype(int).groupby(_par).transform("sum")
    parent_gross_sum = gross_num2.groupby(_par).transform("sum")
    ratio_from_po = np.where(parent_po_sum > 0, po_df["PO_Qty"] / parent_po_sum, 0.0)
    ratio_from_ads = np.where(
        parent_ads_sum > 0,
        po_df["ADS"] / parent_ads_sum,
        np.where(parent_gross_sum > 0, gross_num2 / parent_gross_sum, 0.0),
    )
    ratio_single_gross = np.where(has_gross_req & (n_gross_sizes == 1), 1.0, 0.0)
    po_df["Cutting_Ratio"] = np.where(
        parent_po_sum > 0,
        ratio_from_po,
        np.where(
            n_gross_sizes >= 2,
            ratio_from_ads,
            np.where(n_gross_sizes == 1, ratio_single_gross, 0.0),
        ),
    ).round(4)

    # Formula-sheet aligned:
    # Projected_Running_Days = current cover before new release.
    if "Total_Inventory" in po_df.columns:
        inv_days_left = pd.to_numeric(po_df["Total_Inventory"], errors="coerce").fillna(0)
    else:
        inv_days_left = pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0)
    po_df["Projected_Running_Days"] = np.where(
        po_df["ADS"] > 0,
        ((inv_days_left + po_df["PO_Pipeline_Effective"]) / po_df["ADS"]).round(1),
        999.0,
    )
    # Post-PO cover after adding new release quantity.
    _post_cover = pd.Series(
        np.where(
            po_df["ADS"] > 0,
            ((inv_days_left + po_df["PO_Pipeline_Effective"] + po_df["PO_Qty"]) / po_df["ADS"]).round(1),
            999.0,
        ),
        index=po_df.index,
    )
    po_df["Post_PO_Cover_Days_Capped"] = pd.to_numeric(_post_cover, errors="coerce").fillna(999.0).round(1)

    try:
        from .sku_status_lead import is_closed_sku_status as _ics_hint
    except Exception:  # pragma: no cover
        def _ics_hint(_x: object) -> bool:
            return False

    _par_col = po_df["Parent_SKU"]
    ads_col = pd.to_numeric(po_df["ADS"], errors="coerce").fillna(0.0)
    sold_c = pd.to_numeric(po_df["Sold_Units"], errors="coerce").fillna(0.0)
    ship_col = pd.to_numeric(po_df["Ship_Units_150d"], errors="coerce").fillna(0.0)
    parent_ads = ads_col.groupby(_par_col).transform("sum")
    parent_sold = sold_c.groupby(_par_col).transform("sum")
    n_sib = po_df.groupby(_par_col)["OMS_SKU"].transform("count")
    share = np.where(parent_ads > 1e-9, ads_col / parent_ads, 0.0)
    rk_dense = ads_col.groupby(_par_col).rank(method="dense", ascending=False)
    row_closed = po_df["SKU_Sheet_Closed"] | po_df["SKU_Sheet_Status"].astype(str).map(_ics_hint)

    suggest = pd.Series("", index=po_df.index, dtype=str)
    suggest.loc[row_closed] = "Closed on sheet — run down stock (no new PO)."
    strong = (~row_closed) & (n_sib >= 2) & (rk_dense == 1) & (share >= 0.35) & (parent_ads >= 0.25)
    suggest.loc[strong] = "Strong seller vs other sizes — keep in depth."
    weak = (
        (~row_closed)
        & (n_sib >= 2)
        & (sold_c == 0)
        & (ship_col == 0)
        & (parent_sold > 10)
        & ~strong
    )
    suggest.loc[weak] = "Low movement vs siblings — consider closing."
    deep_stock = (
        (~row_closed)
        & ~strong
        & ~weak
        & (pd.to_numeric(po_df["Days_Left"], errors="coerce").fillna(0.0) > 120.0)
        & (sold_c == 0)
        & (ship_col == 0)
    )
    suggest.loc[deep_stock] = "High stock cover, no recent sales — review listing."
    po_df["Suggest_Close_SKU"] = suggest.fillna("").astype(str)

    # ── All-sizes expansion (two-part) ────────────────────────────────────────
    #
    # Part A — Multi-size sibling lift:
    #   When ≥2 sizes in a parent already have PO, top up every other size with ADS
    #   that is still below post-PO target cover (works with lead gate on or off).
    #
    # Part B — Missing-size ghost rows:
    #   When any size of a parent has Projected_Running_Days < urgent_all_sizes_days,
    #   add display rows (PO_Qty = 0) for sibling SKUs found in sku_mapping that are
    #   entirely absent from the output (no inventory, no ADS).  These rows let the
    #   operator see and raise for every size even if some aren't in the current data.
    import logging as _log_mod

    # ── Part A ────────────────────────────────────────────────────────────────
    try:
        po_df = _apply_multi_size_sibling_po_lift(po_df, target_cover_days)
    except Exception as _lift_err:
        _log_mod.getLogger(__name__).warning("multi-size sibling PO lift failed (non-fatal): %s", _lift_err)

    # ── Part B ────────────────────────────────────────────────────────────────
    if urgent_all_sizes_days > 0 and sku_mapping:
        try:
            proj_col = pd.to_numeric(po_df["Projected_Running_Days"], errors="coerce").fillna(999.0)
            urgent_parents: set = set(
                po_df.loc[proj_col < float(urgent_all_sizes_days), "Parent_SKU"].dropna().unique()
            )
            if urgent_parents:
                existing_skus: set = set(po_df["OMS_SKU"].astype(str))
                parent_to_children: dict = {}
                for _mid, _mapped in sku_mapping.items():
                    child_oms = canonical_oms_key(_mapped or _mid, sku_mapping)
                    if not child_oms:
                        continue
                    p = get_parent_sku(child_oms)
                    parent_to_children.setdefault(p, set()).add(child_oms)

                ghost_rows = []
                for parent in urgent_parents:
                    for child in sorted(parent_to_children.get(parent, ())):
                        if str(child) not in existing_skus:
                            ghost = {c: 0 for c in po_df.columns}
                            ghost["OMS_SKU"]               = str(child)
                            ghost["Parent_SKU"]            = str(parent)
                            ghost["Total_Inventory"]       = 0
                            ghost["ADS"]                   = 0.0
                            ghost["Days_Left"]             = 999.0
                            ghost["Projected_Running_Days"]    = 999.0
                            ghost["Post_PO_Cover_Days_Capped"] = 999.0
                            ghost["PO_Qty"]                = 0
                            ghost["Gross_PO_Qty"]          = 0
                            ghost["PO_Block_Reason"]       = "Urgent parent — all sizes shown for review"
                            ghost["Suggest_Close_SKU"]     = ""
                            ghost["Priority"]              = ""
                            ghost["SKU_Sheet_Closed"]      = False
                            ghost["SKU_Sheet_Status"]      = ""
                            for col in po_df.columns:
                                if col not in ghost:
                                    ghost[col] = ""
                            ghost_rows.append(ghost)

                if ghost_rows:
                    ghost_df = pd.DataFrame(ghost_rows, columns=po_df.columns)
                    for col in po_df.columns:
                        try:
                            ghost_df[col] = ghost_df[col].astype(po_df[col].dtype)
                        except Exception:
                            pass
                    po_df = pd.concat([po_df, ghost_df], ignore_index=True)
                    _log_mod.getLogger(__name__).info(
                        "all-sizes ghost rows: %d added for %d urgent parent(s) (proj < %dd)",
                        len(ghost_rows), len(urgent_parents), urgent_all_sizes_days,
                    )
        except Exception as _uas_err:
            _log_mod.getLogger(__name__).warning("all-sizes ghost-row expansion failed (non-fatal): %s", _uas_err)

    # Final pack rounding (overlay / gate-lift can leave non-pack quantities) + refresh cover.
    _po_final = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0.0)

    # Re-apply excluded-SKU block after sibling-lift / ghost-row expansion.
    try:
        from .sku_status_lead import is_excluded_po_status as _is_excl_final
    except Exception:  # pragma: no cover
        def _is_excl_final(_x: object) -> bool:
            return False

    _st_final = po_df.get("SKU_Sheet_Status", pd.Series("", index=po_df.index)).astype(str).map(_is_excl_final)
    _cl_final = po_df.get("SKU_Sheet_Closed", pd.Series(False, index=po_df.index)).fillna(False).astype(bool)
    _excl_final = (_st_final | _cl_final).fillna(False)
    if _excl_final.any():
        _po_final = _po_final.where(~_excl_final, 0.0)
        _msg_excl = "SKU excluded from PO recommendation"
        br = po_df.loc[_excl_final, "PO_Block_Reason"].astype(str).str.strip()
        reasons = po_df.loc[_excl_final, "SKU_Sheet_Status"].map(
            lambda s: po_block_reason_for_excluded_status(s) if str(s).strip() else "SKU marked closed on status sheet"
        )
        po_df.loc[_excl_final, "PO_Block_Reason"] = np.where(
            br.eq("") | br.eq("nan"), reasons, br + "; " + reasons.astype(str)
        )

    po_df["PO_Qty"] = np.vectorize(round_po_pack)(_po_final.to_numpy(dtype=float)).astype(int)
    po_df["Gross_PO_Qty"] = po_df["PO_Qty"]

    # Manual Existing PO upload = confirmed raise for the sheet date — do not
    # re-recommend SKUs that appear on that upload within the raise lookback window.
    if manual_existing_po_raise_skus and manual_existing_po_raise_date:
        try:
            _plan_mr = _plan if _plan is not None else pd.Timestamp.now().normalize()
            _raise_mr = pd.Timestamp(str(manual_existing_po_raise_date).strip()[:10]).normalize()
            _lb_mr = max(1, int(raise_ledger_lookback_days))
            _win_start_mr = _plan_mr - pd.Timedelta(days=_lb_mr)
            if _win_start_mr <= _raise_mr <= _plan_mr:
                _sku_keys = po_df["OMS_SKU"].astype(str).str.strip().str.upper()
                _on_manual_sheet = _sku_keys.isin(manual_existing_po_raise_skus)
                _had_po = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0
                _block_mr = _on_manual_sheet & _had_po
                if bool(_block_mr.any()):
                    _msg_mr = (
                        f"On manual Existing PO raise sheet ({_raise_mr.date()}) — "
                        "already raised via uploaded Po sheet"
                    )
                    po_df.loc[_block_mr, "PO_Qty"] = 0
                    po_df.loc[_block_mr, "Gross_PO_Qty"] = 0
                    br_mr = po_df.loc[_block_mr, "PO_Block_Reason"].astype(str).str.strip()
                    po_df.loc[_block_mr, "PO_Block_Reason"] = np.where(
                        br_mr.eq("") | br_mr.eq("nan"),
                        _msg_mr,
                        br_mr + "; " + _msg_mr,
                    )
        except Exception:
            import logging

            logging.getLogger(__name__).exception("manual existing PO raise block failed")

    if "Total_Inventory" in po_df.columns:
        _inv_final = pd.to_numeric(po_df["Total_Inventory"], errors="coerce").fillna(0.0)
    else:
        _inv_final = pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0.0)
    po_df["Post_PO_Cover_Days_Capped"] = np.where(
        po_df["ADS"] > 0,
        ((_inv_final + po_df["PO_Pipeline_Effective"] + po_df["PO_Qty"]) / po_df["ADS"]).round(1),
        999.0,
    )

    # Drop intermediate calc columns (datetime/float cols that break router serialisation)
    po_df.drop(
        columns=["ADS_Sold_Units", "ADS_Net_Units"],
        errors="ignore",
        inplace=True,
    )

    if "OMS_SKU" in po_df.columns:
        po_df = po_df.copy()

        def _output_oms_key(raw) -> str:
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                return ""
            t = clean_sku(str(raw).strip())
            if not t:
                t = str(raw).strip().upper()
            return collapse_duplicate_trailing_size_suffix(t)

        po_df["OMS_SKU"] = po_df["OMS_SKU"].astype(str).map(_output_oms_key)

    from .existing_po import (
        build_bundle_listing_map,
        dedupe_po_rows_by_sku,
        existing_po_merge_key,
    )

    _bundle_sources = []
    if inv_df is not None and not getattr(inv_df, "empty", True) and "OMS_SKU" in inv_df.columns:
        _bundle_sources.append(inv_df["OMS_SKU"])
    if (
        existing_po_df is not None
        and not getattr(existing_po_df, "empty", True)
        and "OMS_SKU" in existing_po_df.columns
    ):
        _bundle_sources.append(existing_po_df["OMS_SKU"])
    if "OMS_SKU" in po_df.columns:
        _bundle_sources.append(po_df["OMS_SKU"])
    _bundle_map = build_bundle_listing_map(*_bundle_sources)
    if _bundle_map:
        po_df["Bundle_Size"] = po_df["OMS_SKU"].astype(str).map(
            lambda s: _bundle_map.get(existing_po_merge_key(s), "")
        )
    else:
        po_df["Bundle_Size"] = ""

    # Final lead-time gate (belt-and-suspenders after pack-round / ghost rows).
    if enforce_lead_time_release_gate and "Projected_Running_Days" in po_df.columns:
        po_df = _apply_multi_size_sibling_po_lift(po_df, target_cover_days)
        _proj_fin = pd.to_numeric(po_df["Projected_Running_Days"], errors="coerce").fillna(999.0)
        _lt_fin_arr = _lead_time_release_gate_days(po_df, lead_time)
        _lt_fin_s = pd.Series(_lt_fin_arr, index=po_df.index)
        _po_fin = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0)
        _msg_lead_fin = _lead_time_gate_block_message(_lead_time_gate_uses_entered(lead_time))
        _par_fin = po_df["Parent_SKU"]
        _n_po_sizes = (_po_fin > 0).astype(int).groupby(_par_fin).transform("sum")
        _multi_size_parent = _n_po_sizes >= 2

        _over_lead = (_proj_fin >= _lt_fin_s) & (_po_fin > 0) & ~_multi_size_parent
        if _over_lead.any():
            po_df.loc[_over_lead, "PO_Qty"] = 0
            po_df.loc[_over_lead, "Gross_PO_Qty"] = 0
            br = po_df.loc[_over_lead, "PO_Block_Reason"].astype(str).str.strip()
            po_df.loc[_over_lead, "PO_Block_Reason"] = np.where(
                br.eq("") | br.eq("nan"),
                _msg_lead_fin,
                br + "; " + _msg_lead_fin,
            )

        if enforce_two_size_minimum:
            _par_fin = po_df["Parent_SKU"]
            _below_lead = (_proj_fin < _lt_fin_s).astype(int)
            _n_below_lead = _below_lead.groupby(_par_fin).transform("sum")
            _po_fin2 = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0)
            _single_parent = (_n_below_lead < 2) & (_po_fin2 > 0)
            if _single_parent.any():
                po_df.loc[_single_parent, "PO_Qty"] = 0
                po_df.loc[_single_parent, "Gross_PO_Qty"] = 0
                _msg_two = (
                    "Fewer than 2 sizes below factory lead — no PO until multiple sizes need cover"
                )
                br = po_df.loc[_single_parent, "PO_Block_Reason"].astype(str).str.strip()
                po_df.loc[_single_parent, "PO_Block_Reason"] = np.where(
                    br.eq("") | br.eq("nan"),
                    _msg_two,
                    br + "; " + _msg_two,
                )

        if "Total_Inventory" in po_df.columns:
            _inv_fin2 = pd.to_numeric(po_df["Total_Inventory"], errors="coerce").fillna(0.0)
        else:
            _inv_fin2 = pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0.0)
        po_df["Post_PO_Cover_Days_Capped"] = np.where(
            po_df["ADS"] > 0,
            ((_inv_fin2 + po_df["PO_Pipeline_Effective"] + po_df["PO_Qty"]) / po_df["ADS"]).round(1),
            999.0,
        )

    if stage_timer is not None:
        stage_timer.mark("po logic")

    po_df = _apply_sibling_cut_from_pending(po_df, target_cover_days)

    return dedupe_po_rows_by_sku(po_df)
