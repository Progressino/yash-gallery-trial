"""
PO Engine — extracted 1-for-1 from app.py.
calculate_quarterly_history + calculate_po_base.
"""
from collections import defaultdict
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

import re

from .helpers import map_to_oms_sku, get_parent_sku, clean_sku, normalize_id_token_for_mapping
from .myntra import myntra_to_sales_rows

# Strip "PL" infix in Amazon seller SKUs (e.g. 1001PLYKBEIGE-3XL → 1001YKBEIGE-3XL)
# Must match the same pattern used in inventory.py _resolve_amz_sku.
_PL_RE = re.compile(r'^(\d+)PL(YK)', re.I)
_SIZE_SUFFIX_RE = re.compile(r"(?:-|_)?(XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|5XL|6XL)$", re.I)


def _strip_pl(sku: str, mapping: Dict[str, str]) -> str:
    """Map an Amazon seller SKU to OMS SKU, stripping PL infix if needed."""
    raw = str(sku).strip().upper()
    stripped = _PL_RE.sub(r"\1\2", raw)
    return mapping.get(stripped, mapping.get(raw, stripped))


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
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> pd.DataFrame:
    parts = []

    # Unified sales_df already contains Amazon + Myntra (via build_sales_df). Appending
    # raw mtr_df / myntra_df on top doubled those channels in quarterly / Forecast.
    if not sales_df.empty and "Sku" in sales_df.columns:
        tmp = sales_df[["Sku", "TxnDate", "Quantity", "Transaction Type"]].copy()
        tmp.columns = ["SKU", "Date", "Qty", "TxnType"]
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp["Qty"]  = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
        parts.append(tmp.dropna(subset=["Date"]))
    else:
        # Bootstrap: no unified sales yet (e.g. SQLite-only) — stitch raw platform frames.
        if mtr_df is not None and not mtr_df.empty:
            mtr_sku_col  = next((c for c in mtr_df.columns if c in ["SKU", "Sku", "OMS_SKU"]),  None)
            mtr_date_col = next((c for c in mtr_df.columns if c in ["Date", "TxnDate"]),         None)
            mtr_qty_col  = next((c for c in mtr_df.columns if c in ["Quantity", "Qty"]),          None)
            mtr_txn_col  = next((c for c in mtr_df.columns if c in ["Transaction_Type", "Transaction Type", "TxnType"]), None)
            if mtr_sku_col and mtr_date_col and mtr_qty_col:
                tmp = mtr_df[[mtr_sku_col, mtr_date_col, mtr_qty_col]].copy()
                tmp.columns = ["SKU", "Date", "Qty"]
                tmp["Date"]    = pd.to_datetime(tmp["Date"], errors="coerce")
                tmp["Qty"]     = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
                tmp["TxnType"] = mtr_df[mtr_txn_col].values if mtr_txn_col else "Shipment"
                if sku_mapping:
                    tmp["SKU"] = tmp["SKU"].apply(lambda x: _strip_pl(x, sku_mapping))
                else:
                    tmp["SKU"] = tmp["SKU"].apply(lambda x: _PL_RE.sub(r"\1\2", str(x).strip().upper()))
                parts.append(tmp.dropna(subset=["Date"]))
        if myntra_df is not None and not myntra_df.empty:
            myn_sku_col  = next((c for c in myntra_df.columns if c in ["OMS_SKU", "Sku", "SKU"]), None)
            myn_date_col = next((c for c in myntra_df.columns if c in ["Date", "TxnDate"]),        None)
            myn_qty_col  = next((c for c in myntra_df.columns if c in ["Quantity", "Qty"]),        None)
            myn_txn_col  = next((c for c in myntra_df.columns if c in ["TxnType", "Transaction Type"]), None)
            if myn_sku_col and myn_date_col and myn_qty_col:
                tmp = myntra_df[[myn_sku_col, myn_date_col, myn_qty_col]].copy()
                tmp.columns = ["SKU", "Date", "Qty"]
                tmp["Date"]    = pd.to_datetime(tmp["Date"], errors="coerce")
                tmp["Qty"]     = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
                tmp["TxnType"] = myntra_df[myn_txn_col].values if myn_txn_col else "Shipment"
                parts.append(tmp.dropna(subset=["Date"]))

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

    # Normalize SKUs: strip PL infix — use unique-value cache (18x faster than row-by-row apply)
    _uniq_hist = hist["SKU"].unique()
    _pl_norm_cache = {
        s: (_PL_RE.sub(r"\1\2", str(s).strip().upper()) if isinstance(s, str) else str(s))
        for s in _uniq_hist
    }
    hist["SKU"] = hist["SKU"].map(_pl_norm_cache)

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
    return pivot


def _seasonal_adjacent_months_ads(
    sales_df: pd.DataFrame,
    max_date: pd.Timestamp,
    group_by_parent: bool,
    demand_basis: str,
    years_lookback: int = 2,
    min_denominator: int = 7,
) -> pd.DataFrame:
    """
    Daily ADS from the same calendar month + following month in prior years.

    Example: max_date in late March 2026 → use Mar+Apr of 2025 and 2024 per SKU.
    If the last 30–90 days look weak but March/April historically move 4–5+ units
    per day in aggregate, this lifts ADS so PO is appropriate before the season.

    Per prior year we compute (units in that two-month window) / (days in window),
    then take the mean across years (only years with any sales in the window
    contribute — avoids diluting a clear seasonal peak with empty years).
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
    rate_lists: Dict[str, list] = defaultdict(list)

    for yo in range(1, years_lookback + 1):
        y = max_date.year - yo
        if m0 == 12:
            start = pd.Timestamp(year=y, month=12, day=1)
            end = pd.Timestamp(year=y + 1, month=1, day=31)
        else:
            start = pd.Timestamp(year=y, month=m0, day=1)
            end = pd.Timestamp(year=y, month=m0 + 1, day=1) + MonthEnd(0)
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


def calculate_po_base(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    period_days: int,
    lead_time: int,
    target_days: int,
    demand_basis: str = "Sold",
    min_denominator: int = 7,
    grace_days: int = 7,
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
) -> pd.DataFrame:
    if sales_df.empty or inv_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()
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
        return _strip_pl(str(t).strip(), _map)

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
    _unique_inv_skus = inv_work["OMS_SKU"].unique()
    _inv_canon_cache = {s: _canonical_oms_key(s) for s in _unique_inv_skus}
    inv_work["OMS_SKU"] = inv_work["OMS_SKU"].map(_inv_canon_cache).fillna("")
    inv_work = inv_work[inv_work["OMS_SKU"].str.len() > 0]
    if inv_work["OMS_SKU"].duplicated().any():
        inv_work = inv_work.drop_duplicates(subset=["OMS_SKU"], keep="last")

    max_date = df["TxnDate"].max()
    # PO only needs recent and LY windows; trimming old rows avoids multi-year
    # full-table groupbys that make calculation feel stuck for large histories.
    lookback_days = int(max(30, period_days) + 365 + period_days + 7)
    hist_cutoff = max_date - timedelta(days=lookback_days)
    df = df[df["TxnDate"] >= hist_cutoff].copy()
    cutoff   = max_date - timedelta(days=period_days)
    recent   = df[df["TxnDate"] >= cutoff].copy()

    sold = recent[recent["_is_ship"]].groupby("Sku")["Quantity"].sum().reset_index()
    sold.columns = ["OMS_SKU", "Sold_Units"]
    _is_ref = recent["Transaction Type"].astype(str).str.strip().str.lower().eq("refund")
    returns = recent[_is_ref].groupby("Sku")["Quantity"].sum().reset_index()
    returns.columns = ["OMS_SKU", "Return_Units"]
    net = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU", "Net_Units"]

    summary = sold.merge(returns, on="OMS_SKU", how="outer").merge(net, on="OMS_SKU", how="outer").fillna(0)
    po_df = _merge_metric_with_parent_fallback(
        inv_work, summary, ["Sold_Units", "Return_Units", "Net_Units"]
    ).fillna({"Sold_Units": 0, "Return_Units": 0, "Net_Units": 0})

    # Shipment context column for operators (not used in PO math):
    # show broader last ~5 months shipments even when period window is shorter.
    ship_150_cutoff = max_date - timedelta(days=150)
    ship_150 = (
        df[(df["TxnDate"] >= ship_150_cutoff) & (df["_is_ship"])]
        .groupby("Sku")["Quantity"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Quantity": "Ship_Units_150d"})
    )
    po_df = _merge_metric_with_parent_fallback(
        po_df, ship_150, ["Ship_Units_150d"]
    )
    po_df["Ship_Units_150d"] = pd.to_numeric(po_df["Ship_Units_150d"], errors="coerce").fillna(0).astype(int)

    # ── SKU Status / Lead sheet (optional) ────────────────────────────────
    # Upload only changes per-SKU lead days for the PO formula. Status / closed flags
    # are merged for display in the UI; they do not change Gross_PO_Qty or ADS.
    po_df["Lead_Time_Days"] = int(max(1, int(lead_time)))
    po_df["PO_Block_Reason"] = ""
    if sku_status_df is not None and not sku_status_df.empty:
        m = sku_status_df.copy()
        _uniq_ss = m["OMS_SKU"].unique()
        _ss_canon = {s: _canonical_oms_key(s) for s in _uniq_ss}
        m["OMS_SKU"] = m["OMS_SKU"].map(_ss_canon).fillna("")
        m = m[m["OMS_SKU"].str.len() > 0]

        def _max_positive_lead(series: pd.Series) -> float:
            v = pd.to_numeric(series, errors="coerce")
            v = v[v > 0]
            return float(v.max()) if len(v) else float("nan")

        # Max lead per parent style — used when sheet lists parent SKU only, inventory is per-size.
        m["_par_key"] = m["OMS_SKU"].map(get_parent_sku)
        lead_by_parent = m.groupby("_par_key")["Lead_Time_From_Sheet"].apply(_max_positive_lead).to_dict()
        _sorted_lead_parent_keys = sorted(
            (str(sk).strip() for sk in lead_by_parent if str(sk).strip()),
            key=len,
            reverse=True,
        )

        # Sheet may use numeric style only (e.g. ``1657``/``1394``) while inventory is
        # ``1657YK…-SIZE`` or ``AK-1394BROWN-L``.
        lead_by_digit_token: dict[str, float] = {}
        for _, rw in m.iterrows():
            lt_one = float(pd.to_numeric(rw.get("Lead_Time_From_Sheet"), errors="coerce"))
            if not np.isfinite(lt_one) or lt_one <= 0:
                continue
            oms = str(rw.get("OMS_SKU") or "")
            par = str(rw.get("_par_key") or "")
            for tok in {_style_digit_token(oms), _style_digit_token(par)}:
                if not tok:
                    continue
                prev = lead_by_digit_token.get(tok, float("nan"))
                if not np.isfinite(prev) or lt_one > prev:
                    lead_by_digit_token[tok] = float(lt_one)

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
        po_df = po_df.merge(m, on="OMS_SKU", how="left")
        po_df["SKU_Sheet_Status"] = po_df["SKU_Sheet_Status"].fillna("").astype(str)
        po_df["SKU_Sheet_Closed"] = po_df["SKU_Sheet_Closed"].fillna(False).astype(bool)
        if "Lead_Time_From_Sheet" in po_df.columns:
            lt_vals = pd.to_numeric(po_df["Lead_Time_From_Sheet"], errors="coerce")
            bad = lt_vals.isna() | (lt_vals <= 0)
            if bad.any() and lead_by_parent:
                pk = po_df["OMS_SKU"].astype(str).map(get_parent_sku)
                fill_s = pd.to_numeric(pk.map(lead_by_parent), errors="coerce")
                use = bad & (fill_s > 0)
                lt_vals = lt_vals.where(~use, fill_s)
                bad = lt_vals.isna() | (lt_vals <= 0)
                # Sheet lists ``AK-139`` while inventory rows are ``AK-139BROWN-L`` → parent
                # ``AK-139BROWN`` must inherit lead from longest matching sheet parent key.
                if bad.any():
                    # Only unique inventory parents still missing lead — avoids calling
                    # prefix matching once per row (was freezing PO with 10k+ rows × 1k+ keys).
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
            if bad.any() and lead_by_digit_token:
                pk2 = po_df["OMS_SKU"].astype(str).map(get_parent_sku)
                dig = pk2.map(_style_digit_token)
                fill2 = pd.to_numeric(dig.map(lead_by_digit_token), errors="coerce")
                use2 = bad & (fill2 > 0)
                lt_vals = lt_vals.where(~use2, fill2)
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
        po_df["SKU_Sheet_Status"] = ""
        po_df["SKU_Sheet_Closed"] = False
    po_df["Lead_Time_Days"] = pd.to_numeric(po_df["Lead_Time_Days"], errors="coerce").fillna(int(max(1, int(lead_time)))).clip(lower=1, upper=730).astype(int)

    # ADS starts from period_days-window Recent_ADS / LY blend, is floored by
    # seasonal same-month+next-month history, and by Flat30_ADS (Req.xlsx FREQ).
    ADS_WINDOW = period_days
    ads_cutoff = max_date - timedelta(days=ADS_WINDOW)
    ads_recent = df[df["TxnDate"] >= ads_cutoff].copy()

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
        ads_active_span = pd.DataFrame(columns=["OMS_SKU", "_eff_days_active"])
    else:
        ads_active_span = (
            act.groupby("Sku", as_index=False)
            .agg(ADS_First_Active=("TxnDate", "min"), ADS_Last_Active=("TxnDate", "max"))
            .rename(columns={"Sku": "OMS_SKU"})
        )
        ads_active_span["_eff_days_active"] = (
            (ads_active_span["ADS_Last_Active"] - ads_active_span["ADS_First_Active"]).dt.days + 1
        ).astype(int)

    ads_summary = ads_sold.merge(ads_net, on="OMS_SKU", how="outer").fillna(0)
    po_df = _merge_metric_with_parent_fallback(
        po_df, ads_summary, ["ADS_Sold_Units", "ADS_Net_Units"]
    )
    po_df = po_df.merge(ads_active_span[["OMS_SKU", "_eff_days_active"]], on="OMS_SKU", how="left")
    if not group_by_parent and not ads_recent.empty:
        span_exact_keys = set(ads_active_span["OMS_SKU"].astype(str)) if not ads_active_span.empty else set()
        _p_act = act.copy()
        if not _p_act.empty:
            _p_act["_Parent_SKU"] = _p_act["Sku"].map(_fallback_parent_key)
            p_ads_active_span = (
                _p_act.groupby("_Parent_SKU", as_index=False)
                .agg(P_ADS_First_Active=("TxnDate", "min"), P_ADS_Last_Active=("TxnDate", "max"))
                .rename(columns={"_Parent_SKU": "OMS_SKU"})
            )
            p_ads_active_span["P__eff_days_active"] = (
                (p_ads_active_span["P_ADS_Last_Active"] - p_ads_active_span["P_ADS_First_Active"]).dt.days + 1
            ).astype(int)
            po_df = po_df.merge(p_ads_active_span[["OMS_SKU", "P__eff_days_active"]], on="OMS_SKU", how="left")
            out_sku = po_df["OMS_SKU"].astype(str)
            use_parent_days = out_sku.map(_fallback_parent_key).eq(out_sku) | (~out_sku.isin(span_exact_keys))
            po_df["_eff_days_active"] = np.where(
                use_parent_days,
                pd.to_numeric(po_df["P__eff_days_active"], errors="coerce"),
                pd.to_numeric(po_df["_eff_days_active"], errors="coerce"),
            )
            po_df.drop(columns=["P__eff_days_active"], inplace=True, errors="ignore")
    po_df[["ADS_Sold_Units", "ADS_Net_Units"]] = po_df[["ADS_Sold_Units", "ADS_Net_Units"]].fillna(0)

    # Sold_Units / Net_Units reflect the full ADS window (period_days).
    po_df["Sold_Units"] = po_df["ADS_Sold_Units"].astype(int)
    po_df["Net_Units"]  = po_df["ADS_Net_Units"].clip(lower=0).astype(int)

    # Use true active-day span per SKU. Old min_denominator floor (often 7) forced many SKUs
    # to show only 7/30 and diluted ADS, which distorted PO suggestions.
    po_df["Eff_Days"] = (
        pd.to_numeric(po_df["_eff_days_active"], errors="coerce")
        .fillna(float(ADS_WINDOW))
        .clip(lower=1.0, upper=float(ADS_WINDOW))
    )
    po_df.drop(columns=["_eff_days_active"], inplace=True, errors="ignore")
    ads_demand = po_df["ADS_Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["ADS_Sold_Units"]
    po_df["Recent_ADS"] = (ads_demand / po_df["Eff_Days"]).fillna(0)

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
        blended = np.maximum(po_df["Recent_ADS"], po_df["LY_ADS"])

    # Final ADS for PO should not collapse to zero only because recent window is quiet.
    # Use recent as primary signal, then floor with LY/seasonal/flat diagnostics.
    recent_ads = pd.to_numeric(po_df["Recent_ADS"], errors="coerce").fillna(0.0)
    ly_ads = pd.to_numeric(po_df["LY_ADS"], errors="coerce").fillna(0.0)
    seasonal_ads = pd.to_numeric(po_df["Seasonal_Month_ADS"], errors="coerce").fillna(0.0)
    flat_ads = pd.to_numeric(po_df["Flat30_ADS"], errors="coerce").fillna(0.0)
    if use_seasonality:
        prim_ads = pd.to_numeric(pd.Series(blended, index=po_df.index), errors="coerce").fillna(0.0)
    else:
        prim_ads = np.maximum(recent_ads, ly_ads)
    po_df["ADS"] = np.maximum.reduce([prim_ads, seasonal_ads, flat_ads]).round(3)

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
    # "Running days" for operators: use total sellable stock when present (FBA + warehouse),
    # while PO qty still uses OMS_Inventory only (inv_vals above).
    if "Total_Inventory" in po_df.columns:
        inv_days_left = pd.to_numeric(po_df["Total_Inventory"], errors="coerce").fillna(0)
    else:
        inv_days_left = inv_vals

    # PO calculation — per-SKU lead from sheet when uploaded, else global lead_time
    lt_vec = pd.to_numeric(po_df["Lead_Time_Days"], errors="coerce").fillna(int(max(1, int(lead_time)))).clip(lower=1)
    lead_demand  = po_df["ADS"] * lt_vec
    target_stock = po_df["ADS"] * (target_days + grace_days)
    base_req     = lead_demand + target_stock
    safety       = base_req * (safety_pct / 100.0)
    total_req    = base_req + safety
    gross_po     = (total_req - inv_vals).clip(lower=0)
    po_df["Gross_PO_Qty"] = (np.ceil(gross_po / 10) * 10).astype(int)

    if enforce_two_size_minimum:
        _par_key = po_df["OMS_SKU"].apply(get_parent_sku)
        has_g = pd.to_numeric(po_df["Gross_PO_Qty"], errors="coerce").fillna(0) > 0
        n_sizes = has_g.astype(int).groupby(_par_key).transform("sum")
        single_only = has_g & (n_sizes == 1)
        po_df.loc[single_only, "Gross_PO_Qty"] = 0
        br = po_df.loc[single_only, "PO_Block_Reason"].astype(str).str.strip()
        add = "Single size only (need ≥2 sizes with demand)"
        po_df.loc[single_only, "PO_Block_Reason"] = np.where(
            br.eq("") | br.eq("nan"),
            add,
            br + "; " + add,
        )

    # Pipeline deduction from existing PO sheet
    if existing_po_df is not None and not existing_po_df.empty and "PO_Pipeline_Total" in existing_po_df.columns:
        # Pull PO_Pipeline_Total + any breakdown columns present in the uploaded sheet
        _breakdown_cols = [c for c in ["PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch"]
                           if c in existing_po_df.columns]
        _merge_cols = ["OMS_SKU", "PO_Pipeline_Total"] + _breakdown_cols
        po_df = pd.merge(
            po_df,
            existing_po_df[_merge_cols],
            on="OMS_SKU", how="left",
        )
        po_df["PO_Pipeline_Total"] = pd.to_numeric(
            po_df["PO_Pipeline_Total"], errors="coerce"
        ).fillna(0).astype(int)
        for _bc in _breakdown_cols:
            po_df[_bc] = pd.to_numeric(po_df[_bc], errors="coerce").fillna(0).astype(int)
    else:
        po_df["PO_Pipeline_Total"] = 0

    # Days of stock remaining = current inventory cover only (no pipeline).
    po_df["Days_Left"] = np.where(
        po_df["ADS"] > 0,
        (inv_days_left / po_df["ADS"]).round(1),
        999.0,
    )

    # ── Inject PO-sheet SKUs missing from inventory ──────────────
    # If a SKU has an active pipeline order but isn't in the inventory file
    # (e.g. out of stock, removed from listing), add it as a ghost row so it
    # still shows up as "🔄 In Pipeline" and isn't invisible to the user.
    if existing_po_df is not None and not existing_po_df.empty and "PO_Pipeline_Total" in existing_po_df.columns:
        missing_mask = (
            ~existing_po_df["OMS_SKU"].isin(po_df["OMS_SKU"])
            & (existing_po_df["PO_Pipeline_Total"] > 0)
        )
        missing_po = existing_po_df[missing_mask].copy()
        if not missing_po.empty:
            ghost = pd.DataFrame({"OMS_SKU": missing_po["OMS_SKU"].values})
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
            ghost["Lead_Time_Days"] = int(max(1, int(lead_time)))
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

    net_po = (po_df["Gross_PO_Qty"] - po_df["PO_Pipeline_Total"]).clip(lower=0)
    po_df["PO_Qty"] = (np.ceil(net_po / 10) * 10).astype(int)

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
        po_df["PO_Pipeline_Total"] > 0,
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
    gross_num = pd.to_numeric(po_df["Gross_PO_Qty"], errors="coerce").fillna(0)
    has_gross_req = gross_num > 0
    _par = po_df["Parent_SKU"]
    n_gross_sizes = has_gross_req.astype(int).groupby(_par).transform("sum")
    parent_gross_sum = gross_num.groupby(_par).transform("sum")
    ratio_from_po = np.where(parent_po_sum > 0, po_df["PO_Qty"] / parent_po_sum, 0.0)
    ratio_from_ads = np.where(
        parent_ads_sum > 0,
        po_df["ADS"] / parent_ads_sum,
        np.where(parent_gross_sum > 0, gross_num / parent_gross_sum, 0.0),
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

    # Projected Running Days includes current inventory + pipeline.
    po_df["Projected_Running_Days"] = np.where(
        po_df["ADS"] > 0,
        ((inv_days_left + po_df["PO_Pipeline_Total"]) / po_df["ADS"]).round(1),
        999.0,
    )

    po_df["Suggest_Close_SKU"] = ""

    # Drop intermediate calc columns (datetime/float cols that break router serialisation)
    po_df.drop(
        columns=["ADS_Sold_Units", "ADS_Net_Units"],
        errors="ignore",
        inplace=True,
    )

    return po_df
