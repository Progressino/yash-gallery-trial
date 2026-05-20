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
    return _strip_pl(str(t).strip(), sku_mapping or {})


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
) -> pd.DataFrame:
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

    _plan = None
    if planning_date:
        try:
            _plan = pd.Timestamp(pd.to_datetime(planning_date).normalize())
        except Exception:
            _plan = None

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

    if po_return_overlay_df is not None and not po_return_overlay_df.empty:
        ov = po_return_overlay_df.copy()
        if "OMS_SKU" in ov.columns and "Return_Units" in ov.columns:
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
    po_df = _merge_metric_with_parent_fallback(
        po_df, ship_150, ["Ship_Units_150d"]
    )
    po_df["Ship_Units_150d"] = pd.to_numeric(po_df["Ship_Units_150d"], errors="coerce").fillna(0).astype(int)

    # Defaults until SKU Status / Lead merge — applied later after pipeline ghost rows are appended,
    # so pipeline-only SKUs still inherit per-sheet lead times.
    po_df["Lead_Time_Days"] = int(max(1, int(lead_time)))
    po_df["PO_Block_Reason"] = ""
    po_df["SKU_Sheet_Status"] = ""
    po_df["SKU_Sheet_Closed"] = False

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
    if "Return_Overlay_Units" in po_df.columns:
        _rov = pd.to_numeric(po_df["Return_Overlay_Units"], errors="coerce").fillna(0).astype(int)
        if int(_rov.sum()) > 0:
            po_df["ADS_Net_Units"] = (
                pd.to_numeric(po_df["ADS_Net_Units"], errors="coerce").fillna(0).astype(int) - _rov
            ).clip(lower=0)
            po_df["Net_Units"] = po_df["ADS_Net_Units"].astype(int)

    # Use true active-day span per SKU. Old min_denominator floor (often 7) forced many SKUs
    # to show only 7/30 and diluted ADS, which distorted PO suggestions.
    po_df["Eff_Days"] = (
        pd.to_numeric(po_df["_eff_days_active"], errors="coerce")
        .fillna(float(ADS_WINDOW))
        .clip(lower=1.0, upper=float(ADS_WINDOW))
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
            ih = trim_inventory_history_for_po(
                inventory_history_df,
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
                skip_extend = bool(
                    pd.notna(sheet_max)
                    and sheet_max >= inv_window_end - timedelta(days=1)
                ) or coverage_in_window >= min(int(ADS_WINDOW), 14)

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
                    scale = float(ADS_WINDOW) / float(coverage_days) if coverage_days else 1.0
                    use_inv = inv_days.notna() & (inv_days > 0)
                    inv_eff = (inv_days * scale).round()
                    inv_clipped = inv_eff.clip(lower=1.0, upper=float(ADS_WINDOW))
                    po_df["Eff_Days"] = np.where(
                        use_inv,
                        inv_clipped.fillna(po_df["Eff_Days"]),
                        po_df["Eff_Days"],
                    )
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

    # Gross/Net PO is finalised after pipeline merge below (sheet-style balance-days formula).
    po_df["Gross_PO_Qty"] = 0

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
        _po_keys = set(po_df["OMS_SKU"].astype(str).str.strip())
        _pipe_canon = existing_po_df["OMS_SKU"].map(_canonical_oms_key).astype(str).str.strip()
        missing_mask = ~_pipe_canon.isin(_po_keys) & (
            pd.to_numeric(existing_po_df["PO_Pipeline_Total"], errors="coerce").fillna(0) > 0
        )
        missing_po = existing_po_df[missing_mask].copy()
        if not missing_po.empty:
            ghost = pd.DataFrame(
                {"OMS_SKU": [_canonical_oms_key(x) for x in missing_po["OMS_SKU"].values]},
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

    # ── SKU Status / Lead sheet (optional, after pipeline ghost rows) ────────────
    # Upload overrides lead days per SKU / parent style. Without this pass, every row
    # keeps the global ``lead_time`` default — operators often mistake that for “missing”.
    if sku_status_df is not None and not sku_status_df.empty:
        po_df.drop(columns=["SKU_Sheet_Status", "SKU_Sheet_Closed"], inplace=True, errors="ignore")
        m = sku_status_df.copy()
        _uniq_ss = m["OMS_SKU"].unique()
        _ss_canon = {s: _canonical_oms_key(s) for s in _uniq_ss}
        m["OMS_SKU"] = m["OMS_SKU"].map(_ss_canon).fillna("")
        m = m[m["OMS_SKU"].str.len() > 0]

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

        par_status_candidates: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        for _, rw in m.iterrows():
            par = str(rw.get("_par_key") or "").strip()
            if not par:
                continue
            st = str(rw.get("SKU_Sheet_Status") or "").strip()
            cl_row = bool(rw.get("SKU_Sheet_Closed", False)) or (bool(st) and _is_closed_st(st))
            par_status_candidates[par].append((st, cl_row))
        status_by_parent: dict[str, str] = {}
        closed_by_parent: dict[str, bool] = {}
        for par, lst in par_status_candidates.items():
            closed_by_parent[par] = any(x[1] for x in lst)
            closed_statuses = [x[0] for x in lst if x[1] and x[0]]
            if closed_statuses:
                status_by_parent[par] = closed_statuses[0]
            else:
                nonempty = [x[0] for x in lst if x[0]]
                status_by_parent[par] = nonempty[-1] if nonempty else ""

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

        # Sheet rows whose entire OMS key is a style numeric (e.g. ``1394``) are a direct
        # factory style code — match inventory by the same style digit token as parent/OMS.
        # This is **not** the same as ``lead_by_digit_token`` (which keys off any substring
        # token from keys like ``PREFIX-4002`` and must not satisfy the sheet PO gate alone).
        lead_by_pure_digit_style: dict[str, float] = {}
        status_by_pure_digit_style: dict[str, tuple[str, bool]] = {}
        for _, rw in m.iterrows():
            oms_c = str(rw.get("OMS_SKU") or "").strip()
            if not oms_c or not re.fullmatch(r"\d{3,}", oms_c):
                continue
            st_one = str(rw.get("SKU_Sheet_Status") or "").strip()
            cl_one = bool(rw.get("SKU_Sheet_Closed", False)) or (bool(st_one) and _is_closed_st(st_one))
            status_by_pure_digit_style[oms_c] = (st_one, cl_one)
            lt_one = float(pd.to_numeric(rw.get("Lead_Time_From_Sheet"), errors="coerce"))
            if not np.isfinite(lt_one) or lt_one <= 0:
                continue
            prev = lead_by_pure_digit_style.get(oms_c, float("nan"))
            if not np.isfinite(prev) or lt_one > prev:
                lead_by_pure_digit_style[oms_c] = float(lt_one)

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
        for _, rw in m.iterrows():
            k = str(rw["OMS_SKU"]).strip()
            stc = str(rw.get("SKU_Sheet_Status") or "").strip()
            clc = bool(rw.get("SKU_Sheet_Closed", False)) or (bool(stc) and _is_closed_st(stc))
            if k:
                sheet_key_meta[k] = (stc, clc)
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

    po_df["Lead_Time_Days"] = pd.to_numeric(po_df["Lead_Time_Days"], errors="coerce").fillna(int(max(1, int(lead_time)))).clip(lower=1, upper=730).astype(int)

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

    # Sheet formula:
    # projected_days_now = (Total_Inventory + effective pipeline) / ADS
    # balance_days       = target_cover_days - projected_days_now
    # po_qty_raw         = ADS * balance_days
    # po_qty_round       = ceil(max(po_qty_raw, 0) / pack) * pack
    target_cover_days = float(max(0, target_days + grace_days))
    if "Total_Inventory" in po_df.columns:
        inv_for_cover = pd.to_numeric(po_df["Total_Inventory"], errors="coerce").fillna(0.0)
    else:
        inv_for_cover = pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0.0)
    ads_num = pd.to_numeric(po_df["ADS"], errors="coerce").fillna(0.0)
    pipe_num = pd.to_numeric(po_df["PO_Pipeline_Effective"], errors="coerce").fillna(0.0)
    projected_days_now = np.where(ads_num > 0, (inv_for_cover + pipe_num) / ads_num, 999.0)
    balance_days = target_cover_days - projected_days_now
    raw_po = ads_num * balance_days
    _pack = 5.0
    po_qty_round = np.ceil(np.maximum(raw_po, 0.0) / _pack) * _pack
    po_df["Gross_PO_Qty"] = np.floor(np.maximum(po_qty_round, 0.0)).astype(int)
    po_df["PO_Qty"] = po_df["Gross_PO_Qty"].astype(int)

    if "Return_Overlay_Units" in po_df.columns:
        overlay = pd.to_numeric(po_df["Return_Overlay_Units"], errors="coerce").fillna(0).astype(int)
        po_df["PO_Qty"] = np.maximum(
            pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0).astype(int) - overlay,
            0,
        ).astype(int)
        po_df["Gross_PO_Qty"] = po_df["PO_Qty"]

    # Lead-time release (sheet-resolved lead only): no fresh PO while projected cover
    # (inv + eff. pipeline) is still *above* sheet ``Lead_Time_Days`` — then top up
    # toward ``target_days`` using the balance formula above.
    _msg_lead_window = (
        "Projected cover exceeds factory lead time — no PO until cover is within lead days"
    )
    if enforce_lead_time_release_gate:
        _from_sheet = po_df.get("Lead_Time_From_Status_Sheet")
        if _from_sheet is not None:
            _from_sheet_ok = _from_sheet.fillna(False).astype(bool).to_numpy()
        else:
            _from_sheet_ok = np.zeros(len(po_df), dtype=bool)
        _lt_gate = (
            pd.to_numeric(po_df["Lead_Time_Days"], errors="coerce")
            .fillna(float(max(1, int(lead_time))))
            .clip(lower=1.0, upper=730.0)
            .to_numpy(dtype=float)
        )
        _proj_gate = np.asarray(projected_days_now, dtype=float)
        _ads_gate = ads_num.to_numpy(dtype=float)
        _po_gate = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0).to_numpy(dtype=int)
        _block_lead_win = (
            _from_sheet_ok
            & (_po_gate > 0)
            & (_ads_gate > 0)
            & (_proj_gate > _lt_gate)
        )
        if bool(np.any(_block_lead_win)):
            _idx_bw = po_df.index[_block_lead_win]
            po_df.loc[_idx_bw, "PO_Qty"] = 0
            po_df.loc[_idx_bw, "Gross_PO_Qty"] = 0
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

    # SKU Status sheet rules (after every automated PO_Qty adjustment):
    # • Rows marked CLOSED must never receive a fresh PO release.
    # • When a status sheet IS loaded, a positive lead must have been resolvable from that
    #   sheet (direct row, parent rollup, longest-prefix, or pure numeric style code row) —
    #   not merely the global default, and not digit-token-only inference from non-numeric
    #   keys (which can borrow an unrelated style).
    _closed = po_df.get("SKU_Sheet_Closed", pd.Series(False, index=po_df.index)).fillna(False).astype(bool)
    _hot = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0
    _msg_closed = "SKU marked closed on status sheet"
    _msg_nolead = "No lead time resolved from SKU status sheet for this SKU"
    block_closed = _closed & _hot
    if block_closed.any():
        po_df.loc[block_closed, "PO_Qty"] = 0
        po_df.loc[block_closed, "Gross_PO_Qty"] = 0
        br = po_df.loc[block_closed, "PO_Block_Reason"].astype(str).str.strip()
        po_df.loc[block_closed, "PO_Block_Reason"] = np.where(
            br.eq("") | br.eq("nan"),
            _msg_closed,
            br + "; " + _msg_closed,
        )
        po_df.loc[block_closed, "Lead_Time_Days"] = 0
    if "Lead_Time_From_Status_Sheet" in po_df.columns:
        _sheet_ok = po_df["Lead_Time_From_Status_Sheet"].fillna(False).astype(bool)
        _hot2 = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0
        block_lt = (~_sheet_ok) & _hot2
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

    # Drop intermediate calc columns (datetime/float cols that break router serialisation)
    po_df.drop(
        columns=["ADS_Sold_Units", "ADS_Net_Units"],
        errors="ignore",
        inplace=True,
    )

    return po_df
