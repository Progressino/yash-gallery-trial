"""SKU Deepdive — merge unified sales with platform bulk history (same source as PO quarterly)."""
from __future__ import annotations

import re
from typing import Set

import pandas as pd

from .helpers import get_parent_sku, normalized_sku_forms_for_lookup
from .sales import (
    canonical_sales_sku,
    canonical_sales_sku_series,
    txn_reporting_naive_ist,
)
from .shared_frames import (
    session_platform_df,
    session_sales_df,
)


def _yk_style_hyphen_variants(token: str) -> Set[str]:
    """165YK-251MUSTRAD ↔ 165YK251MUSTRAD (Meesho vs Amazon spelling)."""
    out: Set[str] = set()
    u = (token or "").strip().upper()
    m = re.match(r"^(\d+YK)-(\d+[A-Z0-9\-]*)$", u)
    if m:
        out.add(f"{m.group(1)}{m.group(2)}")
    m2 = re.match(r"^(\d+YK)(\d+[A-Z0-9\-]*)$", u)
    if m2 and "-" not in u[len(m2.group(1)) :]:
        out.add(f"{m2.group(1)}-{m2.group(2)}")
    return out


def deepdive_sku_alias_tokens(raw: str) -> Set[str]:
    """All SKU spellings that should match one listing (PL strip, hyphen variants, MUSTRAD↔MUSTARD)."""
    u = (raw or "").strip().upper()
    out: Set[str] = set()
    if not u or u == "NAN":
        return out
    out.add(u)
    out.add(canonical_sales_sku(u))
    for form in normalized_sku_forms_for_lookup(u):
        out.add(form)
        out.add(canonical_sales_sku(form))
    i = 0
    snap = list(out)
    while i < len(snap):
        out |= _yk_style_hyphen_variants(snap[i])
        i += 1
    return {x for x in out if x and x != "NAN"}


def deepdive_parent_tokens(raw: str) -> Set[str]:
    """Parent/base SKU tokens including hyphen-normalised forms (165YK-251… ↔ 165YK251…)."""
    out: Set[str] = set()
    for alias in deepdive_sku_alias_tokens(raw):
        parent = get_parent_sku(alias)
        if parent is None or (isinstance(parent, float) and pd.isna(parent)):
            continue
        for form in deepdive_sku_alias_tokens(str(parent).strip()):
            out.add(form)
    return {x for x in out if x}


def _platform_raw_sku_series(df: pd.DataFrame) -> pd.Series:
    col = next((c for c in df.columns if c in ("OMS_SKU", "SKU", "Sku")), None)
    if not col:
        return pd.Series("", index=df.index, dtype=str)
    return df[col].fillna("").astype(str).str.strip()


def _sku_match_mask(
    sku_series: pd.Series,
    *,
    exact_targets: Set[str],
    parent_targets: Set[str],
    all_sizes: bool,
) -> pd.Series:
    if sku_series.empty:
        return pd.Series(dtype=bool, index=sku_series.index)
    canon = canonical_sales_sku_series(sku_series)
    hit = canon.isin(exact_targets)
    if not all_sizes:
        return hit
    uniq = sku_series.unique()
    parent_map: dict[str, Set[str]] = {}
    for u in uniq:
        parent_map[str(u)] = deepdive_parent_tokens(str(u).strip())
    def _row_parent_hit(val: str) -> bool:
        return bool(parent_map.get(val, set()) & parent_targets)
    parent_hit = sku_series.map(lambda v: _row_parent_hit(str(v)))
    return hit | parent_hit


def _filter_platform_df(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    if df.empty or not mask.any():
        return df.iloc[0:0]
    return df.loc[mask]


def _merge_platform_and_sales(plat: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Platform bulk history wins on (canonical SKU, day); append non-overlapping sales rows."""
    if plat.empty:
        return sales.copy() if sales is not None else pd.DataFrame()
    if sales is None or sales.empty:
        return plat.copy()

    plat = plat.copy()
    sales = sales.copy()
    plat["_day"] = txn_reporting_naive_ist(plat["TxnDate"]).dt.normalize()
    sales["_day"] = txn_reporting_naive_ist(sales["TxnDate"]).dt.normalize()
    plat["_skukey"] = canonical_sales_sku_series(plat["Sku"])
    sales["_skukey"] = canonical_sales_sku_series(sales["Sku"])

    plat_keys = plat[["_skukey", "_day"]].drop_duplicates()
    plat_keys["_in_plat"] = True
    merged = sales.merge(plat_keys, on=["_skukey", "_day"], how="left")
    extra = merged[merged["_in_plat"].isna()].drop(
        columns=["_day", "_skukey", "_in_plat"], errors="ignore"
    )
    return pd.concat(
        [plat.drop(columns=["_day", "_skukey"], errors="ignore"), extra],
        ignore_index=True,
    )


def _build_platform_sales_parts(sess, sku_mask_fn) -> pd.DataFrame:
    """Convert matching rows from each platform frame to unified sales schema."""
    from .sales import (
        _build_flipkart_sales_part,
        _build_mtr_sales_tagged,
        _build_myntra_sales_part,
        _build_snapdeal_sales_part,
    )
    from .meesho import meesho_to_sales_rows

    mapping = getattr(sess, "sku_mapping", None) or {}
    parts: list[pd.DataFrame] = []

    mtr = session_platform_df(sess, "mtr_df")
    if not mtr.empty:
        raw = _platform_raw_sku_series(mtr)
        sub = _filter_platform_df(mtr, sku_mask_fn(raw))
        if not sub.empty:
            part = _build_mtr_sales_tagged(sub, mapping)
            if not part.empty:
                parts.append(part)

    myntra = session_platform_df(sess, "myntra_df")
    if not myntra.empty:
        raw = _platform_raw_sku_series(myntra)
        sub = _filter_platform_df(myntra, sku_mask_fn(raw))
        if not sub.empty:
            part = _build_myntra_sales_part(sub)
            if not part.empty:
                parts.append(part)

    meesho = session_platform_df(sess, "meesho_df")
    if not meesho.empty:
        raw = _platform_raw_sku_series(meesho)
        sub = _filter_platform_df(meesho, sku_mask_fn(raw))
        if not sub.empty:
            part = meesho_to_sales_rows(sub, sku_mapping=mapping or None)
            if not part.empty:
                parts.append(part)

    flipkart = session_platform_df(sess, "flipkart_df")
    if not flipkart.empty:
        raw = _platform_raw_sku_series(flipkart)
        sub = _filter_platform_df(flipkart, sku_mask_fn(raw))
        if not sub.empty:
            part = _build_flipkart_sales_part(sub)
            if not part.empty:
                parts.append(part)

    snapdeal = session_platform_df(sess, "snapdeal_df")
    if not snapdeal.empty:
        raw = _platform_raw_sku_series(snapdeal)
        sub = _filter_platform_df(snapdeal, sku_mask_fn(raw))
        if not sub.empty:
            part = _build_snapdeal_sales_part(sub)
            if not part.empty:
                parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_deepdive_sales_frame(sess, sku: str, *, all_sizes: bool) -> pd.DataFrame:
    """
    Rows for one deep-dive query: platform upload history plus unified sales gaps.
  """
    aliases = deepdive_sku_alias_tokens(sku)
    exact_targets: Set[str] = set(aliases)
    parent_targets: Set[str] = set()
    if all_sizes:
        for a in aliases:
            parent_targets |= deepdive_parent_tokens(a)
    else:
        parent_targets = set()

    def sku_mask_fn(raw: pd.Series) -> pd.Series:
        return _sku_match_mask(
            raw,
            exact_targets=exact_targets,
            parent_targets=parent_targets,
            all_sizes=all_sizes,
        )

    plat = _build_platform_sales_parts(sess, sku_mask_fn)

    sales = session_sales_df(sess)
    if sales is None or sales.empty:
        return plat

    canon = canonical_sales_sku_series(sales["Sku"])
    if all_sizes:
        uniq = sales["Sku"].astype(str).unique()
        parent_map = {u: deepdive_parent_tokens(str(u).strip()) for u in uniq}
        sales_mask = canon.isin(exact_targets) | sales["Sku"].astype(str).map(
            lambda v: bool(parent_map.get(v, set()) & parent_targets)
        )
    else:
        sales_mask = canon.isin(exact_targets)

    sales_part = sales.loc[sales_mask].copy()
    return _merge_platform_and_sales(plat, sales_part)
