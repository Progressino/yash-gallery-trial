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
    platform_frame_for_window,
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


def deepdive_mapping_seller_aliases(oms: str, mapping: dict | None) -> Set[str]:
    """Seller / Meesho keys that map to the searched OMS (e.g. 1057YKNBLUE-5XL → 1057YKBLUE-5XL)."""
    if not mapping:
        return set()
    targets = {t for t in deepdive_sku_alias_tokens(oms) if t}
    if not targets:
        return set()
    out: Set[str] = set()
    for k, v in mapping.items():
        vv = str(v or "").strip().upper()
        if not vv:
            continue
        if vv in targets or canonical_sales_sku(vv) in targets:
            out |= deepdive_sku_alias_tokens(str(k))
    return out


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
    """Prefer OMS_SKU (post-map) over raw seller SKU — Meesho often stores YKN… in SKU."""
    for col in ("OMS_SKU", "SKU", "Sku"):
        if col in df.columns:
            return df[col].fillna("").astype(str).str.strip()
    return pd.Series("", index=df.index, dtype=str)


def _platform_sku_match_mask(df: pd.DataFrame, sku_mask_fn) -> pd.Series:
    """Match if ANY of OMS_SKU / SKU / Sku hits — covers Meesho YKN→YK mapping gaps."""
    masks: list[pd.Series] = []
    for col in ("OMS_SKU", "SKU", "Sku"):
        if col not in df.columns:
            continue
        raw = df[col].fillna("").astype(str).str.strip()
        masks.append(sku_mask_fn(raw))
    if not masks:
        return pd.Series(False, index=df.index)
    out = masks[0].copy()
    for m in masks[1:]:
        out = out | m
    return out


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
    """Platform bulk history wins on (canonical SKU, day); append non-overlapping sales rows.

    When Amazon platform history is present for a SKU, also drop extra Amazon sales
    rows on days the platform frame already covers — stale unified sales_df can still
    carry PL-twin shipments that were wrongly mapped to the OMS SKU.
    """
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

    # When Amazon MTR platform history exists for a SKU, unified sales_df must not
    # supplement it — stale sales_df rows often include PL-twin shipments that were
    # wrongly folded onto the OMS token (inflates deepdive, e.g. 82 vs 60 for XXL).
    if (
        not plat.empty
        and not sales.empty
        and "Source" in plat.columns
        and "Source" in sales.columns
    ):
        amz_skus = set(
            plat.loc[plat["Source"].astype(str) == "Amazon", "_skukey"].astype(str)
        )
        if amz_skus:
            drop = (sales["Source"].astype(str) == "Amazon") & sales["_skukey"].isin(
                amz_skus
            )
            sales = sales.loc[~drop]

    plat_keys = plat[["_skukey", "_day"]].drop_duplicates()
    plat_keys["_in_plat"] = True
    merged = sales.merge(plat_keys, on=["_skukey", "_day"], how="left")
    extra = merged[merged["_in_plat"].isna()].drop(
        columns=["_day", "_skukey", "_in_plat"], errors="ignore"
    )

    # If Amazon plat history is present for a skukey, do not supplement more Amazon
    # sales days for that skukey — those gap days are typically PL twins that share
    # the OMS token in a stale sales_df after PL-strip.
    if (
        not plat.empty
        and "Source" in plat.columns
        and not extra.empty
        and "Source" in extra.columns
    ):
        amz_skus = set(
            plat.loc[plat["Source"].astype(str) == "Amazon", "_skukey"].astype(str)
        )
        if amz_skus:
            extra_skukey = canonical_sales_sku_series(extra["Sku"])
            drop = (extra["Source"].astype(str) == "Amazon") & extra_skukey.isin(amz_skus)
            extra = extra.loc[~drop]

    return pd.concat(
        [plat.drop(columns=["_day", "_skukey"], errors="ignore"), extra],
        ignore_index=True,
    )


def _build_platform_sales_parts(
    sess,
    sku_mask_fn,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
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

    mtr = platform_frame_for_window("mtr_df", sess, start_date=start_date, end_date=end_date)
    if not mtr.empty:
        sub = _filter_platform_df(mtr, _platform_sku_match_mask(mtr, sku_mask_fn))
        if not sub.empty:
            # Re-apply FBA shadow-row dedup on the small per-SKU slice to guarantee
            # correctness regardless of whether the session-level mtr_df was already
            # cleaned (the full-DataFrame dedup may have missed rows due to
            # session-merge ordering or warm-cache generation timing).
            if (
                "Invoice_Number" in sub.columns
                and "Order_Id" in sub.columns
                and len(sub) > 1
            ):
                try:
                    from .mtr import dedup_amazon_mtr_dataframe
                    sub = dedup_amazon_mtr_dataframe(sub)
                except Exception:
                    pass
            part = _build_mtr_sales_tagged(sub, mapping)
            if not part.empty:
                parts.append(part)

    myntra = platform_frame_for_window("myntra_df", sess, start_date=start_date, end_date=end_date)
    if not myntra.empty:
        sub = _filter_platform_df(myntra, _platform_sku_match_mask(myntra, sku_mask_fn))
        if not sub.empty:
            part = _build_myntra_sales_part(sub)
            if not part.empty:
                parts.append(part)

    meesho = platform_frame_for_window("meesho_df", sess, start_date=start_date, end_date=end_date)
    if not meesho.empty:
        sub = _filter_platform_df(meesho, _platform_sku_match_mask(meesho, sku_mask_fn))
        if not sub.empty:
            part = meesho_to_sales_rows(sub, sku_mapping=mapping or None)
            if not part.empty:
                parts.append(part)

    flipkart = platform_frame_for_window("flipkart_df", sess, start_date=start_date, end_date=end_date)
    if not flipkart.empty:
        sub = _filter_platform_df(flipkart, _platform_sku_match_mask(flipkart, sku_mask_fn))
        if not sub.empty:
            part = _build_flipkart_sales_part(sub)
            if not part.empty:
                parts.append(part)

    snapdeal = platform_frame_for_window("snapdeal_df", sess, start_date=start_date, end_date=end_date)
    if not snapdeal.empty:
        sub = _filter_platform_df(snapdeal, _platform_sku_match_mask(snapdeal, sku_mask_fn))
        if not sub.empty:
            part = _build_snapdeal_sales_part(sub)
            if not part.empty:
                parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_deepdive_sales_frame(
    sess,
    sku: str,
    *,
    all_sizes: bool,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Rows for one deep-dive query: platform upload history plus unified sales gaps.
  """
    aliases = deepdive_sku_alias_tokens(sku)
    mapping = getattr(sess, "sku_mapping", None) or {}
    # Seller keys (Meesho YKN…) for finding rows — do NOT use for post-filter, or
    # Amazon distinct-ASIN PL listings (PLYK…) would inflate the OMS deepdive.
    match_targets: Set[str] = set(aliases) | deepdive_mapping_seller_aliases(sku, mapping)
    exact_targets: Set[str] = set(aliases)
    parent_targets: Set[str] = set()
    if all_sizes:
        for a in list(aliases)[:50]:
            parent_targets |= deepdive_parent_tokens(a)
    else:
        parent_targets = set()

    def sku_mask_fn(raw: pd.Series) -> pd.Series:
        return _sku_match_mask(
            raw,
            exact_targets=match_targets,
            parent_targets=parent_targets,
            all_sizes=all_sizes,
        )

    plat = _build_platform_sales_parts(
        sess,
        sku_mask_fn,
        start_date=start_date,
        end_date=end_date,
    )

    # Exact-SKU mode: keep only rows whose resolved Sku equals the searched OMS token.
    # Distinct-ASIN PL listings stay on their PL seller Sku (protect_distinct_asin_pl_skus)
    # and must not inflate OMS — do not PL-strip here.
    # Meesho YKN rows resolve to OMS in meesho_to_sales_rows, so they pass this filter.
    if not all_sizes and plat is not None and not plat.empty and "Sku" in plat.columns:
        plat_skus = plat["Sku"].astype(str).str.strip().str.upper()
        plat = plat.loc[plat_skus.isin(exact_targets)].copy()

    sales = session_sales_df(sess)
    if sales is None or sales.empty:
        out = plat
    else:
        # Narrow sales early when a date window is set — avoids scanning full history.
        sales_work = sales
        if start_date or end_date:
            td = txn_reporting_naive_ist(sales["TxnDate"])
            mask = td.notna()
            if start_date:
                mask &= td >= pd.Timestamp(str(start_date)[:10])
            if end_date:
                mask &= td <= pd.Timestamp(str(end_date)[:10]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            sales_work = sales.loc[mask]
        if sales_work.empty:
            out = plat
        else:
            canon = canonical_sales_sku_series(sales_work["Sku"])
            if all_sizes:
                uniq = sales_work["Sku"].astype(str).unique()
                parent_map = {u: deepdive_parent_tokens(str(u).strip()) for u in uniq}
                sales_mask = canon.isin(exact_targets) | sales_work["Sku"].astype(str).map(
                    lambda v: bool(parent_map.get(v, set()) & parent_targets)
                )
            else:
                sales_mask = canon.isin(exact_targets)

            sales_part = sales_work.loc[sales_mask].copy()
            out = _merge_platform_and_sales(plat, sales_part)

    if out is None or out.empty:
        return pd.DataFrame()
    from .sales import _drop_amazon_unkeyed_shadows
    return _drop_amazon_unkeyed_shadows(out)


def amazon_mtr_b2b_b2c_monthly(
    sess,
    sku: str,
    *,
    all_sizes: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Amazon MTR units by invoice month and B2B/B2C (gross shipments − refunds)."""
    from .sales import (
        amazon_mtr_reporting_date,
        apply_amazon_free_replacement_txn,
        canonical_sales_sku_series,
    )

    mtr = platform_frame_for_window("mtr_df", sess, start_date=start_date, end_date=end_date)
    if mtr.empty:
        return []

    aliases = deepdive_sku_alias_tokens(sku)
    exact_targets: Set[str] = set(aliases)
    parent_targets: Set[str] = set()
    if all_sizes:
        for a in aliases:
            parent_targets |= deepdive_parent_tokens(a)

    raw = _platform_raw_sku_series(mtr)
    mask = _sku_match_mask(
        raw,
        exact_targets=exact_targets,
        parent_targets=parent_targets,
        all_sizes=all_sizes,
    )
    sub = _filter_platform_df(mtr, mask)
    if sub.empty:
        return []

    if len(sub) > 1 and {"Invoice_Number", "Order_Id"}.issubset(sub.columns):
        try:
            from .mtr import dedup_amazon_mtr_dataframe
            sub = dedup_amazon_mtr_dataframe(sub)
        except Exception:
            pass

    sub = apply_amazon_free_replacement_txn(sub)
    sub = sub.copy()
    sub["_rep"] = amazon_mtr_reporting_date(sub)
    sub = sub.dropna(subset=["_rep"])
    if start_date:
        sub = sub[sub["_rep"] >= pd.Timestamp(start_date)]
    if end_date:
        sub = sub[sub["_rep"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if sub.empty or "Report_Type" not in sub.columns:
        return []

    sub["_month"] = sub["_rep"].dt.to_period("M").astype(str)
    rows: list[dict] = []
    for (month, rt), grp in sub.groupby(["_month", "Report_Type"], sort=True):
        qty = pd.to_numeric(grp["Quantity"], errors="coerce").fillna(0)
        txn = grp["Transaction_Type"].astype(str).str.strip()
        ship = int(qty[txn == "Shipment"].sum())
        ret = int(qty[txn == "Refund"].sum())
        can = int(qty[txn == "Cancel"].sum())
        free = int(qty[txn == "FreeReplacement"].sum())
        rows.append({
            "month": str(month),
            "channel": str(rt),
            "units": ship - ret,
            "shipments": ship,
            "returns": ret,
            "cancelled": can,
            "free_replacements": free,
        })
    return rows
