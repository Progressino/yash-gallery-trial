"""
Data query router — analytics endpoints.
GET /api/data/coverage, sales-summary, sales-export, sales-by-source, daily-dsr, daily-dsr-export,
dsr-brand-monthly, dsr-brand-monthly-export, top-skus,
mtr-analytics, myntra-analytics, meesho-analytics, flipkart-analytics, inventory
"""
import csv
import io
import re
from typing import List, Optional, Set
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from ..models.schemas import CoverageResponse
from ..services.helpers import (
    clean_sku,
    get_parent_sku,
    is_likely_non_sku_notes_value,
    map_to_oms_sku,
    mapping_lookup_sets,
    normalize_id_token_for_mapping,
    sku_recognized_in_master,
)
from ..services.meesho import apply_meesho_listing_sku_recovery_for_export
from ..services.sales import (
    _filter_by_reporting_days,
    canonical_sales_sku,
    canonical_sales_sku_series,
    daily_dsr_report_to_csv_rows,
    filter_sales_for_export,
    get_anomalies,
    get_daily_dsr_report,
    dsr_brand_monthly_to_csv_rows,
    get_dsr_brand_monthly_comparison,
    get_platform_summary,
    get_sales_by_source,
    get_sales_summary,
    get_top_skus,
    txn_reporting_naive_ist,
)
from ..services.daily_store import list_uploads, get_summary, delete_upload
from ..session import AppSession

router = APIRouter()


def _sku_deepdive_aliases(raw: str) -> Set[str]:
    """Return SKU tokens that should match the same row after PL/YK normalisation."""
    u = raw.strip().upper()
    out = {u, canonical_sales_sku(u)}
    m = re.match(r"^(\d+)(YK[A-Z0-9\-]+)$", u)
    if m and "PL" not in m.group(0):
        out.add(f"{m.group(1)}PL{m.group(2)}")
    return {x for x in out if x and x != "NAN"}


def _sess(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")
    return sess


def _restore_daily_if_needed(sess: AppSession) -> None:
    """
    On first coverage check per session, load any persisted daily SQLite data
    into the session DFs (fills in data lost on server restart).
    Skips platforms that already have session data.
    Also auto-restores SKU mapping from GitHub cache if missing.
    """
    if getattr(sess, "pause_auto_data_restore", False):
        return
    if sess.daily_restored:
        return

    with sess._daily_restore_lock:
        if sess.daily_restored:
            return

        import pandas as pd
        from ..services.daily_store import load_platform_data
        from ..services.sales import build_sales_df

        # Auto-restore SKU mapping from GitHub cache if missing (lightweight — JSON only)
        if not sess.sku_mapping:
            try:
                from ..services.github_cache import load_sku_mapping_from_drive
                mapping = load_sku_mapping_from_drive()
                if mapping:
                    sess.sku_mapping = mapping
            except Exception:
                pass  # GitHub not configured or network error — skip silently

        changed = False
        for platform, attr in [
            ("amazon",   "mtr_df"),
            ("myntra",   "myntra_df"),
            ("meesho",   "meesho_df"),
            ("flipkart", "flipkart_df"),
            ("snapdeal", "snapdeal_df"),
        ]:
            if getattr(sess, attr).empty:
                df = load_platform_data(platform)
                if not df.empty:
                    setattr(sess, attr, df)
                    changed = True

        if changed:
            try:
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df,
                    myntra_df=sess.myntra_df,
                    meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df,
                    snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping,
                )
                sess._quarterly_cache.clear()
            except Exception:
                pass

        # Restore inventory from warm cache (fast — already in memory) or GitHub as fallback.
        # Inventory has no SQLite backing so it's always lost on server restart.
        need_inventory = sess.inventory_df_variant.empty
        if need_inventory:
            try:
                import backend.main as _main
                if _main._warm_cache:
                    # Fast path: copy from in-memory warm cache — no network call
                    for key in ["inventory_df_variant", "inventory_df_parent"]:
                        val = _main._warm_cache.get(key)
                        if val is not None and not (isinstance(val, pd.DataFrame) and val.empty):
                            setattr(sess, key, val)
                    if not sess.sku_mapping and _main._warm_cache.get("sku_mapping"):
                        sess.sku_mapping = _main._warm_cache["sku_mapping"]
                else:
                    # Warm cache not ready yet — fall back to GitHub download
                    from ..services.github_cache import load_cache_from_drive
                    ok, _, loaded = load_cache_from_drive()
                    if ok and loaded:
                        for key in ["inventory_df_variant", "inventory_df_parent"]:
                            val = loaded.get(key)
                            if val is not None and not (isinstance(val, pd.DataFrame) and val.empty):
                                setattr(sess, key, val)
                        if not sess.sku_mapping and loaded.get("sku_mapping"):
                            sess.sku_mapping = loaded["sku_mapping"]
            except Exception:
                pass

        sess.daily_restored = True


# ── Coverage ──────────────────────────────────────────────────

@router.get("/coverage", response_model=CoverageResponse)
def get_coverage(request: Request):
    sess = _sess(request)
    _restore_daily_if_needed(sess)   # auto-load persisted daily data on first access
    paused = getattr(sess, "pause_auto_data_restore", False)
    from ..services.daily_store import get_summary

    tier3_any = bool(get_summary())
    return CoverageResponse(
        sku_mapping=bool(sess.sku_mapping),
        mtr=not sess.mtr_df.empty,
        sales=not sess.sales_df.empty,
        myntra=not sess.myntra_df.empty,
        meesho=not sess.meesho_df.empty,
        flipkart=not sess.flipkart_df.empty,
        snapdeal=not sess.snapdeal_df.empty,
        inventory=not sess.inventory_df_variant.empty,
        daily_orders=len(sess.daily_sales_sources) > 0 or tier3_any,
        existing_po=not sess.existing_po_df.empty,
        mtr_rows=len(sess.mtr_df),
        sales_rows=len(sess.sales_df),
        myntra_rows=len(sess.myntra_df),
        meesho_rows=len(sess.meesho_df),
        flipkart_rows=len(sess.flipkart_df),
        snapdeal_rows=len(sess.snapdeal_df),
        pause_auto_data_restore=paused,
    )


# ── Data quality (duplicate / sanity checks) ──────────────────

@router.get("/data-quality")
def data_quality_report(request: Request):
    """
    Lightweight diagnostics to spot overlapping uploads and sanity-check totals.
    Does not modify session data.
    """
    import pandas as pd
    sess = _sess(request)
    _restore_daily_if_needed(sess)

    hints = [
        "Pick one SKU and date range, then in SKU Deepdive choose “Amazon only” (or another channel) and compare units to your marketplace export for the same window.",
        "If “All channels” is much higher than a single channel export, you are summing every marketplace — that is expected, not a bug.",
        "Amazon MTR “potential duplicate rows” uses the same collapse rules as uploads: if this number is large, overlapping Tier‑1 ZIPs were merging extra lines (now deduped when you re-upload or Rebuild).",
        "After “Reset all data”, upload Tier‑1 first, click Build Sales, then add Tier‑3 dailies so history is not loaded twice from SQLite + bulk.",
    ]

    checks: dict = {}
    loaded = (
        bool(sess.sku_mapping)
        or not sess.mtr_df.empty
        or not sess.sales_df.empty
        or not sess.myntra_df.empty
    )

    if not sess.mtr_df.empty and {
        "Invoice_Number", "Order_Id", "SKU", "Transaction_Type", "Quantity", "Date",
    }.issubset(sess.mtr_df.columns):
        from ..services.mtr import dedup_amazon_mtr_dataframe

        raw_n = len(sess.mtr_df)
        ded_n = len(dedup_amazon_mtr_dataframe(sess.mtr_df.copy()))
        checks["amazon_mtr"] = {
            "rows_in_session":     raw_n,
            "rows_after_dedup_key": ded_n,
            "rows_collapsible":    max(0, raw_n - ded_n),
            "note": (
                "Rows that would merge if dedup rules were applied again to the current "
                "Amazon frame (overlapping ZIPs / duplicate lines)."
            ),
        }

    if not sess.sales_df.empty and "Sku" in sess.sales_df.columns:
        s = sess.sales_df
        key = [c for c in ("Sku", "TxnDate", "Transaction Type", "Quantity", "Source", "OrderId") if c in s.columns]
        dup_extra = 0
        if key:
            uniq = s.drop_duplicates(subset=key, keep="first")
            dup_extra = int(len(s) - len(uniq))
        ship = s[s["Transaction Type"].astype(str).str.strip() == "Shipment"]
        ship_units = int(pd.to_numeric(ship["Quantity"], errors="coerce").fillna(0).sum()) if not ship.empty else 0
        checks["unified_sales_df"] = {
            "rows":              len(s),
            "exact_duplicate_rows": dup_extra,
            "shipment_units_sum": ship_units,
            "by_source":         get_sales_by_source(sess.sales_df),
        }

    try:
        checks["tier3_sqlite_summary"] = get_summary()
    except Exception:
        checks["tier3_sqlite_summary"] = {}

    return {
        "loaded": loaded,
        "checks": checks,
        "hints":  hints,
    }


# ── Sales Dashboard KPIs ──────────────────────────────────────

@router.get("/sales-summary")
def sales_summary(
    request: Request,
    months: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_sales_summary(sess.sales_df, months=months, start_date=start_date, end_date=end_date)


@router.get("/sales-export")
def sales_export(
    request: Request,
    months: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    platforms: Optional[str] = None,
):
    """CSV of unified `sales_df` rows for the dashboard date range (and optional platform list)."""
    import pandas as pd

    sess = _sess(request)
    _restore_daily_if_needed(sess)
    if sess.sales_df.empty:
        raise HTTPException(status_code=404, detail="No sales data loaded — upload or rebuild sales first.")

    src_list: Optional[List[str]] = None
    if platforms and platforms.strip():
        src_list = [p.strip() for p in platforms.split(",") if p.strip()]

    df = filter_sales_for_export(
        sess.sales_df,
        months=months,
        start_date=start_date,
        end_date=end_date,
        sources=src_list,
    )
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No rows in this date range / platform filter — widen dates or include more platforms.",
        )

    out = df.copy()
    out["TxnDate"] = pd.to_datetime(out["TxnDate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    base_cols = ["TxnDate", "Sku", "Transaction Type", "Quantity", "Units_Effective", "Source"]
    extra = [c for c in ("OrderId",) if c in out.columns]
    cols = [c for c in base_cols + extra if c in out.columns]
    export_df = out[cols].copy()
    export_df = apply_meesho_listing_sku_recovery_for_export(export_df, sess.meesho_df)
    cmap = sess.sku_mapping or {}
    _map_keys, _map_vals, _map_num = mapping_lookup_sets(cmap) if cmap else (set(), set(), {})

    def _export_oms_sku_cell(v) -> str:
        if pd.isna(v):
            return ""
        s = normalize_id_token_for_mapping(str(v).strip())
        if s.lower() in ("", "nan", "none"):
            return ""
        if is_likely_non_sku_notes_value(s):
            return ""
        resolved = canonical_sales_sku(map_to_oms_sku(s, cmap))
        if is_likely_non_sku_notes_value(resolved):
            return ""
        # If lookup is a no-op and this token never appears on the master (key or OMS),
        # leave OMS_Sku blank so exports don't fake a match (common for raw Myntra style IDs).
        if (
            cmap
            and clean_sku(resolved) == clean_sku(s)
            and not sku_recognized_in_master(
                s, cmap, key_set=_map_keys, val_set=_map_vals, numeric_embed=_map_num
            )
        ):
            return ""
        return resolved

    export_df["OMS_Sku"] = export_df["Sku"].apply(_export_oms_sku_cell)
    sku_pos = cols.index("Sku") + 1
    ordered = cols[:sku_pos] + ["OMS_Sku"] + cols[sku_pos:]
    export_df = export_df[ordered]

    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    body = buf.getvalue().encode("utf-8")

    part_start = (start_date or "all").replace(":", "")
    part_end = (end_date or "all").replace(":", "")
    fname = f"intelligence-sales_{part_start}_{part_end}_{len(export_df)}_rows.csv"

    return StreamingResponse(
        iter([body]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/sales-by-source")
def sales_by_source(request: Request):
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_sales_by_source(sess.sales_df)


@router.get("/top-skus")
def top_skus(
    request: Request,
    limit: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    basis: Optional[str] = None,
):
    """``basis=gross`` (default): rank by shipment quantity. ``basis=net``: by ``Units_Effective`` sum."""
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_top_skus(
        sess.sales_df,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        basis=basis or "gross",
    )


# ── SKU List (for search autocomplete) ───────────────────────

@router.get("/sku-list")
def sku_list(request: Request, q: Optional[str] = None, limit: int = 100, include_parents: bool = False):
    """Return unique SKUs in sales_df, optionally filtered by search query.
    When include_parents=True also returns deduplicated parent/base SKUs marked with a flag."""
    import pandas as pd

    sess = _sess(request)
    df   = sess.sales_df
    if df.empty or "Sku" not in df.columns:
        return []
    shipped = df[df["Transaction Type"].astype(str) == "Shipment"]["Sku"].astype(str)
    skus = shipped.unique().tolist()
    # Filter out noise rows
    skus = [s for s in skus if s and s.lower() not in ("nan", "none", "") and not s.lower().endswith("_total")]
    if q:
        q_lower = q.strip().lower()
        skus = [s for s in skus if q_lower in s.lower()]
    skus = sorted(skus)

    if include_parents:
        parents = sorted({get_parent_sku(s) for s in skus if get_parent_sku(s) != s})
        if q:
            q_lower = q.strip().lower()
            parents = [p for p in parents if q_lower in p.lower()]
        # Return as dicts with type flag so frontend can distinguish
        result = [{"sku": s, "type": "variant"} for s in skus[:limit]]
        result = [{"sku": p, "type": "parent"} for p in parents[:20]] + result
        return result[:limit]

    return skus[:limit]


# ── SKU Deepdive ──────────────────────────────────────────────

@router.get("/sku-deepdive")
def sku_deepdive(
    request: Request,
    sku: str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    all_sizes:  bool = False,   # if True, match all SKUs that share the same base (parent) SKU
    source: Optional[str] = None,  # if set (e.g. "Amazon"), only rows from that channel — matches single-market exports
):
    """
    Full sales breakdown for a single SKU (or all sizes of a base SKU).
    Returns: summary KPIs, monthly trend, platform breakdown, daily trend, sizes breakdown.
    Default window: last 90 days.
    When all_sizes=True the `sku` param is treated as the base/parent SKU and all
    size variants (e.g. 1898YKYELLOW-3XL, 1898YKYELLOW-2XL …) are aggregated together.
    """
    import pandas as pd

    sess = _sess(request)
    df0 = sess.sales_df

    # Detect whether Meesho is loaded but has no per-SKU data (TCS ZIP format)
    meesho_note: str | None = None
    if not sess.meesho_df.empty:
        meesho_skus_in_sales = (
            sess.sales_df[sess.sales_df["Source"].astype(str) == "Meesho"]["Sku"]
            .dropna().unique().tolist()
            if not sess.sales_df.empty and "Source" in sess.sales_df.columns else []
        )
        if meesho_skus_in_sales == ["MEESHO_TOTAL"] or set(meesho_skus_in_sales) == {"MEESHO_TOTAL"}:
            meesho_total_units = int(
                sess.sales_df[
                    (sess.sales_df["Source"].astype(str) == "Meesho") &
                    (sess.sales_df["Transaction Type"].astype(str) == "Shipment")
                ]["Quantity"].sum()
            ) if not sess.sales_df.empty else 0
            meesho_note = (
                f"Meesho data loaded ({meesho_total_units:,} total units) but your uploaded "
                f"Meesho TCS ZIP reports don't include per-SKU data. "
                f"To see Meesho in SKU breakdown, upload the Meesho Order Report CSV "
                f"(Supplier Panel → Reports → Order Reports)."
            )

    if df0.empty:
        return {"loaded": False, "message": "No sales data loaded"}

    # Parse dates once; avoid copying the full sales table (was the main latency on large sessions).
    txn_dates = txn_reporting_naive_ist(df0["TxnDate"])

    valid_dt = txn_dates.notna()
    source_filter: Optional[str] = None
    if source and str(source).strip():
        source_filter = str(source).strip()
        src_mask = df0["Source"].astype(str).str.strip() == source_filter
    else:
        src_mask = pd.Series(True, index=df0.index)

    base_mask = valid_dt & src_mask
    if not base_mask.any():
        return {
            "loaded":        bool(source_filter),
            "sku":           sku,
            "all_sizes":     all_sizes,
            "matched_skus":  [],
            "summary":       {"shipped": 0, "returns": 0, "net_units": 0, "return_rate": 0.0, "ads": 0.0},
            "monthly":       [],
            "by_platform":   [],
            "by_size":       [],
            "daily":         [],
            "first_sale":    None,
            "last_sale":     None,
            "meesho_note":   meesho_note,
            "source_filter": source_filter,
            "filter_note":   (
                f"No unified sales rows for channel “{source_filter}”. Try “All channels”."
                if source_filter
                else "No sales rows after date filter."
            ),
        }

    # Default: full loaded history (matches Excel "total sales" exports). Use explicit
    # start_date / end_date query params for a shorter window (e.g. last 90 days).
    if not start_date and not end_date:
        start_ts = txn_dates.loc[base_mask].min()
        end_ts = txn_dates.loc[base_mask].max()
    else:
        start_ts = pd.Timestamp(start_date) if start_date else txn_dates.loc[base_mask].min()
        end_ts = pd.Timestamp(end_date) if end_date else txn_dates.loc[base_mask].max()

    end_inclusive = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    date_mask = (txn_dates >= start_ts) & (txn_dates <= end_inclusive)
    pre_mask = base_mask & date_mask

    sku_variants = _sku_deepdive_aliases(sku)
    if all_sizes:
        parent_targets = {
            canonical_sales_sku(str(get_parent_sku(s)).strip()) for s in sku_variants
        }
        sub_skus = df0.loc[pre_mask, "Sku"].astype(str)
        uniq = sub_skus.unique()
        parent_map = {u: canonical_sales_sku(str(get_parent_sku(u)).strip()) for u in uniq}
        sub_parents = sub_skus.map(parent_map)
        sku_match_sub = sub_parents.isin(parent_targets)
        sku_mask = pd.Series(False, index=df0.index)
        sku_mask.loc[pre_mask] = sku_match_sub.values
    else:
        targets = {canonical_sales_sku(s) for s in sku_variants}
        sku_mask = canonical_sales_sku_series(df0["Sku"]).isin(targets)

    final_mask = pre_mask & sku_mask
    sku_df = df0.loc[final_mask].copy()
    sku_df["TxnDate"] = txn_dates.loc[final_mask]

    if sku_df.empty:
        return {
            "loaded":       True,
            "sku":          sku,
            "all_sizes":    all_sizes,
            "matched_skus": [],
            "summary":      {"shipped": 0, "returns": 0, "net_units": 0, "return_rate": 0.0, "ads": 0.0},
            "monthly":      [],
            "by_platform":  [],
            "by_size":      [],
            "daily":        [],
            "first_sale":   None,
            "last_sale":    None,
            "meesho_note":  meesho_note,
            "source_filter": source_filter,
            "filter_note":  None,
        }

    qty    = pd.to_numeric(sku_df["Quantity"],       errors="coerce").fillna(0)
    eff    = pd.to_numeric(sku_df["Units_Effective"], errors="coerce").fillna(0)
    txn    = sku_df["Transaction Type"].astype(str).str.strip()
    shipped  = int(qty[txn == "Shipment"].sum())
    returns  = int(qty[txn == "Refund"].sum())
    net_units = int(eff.sum())
    rr       = round(returns / shipped * 100, 1) if shipped > 0 else 0.0
    period_days = max((end_ts - start_ts).days, 1)
    # Use effective days (first shipment → end of period) so ADS isn't diluted
    # for products that launched mid-window — matches PO engine Eff_Days logic.
    ship_dates = sku_df.loc[txn == "Shipment", "TxnDate"]
    if not ship_dates.empty:
        first_ship_ts = ship_dates.min()
        eff_days = max((end_ts - first_ship_ts).days, 7)
        eff_days = min(eff_days, period_days)
    else:
        eff_days = period_days
    ads      = round(shipped / eff_days, 2) if shipped > 0 else 0.0

    # Monthly trend
    sku_df["_month"] = sku_df["TxnDate"].dt.to_period("M").astype(str)
    monthly_raw = (
        sku_df.assign(_qty=qty, _eff=eff)
        .groupby(["_month", "Transaction Type"])
        .agg(units=("_qty", "sum"))
        .reset_index()
        .pivot_table(index="_month", columns="Transaction Type", values="units", fill_value=0)
        .reset_index()
    )
    monthly_raw.columns.name = None
    monthly_raw = monthly_raw.rename(columns={
        "_month":   "month",
        "Shipment": "shipped",
        "Refund":   "returns",
        "Cancel":   "cancels",
    })
    for col in ["shipped", "returns", "cancels"]:
        if col not in monthly_raw.columns:
            monthly_raw[col] = 0
    monthly_raw["net"] = monthly_raw["shipped"] - monthly_raw.get("returns", 0)
    monthly = monthly_raw.sort_values("month")[["month", "shipped", "returns", "cancels", "net"]].to_dict("records")

    # Platform breakdown
    plat_grp = (
        sku_df.assign(_qty=qty)
        .groupby(["Source", "Transaction Type"])
        .agg(units=("_qty", "sum"))
        .reset_index()
        .pivot_table(index="Source", columns="Transaction Type", values="units", fill_value=0)
        .reset_index()
    )
    plat_grp.columns.name = None
    plat_grp = plat_grp.rename(columns={"Shipment": "shipped", "Refund": "returns"})
    if "shipped" not in plat_grp.columns:
        plat_grp["shipped"] = 0
    if "returns" not in plat_grp.columns:
        plat_grp["returns"] = 0
    plat_grp["return_rate"] = (plat_grp["returns"] / plat_grp["shipped"].replace(0, float("nan")) * 100).fillna(0).round(1)
    plat_grp = plat_grp.rename(columns={"Source": "platform"})
    by_platform = plat_grp[["platform", "shipped", "returns", "return_rate"]].sort_values("shipped", ascending=False).to_dict("records")

    # Daily trend (shipments only)
    _ship_m = txn == "Shipment"
    _ship_df = sku_df.loc[_ship_m]
    daily_grp = (
        _ship_df.assign(_qty=qty.loc[_ship_m], _day=_ship_df["TxnDate"].dt.strftime("%Y-%m-%d"))
        .groupby("_day", as_index=False)
        .agg(units=("_qty", "sum"))
        .rename(columns={"_day": "date"})
        .sort_values("date")
    )
    daily = daily_grp.to_dict("records")

    # Sizes breakdown (only meaningful in all_sizes mode)
    matched_skus = sorted(sku_df["Sku"].astype(str).unique().tolist())
    if all_sizes and len(matched_skus) > 1:
        sz_grp = (
            sku_df[txn == "Shipment"]
            .assign(_qty=qty[txn == "Shipment"])
            .groupby("Sku")
            .agg(shipped=("_qty", "sum"))
            .reset_index()
            .sort_values("shipped", ascending=False)
        )
        by_size = sz_grp.rename(columns={"Sku": "sku"}).to_dict("records")
    else:
        by_size = []

    return {
        "loaded":     True,
        "sku":        sku,
        "all_sizes":  all_sizes,
        "matched_skus": matched_skus,
        "start_date": str(start_ts.date()),
        "end_date":   str(end_ts.date()),
        "summary": {
            "shipped":     shipped,
            "returns":     returns,
            "net_units":   net_units,
            "return_rate": rr,
            "ads":         ads,
        },
        "monthly":     monthly,
        "by_platform": by_platform,
        "by_size":     by_size,
        "daily":       daily,
        "first_sale":  str(sku_df["TxnDate"].min().date()),
        "last_sale":   str(sku_df["TxnDate"].max().date()),
        "meesho_note": meesho_note,
        "source_filter": source_filter,
        "filter_note":  (
            f"Showing {source_filter} only — totals exclude other marketplaces."
            if source_filter
            else None
        ),
    }


# ── Daily Breakdown ───────────────────────────────────────────

@router.get("/daily-breakdown")
def daily_breakdown(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    platform: Optional[str] = None,   # comma-sep list, e.g. "Amazon,Meesho"
):
    """
    Per-day shipment/refund counts broken down by platform.
    Returns [{date, platform, units, returns}] sorted by date.
    """
    import pandas as pd
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    df = sess.sales_df
    if df.empty:
        return []

    try:
        d = df.copy()
        d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
        d = d.dropna(subset=["TxnDate"])
        if start_date or end_date:
            d = _filter_by_reporting_days(d, "TxnDate", start_date, end_date)
        if platform:
            plats = [p.strip() for p in platform.split(",")]
            d = d[d["Source"].isin(plats)]

        if d.empty:
            return []

        d["_day"] = d["TxnDate"].dt.strftime("%Y-%m-%d")
        qty = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)
        ship_mask   = d["Transaction Type"].astype(str).str.strip() == "Shipment"
        refund_mask = d["Transaction Type"].astype(str).str.strip() == "Refund"

        grp = (
            d.assign(_qty=qty)
            .groupby(["_day", "Source"])
            .apply(lambda g: pd.Series({
                "units":   int(g.loc[ship_mask.loc[g.index], "_qty"].sum()),
                "returns": int(g.loc[refund_mask.loc[g.index], "_qty"].sum()),
            }))
            .reset_index()
            .rename(columns={"_day": "date", "Source": "platform"})
            .sort_values("date")
        )
        return grp.to_dict("records")
    except Exception:
        return []


def _resolve_daily_dsr_date(sess: AppSession, date: Optional[str]) -> tuple:
    """Return (sales_df, iso_date_str) for DSR helpers."""
    import pandas as pd

    df = sess.sales_df
    if df.empty:
        return df, (date or "").strip()

    if not date or not str(date).strip():
        d = df.copy()
        d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
        d = d.dropna(subset=["TxnDate"])
        if d.empty:
            return df, ""
        latest = d["TxnDate"].max()
        if pd.isna(latest):
            return df, ""
        date = str(latest.normalize().date())

    return df, str(date).strip()


@router.get("/daily-dsr")
def daily_dsr(request: Request, date: Optional[str] = None):
    """
    Daily DSR-style report for one calendar day: marketplace sections with optional
    segment rows (Flipkart Brand, Snapdeal Company, etc.) and an Others bucket.
    Query: ``date`` = ISO ``YYYY-MM-DD`` (defaults to latest day with sales if omitted).
    """
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    df, iso = _resolve_daily_dsr_date(sess, date)
    return get_daily_dsr_report(df, iso)


@router.get("/daily-dsr-export")
def daily_dsr_export(request: Request, date: Optional[str] = None):
    """CSV download matching the on-screen Daily DSR table."""
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    df, iso = _resolve_daily_dsr_date(sess, date)
    report = get_daily_dsr_report(df, iso)
    buf = io.StringIO()
    w = csv.writer(buf)
    for row in daily_dsr_report_to_csv_rows(report):
        w.writerow(row)
    body = buf.getvalue().encode("utf-8")
    fname = f"daily-dsr_{report.get('date') or 'nodate'}.csv"
    return StreamingResponse(
        iter([body]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/dsr-brand-monthly")
def dsr_brand_monthly(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """YG vs Akiko shipment units by calendar month (DSR segment / brand labels)."""
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_dsr_brand_monthly_comparison(sess.sales_df, start_date, end_date)


@router.get("/dsr-brand-monthly-export")
def dsr_brand_monthly_export(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """CSV of YG vs Akiko monthly comparison (same logic as ``/dsr-brand-monthly``)."""
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    result = get_dsr_brand_monthly_comparison(sess.sales_df, start_date, end_date)
    buf = io.StringIO()
    w = csv.writer(buf)
    for row in dsr_brand_monthly_to_csv_rows(result):
        w.writerow(row)
    body = buf.getvalue().encode("utf-8")
    part = f"{start_date or 'all'}_{end_date or 'all'}".replace("/", "-")
    fname = f"dsr-yg-akiko-monthly_{part}.csv"
    return StreamingResponse(
        iter([body]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── MTR Analytics ─────────────────────────────────────────────

@router.get("/mtr-analytics")
def mtr_analytics(request: Request):
    sess = _sess(request)
    df = sess.mtr_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd
    import numpy as np

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Monthly shipments vs refunds
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby(["Month", "Transaction_Type"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="Transaction_Type", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={
        "Shipment": "shipments",
        "Refund":   "refunds",
        "Cancel":   "cancels",
    })

    # Top SKUs
    top = (
        df[df["Transaction_Type"] == "Shipment"]
        .groupby("SKU")["Quantity"].sum()
        .sort_values(ascending=False).head(20).reset_index()
    )
    top.columns = ["sku", "units"]

    # Summary
    shipped  = float(df[df["Transaction_Type"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["Transaction_Type"] == "Refund"]["Quantity"].sum())
    net_units = int(shipped - returned)

    if "shipments" in monthly.columns and "refunds" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly["refunds"]
    elif "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"]

    return {
        "loaded":       True,
        "rows":         len(df),
        "date_range":   [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":      int(shipped),
        "returned":     int(returned),
        "net_units":    net_units,
        "return_rate":  round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":      monthly.to_dict("records"),
        "top_skus":     top.to_dict("records"),
    }


# ── Myntra Analytics ─────────────────────────────────────────

@router.get("/myntra-analytics")
def myntra_analytics(request: Request):
    sess = _sess(request)
    df = sess.myntra_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})

    top_skus = (
        df[df["TxnType"] == "Shipment"].groupby("OMS_SKU")["Quantity"]
        .sum().sort_values(ascending=False).head(20).reset_index()
    )
    top_skus.columns = ["sku", "units"]

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
        "top_skus":    top_skus.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Meesho Analytics ─────────────────────────────────────────

@router.get("/meesho-analytics")
def meesho_analytics(request: Request):
    sess = _sess(request)
    df = sess.meesho_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Flipkart Analytics ────────────────────────────────────────

@router.get("/flipkart-analytics")
def flipkart_analytics(request: Request):
    sess = _sess(request)
    df = sess.flipkart_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})

    top_skus = (
        df[df["TxnType"] == "Shipment"].groupby("OMS_SKU")["Quantity"]
        .sum().sort_values(ascending=False).head(20).reset_index()
    )
    top_skus.columns = ["sku", "units"]

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
        "top_skus":    top_skus.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Inventory ─────────────────────────────────────────────────

@router.get("/inventory")
def get_inventory(request: Request):
    sess = _sess(request)
    df = sess.inventory_df_variant
    if df.empty:
        return {"loaded": False, "rows": []}

    import pandas as pd
    cols = [c for c in df.columns if c != "OMS_SKU"]

    # Per-source totals for debugging discrepancies
    totals = {}
    for c in cols:
        try:
            totals[c] = int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        except Exception:
            totals[c] = 0

    return {
        "loaded":   True,
        "rows":     df.fillna(0).to_dict("records"),
        "columns":  ["OMS_SKU"] + cols,
        "totals":   totals,
        "debug":    getattr(sess, "inventory_debug", {}),
    }


# ── Snapdeal Analytics ────────────────────────────────────────

@router.get("/snapdeal-analytics")
def snapdeal_analytics(request: Request, company: Optional[str] = None):
    sess = _sess(request)
    df = sess.snapdeal_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Collect unique companies before filtering
    companies: list = []
    if "Company" in df.columns:
        companies = sorted(df["Company"].dropna().str.strip().unique().tolist())
        companies = [c for c in companies if c]

    # Apply company filter
    if company and "Company" in df.columns:
        df = df[df["Company"].str.strip() == company.strip()]

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})
    if "shipments" not in monthly.columns:
        monthly["shipments"] = 0
    if "refunds" not in monthly.columns:
        monthly["refunds"] = 0

    top_skus = (
        df[df["TxnType"] == "Shipment"].groupby("OMS_SKU")["Quantity"]
        .sum().sort_values(ascending=False).head(20).reset_index()
    )
    top_skus.columns = ["sku", "units"]

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]
    by_state = by_state[by_state["state"].str.strip() != ""]

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "companies":   companies,
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
        "top_skus":    top_skus.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Snapdeal Debug (column inspection) ───────────────────────

@router.get("/snapdeal-debug")
def snapdeal_debug(request: Request):
    """Returns column names, TxnType distribution, and SKU sample from the loaded snapdeal_df."""
    sess = _sess(request)
    df = sess.snapdeal_df
    if df.empty:
        return {"loaded": False}
    return {
        "loaded":       True,
        "rows":         len(df),
        "txn_types":    df["TxnType"].value_counts().to_dict(),
        "sku_sample":   df["OMS_SKU"].value_counts().head(15).to_dict(),
        "state_sample": df["State"].value_counts().head(10).to_dict(),
        "parse_info":   sess.snapdeal_parse_info,   # raw cols + detected fields per file
        "sample_rows":  df.head(3).fillna("").to_dict("records"),
    }


# ── Daily Sales Management ───────────────────────────────────

@router.get("/daily-summary")
def daily_summary(_request: Request):
    """Per-platform summary of persisted daily uploads."""
    return get_summary()


@router.get("/daily-uploads")
def daily_uploads(_request: Request):
    """Full list of persisted daily upload records (newest first)."""
    return list_uploads()


@router.delete("/daily-uploads/{upload_id}")
def delete_daily_upload(upload_id: int, _request: Request):
    ok = delete_upload(upload_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {"ok": True, "message": f"Deleted upload {upload_id}"}


# ── Data Debug / Coverage ────────────────────────────────────

@router.get("/debug-coverage")
def debug_coverage(request: Request):
    """
    Returns row counts, date ranges, and sample transaction types
    for each loaded DataFrame. Useful for diagnosing data integrity
    issues on production without redeploying.
    """
    import pandas as pd
    sess = _sess(request)

    def _df_info(df: pd.DataFrame, date_col: str, txn_col: str | None = None) -> dict:
        if df.empty:
            return {"loaded": False, "rows": 0}
        out: dict = {"loaded": True, "rows": len(df)}
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                out["min_date"] = str(dates.min().date())
                out["max_date"] = str(dates.max().date())
                out["tz_aware"] = dates.dt.tz is not None
        except Exception as e:
            out["date_error"] = str(e)
        if txn_col and txn_col in df.columns:
            out["txn_type_counts"] = df[txn_col].astype(str).value_counts().head(10).to_dict()
        return out

    from backend.main import _warm_cache, _warm_cache_loaded_at  # type: ignore
    return {
        "session": {
            "mtr_df":      _df_info(sess.mtr_df,      "Date", "Transaction_Type"),
            "myntra_df":   _df_info(sess.myntra_df,   "Date", "TxnType"),
            "meesho_df":   {**_df_info(sess.meesho_df, "Date", "TxnType"),
                            "columns": list(sess.meesho_df.columns) if not sess.meesho_df.empty else [],
                            "sku_sample": sess.meesho_df["SKU"].dropna().unique()[:5].tolist()
                                          if not sess.meesho_df.empty and "SKU" in sess.meesho_df.columns else "NO SKU COLUMN"},
            "flipkart_df": _df_info(sess.flipkart_df, "Date", "TxnType"),
            "snapdeal_df": _df_info(sess.snapdeal_df, "Date", "TxnType"),
            "sales_df":    {**_df_info(sess.sales_df, "TxnDate", "Transaction Type"),
                            "meesho_skus": sess.sales_df[sess.sales_df["Source"].astype(str) == "Meesho"]["Sku"]
                                           .value_counts().head(5).to_dict()
                                           if not sess.sales_df.empty and "Source" in sess.sales_df.columns else {}},
            "sku_mapping_len": len(sess.sku_mapping),
        },
        "warm_cache": {
            "loaded_at": _warm_cache_loaded_at.isoformat() if _warm_cache_loaded_at else None,
            "keys":      list(_warm_cache.keys()),
        },
    }


# ── AI Dashboard Endpoints ────────────────────────────────────

@router.get("/platform-summary")
def platform_summary(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_platform_summary(
        sess.mtr_df, sess.myntra_df, sess.meesho_df,
        sess.flipkart_df, sess.snapdeal_df,
        start_date=start_date, end_date=end_date,
        sales_df=sess.sales_df,
    )


@router.get("/anomalies")
def anomalies_endpoint(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_anomalies(
        sess.mtr_df, sess.myntra_df, sess.meesho_df,
        sess.flipkart_df, sess.snapdeal_df,
        sess.inventory_df_variant, sess.sales_df,
        start_date=start_date,
        end_date=end_date,
    )


# ── Quarterly History (for PO Engine) ────────────────────────

@router.get("/quarterly-history")
def quarterly_history(request: Request, group_by_parent: bool = False, n_quarters: int = 8):
    sess = _sess(request)
    if sess.sales_df.empty and sess.mtr_df.empty:
        return {"loaded": False, "rows": []}

    from ..services.po_engine import calculate_quarterly_history
    # When sales_df exists it already includes Amazon & Myntra — never pass raw DFs (was doubling).
    # When sales_df is empty, pass platform frames so quarterly still works before first build-sales.
    _boot = sess.sales_df.empty or "Sku" not in sess.sales_df.columns
    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=sess.mtr_df if _boot and not sess.mtr_df.empty else None,
        myntra_df=sess.myntra_df if _boot and not sess.myntra_df.empty else None,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
    )
    if pivot.empty:
        return {"loaded": False, "rows": []}

    return {
        "loaded":   True,
        "columns":  list(pivot.columns),
        "rows":     pivot.fillna(0).to_dict("records"),
    }
