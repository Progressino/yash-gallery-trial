"""
Data query router — analytics endpoints.
GET /api/data/coverage, sales-summary, sales-by-source, top-skus,
mtr-analytics, myntra-analytics, meesho-analytics, flipkart-analytics, inventory
"""
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from ..models.schemas import CoverageResponse
from ..services.sales import get_sales_summary, get_sales_by_source, get_top_skus, get_platform_summary, get_anomalies
from ..services.daily_store import list_uploads, get_summary, delete_upload
from ..session import AppSession

router = APIRouter()


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
    if sess.daily_restored:
        return
    sess.daily_restored = True  # Only attempt once per session

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

    # Restore inventory from GitHub cache (inventory has no SQLite backing so it's always
    # lost on server restart — this is the only place it can be auto-recovered).
    # Sales/platform data is intentionally NOT auto-merged from cache here; it is loaded
    # exclusively from the SQLite daily_store above so that metrics are always consistent
    # regardless of which GitHub cache snapshot was last saved.
    # Users can explicitly pull historical data via the "Load Cache" button when needed.
    need_inventory = sess.inventory_df_variant.empty
    if need_inventory:
        try:
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


# ── Coverage ──────────────────────────────────────────────────

@router.get("/coverage", response_model=CoverageResponse)
def get_coverage(request: Request):
    sess = _sess(request)
    _restore_daily_if_needed(sess)   # auto-load persisted daily data on first access
    return CoverageResponse(
        sku_mapping=bool(sess.sku_mapping),
        mtr=not sess.mtr_df.empty,
        sales=not sess.sales_df.empty,
        myntra=not sess.myntra_df.empty,
        meesho=not sess.meesho_df.empty,
        flipkart=not sess.flipkart_df.empty,
        snapdeal=not sess.snapdeal_df.empty,
        inventory=not sess.inventory_df_variant.empty,
        daily_orders=len(sess.daily_sales_sources) > 0,
        existing_po=not sess.existing_po_df.empty,
        mtr_rows=len(sess.mtr_df),
        sales_rows=len(sess.sales_df),
        myntra_rows=len(sess.myntra_df),
        meesho_rows=len(sess.meesho_df),
        flipkart_rows=len(sess.flipkart_df),
        snapdeal_rows=len(sess.snapdeal_df),
    )


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


@router.get("/sales-by-source")
def sales_by_source(request: Request):
    sess = _sess(request)
    return get_sales_by_source(sess.sales_df)


@router.get("/top-skus")
def top_skus(
    request: Request,
    limit: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    sess = _sess(request)
    return get_top_skus(sess.sales_df, limit=limit, start_date=start_date, end_date=end_date)


# ── SKU List (for search autocomplete) ───────────────────────

@router.get("/sku-list")
def sku_list(request: Request, q: Optional[str] = None, limit: int = 100):
    """Return unique SKUs in sales_df, optionally filtered by search query."""
    import pandas as pd
    sess = _sess(request)
    df   = sess.sales_df
    if df.empty or "Sku" not in df.columns:
        return []
    skus = (
        df[df["Transaction Type"].astype(str) == "Shipment"]["Sku"]
        .astype(str)
        .unique()
        .tolist()
    )
    # Filter out noise rows
    skus = [s for s in skus if s and s.lower() not in ("nan", "none", "") and not s.lower().endswith("_total")]
    if q:
        q_lower = q.strip().lower()
        skus = [s for s in skus if q_lower in s.lower()]
    skus.sort()
    return skus[:limit]


# ── SKU Deepdive ──────────────────────────────────────────────

@router.get("/sku-deepdive")
def sku_deepdive(
    request: Request,
    sku: str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
):
    """
    Full sales breakdown for a single SKU.
    Returns: summary KPIs, monthly trend, platform breakdown, daily trend.
    Default window: last 90 days.
    """
    import pandas as pd
    sess = _sess(request)
    df   = sess.sales_df

    if df.empty:
        return {"loaded": False, "message": "No sales data loaded"}

    # Normalise dates
    df = df.copy()
    df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
    if df["TxnDate"].dt.tz is not None:
        df["TxnDate"] = df["TxnDate"].dt.tz_localize(None)
    df = df.dropna(subset=["TxnDate"])

    # Default to last 90 days if no range supplied
    if not start_date and not end_date:
        end_ts   = df["TxnDate"].max()
        start_ts = end_ts - pd.Timedelta(days=90)
    else:
        start_ts = pd.Timestamp(start_date) if start_date else df["TxnDate"].min()
        end_ts   = pd.Timestamp(end_date)   if end_date   else df["TxnDate"].max()

    # Filter to SKU + date window
    sku_df = df[
        (df["Sku"].astype(str) == sku) &
        (df["TxnDate"] >= start_ts) &
        (df["TxnDate"] <= end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    ].copy()

    if sku_df.empty:
        return {
            "loaded":   True,
            "sku":      sku,
            "summary":  {"shipped": 0, "returns": 0, "net_units": 0, "return_rate": 0.0, "ads": 0.0},
            "monthly":  [],
            "by_platform": [],
            "daily":    [],
            "first_sale": None,
            "last_sale":  None,
        }

    qty    = pd.to_numeric(sku_df["Quantity"],       errors="coerce").fillna(0)
    eff    = pd.to_numeric(sku_df["Units_Effective"], errors="coerce").fillna(0)
    txn    = sku_df["Transaction Type"].astype(str).str.strip()
    shipped  = int(qty[txn == "Shipment"].sum())
    returns  = int(qty[txn == "Refund"].sum())
    net_units = int(eff.sum())
    rr       = round(returns / shipped * 100, 1) if shipped > 0 else 0.0
    period_days = max((end_ts - start_ts).days, 1)
    ads      = round(shipped / period_days, 2)

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
    daily_grp = (
        sku_df[txn == "Shipment"]
        .assign(_qty=qty[txn == "Shipment"])
        .groupby(sku_df["TxnDate"].dt.strftime("%Y-%m-%d"))
        .agg(units=("_qty", "sum"))
        .reset_index()
        .rename(columns={"TxnDate": "date"})
        .sort_values("date")
    )
    daily = daily_grp.to_dict("records")

    return {
        "loaded":   True,
        "sku":      sku,
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
        "daily":       daily,
        "first_sale":  str(sku_df["TxnDate"].min().date()),
        "last_sale":   str(sku_df["TxnDate"].max().date()),
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
    df = sess.sales_df
    if df.empty:
        return []

    try:
        d = df.copy()
        d["TxnDate"] = pd.to_datetime(d["TxnDate"], errors="coerce")
        d = d.dropna(subset=["TxnDate"])

        if start_date:
            d = d[d["TxnDate"] >= pd.Timestamp(start_date)]
        if end_date:
            d = d[d["TxnDate"] <= pd.Timestamp(end_date)]
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

    return {
        "loaded":       True,
        "rows":         len(df),
        "date_range":   [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":      int(shipped),
        "returned":     int(returned),
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

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
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

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
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

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
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

    return {
        "loaded":      True,
        "rows":        len(df),
        "companies":   companies,
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
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
            "meesho_df":   _df_info(sess.meesho_df,   "Date", "TxnType"),
            "flipkart_df": _df_info(sess.flipkart_df, "Date", "TxnType"),
            "snapdeal_df": _df_info(sess.snapdeal_df, "Date", "TxnType"),
            "sales_df":    _df_info(sess.sales_df,    "TxnDate", "Transaction Type"),
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
    )


@router.get("/anomalies")
def anomalies_endpoint(request: Request):
    sess = _sess(request)
    return get_anomalies(
        sess.mtr_df, sess.myntra_df, sess.meesho_df,
        sess.flipkart_df, sess.snapdeal_df,
        sess.inventory_df_variant, sess.sales_df,
    )


# ── Quarterly History (for PO Engine) ────────────────────────

@router.get("/quarterly-history")
def quarterly_history(request: Request, group_by_parent: bool = False, n_quarters: int = 8):
    sess = _sess(request)
    if sess.sales_df.empty and sess.mtr_df.empty:
        return {"loaded": False, "rows": []}

    from ..services.po_engine import calculate_quarterly_history
    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=sess.mtr_df if not sess.mtr_df.empty else None,
        myntra_df=sess.myntra_df if not sess.myntra_df.empty else None,
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
