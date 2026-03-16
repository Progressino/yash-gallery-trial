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

    cols = [c for c in df.columns if c != "OMS_SKU"]
    return {
        "loaded":   True,
        "rows":     df.fillna(0).to_dict("records"),
        "columns":  ["OMS_SKU"] + cols,
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


# ── AI Dashboard Endpoints ────────────────────────────────────

@router.get("/platform-summary")
def platform_summary(request: Request):
    sess = _sess(request)
    return get_platform_summary(
        sess.mtr_df, sess.myntra_df, sess.meesho_df,
        sess.flipkart_df, sess.snapdeal_df,
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
