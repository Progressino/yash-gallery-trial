"""
Finance service — P&L, GST summary, platform revenue reconciliation, COGS sheet parser.
"""
import io
from typing import Optional

import pandas as pd

from ..db.finance_db import list_expenses


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_by_date(df: pd.DataFrame, date_col: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Filter a DataFrame by a datetime column using ISO date strings."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if start_date:
        df = df[df[date_col] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return df


def _platform_rev(df: pd.DataFrame, txn_col: str, ship_val: str = "Shipment", refund_val: str = "Refund") -> tuple[float, float]:
    """Return (gross_revenue, returns_value) from a platform DataFrame."""
    if df.empty or "Invoice_Amount" not in df.columns or txn_col not in df.columns:
        return 0.0, 0.0
    gross = float(df[df[txn_col] == ship_val]["Invoice_Amount"].sum())
    ret   = float(df[df[txn_col] == refund_val]["Invoice_Amount"].sum())
    return gross, ret


# ── Public API ────────────────────────────────────────────────────────────────

def get_platform_revenue(
    mtr_df:     pd.DataFrame,
    myntra_df:  pd.DataFrame,
    meesho_df:  pd.DataFrame,
    flipkart_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> list[dict]:
    """Per-platform gross revenue, returns, and net revenue."""
    platforms = [
        ("Amazon",   mtr_df,      "Transaction_Type"),
        ("Myntra",   myntra_df,   "TxnType"),
        ("Meesho",   meesho_df,   "TxnType"),
        ("Flipkart", flipkart_df, "TxnType"),
    ]
    result = []
    for name, df, txn_col in platforms:
        loaded = not df.empty
        if loaded:
            df_f = _filter_by_date(df, "Date", start_date, end_date)
            gross, ret = _platform_rev(df_f, txn_col)
        else:
            gross, ret = 0.0, 0.0
        net = gross - ret
        rr  = round(ret / gross * 100, 1) if gross > 0 else 0.0
        result.append({
            "platform":        name,
            "loaded":          loaded,
            "gross_revenue":   round(gross, 2),
            "returns_value":   round(ret, 2),
            "net_revenue":     round(net, 2),
            "return_rate_pct": rr,
        })
    return result


def get_gst_summary(
    mtr_df:     pd.DataFrame,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> dict:
    """GST breakdown from Amazon MTR (only platform with CGST/SGST/IGST)."""
    if mtr_df.empty:
        return {"loaded": False, "months": [], "totals": {"cgst": 0, "sgst": 0, "igst": 0, "total_tax": 0}}

    df = _filter_by_date(mtr_df, "Date", start_date, end_date)
    # Only Shipment rows for GST liability
    df = df[df["Transaction_Type"] == "Shipment"]

    tax_cols = ["CGST", "SGST", "IGST", "Total_Tax"]
    missing  = [c for c in tax_cols if c not in df.columns]
    if missing:
        return {"loaded": False, "months": [], "totals": {"cgst": 0, "sgst": 0, "igst": 0, "total_tax": 0}}

    # Group by Month (already "YYYY-MM" string in mtr_df)
    if "Month" not in df.columns:
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

    grp = df.groupby("Month")[["CGST", "SGST", "IGST", "Total_Tax"]].sum().reset_index()
    grp = grp.sort_values("Month")

    months = [
        {
            "month":     row["Month"],
            "cgst":      round(float(row["CGST"]), 2),
            "sgst":      round(float(row["SGST"]), 2),
            "igst":      round(float(row["IGST"]), 2),
            "total_tax": round(float(row["Total_Tax"]), 2),
        }
        for _, row in grp.iterrows()
    ]
    totals = {
        "cgst":      round(float(grp["CGST"].sum()), 2),
        "sgst":      round(float(grp["SGST"].sum()), 2),
        "igst":      round(float(grp["IGST"].sum()), 2),
        "total_tax": round(float(grp["Total_Tax"].sum()), 2),
    }
    return {"loaded": True, "months": months, "totals": totals}


def get_pl_statement(
    mtr_df:     pd.DataFrame,
    myntra_df:  pd.DataFrame,
    meesho_df:  pd.DataFrame,
    flipkart_df: pd.DataFrame,
    sales_df:   pd.DataFrame,
    cogs_df:    pd.DataFrame,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> dict:
    """Full P&L statement."""
    platform_data = get_platform_revenue(mtr_df, myntra_df, meesho_df, flipkart_df, start_date, end_date)

    gross_revenue  = sum(p["gross_revenue"] for p in platform_data)
    returns_value  = sum(p["returns_value"]  for p in platform_data)
    net_revenue    = gross_revenue - returns_value

    # ── COGS ─────────────────────────────────────────────────────
    cogs = 0.0
    if not cogs_df.empty and not sales_df.empty and "OMS_SKU" in cogs_df.columns and "Cost_Price" in cogs_df.columns:
        sf = sales_df[sales_df["Transaction Type"] == "Shipment"].copy()
        if start_date or end_date:
            sf["TxnDate"] = pd.to_datetime(sf["TxnDate"], errors="coerce")
            if start_date:
                sf = sf[sf["TxnDate"] >= pd.Timestamp(start_date)]
            if end_date:
                sf = sf[sf["TxnDate"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
        qty_by_sku = sf.groupby("Sku")["Quantity"].sum().reset_index()
        qty_by_sku.columns = ["OMS_SKU", "qty"]
        merged = qty_by_sku.merge(cogs_df[["OMS_SKU", "Cost_Price"]], on="OMS_SKU", how="inner")
        cogs = float((merged["qty"] * merged["Cost_Price"]).sum())

    gross_profit     = net_revenue - cogs
    gross_margin_pct = round(gross_profit / net_revenue * 100, 1) if net_revenue > 0 else 0.0

    # ── Expenses ──────────────────────────────────────────────────
    expense_rows     = list_expenses(start_date, end_date)
    total_expenses   = sum(r["amount"] + r["gst_amount"] for r in expense_rows)
    net_profit       = gross_profit - total_expenses

    return {
        "gross_revenue":    round(gross_revenue, 2),
        "returns_value":    round(returns_value, 2),
        "net_revenue":      round(net_revenue, 2),
        "cogs":             round(cogs, 2),
        "cogs_available":   not cogs_df.empty,
        "gross_profit":     round(gross_profit, 2),
        "gross_margin_pct": gross_margin_pct,
        "total_expenses":   round(total_expenses, 2),
        "net_profit":       round(net_profit, 2),
        "platform_breakdown": platform_data,
    }


def parse_cogs_sheet(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse a COGS sheet (Excel or CSV).
    Expected columns: SKU/OMS SKU + one of: Cost Price / Cost Per Unit / COGS / Cost / Unit Cost
    Returns DataFrame[OMS_SKU, Cost_Price].
    """
    fn_lower = filename.lower()
    try:
        if fn_lower.endswith(".csv"):
            try:
                raw = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                raw = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding="ISO-8859-1", on_bad_lines="skip")
        else:
            raw = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
    except Exception as e:
        raise ValueError(f"Cannot read COGS file: {e}")

    if raw.empty:
        raise ValueError("COGS file is empty.")

    raw.columns = raw.columns.astype(str).str.strip()
    cols_lower = {c.lower(): c for c in raw.columns}

    # Find SKU column
    sku_col = next(
        (cols_lower[k] for k in ["oms sku", "oms_sku", "sku", "style code", "item code", "product code"] if k in cols_lower),
        None,
    )
    if sku_col is None:
        raise ValueError(f"No SKU column found in COGS sheet. Columns: {list(raw.columns)[:15]}")

    # Find cost column
    cost_col = next(
        (cols_lower[k] for k in ["cost price", "cost per unit", "cost_price", "cogs", "unit cost", "cost"] if k in cols_lower),
        None,
    )
    if cost_col is None:
        raise ValueError(f"No cost column found. Expected: Cost Price, Cost Per Unit, COGS, etc. Columns: {list(raw.columns)[:15]}")

    df = raw[[sku_col, cost_col]].copy()
    df[sku_col]  = df[sku_col].astype(str).str.strip()
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0)
    df = df[df[sku_col].str.len() > 0]
    df = df.rename(columns={sku_col: "OMS_SKU", cost_col: "Cost_Price"})
    df = df.groupby("OMS_SKU", as_index=False)["Cost_Price"].mean()
    return df
