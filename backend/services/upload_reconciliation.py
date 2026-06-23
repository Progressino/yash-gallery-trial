"""Compare daily vs monthly uploads — highlight mismatches and dedup savings."""
from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

_PLATFORMS = ("amazon", "myntra", "meesho", "flipkart", "snapdeal")


def _classify_upload_kind(filename: str, date_from: str, date_to: str) -> str:
    """Best-effort daily vs monthly label from filename and declared row span."""
    fn = (filename or "").lower().replace("\\", "/")
    if re.search(r"orders_\d{4}-\d{2}-\d{2}", fn):
        return "daily"
    if any(
        k in fn
        for k in (
            "mtr",
            "jan to",
            "jan -",
            "to jun",
            "to may",
            "tcs_sales",
            "ppmp",
            "seller_orders",
            "seller orders",
            "monthly",
            "bulk",
        )
    ):
        return "monthly"
    try:
        d0 = pd.Timestamp(str(date_from or "")[:10])
        d1 = pd.Timestamp(str(date_to or date_from or "")[:10])
        span = int((d1 - d0).days) + 1
        if span <= 2:
            return "daily"
        if span >= 14:
            return "monthly"
    except Exception:
        pass
    return "other"


def _txn_col(df: pd.DataFrame, platform: str) -> str | None:
    for c in (
        "Transaction_Type",
        "TxnType",
        "Transaction Type",
        "transaction_type",
        "txn_type",
    ):
        if c in df.columns:
            return c
    return None


def _date_series(df: pd.DataFrame) -> pd.Series:
    for c in ("Date", "TxnDate", "Order Date", "order_date", "Report_Date"):
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce").dt.normalize()
    return pd.Series(dtype="datetime64[ns]")


def _segment_series(df: pd.DataFrame) -> pd.Series:
    if "DSR_Segment" in df.columns:
        seg = df["DSR_Segment"].fillna("").astype(str).str.strip()
        seg = seg.where(~seg.str.casefold().isin({"", "nan", "none", "all"}), "Unknown")
        return seg
    return pd.Series(["Unknown"] * len(df), index=df.index)


def _amount_series(df: pd.DataFrame) -> pd.Series:
    for c in ("Invoice_Amount", "Invoice Amount", "invoice_amount", "Amount", "amount"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _qty_series(df: pd.DataFrame) -> pd.Series:
    for c in ("Quantity", "quantity", "Qty", "qty"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return pd.Series(0, index=df.index, dtype="int64")


def _bucket_txn(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if "ship" in s:
        return "shipment"
    if "refund" in s or "return" in s:
        return "refund"
    if "cancel" in s:
        return "cancel"
    return "other"


def _metrics_for_df(df: pd.DataFrame, platform: str) -> dict[str, dict[str, float]]:
    """Per YYYY-MM bucket: units/amount by txn class."""
    if df is None or df.empty:
        return {}
    work = df.copy()
    dates = _date_series(work)
    work["_month"] = dates.dt.strftime("%Y-%m")
    work = work[work["_month"].notna() & (work["_month"] != "NaT")]
    if work.empty:
        return {}
    txn_col = _txn_col(work, platform)
    work["_txn"] = work[txn_col].map(_bucket_txn) if txn_col else "shipment"
    work["_qty"] = _qty_series(work)
    work["_amt"] = _amount_series(work)
    work["_seg"] = _segment_series(work)
    out: dict[str, dict[str, float]] = {}
    for (month, seg, txn), g in work.groupby(["_month", "_seg", "_txn"], sort=True, observed=True):
        key = f"{month}|{seg}|{txn}"
        out[key] = {
            "month": month,
            "segment": seg,
            "txn": txn,
            "units": float(g["_qty"].sum()),
            "amount": float(g["_amt"].sum()),
            "rows": float(len(g)),
        }
    return out


def _load_upload_df(platform: str, filename: str) -> pd.DataFrame:
    from .daily_store import _get_conn

    conn = _get_conn()
    row = conn.execute(
        "SELECT data_parquet FROM daily_uploads WHERE platform=? AND filename=? "
        "ORDER BY id DESC LIMIT 1",
        (platform, filename),
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(row[0]), engine="pyarrow")
    except Exception:
        return pd.DataFrame()


def _dedup_stats(platform: str) -> dict[str, int]:
    from .daily_store import _dedup_platform_df, get_summary

    summary = get_summary() or {}
    plat_sum = summary.get(platform) if isinstance(summary.get(platform), dict) else {}
    est_rows = int(plat_sum.get("total_rows") or 0)
    # Full concat+dedup on multi-million-row Amazon history blocks the API for minutes.
    if est_rows > 150_000:
        return {
            "raw_rows": est_rows,
            "deduped_rows": est_rows,
            "collapsible_rows": 0,
            "note": "skipped_full_scan",
        }

    from .daily_store import load_platform_data

    raw = load_platform_data(platform, months=None, dedup=False)
    if raw.empty:
        return {"raw_rows": 0, "deduped_rows": 0, "collapsible_rows": 0}
    ded = _dedup_platform_df(raw.copy(), platform, is_merge=True)
    raw_n = int(len(raw))
    ded_n = int(len(ded))
    return {
        "raw_rows": raw_n,
        "deduped_rows": ded_n,
        "collapsible_rows": max(0, raw_n - ded_n),
    }


def build_upload_reconciliation_report(sess) -> dict[str, Any]:
    """
    Compare daily-classified vs monthly-classified Tier-3 uploads per platform/segment/month.
    Highlights unit/amount gaps; reports rows removed by dedup (no double-count in app totals).
    """
    from .daily_store import list_uploads

    uploads = list_uploads()
    files: list[dict[str, Any]] = []
    # month|segment|txn -> {daily: metrics, monthly: metrics}
    compare: dict[str, dict[str, dict[str, float]]] = {}

    for meta in uploads:
        plat = str(meta.get("platform") or "").strip().lower()
        if plat not in _PLATFORMS:
            continue
        fn = str(meta.get("filename") or "")
        df = _load_upload_df(plat, fn)
        kind = _classify_upload_kind(fn, str(meta.get("date_from") or ""), str(meta.get("date_to") or ""))
        metrics = _metrics_for_df(df, plat)
        files.append(
            {
                "platform": plat,
                "filename": fn,
                "upload_kind": kind,
                "rows": int(meta.get("rows") or 0),
                "date_from": meta.get("date_from"),
                "date_to": meta.get("date_to"),
                "uploaded_at": meta.get("uploaded_at"),
            }
        )
        if kind not in ("daily", "monthly"):
            continue
        for key, m in metrics.items():
            compare.setdefault(key, {}).setdefault(kind, {"units": 0.0, "amount": 0.0, "rows": 0.0})
            for fld in ("units", "amount", "rows"):
                compare[key][kind][fld] = compare[key][kind].get(fld, 0.0) + float(m.get(fld, 0.0))

    mismatches: list[dict[str, Any]] = []
    for key, sides in compare.items():
        daily = sides.get("daily") or {}
        monthly = sides.get("monthly") or {}
        if not daily or not monthly:
            continue
        du = float(daily.get("units") or 0)
        mu = float(monthly.get("units") or 0)
        da = float(daily.get("amount") or 0)
        ma = float(monthly.get("amount") or 0)
        if du <= 0 and mu <= 0:
            continue
        parts = key.split("|", 2)
        month = parts[0] if parts else ""
        segment = parts[1] if len(parts) > 1 else ""
        txn = parts[2] if len(parts) > 2 else ""
        unit_diff = int(round(mu - du))
        amt_diff = round(ma - da, 2)
        if unit_diff == 0 and abs(amt_diff) < 1.0:
            continue
        mismatches.append(
            {
                "month": month,
                "segment": segment,
                "txn": txn,
                "daily_units": int(round(du)),
                "monthly_units": int(round(mu)),
                "unit_diff": unit_diff,
                "daily_amount": round(da, 2),
                "monthly_amount": round(ma, 2),
                "amount_diff": amt_diff,
            }
        )

    mismatches.sort(key=lambda r: (r["month"], r["segment"], r["txn"]))

    dedup_by_platform = {p: _dedup_stats(p) for p in _PLATFORMS}
    total_collapsible = sum(int(v.get("collapsible_rows") or 0) for v in dedup_by_platform.values())

    return_overlay = {}
    try:
        ov = getattr(sess, "po_return_overlay_df", None)
        if ov is not None and not getattr(ov, "empty", True):
            return_overlay = {
                "skus": int(ov["OMS_SKU"].nunique()),
                "units": int(pd.to_numeric(ov.get("Return_Units"), errors="coerce").fillna(0).sum()),
            }
    except Exception:
        pass

    hints = [
        "App totals use deduplicated merges — overlapping daily + monthly lines collapse to one row per order/SKU/day.",
        "Mismatches below mean daily exports and monthly archives disagree for the same month — review before trusting PO.",
        "Meesho tcs_sales_return and Amazon FBA return CSVs belong under Returns (for PO), not sales uploads.",
        "After fixing files, click Build Sales and run Calculate PO.",
    ]

    return {
        "ok": True,
        "file_count": len(files),
        "files": sorted(files, key=lambda f: (f["platform"], f["filename"].lower())),
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "dedup_by_platform": dedup_by_platform,
        "total_collapsible_rows": total_collapsible,
        "return_overlay": return_overlay,
        "hints": hints,
    }
