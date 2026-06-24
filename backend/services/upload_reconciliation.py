"""Compare daily vs monthly uploads — highlight mismatches and dedup savings."""
from __future__ import annotations

import io
import re
import time
from typing import Any

import pandas as pd

_PLATFORMS = ("amazon", "myntra", "meesho", "flipkart", "snapdeal")

_PARQUET_COL_CANDIDATES = (
    "Date",
    "TxnDate",
    "Order Date",
    "order_date",
    "Report_Date",
    "Transaction_Type",
    "TxnType",
    "Transaction Type",
    "transaction_type",
    "txn_type",
    "Quantity",
    "quantity",
    "Qty",
    "qty",
    "Invoice_Amount",
    "Invoice Amount",
    "invoice_amount",
    "Amount",
    "amount",
    "DSR_Segment",
)

_REPORT_CACHE: tuple[float, str, dict[str, Any]] | None = None
_REPORT_CACHE_TTL_SEC = 60.0


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


def _months_in_span(date_from: str, date_to: str) -> set[str]:
    try:
        d0 = pd.Timestamp(str(date_from or "")[:10])
        d1 = pd.Timestamp(str(date_to or date_from or "")[:10])
        if pd.isna(d0) or pd.isna(d1):
            return set()
        if d1 < d0:
            d0, d1 = d1, d0
        months: set[str] = set()
        cur = d0.to_period("M")
        end = d1.to_period("M")
        while cur <= end:
            months.add(str(cur))
            cur += 1
        return months
    except Exception:
        return set()


def _month_in_window(month: str, start_month: str | None, end_month: str | None) -> bool:
    if not start_month and not end_month:
        return True
    if start_month and month < start_month:
        return False
    if end_month and month > end_month:
        return False
    return True


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


def _metrics_for_df(
    df: pd.DataFrame,
    platform: str,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, dict[str, float]]:
    """Per YYYY-MM bucket: units/amount by txn class."""
    if df is None or df.empty:
        return {}
    dates = _date_series(df)
    month = dates.dt.strftime("%Y-%m")
    valid = month.notna() & (month != "NaT")
    if not valid.any():
        return {}
    txn_col = _txn_col(df, platform)
    txn = df[txn_col].map(_bucket_txn) if txn_col else pd.Series("shipment", index=df.index)
    qty = _qty_series(df)
    amt = _amount_series(df)
    seg = _segment_series(df)
    work = pd.DataFrame({"_month": month, "_txn": txn, "_qty": qty, "_amt": amt, "_seg": seg})
    work = work.loc[valid]
    if start_month or end_month:
        work = work[work["_month"].map(lambda m: _month_in_window(m, start_month, end_month))]
    if work.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for (m, s, t), g in work.groupby(["_month", "_seg", "_txn"], sort=True, observed=True):
        key = f"{m}|{s}|{t}"
        out[key] = {
            "month": m,
            "segment": s,
            "txn": t,
            "units": float(g["_qty"].sum()),
            "amount": float(g["_amt"].sum()),
            "rows": float(len(g)),
        }
    return out


def _read_parquet_metrics(blob: bytes) -> pd.DataFrame:
    if not blob:
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(io.BytesIO(blob))
        want = [c for c in _PARQUET_COL_CANDIDATES if c in pf.schema_arrow.names]
        if not want:
            return pd.DataFrame()
        return pf.read(columns=want).to_pandas()
    except Exception:
        try:
            return pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
        except Exception:
            return pd.DataFrame()


def _load_upload_dfs_batch(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], pd.DataFrame]:
    """Load latest parquet per (platform, filename) in one SQLite round-trip."""
    if not pairs:
        return {}
    from .daily_store import _get_conn

    unique_pairs = list(dict.fromkeys(pairs))
    out: dict[tuple[str, str], pd.DataFrame] = {}
    conn = _get_conn()
    chunk_size = 40
    for i in range(0, len(unique_pairs), chunk_size):
        chunk = unique_pairs[i : i + chunk_size]
        placeholders = ",".join(["(?,?)"] * len(chunk))
        params = [x for pair in chunk for x in pair]
        rows = conn.execute(
            f"""
            SELECT d.platform, d.filename, d.data_parquet
            FROM daily_uploads d
            INNER JOIN (
                SELECT platform, filename, MAX(id) AS max_id
                FROM daily_uploads
                GROUP BY platform, filename
            ) latest ON d.id = latest.max_id
            WHERE (d.platform, d.filename) IN ({placeholders})
            """,
            params,
        ).fetchall()
        for plat, fn, blob in rows:
            out[(str(plat), str(fn))] = _read_parquet_metrics(blob)
    conn.close()
    return out


def _enrich_upload_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    plat = str(meta.get("platform") or "").strip().lower()
    if plat not in _PLATFORMS:
        return None
    fn = str(meta.get("filename") or "")
    date_from = str(meta.get("date_from") or "")
    date_to = str(meta.get("date_to") or "")
    kind = _classify_upload_kind(fn, date_from, date_to)
    return {
        **meta,
        "platform": plat,
        "filename": fn,
        "upload_kind": kind,
        "months": _months_in_span(date_from, date_to),
    }


def _uploads_needing_parquet(
    enriched: list[dict[str, Any]],
    start_month: str | None,
    end_month: str | None,
) -> list[dict[str, Any]]:
    """Only load parquets for monthly files and daily files that overlap a monthly month."""
    monthly_months_by_plat: dict[str, set[str]] = {}
    for item in enriched:
        if item["upload_kind"] != "monthly":
            continue
        months = {m for m in item["months"] if _month_in_window(m, start_month, end_month)}
        if months:
            monthly_months_by_plat.setdefault(item["platform"], set()).update(months)

    need: list[dict[str, Any]] = []
    for item in enriched:
        kind = item["upload_kind"]
        if kind == "monthly":
            months = {m for m in item["months"] if _month_in_window(m, start_month, end_month)}
            if months:
                need.append(item)
        elif kind == "daily":
            plat_months = monthly_months_by_plat.get(item["platform"], set())
            if not plat_months:
                continue
            overlap = item["months"] & plat_months
            if overlap:
                need.append(item)
    return need


def _dedup_stats(platform: str) -> dict[str, int | str]:
    """Fast summary-only dedup estimate — never loads full platform history."""
    from .daily_store import get_summary

    summary = get_summary() or {}
    plat_sum = summary.get(platform) if isinstance(summary.get(platform), dict) else {}
    est_rows = int(plat_sum.get("total_rows") or 0)
    return {
        "raw_rows": est_rows,
        "deduped_rows": est_rows,
        "collapsible_rows": 0,
        "note": "summary_estimate",
    }


def _report_fingerprint(uploads: list[dict[str, Any]], start_month: str | None, end_month: str | None) -> str:
    if not uploads:
        return "empty"
    newest = max((str(u.get("uploaded_at") or "") for u in uploads), default="")
    return f"{len(uploads)}|{newest}|{start_month or ''}|{end_month or ''}"


def invalidate_upload_reconciliation_cache() -> None:
    global _REPORT_CACHE
    _REPORT_CACHE = None


def build_upload_reconciliation_report(
    sess,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, Any]:
    """
    Compare daily-classified vs monthly-classified Tier-3 uploads per platform/segment/month.
    Highlights unit/amount gaps; reports rows removed by dedup (no double-count in app totals).

    Only loads parquet blobs for monthly uploads and daily uploads that overlap a monthly
    month on the same platform — metadata alone backs the file registry list.
    """
    global _REPORT_CACHE
    from .daily_store import list_uploads

    t0 = time.perf_counter()
    uploads = list_uploads()
    fingerprint = _report_fingerprint(uploads, start_month, end_month)
    now = time.time()
    if _REPORT_CACHE and _REPORT_CACHE[0] > now - _REPORT_CACHE_TTL_SEC and _REPORT_CACHE[1] == fingerprint:
        cached = dict(_REPORT_CACHE[2])
        cached["cached"] = True
        return cached

    enriched: list[dict[str, Any]] = []
    for meta in uploads:
        item = _enrich_upload_meta(meta)
        if item:
            enriched.append(item)

    files: list[dict[str, Any]] = [
        {
            "platform": e["platform"],
            "filename": e["filename"],
            "upload_kind": e["upload_kind"],
            "rows": int(e.get("rows") or 0),
            "date_from": e.get("date_from"),
            "date_to": e.get("date_to"),
            "uploaded_at": e.get("uploaded_at"),
        }
        for e in enriched
    ]

    to_load = _uploads_needing_parquet(enriched, start_month, end_month)
    pairs = [(e["platform"], e["filename"]) for e in to_load]
    dfs = _load_upload_dfs_batch(pairs)

    compare: dict[str, dict[str, dict[str, float]]] = {}
    for item in to_load:
        kind = item["upload_kind"]
        if kind not in ("daily", "monthly"):
            continue
        df = dfs.get((item["platform"], item["filename"]), pd.DataFrame())
        metrics = _metrics_for_df(df, item["platform"], start_month, end_month)
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

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    report = {
        "ok": True,
        "file_count": len(files),
        "files": sorted(files, key=lambda f: (f["platform"], f["filename"].lower())),
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "dedup_by_platform": dedup_by_platform,
        "total_collapsible_rows": total_collapsible,
        "return_overlay": return_overlay,
        "hints": hints,
        "parquet_files_loaded": len(to_load),
        "parquet_files_skipped": max(0, len(enriched) - len(to_load)),
        "elapsed_ms": elapsed_ms,
        "cached": False,
    }
    _REPORT_CACHE = (now, fingerprint, report)
    return report
