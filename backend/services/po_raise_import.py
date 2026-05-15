"""Parse PO recommendation CSV / Excel exports into raise-ledger rows."""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
from typing import Dict, Optional, Tuple

import pandas as pd

from .po_raise_ledger import append_raise_confirm_rows


def pick_csv_column(fieldnames: list, candidates: tuple[str, ...]) -> str | None:
    if not fieldnames:
        return None
    norm_map: dict[str, str] = {}
    for f in fieldnames:
        if f is None:
            continue
        key = str(f).replace("\ufeff", "").strip().lower().replace(" ", "_")
        norm_map[key] = f
    for cand in candidates:
        k = cand.strip().lower().replace(" ", "_")
        if k in norm_map:
            return norm_map[k]
    for f in fieldnames:
        if f is None:
            continue
        fl = str(f).replace("\ufeff", "").strip().lower()
        for cand in candidates:
            if cand.lower() in fl:
                return f
    return None


_SKU_CANDS = ("oms_sku", "sku", "oms sku", "item_sku", "item sku", "variant_sku")
_QTY_CANDS = (
    "po_qty",
    "final_po_qty",
    "gross_po_qty",
    "po qty",
    "net_po_qty",
    "recommended_po_qty",
    "raised_qty",
    "confirmed_qty",
)


def _accum_from_columns(df: pd.DataFrame, sku_col: str, qty_col: str) -> Dict[str, int]:
    accum: dict[str, int] = {}
    for _, row in df.iterrows():
        sku = str(row.get(sku_col) or "").strip()
        if not sku or sku.lower() in ("nan", "none"):
            continue
        try:
            q = int(float(pd.to_numeric(row.get(qty_col), errors="coerce") or 0))
        except (TypeError, ValueError):
            q = 0
        if q <= 0:
            continue
        accum[sku] = accum.get(sku, 0) + q
    return accum


def parse_ledger_dataframe(df: pd.DataFrame) -> Tuple[Dict[str, int], Optional[str]]:
    """Return SKU→qty map from a PO recommendation table (CSV or Excel)."""
    if df is None or df.empty:
        return {}, "No rows in file."
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    fieldnames = list(work.columns)
    sku_col = pick_csv_column(fieldnames, _SKU_CANDS)
    qty_col = pick_csv_column(fieldnames, _QTY_CANDS)
    if not sku_col or not qty_col:
        return {}, f"Need OMS_SKU (or SKU) and PO_Qty columns. Found: {fieldnames[:40]}"
    accum = _accum_from_columns(work, sku_col, qty_col)
    if not accum:
        return {}, "No positive PO_Qty rows found (check PO_Qty / Final_PO_Qty column)."
    return accum, None


def parse_ledger_csv_text(text: str) -> Tuple[Dict[str, int], Optional[str]]:
    """Return SKU→qty map or error message."""
    reader = csv.DictReader(StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    sku_col = pick_csv_column(fieldnames, _SKU_CANDS)
    qty_col = pick_csv_column(fieldnames, _QTY_CANDS)
    if not sku_col or not qty_col:
        return {}, f"Need OMS_SKU (or SKU) and PO_Qty columns. Found: {fieldnames[:40]}"

    accum: dict[str, int] = {}
    for row in reader:
        sku = str(row.get(sku_col) or "").strip()
        if not sku:
            continue
        qraw = row.get(qty_col)
        try:
            q = int(float(str(qraw).replace(",", "").strip() or 0))
        except (TypeError, ValueError):
            q = 0
        if q <= 0:
            continue
        accum[sku] = accum.get(sku, 0) + q

    if not accum:
        return {}, "No positive PO_Qty rows found in CSV."
    return accum, None


def parse_ledger_upload_bytes(raw: bytes, filename: str = "") -> Tuple[Dict[str, int], Optional[str]]:
    """Parse CSV or Excel (.xlsx / .xls) PO recommendation export."""
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            df = pd.read_excel(BytesIO(raw))
        except Exception as e:
            return {}, f"Excel parse error: {e}"
        return parse_ledger_dataframe(df)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return parse_ledger_csv_text(text)


def ledger_rows_for_date(ledger: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    if ledger is None or getattr(ledger, "empty", True):
        return pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])
    d = pd.Timestamp(day).normalize()
    ld = pd.to_datetime(ledger["Raised_Date"], errors="coerce").dt.normalize()
    out = ledger[ld != d].copy()
    if out.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])
    return out.reset_index(drop=True)


def ledger_has_positive_qty_on_day(ledger: pd.DataFrame, day: pd.Timestamp) -> bool:
    if ledger is None or ledger.empty:
        return False
    d = pd.Timestamp(day).normalize()
    ld = pd.to_datetime(ledger["Raised_Date"], errors="coerce").dt.normalize()
    sub = ledger[ld == d]
    if sub.empty:
        return False
    return int(pd.to_numeric(sub["Raised_Qty"], errors="coerce").fillna(0).sum()) > 0


def apply_ledger_import(
    sess,
    accum: Dict[str, int],
    raised_date: pd.Timestamp,
    *,
    group_by_parent: bool = False,
    replace_day: bool = True,
) -> dict:
    base = getattr(sess, "po_raise_ledger_df", pd.DataFrame())
    if replace_day:
        base = ledger_rows_for_date(base if base is not None else pd.DataFrame(), raised_date)
    tuples = list(accum.items())
    sess.po_raise_ledger_df = append_raise_confirm_rows(
        base,
        tuples,
        raised_date,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=group_by_parent,
    )
    sess._quarterly_cache.clear()
    n = int(len(sess.po_raise_ledger_df))
    tot_units = int(sum(accum.values()))
    return {
        "ok": True,
        "ledger_rows": n,
        "imported_skus": len(accum),
        "total_units": tot_units,
        "raised_date": str(raised_date.date()),
        "message": (
            f"Recorded {len(accum):,} SKU(s) / {tot_units:,} units for {raised_date.date()} "
            f"— ledger now {n:,} SKU-day row(s)."
        ),
    }
