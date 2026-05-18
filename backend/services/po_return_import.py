"""Parse marketplace return reports (CSV / Excel) for PO return overlay."""
from __future__ import annotations

from io import BytesIO
from typing import Dict, Optional, Tuple

import pandas as pd

from .po_raise_import import pick_csv_column
from .po_engine import canonical_oms_key


_SKU_CANDS = (
    "oms_sku",
    "sku",
    "oms sku",
    "item_sku",
    "seller_sku",
    "asin",
    "style_id",
    "product_id",
    "returned_sku",
)
_QTY_CANDS = (
    "return_units",
    "return_qty",
    "returned_qty",
    "return quantity",
    "returns",
    "qty",
    "quantity",
    "units",
)


def parse_return_upload_bytes(
    raw: bytes,
    filename: str,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return DataFrame with OMS_SKU, Return_Units (summed per SKU)."""
    if not raw:
        return pd.DataFrame(), "Empty file."
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(raw))
        else:
            df = pd.read_csv(BytesIO(raw))
    except Exception as e:
        return pd.DataFrame(), f"Could not read file: {e}"
    if df is None or df.empty:
        return pd.DataFrame(), "No rows in file."

    sku_col = pick_csv_column(list(df.columns), _SKU_CANDS)
    qty_col = pick_csv_column(list(df.columns), _QTY_CANDS)
    if not sku_col:
        return pd.DataFrame(), "Could not find SKU column (expected SKU / OMS_SKU / seller_sku)."
    if not qty_col:
        return pd.DataFrame(), "Could not find return quantity column (expected Return_Qty / Qty / Units)."

    out = pd.DataFrame()
    out["OMS_SKU"] = df[sku_col].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    out["Return_Units"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0).astype(int)
    out = out[out["OMS_SKU"].str.len() > 0]
    out = out[out["Return_Units"] > 0]
    if out.empty:
        return pd.DataFrame(), "No positive return quantities found."

    if group_by_parent:
        from .helpers import get_parent_sku

        out["OMS_SKU"] = out["OMS_SKU"].map(
            lambda s: str(get_parent_sku(s) or s).strip().upper()
        )

    out = out.groupby("OMS_SKU", as_index=False)["Return_Units"].sum()
    return out, None


def apply_return_overlay_import(
    sess,
    overlay_df: pd.DataFrame,
    *,
    replace: bool = True,
) -> dict:
    if overlay_df is None or overlay_df.empty:
        return {"ok": False, "message": "No return rows to import."}
    if replace:
        sess.po_return_overlay_df = overlay_df.copy()
    else:
        base = getattr(sess, "po_return_overlay_df", pd.DataFrame())
        merged = pd.concat([base, overlay_df], ignore_index=True)
        merged = merged.groupby("OMS_SKU", as_index=False)["Return_Units"].sum()
        sess.po_return_overlay_df = merged
    n = int(len(sess.po_return_overlay_df))
    units = int(sess.po_return_overlay_df["Return_Units"].sum())
    sess._quarterly_cache.clear()
    return {
        "ok": True,
        "message": f"Return overlay: {n} SKU(s), {units:,} units. Run Calculate PO to apply.",
        "skus": n,
        "total_units": units,
    }
