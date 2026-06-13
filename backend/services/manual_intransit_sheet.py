"""Parse admin 'Intrasit / Not In Inventory' workbook and apply to session inventory."""
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .po_engine import canonical_oms_key
from .po_raise_import import pick_csv_column

_INTRASIT_SHEET_NAMES = (
    "intrasit inventory",
    "in transit inventory",
    "intransit inventory",
    "intrasit",
    "in transit",
    "intransit",
)
_NOT_IN_SHEET_NAMES = (
    "not in inventory",
    "not-in inventory",
    "not in inv",
)

_SKU_CANDS = (
    "sku",
    "oms_sku",
    "oms sku",
    "seller_sku",
    "listing sku",
    "listing_sku",
    "item_sku",
)
_QTY_CANDS = (
    "qty",
    "quantity",
    "units",
    "intransit",
    "in transit",
    "in_transit",
    "not in inventory",
)


def _normalize_sheet_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name or "").strip().lower()).strip()


def _classify_sheet(name: str) -> Optional[str]:
    key = _normalize_sheet_key(name)
    if key in _INTRASIT_SHEET_NAMES or "intrasit" in key or "in transit" in key:
        return "intransit"
    if key in _NOT_IN_SHEET_NAMES or "not in inventory" in key:
        return "not_in"
    return None


def _parse_qty_series(df: pd.DataFrame, qty_col: str) -> pd.Series:
    raw = df[qty_col].astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(raw, errors="coerce").fillna(0).astype(int)


def _parse_sheet_table(
    df: pd.DataFrame,
    *,
    sheet_name: str,
    sheet_kind: str,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Return OMS_SKU + qty column and per-row skip audit entries."""
    skips: List[dict] = []
    if df is None or df.empty:
        skips.append(
            {
                "sheet": sheet_name,
                "kind": sheet_kind,
                "reason": "Sheet is empty",
                "rows_affected": 0,
            }
        )
        return pd.DataFrame(), skips

    cols = [str(c).strip() for c in df.columns]
    sku_col = pick_csv_column(cols, _SKU_CANDS)
    qty_col = pick_csv_column(cols, _QTY_CANDS)

    unrecognized = [c for c in cols if c not in (sku_col, qty_col)]
    if unrecognized:
        skips.append(
            {
                "sheet": sheet_name,
                "kind": "column",
                "reason": f"Unrecognized columns ignored: {', '.join(unrecognized[:12])}"
                + ("…" if len(unrecognized) > 12 else ""),
                "rows_affected": len(df),
            }
        )

    if not sku_col:
        skips.append(
            {
                "sheet": sheet_name,
                "kind": sheet_kind,
                "reason": "Missing SKU column (expected Sku / OMS_SKU / seller_sku)",
                "rows_affected": len(df),
            }
        )
        return pd.DataFrame(), skips
    if not qty_col:
        skips.append(
            {
                "sheet": sheet_name,
                "kind": sheet_kind,
                "reason": "Missing quantity column (expected Qty / Quantity / Units)",
                "rows_affected": len(df),
            }
        )
        return pd.DataFrame(), skips

    out = pd.DataFrame()
    out["OMS_SKU"] = df[sku_col].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    out["_Qty"] = _parse_qty_series(df, qty_col)

    empty_sku = out["OMS_SKU"].str.len().eq(0) | out["OMS_SKU"].str.lower().isin(["nan", "none"])
    if empty_sku.any():
        skips.append(
            {
                "sheet": sheet_name,
                "kind": sheet_kind,
                "reason": "Rows with blank/invalid SKU skipped",
                "rows_affected": int(empty_sku.sum()),
            }
        )
    zero_qty = out["_Qty"].le(0)
    if zero_qty.any():
        skips.append(
            {
                "sheet": sheet_name,
                "kind": sheet_kind,
                "reason": "Rows with zero or negative quantity skipped",
                "rows_affected": int(zero_qty.sum()),
            }
        )

    out = out[~empty_sku & out["_Qty"].gt(0)]
    if out.empty:
        return pd.DataFrame(), skips

    dup_mask = out["OMS_SKU"].duplicated(keep=False)
    if dup_mask.any():
        dup_n = int(dup_mask.sum())
        dup_skus = int(out.loc[dup_mask, "OMS_SKU"].nunique())
        skips.append(
            {
                "sheet": sheet_name,
                "kind": sheet_kind,
                "reason": f"Duplicate SKU rows merged (sum qty) — {dup_skus} SKU(s), {dup_n} row(s)",
                "rows_affected": dup_n,
            }
        )

    qty_col_name = "Manual_InTransit" if sheet_kind == "intransit" else "Not_In_Inventory_Qty"
    grouped = (
        out.groupby("OMS_SKU", as_index=False)["_Qty"]
        .sum()
        .rename(columns={"_Qty": qty_col_name})
    )
    return grouped, skips


def parse_manual_intransit_workbook(
    raw: bytes,
    filename: str = "",
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parse admin workbook with Intrasit + Not In Inventory sheets.

    Returns (intransit_df, not_in_df, parse_report).
    Re-uploading the same file replaces prior data — no duplicate accumulation.
    """
    report: dict[str, Any] = {
        "filename": os.path.basename(str(filename or "").strip()) or "upload.xlsx",
        "sheets_found": [],
        "sheets_skipped": [],
        "skip_details": [],
        "warnings": [],
        "intransit_units": 0,
        "not_in_inventory_units": 0,
        "intransit_skus": 0,
        "not_in_inventory_skus": 0,
    }
    if not raw:
        report["error"] = "Empty file."
        return pd.DataFrame(), pd.DataFrame(), report

    try:
        xl = pd.ExcelFile(BytesIO(raw))
    except Exception as e:
        report["error"] = f"Could not read Excel: {e}"
        return pd.DataFrame(), pd.DataFrame(), report

    intransit_parts: List[pd.DataFrame] = []
    not_in_parts: List[pd.DataFrame] = []

    for sheet in xl.sheet_names:
        kind = _classify_sheet(sheet)
        if not kind:
            report["sheets_skipped"].append(
                {"sheet": sheet, "reason": "Unrecognized sheet name (expected Intrasit / Not In Inventory)"}
            )
            continue
        try:
            df = pd.read_excel(BytesIO(raw), sheet_name=sheet)
        except Exception as e:
            report["sheets_skipped"].append({"sheet": sheet, "reason": f"Read error: {e}"})
            continue
        part, skips = _parse_sheet_table(
            df, sheet_name=sheet, sheet_kind=kind, sku_mapping=sku_mapping
        )
        report["skip_details"].extend(skips)
        if part.empty:
            report["sheets_skipped"].append(
                {"sheet": sheet, "reason": "No usable SKU rows after validation"}
            )
            continue
        report["sheets_found"].append({"sheet": sheet, "kind": kind, "rows": int(len(part))})
        if kind == "intransit":
            intransit_parts.append(part)
        else:
            not_in_parts.append(part)

    intransit_df = (
        pd.concat(intransit_parts, ignore_index=True)
        if intransit_parts
        else pd.DataFrame(columns=["OMS_SKU", "Manual_InTransit"])
    )
    not_in_df = (
        pd.concat(not_in_parts, ignore_index=True)
        if not_in_parts
        else pd.DataFrame(columns=["OMS_SKU", "Not_In_Inventory_Qty"])
    )

    if not intransit_df.empty and "Manual_InTransit" in intransit_df.columns:
        intransit_df = (
            intransit_df.groupby("OMS_SKU", as_index=False)["Manual_InTransit"]
            .sum()
        )
    if not not_in_df.empty and "Not_In_Inventory_Qty" in not_in_df.columns:
        not_in_df = (
            not_in_df.groupby("OMS_SKU", as_index=False)["Not_In_Inventory_Qty"]
            .sum()
        )

    report["intransit_units"] = int(intransit_df["Manual_InTransit"].sum()) if not intransit_df.empty else 0
    report["not_in_inventory_units"] = (
        int(not_in_df["Not_In_Inventory_Qty"].sum()) if not not_in_df.empty else 0
    )
    report["intransit_skus"] = int(len(intransit_df))
    report["not_in_inventory_skus"] = int(len(not_in_df))

    if not intransit_parts and not not_in_parts:
        report["error"] = (
            "No Intrasit or Not In Inventory sheets found. "
            "Expected sheets named like 'Intrasit Inventory' and 'Not In Inventory'."
        )
    elif not intransit_parts:
        report["warnings"].append("Not In Inventory loaded but no Intrasit sheet was parsed.")
    elif not not_in_parts:
        report["warnings"].append("Intrasit loaded but no Not In Inventory sheet was parsed.")

    return intransit_df, not_in_df, report


def _combine_manual_overlay(
    intransit_df: pd.DataFrame,
    not_in_df: pd.DataFrame,
) -> pd.DataFrame:
    if (intransit_df is None or intransit_df.empty) and (not_in_df is None or not_in_df.empty):
        return pd.DataFrame(columns=["OMS_SKU", "Manual_InTransit", "Not_In_Inventory_Qty"])
    overlay = pd.DataFrame({"OMS_SKU": []})
    if intransit_df is not None and not intransit_df.empty:
        overlay = intransit_df[["OMS_SKU", "Manual_InTransit"]].copy()
    if not_in_df is not None and not not_in_df.empty:
        ni = not_in_df[["OMS_SKU", "Not_In_Inventory_Qty"]].copy()
        overlay = ni if overlay.empty else pd.merge(overlay, ni, on="OMS_SKU", how="outer")
    for col in ("Manual_InTransit", "Not_In_Inventory_Qty"):
        if col not in overlay.columns:
            overlay[col] = 0
        overlay[col] = pd.to_numeric(overlay[col], errors="coerce").fillna(0).astype(int)
    return overlay


def apply_manual_intransit_import(
    sess,
    intransit_df: pd.DataFrame,
    not_in_df: pd.DataFrame,
    report: dict,
    *,
    filename: str,
) -> dict:
    """Replace prior manual in-transit upload (same or new filename — no duplicate accumulation)."""
    from datetime import datetime, timezone

    overlay = _combine_manual_overlay(intransit_df, not_in_df)
    if overlay.empty:
        return {"ok": False, "message": report.get("error") or "No usable rows found."}

    file_key = os.path.basename(str(filename or "").strip()) or "manual_intransit.xlsx"
    prev_name = str(getattr(sess, "manual_intransit_filename", "") or "").strip()
    replaced = bool(prev_name)

    sess.manual_intransit_overlay_df = overlay.copy()
    sess.manual_intransit_parse_report = dict(report)
    sess.manual_intransit_uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sess.manual_intransit_filename = file_key
    apply_manual_intransit_overlay_to_inventory(sess)

    units_it = int(report.get("intransit_units") or 0)
    units_ni = int(report.get("not_in_inventory_units") or 0)
    action = "Updated" if replaced else "Loaded"
    return {
        "ok": True,
        "message": (
            f"{action} {file_key}: {len(overlay)} SKU(s) — "
            f"{units_it:,} in-transit units, {units_ni:,} not-in-inventory units."
        ),
        "skus": int(len(overlay)),
        "intransit_units": units_it,
        "not_in_inventory_units": units_ni,
        "parse_report": report,
        "replaced_previous": replaced,
    }


def apply_manual_intransit_overlay_to_inventory(sess) -> None:
    """Merge manual in-transit / not-in-inventory columns into the live inventory snapshot."""
    inv = getattr(sess, "inventory_df_variant", None)
    if inv is None:
        inv = pd.DataFrame()
    overlay = getattr(sess, "manual_intransit_overlay_df", pd.DataFrame())

    if overlay is None or overlay.empty:
        return

    base = inv.copy() if inv is not None and not inv.empty else pd.DataFrame(columns=["OMS_SKU"])
    base = base.drop(columns=["Manual_InTransit", "Not_In_Inventory_Qty"], errors="ignore")

    merged = pd.merge(base, overlay, on="OMS_SKU", how="outer")
    for col in ("Manual_InTransit", "Not_In_Inventory_Qty"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
        else:
            merged[col] = 0

    inv_cols = [
        c
        for c in merged.columns
        if c.endswith("_Inventory")
        or c.endswith("_Live")
        or c.endswith("_InTransit")
        or c == "Buffer_Stock"
    ]
    mkt_cols = [c for c in inv_cols if "OMS" not in c and c != "Buffer_Stock"]
    merged["Marketplace_Total"] = merged[mkt_cols].sum(axis=1) if mkt_cols else 0
    oms_inv = merged.get("OMS_Inventory", pd.Series(0, index=merged.index))
    merged["Total_Inventory"] = pd.to_numeric(oms_inv, errors="coerce").fillna(0) + merged[
        "Marketplace_Total"
    ]

    # Keep SKUs that have on-hand stock OR manual in-transit / not-in-inventory qty.
    keep = (
        merged["Total_Inventory"].gt(0)
        | merged["Manual_InTransit"].gt(0)
        | merged["Not_In_Inventory_Qty"].gt(0)
    )
    sess.inventory_df_variant = merged.loc[keep].reset_index(drop=True)

    parent = getattr(sess, "inventory_df_parent", None)
    if parent is not None and not parent.empty:
        from .helpers import get_parent_sku

        p = sess.inventory_df_variant.copy()
        p["Parent_SKU"] = p["OMS_SKU"].map(get_parent_sku)
        num_cols = [c for c in p.columns if c != "OMS_SKU" and c != "Parent_SKU"]
        sess.inventory_df_parent = (
            p.groupby("Parent_SKU")[num_cols].sum().reset_index().rename(columns={"Parent_SKU": "OMS_SKU"})
        )

    try:
        from .inventory import refresh_inventory_api_cache

        refresh_inventory_api_cache(sess)
    except Exception:
        pass
