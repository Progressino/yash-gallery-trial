"""
Inventory loader — consolidated from all sources.
"""
import io
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

import pandas as pd

from .helpers import map_to_oms_sku, get_parent_sku, read_csv_safe

# Use bsdtar (libarchive-tools) for RAR extraction — supports RAR5 natively
try:
    import rarfile as _rarfile_mod
    _rarfile_mod.UNRAR_TOOL = "bsdtar"
    _rarfile_mod.ALT_TOOL   = "bsdtar"
except Exception:
    pass


_RAR_MAGIC = b"Rar!\x1a\x07"
_PL_RE = re.compile(r'^(\d+)PL(YK)', re.I)


# ── RAR extraction ────────────────────────────────────────────

def _extract_all_from_rar(rar_bytes: bytes) -> dict:
    """
    Extract all relevant inventory files from a RAR archive.
    Tries bsdtar subprocess first (always available), then falls back to rarfile.
    Returns dict with keys: amz_csv, myntra_csv, oms_csv, combo_csv, fba_tsvs (list).
    """
    result: dict = {
        "amz_csv":    None,
        "myntra_csv": None,
        "oms_csv":    None,
        "combo_csv":  None,
        "fba_tsvs":   [],
    }

    # ── Try bsdtar subprocess (libarchive-tools — supports RAR4 & RAR5) ──
    bsdtar = shutil.which("bsdtar")
    if bsdtar:
        tmpdir = tempfile.mkdtemp(prefix="inv_rar_")
        try:
            rar_path = os.path.join(tmpdir, "upload.rar")
            with open(rar_path, "wb") as f:
                f.write(rar_bytes)
            subprocess.run(
                [bsdtar, "xf", rar_path, "-C", tmpdir],
                check=True, capture_output=True,
            )
            for root, _dirs, files in os.walk(tmpdir):
                for fname in files:
                    if fname == "upload.rar":
                        continue
                    base = fname.lower()
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as fh:
                        data = fh.read()
                    if base.endswith(".tsv"):
                        result["fba_tsvs"].append(data)
                    elif "amz" in base and base.endswith(".csv"):
                        result["amz_csv"] = data
                    elif "myntra" in base and base.endswith(".csv"):
                        result["myntra_csv"] = data
                    elif "combo" in base and base.endswith(".csv"):
                        result["combo_csv"] = data
                    elif "oms" in base and base.endswith(".csv"):
                        result["oms_csv"] = data
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return result

    # ── Fallback: rarfile Python module ──────────────────────────────────
    try:
        import rarfile
    except ImportError:
        raise ValueError(
            "Cannot extract RAR: neither bsdtar nor rarfile module found. "
            "Install libarchive-tools (apt) or rarfile (pip)."
        )
    with rarfile.RarFile(io.BytesIO(rar_bytes)) as rf:
        for name in rf.namelist():
            base = name.replace("\\", "/").split("/")[-1].lower()
            if base.endswith(".tsv"):
                result["fba_tsvs"].append(rf.read(name))
            elif "amz" in base and base.endswith(".csv"):
                result["amz_csv"] = rf.read(name)
            elif "myntra" in base and base.endswith(".csv"):
                result["myntra_csv"] = rf.read(name)
            elif "combo" in base and base.endswith(".csv"):
                result["combo_csv"] = rf.read(name)
            elif "oms" in base and base.endswith(".csv"):
                result["oms_csv"] = rf.read(name)
    return result


# ── Per-source parsers ────────────────────────────────────────

def _resolve_amz_sku(msku: str, mapping: Dict[str, str]) -> str:
    """Strip PL prefix then map to OMS SKU (mirrors _parse_fba_tsv logic)."""
    raw = str(msku).strip().upper()
    stripped = _PL_RE.sub(r"\1\2", raw)   # 1001PLYKBEIGE-3XL → 1001YKBEIGE-3XL
    return map_to_oms_sku(stripped, mapping)


def _parse_amz_csv(csv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """Amazon other-warehouse CSV: SELLABLE disposition only to match OMS totals. Returns OMS_SKU, Amazon_Inventory."""
    df = read_csv_safe(csv_bytes)
    if df.empty or not {"MSKU", "Ending Warehouse Balance"}.issubset(df.columns):
        return pd.DataFrame()
    # Filter to SELLABLE disposition only — matches OMS "Amazon Other Warehouse" figure
    # (Inventory Ledger report has SELLABLE, UNSELLABLE, IN_TRANSIT, INBOUND_RECEIVING etc.)
    disp_col = next((c for c in df.columns if c.strip().lower() in ("disposition", "inventory disposition")), None)
    if disp_col:
        df = df[df[disp_col].astype(str).str.strip().str.upper() == "SELLABLE"]
    df["OMS_SKU"]          = df["MSKU"].apply(lambda x: _resolve_amz_sku(x, mapping))
    df["Amazon_Inventory"] = pd.to_numeric(df["Ending Warehouse Balance"], errors="coerce").fillna(0)
    return df.groupby("OMS_SKU")["Amazon_Inventory"].sum().reset_index()


def _parse_fba_tsv(tsv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    FBA in-transit shipment TSV.
    Data table starts at the row beginning with 'Merchant SKU'.
    Returns OMS_SKU, FBA_InTransit.
    """
    text = tsv_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Merchant SKU\t")), None
    )
    if header_idx is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep="\t", dtype=str)
    except Exception:
        return pd.DataFrame()
    if "Merchant SKU" not in df.columns or "Shipped" not in df.columns:
        return pd.DataFrame()

    df = df[df["Merchant SKU"].notna()].copy()
    df["Shipped"] = pd.to_numeric(df["Shipped"], errors="coerce").fillna(0)

    def _resolve(msku: str) -> str:
        raw = str(msku).strip().upper()
        stripped = _PL_RE.sub(r"\1\2", raw)   # 1323PLYKTEAL-6XL → 1323YKTEAL-6XL
        return map_to_oms_sku(stripped, mapping)

    df["OMS_SKU"] = df["Merchant SKU"].apply(_resolve)
    return (
        df.groupby("OMS_SKU")["Shipped"].sum()
        .reset_index()
        .rename(columns={"Shipped": "FBA_InTransit"})
    )


# Flipkart stock columns — only physically available/sellable units (matches OMS Flipkart count).
# "Orders to Dispatch", "Recalls to Dispatch", "Returns Processing" are excluded because
# OMS does not count stock that is already committed/dispatched/in returns.
_FK_STOCK_COLS = [
    "Live on Website",
]


def _parse_fk_inventory_csv(csv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Flipkart warehouse inventory CSV (Current Inventory export).
    Sums all usable stock columns (Live on Website + reserved + receiving + dispatching).
    Strips PL prefix from SKUs (e.g. 1388PLYKMAROON-XL → 1388YKMAROON-XL).
    Returns OMS_SKU, Flipkart_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty or "SKU" not in df.columns:
        return pd.DataFrame()

    # Build stock total from whichever stock columns exist
    present = [c for c in _FK_STOCK_COLS if c in df.columns]
    if not present:
        # Fallback: any column named "Live on Website" only
        if "Live on Website" not in df.columns:
            return pd.DataFrame()
        present = ["Live on Website"]

    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["_total"] = df[present].sum(axis=1)

    def _resolve_fk_sku(raw) -> str:
        cleaned = str(raw).strip().upper()
        if not cleaned or cleaned == "NAN":
            return ""
        if cleaned in mapping:
            return mapping[cleaned]
        stripped = _PL_RE.sub(r"\1\2", cleaned)
        if stripped in mapping:
            return mapping[stripped]
        return stripped

    df["OMS_SKU"] = df["SKU"].apply(_resolve_fk_sku)
    df = df[df["OMS_SKU"].str.strip() != ""]
    return (
        df.groupby("OMS_SKU")["_total"].sum()
        .reset_index()
        .rename(columns={"_total": "Flipkart_Inventory"})
    )


def _parse_myntra_other(csv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Myntra other-warehouse CSV.
    Uses 'seller sku code' (non-numeric only) for SKU mapping; falls back to
    'style id' for rows where seller sku code is a Myntra internal numeric ID.
    Uses 'inventory count' (not 'sellable inventory count') as the stock value.
    Returns OMS_SKU, Myntra_Other_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty:
        return pd.DataFrame()
    sku_col   = next((c for c in df.columns if "seller sku" in c.lower()), None)
    style_col = next((c for c in df.columns if "style" in c.lower() and "id" in c.lower()), None)
    # Prefer exact "inventory count" over "sellable inventory count"
    inv_col = next((c for c in df.columns if c.lower().strip() == "inventory count"), None)
    if inv_col is None:
        inv_col = next((c for c in df.columns if "inventory" in c.lower() and "count" in c.lower()), None)
    if inv_col is None:
        return pd.DataFrame()

    df[inv_col] = pd.to_numeric(df[inv_col], errors="coerce").fillna(0)

    def _resolve_seller_sku(raw: str) -> str:
        """Map a non-numeric Myntra seller SKU to OMS SKU, handling PL prefix."""
        cleaned = raw.strip().upper()
        if not cleaned or cleaned == "NAN":
            return ""
        # Try direct mapping first
        if cleaned in mapping:
            return mapping[cleaned]
        # Strip PL prefix (1001PLYKBEIGE-3XL → 1001YKBEIGE-3XL) and try again
        stripped = _PL_RE.sub(r"\1\2", cleaned)
        if stripped in mapping:
            return mapping[stripped]
        # Use PL-stripped version as-is (it IS the OMS SKU)
        return stripped

    def _resolve(row) -> str:
        # Primary: seller sku code — only use if it's a real seller SKU (non-numeric)
        # Myntra sometimes puts their internal sku_id (pure number) here, skip those
        if sku_col:
            val = str(row.get(sku_col, "")).strip()
            if val and val.lower() not in ("nan", "") and not val.isdigit():
                return _resolve_seller_sku(val)
        # Fallback: style id → mapped via SKU mapping sheet (MYNTRA sheet adds style_id→OMS)
        if style_col:
            val = str(row.get(style_col, "")).strip()
            if val and val.lower() not in ("nan", ""):
                result = map_to_oms_sku(val, mapping)
                # Only use if it resolved to a non-numeric OMS SKU
                if result and not result.isdigit():
                    return result
        return ""

    df["OMS_SKU"] = df.apply(_resolve, axis=1)
    df = df[df["OMS_SKU"].str.strip() != ""]
    return (
        df.groupby("OMS_SKU")[inv_col].sum()
        .reset_index()
        .rename(columns={inv_col: "Myntra_Other_Inventory"})
    )


def _parse_oms_csv(csv_bytes: bytes) -> pd.DataFrame:
    """
    OMS inventory CSV (Item SkuCode, Inventory, Buffer Stock).
    Returns OMS_SKU, OMS_Inventory (Inventory column only — matches OMS export exactly),
    and Buffer_Stock as a separate column so zero-Inventory / nonzero-Buffer items remain visible.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty or "Item SkuCode" not in df.columns or "Inventory" not in df.columns:
        return pd.DataFrame()
    df["OMS_SKU"] = df["Item SkuCode"].astype(str).str.strip()
    df["_inv"]    = pd.to_numeric(df["Inventory"], errors="coerce").fillna(0)
    df["_buf"]    = pd.to_numeric(df.get("Buffer Stock", 0), errors="coerce").fillna(0)
    # Deduplicate by SKU — OMS has one authoritative row per SKU; duplicates are export artifacts
    df = df.drop_duplicates(subset="OMS_SKU", keep="last")
    return (
        df[["OMS_SKU", "_inv", "_buf"]]
        .rename(columns={"_inv": "OMS_Inventory", "_buf": "Buffer_Stock"})
    )


def _parse_combo_csv(csv_bytes: bytes) -> pd.DataFrame:
    """
    OMS combo SKUs CSV (Combo SKU Code, Combo Qty Stock).
    Returns OMS_SKU, OMS_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty or "Combo SKU Code" not in df.columns or "Combo Qty Stock" not in df.columns:
        return pd.DataFrame()
    df["OMS_SKU"]      = df["Combo SKU Code"].astype(str).str.strip()
    df["OMS_Inventory"] = pd.to_numeric(df["Combo Qty Stock"], errors="coerce").fillna(0)
    return df.groupby("OMS_SKU")["OMS_Inventory"].sum().reset_index()


# ── Main loader ───────────────────────────────────────────────

def _parse_oms_or_combo(csv_bytes: bytes) -> pd.DataFrame:
    """Try OMS CSV first, then Combo SKUs CSV. Returns OMS_Inventory column."""
    part = _parse_oms_csv(csv_bytes)
    if not part.empty:
        return part
    return _parse_combo_csv(csv_bytes)


def load_inventory_consolidated(
    oms_bytes: Optional[bytes | List[bytes]],
    fk_bytes: Optional[bytes | List[bytes]],
    myntra_bytes: Optional[bytes],
    amz_bytes: Optional[bytes],
    mapping: Dict[str, str],
    group_by_parent: bool = False,
    return_debug: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, dict]":
    """
    Merge inventory from OMS, Flipkart, Myntra and Amazon/RAR sources.
    When amz_bytes is a RAR archive it is split into:
      - Amazon CSV (SELLABLE + non-ZNNE)        → Amazon_Inventory
      - FBA in-transit TSVs                      → FBA_InTransit
      - Myntra other-warehouse CSV               → Myntra_Other_Inventory
      - OMS inventory CSV (Inventory + Buffer)   → OMS_Inventory
      - Combo SKUs CSV                           → OMS_Inventory (merged)
    """
    inv_dfs: List[pd.DataFrame] = []
    oms_parts: List[pd.DataFrame] = []   # accumulate OMS data before merging
    debug: dict = {}

    # ── Separately uploaded OMS / Combo files ────────────────
    if oms_bytes:
        oms_list = oms_bytes if isinstance(oms_bytes, list) else [oms_bytes]
        for ob in oms_list:
            if ob:
                part = _parse_oms_or_combo(ob)
                if not part.empty:
                    oms_parts.append(part)

    # ── Flipkart ─────────────────────────────────────────────
    if fk_bytes:
        fk_list = fk_bytes if isinstance(fk_bytes, list) else [fk_bytes]
        fk_parts = []
        for fb in fk_list:
            if fb:
                p = _parse_fk_inventory_csv(fb, mapping)
                if not p.empty:
                    fk_parts.append(p)
        if fk_parts:
            combined_fk = pd.concat(fk_parts, ignore_index=True)
            part = combined_fk.groupby("OMS_SKU")["Flipkart_Inventory"].sum().reset_index()
            inv_dfs.append(part)
            debug["flipkart"] = f"{len(part)} SKUs ({len(fk_list)} files)"
        else:
            debug["flipkart"] = f"0 SKUs (no valid FK data)"

    # ── Separately uploaded Myntra file ──────────────────────
    if myntra_bytes:
        part = _parse_myntra_other(myntra_bytes, mapping)
        debug["myntra_upload_cols"] = "via _parse_myntra_other"
        if not part.empty:
            # rename column to Myntra_Inventory for separately-uploaded file
            part = part.rename(columns={"Myntra_Other_Inventory": "Myntra_Inventory"})
            inv_dfs.append(part)
            debug["myntra"] = f"{len(part)} SKUs"
        else:
            debug["myntra"] = "0 SKUs"

    # ── Amazon / RAR ─────────────────────────────────────────
    if amz_bytes:
        raw = amz_bytes
        if raw[:6] == _RAR_MAGIC:
            extracted = _extract_all_from_rar(raw)
            debug["rar_files"] = {
                "amz_csv": bool(extracted["amz_csv"]),
                "myntra_csv": bool(extracted["myntra_csv"]),
                "oms_csv": bool(extracted["oms_csv"]),
                "combo_csv": bool(extracted["combo_csv"]),
                "fba_tsvs": len(extracted["fba_tsvs"]),
            }

            if extracted["amz_csv"]:
                part = _parse_amz_csv(extracted["amz_csv"], mapping)
                if not part.empty:
                    inv_dfs.append(part)
                debug["amz"] = f"{len(part)} SKUs"

            fba_parts = [_parse_fba_tsv(t, mapping) for t in extracted["fba_tsvs"]]
            fba_parts = [d for d in fba_parts if not d.empty]
            if fba_parts:
                combined = pd.concat(fba_parts, ignore_index=True)
                part = combined.groupby("OMS_SKU")["FBA_InTransit"].sum().reset_index()
                inv_dfs.append(part)
                debug["fba"] = f"{len(part)} SKUs"

            if extracted["myntra_csv"]:
                part = _parse_myntra_other(extracted["myntra_csv"], mapping)
                debug["myntra_rar"] = f"{len(part)} SKUs"
                if not part.empty:
                    inv_dfs.append(part)
            else:
                debug["myntra_rar"] = "no myntra csv found in RAR"

            if extracted["oms_csv"]:
                part = _parse_oms_csv(extracted["oms_csv"])
                if not part.empty:
                    oms_parts.append(part)
                debug["oms_rar"] = f"{len(part)} SKUs"

            if extracted["combo_csv"]:
                part = _parse_combo_csv(extracted["combo_csv"])
                if not part.empty:
                    oms_parts.append(part)
                debug["combo_rar"] = f"{len(part)} SKUs"

        else:
            part = _parse_amz_csv(raw, mapping)
            if not part.empty:
                inv_dfs.append(part)
            debug["amz_csv"] = f"{len(part)} SKUs"

    # ── Combine all OMS parts → single OMS_Inventory + Buffer_Stock ──
    if oms_parts:
        combined_oms = pd.concat(oms_parts, ignore_index=True)
        agg_cols = [c for c in ["OMS_Inventory", "Buffer_Stock"] if c in combined_oms.columns]
        oms_part = combined_oms.groupby("OMS_SKU")[agg_cols].sum().reset_index()
        inv_dfs.insert(0, oms_part)
        debug["oms"] = f"{len(oms_part)} SKUs"

    if not inv_dfs:
        if return_debug:
            return pd.DataFrame(), debug
        return pd.DataFrame()

    # ── Normalize OMS_SKU case before merge to prevent case-based duplicates ──
    # (OMS CSV may use lowercase sizes like "6xl" while Amazon uses "6XL")
    for i, df in enumerate(inv_dfs):
        if "OMS_SKU" in df.columns:
            df = df.copy()
            df["OMS_SKU"] = df["OMS_SKU"].str.strip().str.upper()
            num_cols = [c for c in df.columns if c != "OMS_SKU"]
            df = df.groupby("OMS_SKU")[num_cols].sum().reset_index()
            inv_dfs[i] = df

    # ── Outer-merge all sources on OMS_SKU ───────────────────
    consolidated = inv_dfs[0]
    for d in inv_dfs[1:]:
        consolidated = pd.merge(consolidated, d, on="OMS_SKU", how="outer")

    inv_cols = [
        c for c in consolidated.columns
        if c.endswith("_Inventory") or c.endswith("_Live") or c.endswith("_InTransit")
           or c == "Buffer_Stock"
    ]
    consolidated[inv_cols] = consolidated[inv_cols].fillna(0)

    mkt_cols = [c for c in inv_cols if "OMS" not in c and c != "Buffer_Stock"]
    consolidated["Marketplace_Total"] = consolidated[mkt_cols].sum(axis=1) if mkt_cols else 0
    oms_inv    = consolidated.get("OMS_Inventory", pd.Series(0, index=consolidated.index))
    buf_stock  = consolidated.get("Buffer_Stock",  pd.Series(0, index=consolidated.index))
    consolidated["Total_Inventory"] = oms_inv + buf_stock + consolidated["Marketplace_Total"]

    if group_by_parent:
        consolidated["Parent_SKU"] = consolidated["OMS_SKU"].apply(get_parent_sku)
        consolidated = (
            consolidated.groupby("Parent_SKU")[inv_cols + ["Marketplace_Total", "Total_Inventory"]]
            .sum().reset_index()
            .rename(columns={"Parent_SKU": "OMS_SKU"})
        )

    result = consolidated[consolidated["Total_Inventory"] > 0].reset_index(drop=True)
    if return_debug:
        return result, debug
    return result


# ── Incremental merge helper ──────────────────────────────────────────────────

_COMPUTED_COLS = {"Total_Inventory", "Marketplace_Total"}
_FIXED_COLS    = {"OMS_SKU", "OMS_Inventory", "Buffer_Stock"}


def merge_inventory_update(existing: pd.DataFrame, update: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a new partial inventory upload into the existing session inventory.

    Source columns from `update` (e.g. Flipkart_Inventory) replace the same
    columns in `existing`.  All other columns from `existing` are kept.
    Total_Inventory and Marketplace_Total are recomputed from scratch.
    """
    if existing.empty:
        return update
    if update.empty:
        return existing

    # Source columns carried by this update (skip computed totals)
    update_src_cols = [c for c in update.columns if c not in _COMPUTED_COLS]

    # Drop replaced columns from existing, keep OMS_SKU
    drop_from_existing = [
        c for c in update_src_cols
        if c in existing.columns and c != "OMS_SKU"
    ]
    base = existing.drop(columns=drop_from_existing + list(_COMPUTED_COLS), errors="ignore")

    # Outer-merge: keeps all SKUs from both sides
    result = pd.merge(base, update[update_src_cols], on="OMS_SKU", how="outer")

    # Fill numerics with 0
    for col in result.columns:
        if col != "OMS_SKU":
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    # Recompute totals
    inv_cols = [
        c for c in result.columns
        if c.endswith("_Inventory") or c.endswith("_Live") or c.endswith("_InTransit")
           or c == "Buffer_Stock"
    ]
    mkt_cols = [c for c in inv_cols if "OMS" not in c and c != "Buffer_Stock"]
    result["Marketplace_Total"] = result[mkt_cols].sum(axis=1) if mkt_cols else 0
    oms_inv   = result.get("OMS_Inventory", pd.Series(0, index=result.index))
    buf_stock = result.get("Buffer_Stock",  pd.Series(0, index=result.index))
    result["Total_Inventory"] = oms_inv + buf_stock + result["Marketplace_Total"]

    return result[result["Total_Inventory"] > 0].reset_index(drop=True)
