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
    """Amazon other-warehouse CSV: keep SELLABLE, remove ZNNE. Returns OMS_SKU, Amazon_Inventory."""
    df = read_csv_safe(csv_bytes)
    if df.empty or not {"MSKU", "Ending Warehouse Balance"}.issubset(df.columns):
        return pd.DataFrame()
    if "Disposition" in df.columns:
        df = df[df["Disposition"].astype(str).str.strip().str.upper() == "SELLABLE"]
    if "Location" in df.columns:
        df = df[df["Location"].astype(str).str.strip().str.upper() != "ZNNE"]
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


def _parse_myntra_other(csv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Myntra other-warehouse CSV.
    Uses 'seller sku code' for SKU mapping; falls back to 'style id' when empty.
    Returns OMS_SKU, Myntra_Other_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty:
        return pd.DataFrame()
    cols = {c.lower().strip(): c for c in df.columns}
    sku_col   = cols.get("seller sku code")
    style_col = cols.get("style id")
    inv_col   = cols.get("sellable inventory count")
    if inv_col is None:
        return pd.DataFrame()

    df[inv_col] = pd.to_numeric(df[inv_col], errors="coerce").fillna(0)

    def _resolve(row) -> str:
        # Primary: seller sku code
        if sku_col:
            val = str(row.get(sku_col, "")).strip()
            if val and val.lower() != "nan":
                return map_to_oms_sku(val, mapping)
        # Fallback: style id
        if style_col:
            val = str(row.get(style_col, "")).strip()
            if val and val.lower() != "nan":
                return map_to_oms_sku(val, mapping)
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
    return (
        df.groupby("OMS_SKU")[["_inv", "_buf"]].sum()
        .reset_index()
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
    fk_bytes: Optional[bytes],
    myntra_bytes: Optional[bytes],
    amz_bytes: Optional[bytes],
    mapping: Dict[str, str],
    group_by_parent: bool = False,
) -> pd.DataFrame:
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

    # ── Separately uploaded OMS / Combo files ────────────────
    if oms_bytes:
        # Accept either a single bytes object or a list of bytes
        oms_list = oms_bytes if isinstance(oms_bytes, list) else [oms_bytes]
        for ob in oms_list:
            if ob:
                part = _parse_oms_or_combo(ob)
                if not part.empty:
                    oms_parts.append(part)

    # ── Flipkart ─────────────────────────────────────────────
    if fk_bytes:
        df = read_csv_safe(fk_bytes)
        if not df.empty and {"SKU", "Live on Website"}.issubset(df.columns):
            df["OMS_SKU"]       = df["SKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            df["Flipkart_Live"] = pd.to_numeric(df["Live on Website"], errors="coerce").fillna(0)
            inv_dfs.append(df.groupby("OMS_SKU")["Flipkart_Live"].sum().reset_index())

    # ── Separately uploaded Myntra file ──────────────────────
    if myntra_bytes:
        df = read_csv_safe(myntra_bytes)
        if not df.empty:
            sku_col = next(
                (c for c in df.columns
                 if "seller sku code" in c.lower() or "sku code" in c.lower()), None
            )
            inv_col = next(
                (c for c in df.columns if "sellable inventory count" in c.lower()), None
            )
            if sku_col and inv_col:
                df["OMS_SKU"]          = df[sku_col].apply(lambda x: map_to_oms_sku(x, mapping))
                df["Myntra_Inventory"] = pd.to_numeric(df[inv_col], errors="coerce").fillna(0)
                inv_dfs.append(df.groupby("OMS_SKU")["Myntra_Inventory"].sum().reset_index())

    # ── Amazon / RAR ─────────────────────────────────────────
    if amz_bytes:
        raw = amz_bytes
        if raw[:6] == _RAR_MAGIC:
            # Multi-file RAR — extract all sources
            extracted = _extract_all_from_rar(raw)

            # 1. Amazon other-warehouse CSV
            if extracted["amz_csv"]:
                part = _parse_amz_csv(extracted["amz_csv"], mapping)
                if not part.empty:
                    inv_dfs.append(part)

            # 2. FBA in-transit TSVs (combine all shipments)
            fba_parts = [_parse_fba_tsv(t, mapping) for t in extracted["fba_tsvs"]]
            fba_parts = [d for d in fba_parts if not d.empty]
            if fba_parts:
                combined = pd.concat(fba_parts, ignore_index=True)
                inv_dfs.append(
                    combined.groupby("OMS_SKU")["FBA_InTransit"].sum().reset_index()
                )

            # 3. Myntra other-warehouse CSV
            if extracted["myntra_csv"]:
                part = _parse_myntra_other(extracted["myntra_csv"], mapping)
                if not part.empty:
                    inv_dfs.append(part)

            # 4. OMS inventory CSV (Inventory + Buffer Stock)
            if extracted["oms_csv"]:
                part = _parse_oms_csv(extracted["oms_csv"])
                if not part.empty:
                    oms_parts.append(part)

            # 5. Combo SKUs CSV
            if extracted["combo_csv"]:
                part = _parse_combo_csv(extracted["combo_csv"])
                if not part.empty:
                    oms_parts.append(part)

        else:
            # Plain CSV — treat as Amazon inventory
            part = _parse_amz_csv(raw, mapping)
            if not part.empty:
                inv_dfs.append(part)

    # ── Combine all OMS parts → single OMS_Inventory column ──
    if oms_parts:
        combined_oms = pd.concat(oms_parts, ignore_index=True)
        inv_dfs.insert(
            0,
            combined_oms.groupby("OMS_SKU")["OMS_Inventory"].sum().reset_index(),
        )

    if not inv_dfs:
        return pd.DataFrame()

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

    return consolidated[consolidated["Total_Inventory"] > 0].reset_index(drop=True)
