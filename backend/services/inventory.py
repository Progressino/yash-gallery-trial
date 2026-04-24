"""
Inventory loader — consolidated from all sources.
"""
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


def _dedupe_identical_byte_payloads(payloads: List[bytes]) -> Tuple[List[bytes], int]:
    """Drop byte-identical blobs (duplicate exports / (1)(2) copies)."""
    seen: set[bytes] = set()
    out: List[bytes] = []
    for p in payloads:
        h = hashlib.sha256(p).digest()
        if h in seen:
            continue
        seen.add(h)
        out.append(p)
    return out, len(payloads) - len(out)


def _rar_sniff_csv_kind(base_lower: str, data: bytes) -> Optional[str]:
    """
    Classify a CSV inside an inventory RAR. Returns:
      flipkart | myntra | amazon | combo | oms | None
    """
    if "flipkart" in base_lower or base_lower.startswith("fk"):
        return "flipkart"
    if "myntra" in base_lower:
        return "myntra"
    if "amz" in base_lower or "amazon" in base_lower:
        return "amazon"
    if "combo" in base_lower:
        return "combo"
    if "oms" in base_lower:
        return "oms"
    text = data[:4000].decode("utf-8", errors="ignore").lower()
    if "msku" in text and "ending warehouse balance" in text:
        return "amazon"
    if "seller sku code" in text or ("style id" in text and "inventory count" in text):
        return "myntra"
    if "combo sku code" in text and "combo" in text:
        return "combo"
    if "item skucode" in text or "buffer stock" in text:
        return "oms"
    if "live on website" in text:
        return "flipkart"
    return None


def _append_rar_csv(result: dict, kind: str, data: bytes) -> None:
    if kind == "flipkart":
        result["flipkart_csvs"].append(data)
    elif kind == "myntra":
        result["myntra_csvs"].append(data)
    elif kind == "amazon":
        result["amz_csvs"].append(data)
    elif kind == "combo":
        result["combo_csvs"].append(data)
    elif kind == "oms":
        result["oms_csvs"].append(data)


# ── RAR extraction ────────────────────────────────────────────

def _extract_all_from_rar(rar_bytes: bytes) -> dict:
    """
    Extract all relevant inventory files from a RAR archive.
    Tries bsdtar subprocess first (always available), then falls back to rarfile.
    Returns dict with keys:
      amz_csvs, myntra_csvs, oms_csvs, combo_csvs, flipkart_csvs (lists of bytes),
      fba_tsvs (list).
    Every matching CSV is collected — os.walk order no longer drops files.
    """
    result: dict = {
        "amz_csvs":      [],
        "myntra_csvs":   [],
        "oms_csvs":      [],
        "combo_csvs":    [],
        "flipkart_csvs": [],
        "fba_tsvs":      [],
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
                        continue
                    if not base.endswith(".csv"):
                        continue
                    kind = _rar_sniff_csv_kind(base, data)
                    if kind:
                        _append_rar_csv(result, kind, data)
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
                continue
            if not base.endswith(".csv"):
                continue
            data = rf.read(name)
            kind = _rar_sniff_csv_kind(base, data)
            if kind:
                _append_rar_csv(result, kind, data)
    return result


# ── Per-source parsers ────────────────────────────────────────

def _resolve_amz_sku(msku: str, mapping: Dict[str, str]) -> str:
    """Strip PL prefix then map to OMS SKU (mirrors _parse_fba_tsv logic)."""
    raw = str(msku).strip().upper()
    stripped = _PL_RE.sub(r"\1\2", raw)   # 1001PLYKBEIGE-3XL → 1001YKBEIGE-3XL
    return map_to_oms_sku(stripped, mapping)


def _parse_amz_csv(csv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Amazon Inventory Ledger CSV (Summary view).
    The report has one row per MSKU × Disposition × Location × Date — multi-period exports
    contain many date rows per MSKU.  We keep only the MOST RECENT date per
    (MSKU, Disposition, Location) before summing, giving current stock levels.
    Then filter to SELLABLE disposition to match OMS 'Amazon Other Warehouse'.
    Returns OMS_SKU, Amazon_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty or not {"MSKU", "Ending Warehouse Balance"}.issubset(df.columns):
        return pd.DataFrame()

    # ── Deduplicate by keeping the most recent Date per (MSKU, Disposition, Location) ──
    # Amazon Inventory Ledger exported over a date range has one row per period per SKU;
    # summing all periods inflates the total.  Only the latest row = current balance.
    group_keys = ["MSKU"]
    if "Disposition" in df.columns:
        group_keys.append("Disposition")
    if "Location" in df.columns:
        group_keys.append("Location")

    if "Date" in df.columns:
        df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
        # Within each (MSKU, Disposition, Location) group keep only the latest date row
        df = (
            df.sort_values("_date_parsed")
              .groupby(group_keys, sort=False)
              .last()
              .reset_index()
        )
        df = df.drop(columns=["_date_parsed"], errors="ignore")

    # ── Filter to SELLABLE only ──────────────────────────────────────────────────────
    disp_col = next((c for c in df.columns if c.strip().lower() in ("disposition", "inventory disposition")), None)
    if disp_col:
        df = df[df[disp_col].astype(str).str.strip().str.upper() == "SELLABLE"]

    # ── Exclude ZNNE — Amazon virtual/accounting location, not physical warehouse stock ──
    # ZNNE represents ~25,670 units that Amazon tracks internally but OMS excludes from
    # "Amazon Other Warehouse".  All real FCs use 3-letter city codes (BLR7, MAA4, etc.).
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


def _parse_fba_shipment_id(tsv_bytes: bytes) -> Optional[str]:
    """First-line 'Shipment ID' from Amazon FBA inbound TSV, if present."""
    first = tsv_bytes.decode("utf-8", errors="ignore").partition("\n")[0].split("\t")
    if len(first) >= 2 and first[0].strip().lower() == "shipment id":
        sid = first[1].strip().upper()
        return sid or None
    return None


def _parse_fba_tsv_detail(tsv_bytes: bytes) -> pd.DataFrame:
    """
    Line-level rows: Merchant SKU, FNSKU, Shipped (before OMS mapping).
    Used to merge multiple exports for the same shipment without double-counting.
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
    if "FNSKU" not in df.columns:
        df = df.copy()
        df["FNSKU"] = ""
    df = df[df["Merchant SKU"].notna()].copy()
    df["Shipped"] = pd.to_numeric(df["Shipped"], errors="coerce").fillna(0)
    return df[["Merchant SKU", "FNSKU", "Shipped"]]


def _parse_fba_tsv_shipment_group(tsv_chunks: List[bytes], mapping: Dict[str, str]) -> pd.DataFrame:
    """
    One or more TSV exports for the same FBA shipment. Identical files should be
    merged to a single manifest; overlapping line keys take max(Shipped) so
    duplicate rows do not inflate totals.
    """
    if len(tsv_chunks) == 1:
        return _parse_fba_tsv(tsv_chunks[0], mapping)
    detail_parts = [_parse_fba_tsv_detail(t) for t in tsv_chunks]
    detail_parts = [d for d in detail_parts if not d.empty]
    if not detail_parts:
        return pd.DataFrame()
    df = pd.concat(detail_parts, ignore_index=True)
    df = (
        df.groupby(["Merchant SKU", "FNSKU"], sort=False)["Shipped"]
        .max()
        .reset_index()
    )

    def _resolve(msku: str) -> str:
        raw = str(msku).strip().upper()
        stripped = _PL_RE.sub(r"\1\2", raw)
        return map_to_oms_sku(stripped, mapping)

    df["OMS_SKU"] = df["Merchant SKU"].apply(_resolve)
    return (
        df.groupby("OMS_SKU")["Shipped"]
        .sum()
        .reset_index()
        .rename(columns={"Shipped": "FBA_InTransit"})
    )


def _fba_chunk_total_shipped(tsv_bytes: bytes) -> float:
    """Total shipped quantity for one FBA TSV payload (best-effort)."""
    d = _parse_fba_tsv_detail(tsv_bytes)
    if d.empty:
        return 0.0
    return float(pd.to_numeric(d["Shipped"], errors="coerce").fillna(0).sum())


def _dedupe_fba_tsv_payloads(tsv_list: List[bytes]) -> Tuple[List[bytes], int]:
    """Drop byte-identical TSVs (e.g. duplicate downloads named (1)/(2))."""
    seen: set[bytes] = set()
    out: List[bytes] = []
    for t in tsv_list:
        h = hashlib.sha256(t).digest()
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out, len(tsv_list) - len(out)


def _aggregate_fba_intransit_tsvs(tsv_list: List[bytes], mapping: Dict[str, str]) -> Tuple[pd.DataFrame, dict]:
    """
    Parse all FBA in-transit TSVs: hash-dedupe, then merge per Shipment ID, then sum across shipments.
    Returns (dataframe OMS_SKU, FBA_InTransit), debug dict.
    """
    dbg: dict = {}
    if not tsv_list:
        return pd.DataFrame(), dbg
    dbg["fba_tsv_count_raw"] = len(tsv_list)
    deduped, dropped = _dedupe_fba_tsv_payloads(tsv_list)
    dbg["fba_tsv_identical_dropped"] = dropped
    dbg["fba_tsv_count_deduped"] = len(deduped)

    by_ship: Dict[str, List[bytes]] = defaultdict(list)
    no_sid: List[bytes] = []
    for t in deduped:
        sid = _parse_fba_shipment_id(t)
        if sid:
            by_ship[sid].append(t)
        else:
            no_sid.append(t)

    dbg["fba_shipment_ids"] = {k: len(v) for k, v in by_ship.items()}
    dbg["fba_tsv_no_shipment_id"] = len(no_sid)
    dup_ship_ids = sorted([sid for sid, chunks in by_ship.items() if len(chunks) > 1])
    if dup_ship_ids:
        dbg["fba_duplicate_shipment_ids"] = dup_ship_ids
        dbg["fba_duplicate_notice"] = (
            f"Duplicate FBA files detected for shipment IDs: {', '.join(dup_ship_ids)}. "
            "Only one file per shipment is included."
        )

    parts: List[pd.DataFrame] = []
    for _sid, chunks in by_ship.items():
        # If duplicate files for a shipment are present, include only one payload
        # (highest shipped total) and exclude the rest as duplicates.
        chosen = chunks
        if len(chunks) > 1:
            best = max(chunks, key=_fba_chunk_total_shipped)
            chosen = [best]
        p = _parse_fba_tsv_shipment_group(chosen, mapping)
        if not p.empty:
            parts.append(p)
    for t in no_sid:
        p = _parse_fba_tsv(t, mapping)
        if not p.empty:
            parts.append(p)

    if not parts:
        return pd.DataFrame(), dbg
    combined = pd.concat(parts, ignore_index=True)
    out = combined.groupby("OMS_SKU")["FBA_InTransit"].sum().reset_index()
    return out, dbg


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
    # Use "inventory count" (total warehouse stock) — matches OMS Myntra Other Warehouse figure.
    # "sellable inventory count" is per-listing stock (lower); "inventory count" is the physical count.
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
    OMS inventory CSV (Item SkuCode, Inventory, Buffer Stock + optional marketplace columns).
    The Unicommerce export includes per-channel inventory as additional columns
    (e.g. 'Amazon Other Warehouse', 'Flipkart', 'Myntra'). When present, these are
    returned alongside OMS_Inventory so the caller can use them as authoritative figures.
    Returns: OMS_SKU, OMS_Inventory, Buffer_Stock, and optionally
             Amazon_Inventory, Flipkart_Inventory, Myntra_Other_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty or "Item SkuCode" not in df.columns or "Inventory" not in df.columns:
        return pd.DataFrame()
    df["OMS_SKU"] = df["Item SkuCode"].astype(str).str.strip().str.upper()

    # Buffer Stock — flexible column name match (before dedup so we can sum it)
    buf_col = next(
        (c for c in df.columns if c.strip().lower() in ("buffer stock", "buffer_stock", "bufferstock", "buffer")),
        None,
    )

    # Auto-detect marketplace inventory columns
    col_lower = {c.strip().lower(): c for c in df.columns}
    _MKTPLACE = [
        ("Amazon_Inventory",       [
            "amazon other warehouse", "amazon other wh", "amazon other", "amz other warehouse",
            "amz other wh", "amazon_other_warehouse",
        ]),
        ("Flipkart_Inventory",     ["flipkart inventory", "flipkart_inventory", "flipkart"]),
        ("Myntra_Other_Inventory", [
            "myntra other warehouse", "myntra other wh", "myntra other", "myntra inventory",
            "myntra_other_inventory", "myntra_inventory", "myntra",
        ]),
    ]

    # Build numeric columns to aggregate
    agg_cols = {"OMS_Inventory": ("Inventory", "sum")}
    if buf_col:
        agg_cols["Buffer_Stock"] = (buf_col, "sum")
    for out_col, keywords in _MKTPLACE:
        for kw in keywords:
            if kw in col_lower:
                agg_cols[out_col] = (col_lower[kw], "sum")
                break

    # Convert to numeric before groupby
    df["OMS_Inventory"] = pd.to_numeric(df["Inventory"], errors="coerce").fillna(0)
    if buf_col:
        df["Buffer_Stock"] = pd.to_numeric(df[buf_col], errors="coerce").fillna(0)
    for out_col, (src_col, _) in {k: v for k, v in agg_cols.items() if k not in ("OMS_Inventory", "Buffer_Stock")}.items():
        df[out_col] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)

    numeric_cols = list(agg_cols.keys())
    result = df.groupby("OMS_SKU", as_index=False)[numeric_cols].sum()

    if "Buffer_Stock" not in result.columns:
        result["Buffer_Stock"] = 0

    return result


def _parse_combo_csv(csv_bytes: bytes) -> pd.DataFrame:
    """
    OMS combo SKUs CSV (Combo SKU Code, Combo Qty Stock).
    Returns OMS_SKU, OMS_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty or "Combo SKU Code" not in df.columns or "Combo Qty Stock" not in df.columns:
        return pd.DataFrame()
    df["OMS_SKU"]      = df["Combo SKU Code"].astype(str).str.strip().str.upper()
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
    myntra_bytes: Optional[bytes | List[bytes]],
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
    fk_parts: List[pd.DataFrame] = []    # Flipkart — standalone + inside RAR, one merge later
    myntra_other_parts: List[pd.DataFrame] = []  # Myntra PPMP — standalone + RAR (one layer)
    amz_rar_parts: List[pd.DataFrame] = []
    debug: dict = {}

    # ── Separately uploaded OMS / Combo files ────────────────
    if oms_bytes:
        oms_list = oms_bytes if isinstance(oms_bytes, list) else [oms_bytes]
        for ob in oms_list:
            if ob:
                part = _parse_oms_or_combo(ob)
                if not part.empty:
                    oms_parts.append(part)

    # ── Flipkart (standalone uploads — RAR FK files appended into fk_parts below) ──
    if fk_bytes:
        fk_list = fk_bytes if isinstance(fk_bytes, list) else [fk_bytes]
        for fb in fk_list:
            if fb:
                p = _parse_fk_inventory_csv(fb, mapping)
                if not p.empty:
                    fk_parts.append(p)

    # ── Separately uploaded Myntra (multi-file + byte-dedupe) ──────────────────────
    if myntra_bytes:
        m_list = myntra_bytes if isinstance(myntra_bytes, list) else [myntra_bytes]
        m_list = [b for b in m_list if b]
        m_dedup, m_drop = _dedupe_identical_byte_payloads(m_list)
        debug["myntra_upload_deduped"] = m_drop
        for blob in m_dedup:
            p = _parse_myntra_other(blob, mapping)
            if not p.empty:
                myntra_other_parts.append(p)
        debug["myntra_upload_cols"] = "via _parse_myntra_other"

    # ── Amazon / RAR ─────────────────────────────────────────
    if amz_bytes:
        raw = amz_bytes
        if raw[:6] == _RAR_MAGIC:
            extracted = _extract_all_from_rar(raw)
            debug["rar_files"] = {
                "amz_csvs":      len(extracted["amz_csvs"]),
                "myntra_csvs":   len(extracted["myntra_csvs"]),
                "oms_csvs":      len(extracted["oms_csvs"]),
                "combo_csvs":    len(extracted["combo_csvs"]),
                "flipkart_csvs": len(extracted["flipkart_csvs"]),
                "fba_tsvs":      len(extracted["fba_tsvs"]),
            }

            for fc in extracted["flipkart_csvs"]:
                p = _parse_fk_inventory_csv(fc, mapping)
                if not p.empty:
                    fk_parts.append(p)

            amz_blobs, _ = _dedupe_identical_byte_payloads(extracted["amz_csvs"])
            if amz_blobs:
                _amz_peek = read_csv_safe(amz_blobs[0])
                debug["amz_csv_cols"] = list(_amz_peek.columns) if not _amz_peek.empty else []
                debug["amz_csv_sample_dispositions"] = (
                    _amz_peek["Disposition"].dropna().unique().tolist()
                    if "Disposition" in _amz_peek.columns else "no Disposition col"
                )
                if not _amz_peek.empty and "Location" in _amz_peek.columns and "Ending Warehouse Balance" in _amz_peek.columns:
                    _amz_peek["_bal"] = pd.to_numeric(_amz_peek["Ending Warehouse Balance"], errors="coerce").fillna(0)
                    _amz_sellable = _amz_peek[_amz_peek.get("Disposition", pd.Series(dtype=str)).astype(str).str.strip().str.upper() == "SELLABLE"] if "Disposition" in _amz_peek.columns else _amz_peek
                    debug["amz_location_totals"] = (
                        _amz_sellable.groupby("Location")["_bal"].sum()
                        .sort_values(ascending=False).head(30).to_dict()
                    )
                    debug["amz_sellable_total"] = int(_amz_sellable["_bal"].sum())
            for ab in amz_blobs:
                p = _parse_amz_csv(ab, mapping)
                if not p.empty:
                    amz_rar_parts.append(p)
            if amz_rar_parts:
                amz_cat = pd.concat(amz_rar_parts, ignore_index=True)
                part = amz_cat.groupby("OMS_SKU")["Amazon_Inventory"].sum().reset_index()
                inv_dfs.append(part)
                debug["amz"] = f"{len(part)} SKUs ({len(amz_blobs)} ledger file(s))"
            elif amz_blobs:
                debug["amz"] = "0 SKUs (ledger parse empty)"

            part, fba_dbg = _aggregate_fba_intransit_tsvs(extracted["fba_tsvs"], mapping)
            debug.update(fba_dbg)
            if not part.empty:
                inv_dfs.append(part)
                debug["fba"] = f"{len(part)} SKUs"

            myn_blobs, myn_drop = _dedupe_identical_byte_payloads(extracted["myntra_csvs"])
            debug["myntra_rar_deduped"] = myn_drop
            for mb in myn_blobs:
                p = _parse_myntra_other(mb, mapping)
                if not p.empty:
                    myntra_other_parts.append(p)
            debug["myntra_rar"] = (
                f"{len(myn_blobs)} file(s) from RAR → merged with standalone Myntra"
                if myn_blobs
                else "no myntra csv found in RAR"
            )

            if not oms_bytes:
                oms_blobs, oms_dedup_n = _dedupe_identical_byte_payloads(extracted["oms_csvs"])
                debug["oms_rar_deduped_identical"] = oms_dedup_n
                for ob in oms_blobs:
                    _oms_peek = read_csv_safe(ob)
                    debug["oms_csv_all_cols"] = list(_oms_peek.columns)
                    p = _parse_oms_csv(ob)
                    if not p.empty:
                        oms_parts.append(p)
                if oms_blobs:
                    debug["oms_rar"] = f"{len(oms_blobs)} OMS file(s) merged"
            else:
                debug["oms_rar"] = "skipped (separate OMS file takes precedence)"

            if not oms_bytes:
                combo_blobs, combo_dedup = _dedupe_identical_byte_payloads(extracted["combo_csvs"])
                debug["combo_rar_deduped_identical"] = combo_dedup
                for cb in combo_blobs:
                    p = _parse_combo_csv(cb)
                    if not p.empty:
                        oms_parts.append(p)
                if combo_blobs:
                    debug["combo_rar"] = f"{len(combo_blobs)} combo file(s) merged"
            else:
                debug["combo_rar"] = "skipped (separate OMS file takes precedence)"

        else:
            part = _parse_amz_csv(raw, mapping)
            if not part.empty:
                inv_dfs.append(part)
            debug["amz_csv"] = f"{len(part)} SKUs"

    # ── Single Myntra Other layer (standalone + RAR) ───────────────────────────
    if myntra_other_parts:
        m_all = pd.concat(myntra_other_parts, ignore_index=True)
        part = m_all.groupby("OMS_SKU")["Myntra_Other_Inventory"].sum().reset_index()
        inv_dfs.append(part)
        debug["myntra"] = f"{len(part)} SKUs ({len(myntra_other_parts)} file payload(s) merged)"
    else:
        debug["myntra"] = debug.get("myntra_rar") or "0 SKUs (no Myntra PPMP)"

    # ── Single merged Flipkart layer (avoids duplicate Flipkart_Inventory_x / _y columns) ──
    if fk_parts:
        combined_fk = pd.concat(fk_parts, ignore_index=True)
        part = combined_fk.groupby("OMS_SKU")["Flipkart_Inventory"].sum().reset_index()
        inv_dfs.append(part)
        debug["flipkart"] = f"{len(part)} SKUs ({len(fk_parts)} file payloads)"
    elif "flipkart" not in debug:
        debug["flipkart"] = "0 SKUs (no valid FK data)"

    # ── Combine all OMS parts → single OMS_Inventory + Buffer_Stock (+ marketplace cols) ──
    if oms_parts:
        combined_oms = pd.concat(oms_parts, ignore_index=True)
        _all_src = ["OMS_Inventory", "Buffer_Stock", "Amazon_Inventory", "Flipkart_Inventory", "Myntra_Other_Inventory"]
        agg_cols = [c for c in _all_src if c in combined_oms.columns]
        oms_part = combined_oms.groupby("OMS_SKU")[agg_cols].sum().reset_index()
        inv_dfs.insert(0, oms_part)
        debug["oms"] = f"{len(oms_part)} SKUs"

        # If OMS CSV provides marketplace columns, they are authoritative (same source as OMS UI).
        # Remove those columns from separately parsed sources to avoid double-counting.
        oms_mkt_cols = {c for c in ["Amazon_Inventory", "Flipkart_Inventory", "Myntra_Other_Inventory"]
                        if c in oms_part.columns and oms_part[c].sum() > 0}
        if oms_mkt_cols:
            debug["oms_provides_marketplace"] = sorted(oms_mkt_cols)
            for i in range(1, len(inv_dfs)):
                drop = [c for c in oms_mkt_cols if c in inv_dfs[i].columns]
                if drop:
                    inv_dfs[i] = inv_dfs[i].drop(columns=drop)

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
    # Buffer_Stock is shown as a separate informational column but NOT added to Total_Inventory.
    # OMS exports "Inventory" as the complete figure; adding Buffer_Stock would double-count it.
    consolidated["Total_Inventory"] = oms_inv + consolidated["Marketplace_Total"]

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
    # Buffer_Stock is informational only — not added to Total_Inventory (would double-count OMS figure)
    result["Total_Inventory"] = oms_inv + result["Marketplace_Total"]

    return result[result["Total_Inventory"] > 0].reset_index(drop=True)
