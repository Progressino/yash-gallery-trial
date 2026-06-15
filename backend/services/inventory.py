"""
Inventory loader — consolidated from all sources.
"""
import calendar
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .helpers import (
    collapse_duplicate_trailing_size_suffix,
    map_to_oms_sku,
    get_parent_sku,
    read_csv_safe,
)

# Use bsdtar (libarchive-tools) for RAR extraction — supports RAR5 natively
try:
    import rarfile as _rarfile_mod
    _rarfile_mod.UNRAR_TOOL = "bsdtar"
    _rarfile_mod.ALT_TOOL   = "bsdtar"
except Exception:
    pass


_RAR_MAGIC = b"Rar!\x1a\x07"
_PL_RE = re.compile(r'^(\d+)PL(YK)', re.I)

_RE_DATE_DMY = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b")
_RE_DATE_DMONY = re.compile(r"\b(\d{1,2})[-\s]([A-Za-z]{3,9})[-\s](\d{2,4})\b", re.I)
_RE_DATE_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")

_MONTH_NAME_TO_NUM = {
    m.lower(): i
    for i, m in enumerate(calendar.month_abbr)
    if m
}
_MONTH_NAME_TO_NUM.update(
    {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
)


def _normalize_year(y: int) -> int:
    if y < 100:
        return 2000 + y if y < 70 else 1900 + y
    return y


def _dates_in_text(text: str) -> list[date]:
    found: list[date] = []
    for m in _RE_DATE_ISO.finditer(text):
        try:
            found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    for m in _RE_DATE_DMY.finditer(text):
        try:
            d, mo, y = int(m.group(1)), int(m.group(2)), _normalize_year(int(m.group(3)))
            found.append(date(y, mo, d))
        except ValueError:
            continue
    for m in _RE_DATE_DMONY.finditer(text):
        mon = _MONTH_NAME_TO_NUM.get(m.group(2).lower()[:3])
        if not mon:
            continue
        try:
            d, y = int(m.group(1)), _normalize_year(int(m.group(3)))
            found.append(date(y, mon, d))
        except ValueError:
            continue
    return found


def infer_inventory_snapshot_date(
    file_parts: list[tuple[str, bytes]] | None = None,
    debug: dict | None = None,
) -> dict[str, Any]:
    """
    Best-effort snapshot as-of date from upload filenames and Amazon ledger metadata.
    Returns snapshot_date (ISO), snapshot_date_label, snapshot_date_sources.
    """
    dbg = debug or {}
    candidates: list[tuple[date, str, int]] = []

    def _add(d: date, source: str, priority: int) -> None:
        candidates.append((d, source, priority))

    for fname, _raw in file_parts or []:
        base = (fname or "").replace("\\", "/").split("/")[-1]
        if not base:
            continue
        low = base.lower()
        pri = 2
        if low.startswith("oms") or " oms" in low:
            pri = 0
        elif "inventory" in low and low.endswith((".rar", ".zip")):
            pri = 1
        for d in _dates_in_text(base):
            _add(d, base, pri)

    for entry in dbg.get("rar_manifest") or []:
        if entry.get("status") != "loaded":
            continue
        fn = str(entry.get("filename") or "")
        low = fn.lower()
        pri = 3
        if low.startswith("oms"):
            pri = 0
        elif "seller_inventory_report" in low or "current inventory" in low:
            pri = 4
        for d in _dates_in_text(fn):
            _add(d, fn, pri)

    amz = dbg.get("amz_disclaimer") or {}
    amz_day = str(amz.get("latest_report_date") or "").strip()[:10]
    if amz_day:
        try:
            _add(date.fromisoformat(amz_day), "Amazon ledger (latest report day)", 5)
        except ValueError:
            pass

    if not candidates:
        return {
            "snapshot_date": "",
            "snapshot_date_label": "",
            "snapshot_date_sources": [],
        }

    candidates.sort(key=lambda x: (x[2], x[0]))
    primary = candidates[0][0]
    iso = primary.isoformat()
    label = primary.strftime("%d %b %Y")
    sources: list[str] = []
    seen: set[str] = set()
    for d, src, _pri in sorted(candidates, key=lambda x: (x[2], x[1])):
        if d != primary:
            note = f"{src} ({d.strftime('%d %b %Y')})"
        else:
            note = src
        if note not in seen:
            sources.append(note)
            seen.add(note)
    return {
        "snapshot_date": iso,
        "snapshot_date_label": label,
        "snapshot_date_sources": sources[:8],
    }


def apply_inventory_snapshot_metadata(
    sess: Any,
    file_parts: list[tuple[str, bytes]] | None,
    debug: dict | None,
) -> dict:
    """Store snapshot date on session + debug after a successful inventory parse."""
    meta = infer_inventory_snapshot_date(file_parts, debug)
    merged_debug = dict(debug or {})
    merged_debug.update(meta)
    sess.inventory_debug = merged_debug
    sess.inventory_snapshot_date = meta["snapshot_date"]
    sess.inventory_snapshot_date_label = meta["snapshot_date_label"]
    sess.inventory_snapshot_date_sources = list(meta["snapshot_date_sources"])
    sess.inventory_snapshot_uploaded_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return meta


_SNAPSHOT_MARKETPLACE_COLS = (
    ("OMS_Inventory", "OMS warehouse"),
    ("Buffer_Stock", "Buffer stock"),
    ("Amazon_Inventory", "Amazon"),
    ("FBA_InTransit", "FBA in-transit"),
    ("Manual_InTransit", "Manual in-transit"),
    ("Not_In_Inventory_Qty", "Not in inventory"),
    ("Flipkart_Inventory", "Flipkart"),
    ("Myntra_Other_Inventory", "Myntra"),
    ("Meesho_Inventory", "Meesho"),
)


def inventory_marketplace_breakdown(df: pd.DataFrame, debug: dict | None = None) -> list[dict[str, Any]]:
    """Per-channel totals for UI; ``included`` means column present with stock > 0."""
    dbg = debug or {}
    rows: list[dict[str, Any]] = []
    for col, label in _SNAPSHOT_MARKETPLACE_COLS:
        units = 0
        skus = 0
        if not df.empty and col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(0)
            units = int(series.sum())
            skus = int((series > 0).sum())
        rows.append({
            "key": col,
            "label": label,
            "included": units > 0,
            "units": units,
            "skus": skus,
        })
    # Parse-status hints from last upload debug
    parse_hints = {
        "Flipkart_Inventory": dbg.get("flipkart"),
        "Myntra_Other_Inventory": dbg.get("myntra"),
        "Meesho_Inventory": dbg.get("meesho"),
        "Amazon_Inventory": dbg.get("amz"),
    }
    for row in rows:
        hint = parse_hints.get(row["key"])
        if hint is not None:
            row["parse_status"] = str(hint)
    return rows


def inventory_missing_marketplace_warnings(debug: dict | None = None) -> list[str]:
    """Actionable hints when expected marketplace files were not in the last upload."""
    dbg = debug or {}
    warnings: list[str] = []

    def _empty(key: str) -> bool:
        val = str(dbg.get(key) or "")
        return not val or val.startswith("0 SKUs") or "no " in val.lower()

    has_partial = not (_empty("oms") and _empty("flipkart") and _empty("amz"))
    if has_partial and _empty("myntra"):
        warnings.append(
            "Myntra inventory not in this upload — add the Myntra PPMP Seller Inventory CSV "
            "(columns: seller sku code, inventory count, warehouse name with “Myntra”), "
            "or include Myntra columns on the OMS export."
        )
    if _empty("flipkart") and not _empty("oms"):
        warnings.append(
            "Flipkart inventory not parsed — include Flipkart “Current Inventory” CSVs "
            "(SKU + Live on Website) inside the RAR."
        )
    oms_mkt = set(dbg.get("oms_provides_marketplace") or [])
    if has_partial and "Meesho_Inventory" not in oms_mkt:
        warnings.append(
            "Meesho inventory not in this upload — add a Meesho column on the OMS CSV "
            "(e.g. “Meesho inventory”) or upload a Meesho stock file separately."
        )
    if not dbg.get("oms") or str(dbg.get("oms", "")).startswith("0"):
        warnings.append("OMS inventory CSV missing or empty inside the bundle.")
    return warnings


def oms_loaded_in_debug(debug: dict | None) -> bool:
    """True when the last parse loaded OMS warehouse stock."""
    val = str((debug or {}).get("oms") or "").strip()
    return bool(val) and not val.startswith("0 SKUs")


def upload_bundle_expects_oms(file_parts: list[tuple[str, bytes]] | None) -> bool:
    """Daily RAR/ZIP bundles should include an OMS inventory CSV."""
    for fname, _raw in file_parts or []:
        low = (fname or "").lower()
        if low.endswith((".rar", ".zip")):
            return True
        if "oms" in low and low.endswith((".csv", ".xlsx", ".xls")):
            return True
    return False


def inventory_column_totals(df: pd.DataFrame) -> dict[str, int]:
    """Sum numeric inventory columns once per snapshot (API cache)."""
    if df.empty:
        return {}
    totals: dict[str, int] = {}
    for col in df.columns:
        if col == "OMS_SKU":
            continue
        try:
            totals[col] = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
        except Exception:
            totals[col] = 0
    return totals


def refresh_inventory_api_cache(sess: Any) -> None:
    """Cache totals + marketplace cards so GET /inventory stays fast."""
    df = getattr(sess, "inventory_df_variant", None)
    if df is None or getattr(df, "empty", True):
        sess.inventory_api_totals = {}
        sess.inventory_api_marketplaces = []
        return
    dbg = getattr(sess, "inventory_debug", None) or {}
    sess.inventory_api_totals = inventory_column_totals(df)
    sess.inventory_api_marketplaces = inventory_marketplace_breakdown(df, dbg)


def backup_inventory_before_upload(sess: Any) -> None:
    """Keep previous snapshot if a bundle upload fails to include OMS."""
    df = getattr(sess, "inventory_df_variant", None)
    if df is None or getattr(df, "empty", True):
        sess._inventory_pre_upload_backup = None
        return
    parent = getattr(sess, "inventory_df_parent", None)
    sess._inventory_pre_upload_backup = {
        "variant": df.copy(),
        "parent": parent.copy() if parent is not None and not getattr(parent, "empty", True) else parent,
        "meta": inventory_session_meta_bundle(sess),
    }


def inventory_snapshot_upload_epoch(uploaded_at: str) -> float:
    """Parse ``inventory_snapshot_uploaded_at`` (UTC ISO) for comparisons."""
    raw = str(uploaded_at or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def sync_inventory_snapshot_from_warm(sess: Any) -> None:
    """
    Keep session and in-memory warm cache on the newest inventory snapshot.

    Fixes Inventory tab showing stale stock when the session already had an older
    frame loaded but a newer upload was saved to warm cache / another worker.
    """
    if getattr(sess, "inventory_upload_status", "idle") == "running":
        return
    try:
        import backend.main as _main
        import pandas as pd
    except Exception:
        return

    warm = getattr(_main, "_warm_cache", None) or {}
    meta_key = getattr(_main, "_INVENTORY_META_WARM_KEY", "inventory_session_meta")
    warm_meta = warm.get(meta_key) if isinstance(warm.get(meta_key), dict) else {}
    warm_at = inventory_snapshot_upload_epoch(
        str((warm_meta or {}).get("inventory_snapshot_uploaded_at") or "")
    )
    sess_at = inventory_snapshot_upload_epoch(
        getattr(sess, "inventory_snapshot_uploaded_at", "") or ""
    )

    warm_variant = warm.get("inventory_df_variant")
    warm_has = (
        warm_variant is not None
        and hasattr(warm_variant, "empty")
        and not warm_variant.empty
    )
    sess_df = getattr(sess, "inventory_df_variant", None)
    sess_has = sess_df is not None and hasattr(sess_df, "empty") and not sess_df.empty

    if warm_has and (not sess_has or warm_at > sess_at + 1e-6):
        for key in ("inventory_df_variant", "inventory_df_parent"):
            val = warm.get(key)
            if val is not None and hasattr(val, "empty") and not val.empty:
                setattr(sess, key, val.copy() if hasattr(val, "copy") else val)
        if warm_meta:
            apply_inventory_session_meta(sess, warm_meta)
        refresh_inventory_api_cache(sess)
        return

    if sess_has and (not warm_has or sess_at > warm_at + 1e-6):
        try:
            _main.merge_inventory_into_warm_cache(sess)
        except Exception:
            pass


def restore_inventory_upload_backup(sess: Any) -> bool:
    """Restore pre-upload inventory after a rejected bundle parse."""
    bak = getattr(sess, "_inventory_pre_upload_backup", None)
    if not bak:
        return False
    variant = bak.get("variant")
    if variant is None or getattr(variant, "empty", True):
        return False
    sess.inventory_df_variant = variant
    parent = bak.get("parent")
    if parent is not None and not getattr(parent, "empty", True):
        sess.inventory_df_parent = parent
    meta = bak.get("meta")
    if isinstance(meta, dict):
        apply_inventory_session_meta(sess, meta)
    refresh_inventory_api_cache(sess)
    sess._inventory_pre_upload_backup = None
    return True


def inventory_session_meta_bundle(sess: Any) -> dict[str, Any]:
    """Serializable inventory snapshot metadata for warm cache / disk."""
    return {
        "inventory_debug": dict(getattr(sess, "inventory_debug", None) or {}),
        "inventory_snapshot_date": str(getattr(sess, "inventory_snapshot_date", "") or ""),
        "inventory_snapshot_date_label": str(
            getattr(sess, "inventory_snapshot_date_label", "") or ""
        ),
        "inventory_snapshot_date_sources": list(
            getattr(sess, "inventory_snapshot_date_sources", None) or []
        ),
        "inventory_snapshot_uploaded_at": str(
            getattr(sess, "inventory_snapshot_uploaded_at", "") or ""
        ),
    }


def apply_inventory_session_meta(sess: Any, meta: dict[str, Any] | None) -> None:
    """Restore snapshot metadata from warm cache."""
    if not meta:
        return
    dbg = meta.get("inventory_debug")
    if isinstance(dbg, dict):
        sess.inventory_debug = dbg
    for key in (
        "inventory_snapshot_date",
        "inventory_snapshot_date_label",
        "inventory_snapshot_uploaded_at",
    ):
        if meta.get(key):
            setattr(sess, key, meta[key])
    sources = meta.get("inventory_snapshot_date_sources")
    if sources is not None:
        sess.inventory_snapshot_date_sources = list(sources)


def ensure_inventory_snapshot_metadata(sess: Any) -> None:
    """Re-derive snapshot date from stored upload debug when session fields were lost."""
    if (
        getattr(sess, "inventory_snapshot_date_label", "")
        or getattr(sess, "inventory_snapshot_date", "")
    ):
        return
    dbg = getattr(sess, "inventory_debug", None) or {}
    if not dbg:
        return
    meta = infer_inventory_snapshot_date(None, dbg)
    if not meta.get("snapshot_date"):
        return
    merged = dict(dbg)
    merged.update(meta)
    sess.inventory_debug = merged
    sess.inventory_snapshot_date = meta["snapshot_date"]
    sess.inventory_snapshot_date_label = meta["snapshot_date_label"]
    sess.inventory_snapshot_date_sources = list(meta["snapshot_date_sources"])


def inventory_rows_for_api(
    df: pd.DataFrame,
    *,
    search: str = "",
    offset: int = 0,
    limit: int = 500,
) -> tuple[list[dict], int]:
    """Filter/slice inventory for API (avoid shipping 6k+ rows in one JSON blob)."""
    if df.empty:
        return [], 0
    work = df
    q = (search or "").strip().lower()
    if q:
        skus = work["OMS_SKU"].astype(str)
        work = work[skus.str.lower().str.contains(q, na=False, regex=False)]
    total = int(len(work))
    off = max(0, int(offset))
    lim = max(1, min(int(limit), 5000))
    page = work.iloc[off : off + lim]
    return page.fillna(0).to_dict("records"), total


def inventory_snapshot_meta_for_api(sess: Any) -> dict[str, Any]:
    """Fields for /data/inventory and coverage API responses."""
    ensure_inventory_snapshot_metadata(sess)
    dbg = getattr(sess, "inventory_debug", None) or {}
    snap = (
        getattr(sess, "inventory_snapshot_date", "")
        or dbg.get("snapshot_date")
        or ""
    )
    label = (
        getattr(sess, "inventory_snapshot_date_label", "")
        or dbg.get("snapshot_date_label")
        or ""
    )
    sources = (
        getattr(sess, "inventory_snapshot_date_sources", None)
        or dbg.get("snapshot_date_sources")
        or []
    )
    uploaded = (
        getattr(sess, "inventory_snapshot_uploaded_at", "")
        or dbg.get("snapshot_uploaded_at")
        or ""
    )
    if snap and not label:
        try:
            label = date.fromisoformat(str(snap)[:10]).strftime("%d %b %Y")
        except ValueError:
            label = str(snap)
    return {
        "snapshot_date": str(snap) if snap else None,
        "snapshot_date_label": str(label) if label else None,
        "snapshot_date_sources": list(sources) if sources else None,
        "snapshot_uploaded_at": str(uploaded) if uploaded else None,
    }


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


def _content_sniff_csv_kind(data: bytes) -> Optional[str]:
    """Classify by column headers (filename alone is unreliable for PPMP exports)."""
    df = read_csv_safe(data)
    if df.empty:
        text = data[:8000].decode("utf-8", errors="ignore").lower()
    else:
        text = " ".join(str(c).lower() for c in df.columns)
    if "flipkart selling price" in text or (
        "live on website" in text and "listing id" in text and "fsn" in text
    ):
        return "flipkart"
    if "seller sku code" in text and "inventory count" in text:
        # Myntra PPMP Seller Inventory Report (seller id + style id + inventory count).
        # Flipkart uses "Current Inventory" exports with SKU + Live on Website instead.
        if not df.empty:
            wh_col = next(
                (c for c in df.columns if str(c).strip().lower() == "warehouse name"),
                None,
            )
            if wh_col is not None:
                wh = df[wh_col].astype(str).str.lower()
                if wh.str.contains("myntra", na=False).any():
                    return "myntra"
        return "myntra"
    if "item skucode" in text or "buffer stock" in text:
        return "oms"
    if "combo sku code" in text:
        return "combo"
    if "msku" in text and "ending warehouse balance" in text:
        return "amazon"
    return None


def _rar_sniff_csv_kind(base_lower: str, data: bytes) -> Optional[str]:
    """
    Classify a CSV inside an inventory RAR. Returns:
      flipkart | myntra | amazon | combo | oms | None
    """
    by_content = _content_sniff_csv_kind(data)
    if "current inventory" in base_lower:
        return by_content or "flipkart"
    if "seller_inventory_report" in base_lower or "seller_orders_report" in base_lower:
        return by_content or "myntra"
    if "flipkart" in base_lower or base_lower.startswith("fk"):
        return by_content or "flipkart"
    if "myntra" in base_lower:
        return by_content or "myntra"
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


def _manifest_add(manifest: list[dict], filename: str, *, category: str, status: str, reason: str = "") -> None:
    entry: dict = {"filename": filename, "category": category, "status": status}
    if reason:
        entry["reason"] = reason
    manifest.append(entry)


def _extract_all_from_rar(rar_bytes: bytes) -> tuple[dict, list[dict]]:
    """
    Extract all relevant inventory files from a RAR archive.
    Returns (extracted_dict, manifest) where manifest lists every inner file and outcome.
    """
    result: dict = {
        "amz_csvs":      [],
        "myntra_csvs":   [],
        "oms_csvs":      [],
        "combo_csvs":    [],
        "flipkart_csvs": [],
        "fba_tsvs":      [],
    }
    manifest: list[dict] = []

    def _ingest_file(fname: str, data: bytes) -> None:
        base = fname.replace("\\", "/").split("/")[-1]
        base_lower = base.lower()
        if base_lower.endswith(".tsv"):
            result["fba_tsvs"].append(data)
            _manifest_add(manifest, base, category="fba", status="loaded")
            return
        if not base_lower.endswith(".csv"):
            _manifest_add(
                manifest, base, category="other", status="skipped",
                reason="Not a CSV/TSV inventory file",
            )
            return
        kind = _rar_sniff_csv_kind(base_lower, data)
        if kind:
            _append_rar_csv(result, kind, data)
            _manifest_add(manifest, base, category=kind, status="loaded")
        else:
            _manifest_add(
                manifest, base, category="unknown", status="skipped",
                reason="Could not classify inventory CSV (check columns or filename)",
            )

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
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as fh:
                        data = fh.read()
                    _ingest_file(fname, data)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return result, manifest

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
            if name.endswith("/"):
                continue
            base = name.replace("\\", "/").split("/")[-1]
            if not base:
                continue
            data = rf.read(name)
            _ingest_file(base, data)
    return result, manifest


# ── RAR extraction ────────────────────────────────────────────


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
    contain many date rows. Keep only the latest report-day snapshot.
    Then filter to SELLABLE disposition to match OMS 'Amazon Other Warehouse'.
    Returns OMS_SKU, Amazon_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty or not {"MSKU", "Ending Warehouse Balance"}.issubset(df.columns):
        return pd.DataFrame()

    # ── Filter to SELLABLE only ──────────────────────────────────────────────────────
    disp_col = next((c for c in df.columns if c.strip().lower() in ("disposition", "inventory disposition")), None)
    if disp_col:
        df = df[df[disp_col].astype(str).str.strip().str.upper() == "SELLABLE"]

    # ── Exclude ZNNE — Amazon virtual/accounting location, not physical warehouse stock ──
    # ZNNE represents ~25,670 units that Amazon tracks internally but OMS excludes from
    # "Amazon Other Warehouse".  All real FCs use 3-letter city codes (BLR7, MAA4, etc.).
    if "Location" in df.columns:
        df = df[df["Location"].astype(str).str.strip().str.upper() != "ZNNE"]

    # One-day snapshot only: keep rows from latest report date in this file.
    if "Date" in df.columns:
        _d = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
        if _d.notna().any():
            df = df[_d == _d.max()]

    df["OMS_SKU"]          = df["MSKU"].apply(lambda x: _resolve_amz_sku(x, mapping))
    df["Amazon_Inventory"] = pd.to_numeric(df["Ending Warehouse Balance"], errors="coerce").fillna(0)
    return df.groupby("OMS_SKU")["Amazon_Inventory"].sum().reset_index()


def _analyze_amz_ledger_filters(csv_bytes: bytes) -> dict:
    """Build disclaimer metrics for Amazon inventory row exclusions."""
    df = read_csv_safe(csv_bytes)
    if df.empty or "Ending Warehouse Balance" not in df.columns:
        return {}

    work = df.copy()
    work["_bal"] = pd.to_numeric(work["Ending Warehouse Balance"], errors="coerce").fillna(0)
    out: dict = {
        "raw_total_units": float(work["_bal"].sum()),
        "raw_rows": int(len(work)),
    }

    disp_col = next((c for c in work.columns if c.strip().lower() in ("disposition", "inventory disposition")), None)
    if disp_col:
        sell = work[work[disp_col].astype(str).str.strip().str.upper() == "SELLABLE"].copy()
        out["excluded_non_sellable_units"] = float(work["_bal"].sum() - sell["_bal"].sum())
    else:
        sell = work.copy()
        out["excluded_non_sellable_units"] = 0.0

    if "Location" in sell.columns:
        no_znne = sell[sell["Location"].astype(str).str.strip().str.upper() != "ZNNE"].copy()
        out["excluded_znne_units"] = float(sell["_bal"].sum() - no_znne["_bal"].sum())
    else:
        no_znne = sell.copy()
        out["excluded_znne_units"] = 0.0

    out["sellable_non_znne_units"] = float(no_znne["_bal"].sum())

    if "Date" in no_znne.columns:
        d = pd.to_datetime(no_znne["Date"], errors="coerce", dayfirst=False)
        if d.notna().any():
            latest = d.max()
            latest_units = float(no_znne.loc[d == latest, "_bal"].sum())
            out["latest_report_date"] = str(latest.date())
            out["latest_report_units"] = latest_units
            out["excluded_older_date_units"] = float(no_znne["_bal"].sum() - latest_units)
            out["date_count"] = int(d.dt.normalize().nunique())
        else:
            out["latest_report_units"] = float(no_znne["_bal"].sum())
            out["excluded_older_date_units"] = 0.0
            out["date_count"] = 0
    else:
        out["latest_report_units"] = float(no_znne["_bal"].sum())
        out["excluded_older_date_units"] = 0.0
        out["date_count"] = 0

    return out


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


def _resolve_seller_style_sku(raw: str, mapping: Dict[str, str]) -> str:
    """Map marketplace seller SKU / style id to OMS SKU (PL prefix stripped)."""
    cleaned = str(raw).strip().upper()
    if not cleaned or cleaned == "NAN":
        return ""
    if cleaned in mapping:
        return mapping[cleaned]
    stripped = _PL_RE.sub(r"\1\2", cleaned)
    if stripped in mapping:
        return mapping[stripped]
    return stripped


def _parse_fk_ppmp_inventory(csv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Flipkart PPMP Seller Inventory Report (seller sku code + inventory/sellable count).
    Same shape as Myntra PPMP — must not be parsed as Myntra_Other_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty:
        return pd.DataFrame()
    if any("flipkart selling price" in str(c).lower() for c in df.columns):
        return pd.DataFrame()
    sku_col = next((c for c in df.columns if "seller sku" in c.lower()), None)
    if sku_col is None:
        return pd.DataFrame()
    inv_col = next((c for c in df.columns if c.lower().strip() == "inventory count"), None)
    if inv_col is None:
        inv_col = next(
            (c for c in df.columns if c.lower().strip() == "sellable inventory count"),
            None,
        )
    if inv_col is None:
        inv_col = next((c for c in df.columns if "inventory" in c.lower() and "count" in c.lower()), None)
    if inv_col is None:
        return pd.DataFrame()
    df[inv_col] = pd.to_numeric(df[inv_col], errors="coerce").fillna(0)

    def _resolve(row) -> str:
        val = str(row.get(sku_col, "")).strip()
        if val and val.lower() not in ("nan", "") and not val.isdigit():
            return _resolve_seller_style_sku(val, mapping)
        return ""

    df["OMS_SKU"] = df.apply(_resolve, axis=1)
    df = df[df["OMS_SKU"].str.strip() != ""]
    return (
        df.groupby("OMS_SKU")[inv_col].sum()
        .reset_index()
        .rename(columns={inv_col: "Flipkart_Inventory"})
    )


def _parse_fk_inventory_csv(csv_bytes: bytes, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Flipkart warehouse inventory CSV (Current Inventory export) or PPMP seller report.
    Returns OMS_SKU, Flipkart_Inventory.
    """
    df = read_csv_safe(csv_bytes)
    if df.empty:
        return pd.DataFrame()

    if "SKU" in df.columns:
        present = [c for c in _FK_STOCK_COLS if c in df.columns]
        if not present and "Live on Website" in df.columns:
            present = ["Live on Website"]
        if present:
            for c in present:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df["_total"] = df[present].sum(axis=1)
            df["OMS_SKU"] = df["SKU"].apply(lambda r: _resolve_seller_style_sku(r, mapping))
            df = df[df["OMS_SKU"].str.strip() != ""]
            if not df.empty:
                return (
                    df.groupby("OMS_SKU")["_total"].sum()
                    .reset_index()
                    .rename(columns={"_total": "Flipkart_Inventory"})
                )

    return _parse_fk_ppmp_inventory(csv_bytes, mapping)


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
    if any("flipkart selling price" in str(c).lower() for c in df.columns):
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

    def _resolve(row) -> str:
        # Primary: seller sku code — only use if it's a real seller SKU (non-numeric)
        # Myntra sometimes puts their internal sku_id (pure number) here, skip those
        if sku_col:
            val = str(row.get(sku_col, "")).strip()
            if val and val.lower() not in ("nan", "") and not val.isdigit():
                return _resolve_seller_style_sku(val, mapping)
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
        ("Meesho_Inventory", [
            "meesho inventory", "meesho_inventory", "meesho other warehouse", "meesho",
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
            extracted, rar_manifest = _extract_all_from_rar(raw)
            debug["rar_manifest"] = rar_manifest
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
            amz_disclaimer: dict = {
                "raw_total_units": 0.0,
                "excluded_non_sellable_units": 0.0,
                "excluded_znne_units": 0.0,
                "sellable_non_znne_units": 0.0,
                "excluded_older_date_units": 0.0,
                "latest_report_units": 0.0,
                "raw_rows": 0,
                "date_count": 0,
            }
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
                metrics = _analyze_amz_ledger_filters(ab)
                for k in ["raw_rows", "date_count"]:
                    amz_disclaimer[k] += int(metrics.get(k, 0))
                for k in [
                    "raw_total_units",
                    "excluded_non_sellable_units",
                    "excluded_znne_units",
                    "sellable_non_znne_units",
                    "excluded_older_date_units",
                    "latest_report_units",
                ]:
                    amz_disclaimer[k] += float(metrics.get(k, 0.0))
                if metrics.get("latest_report_date"):
                    amz_disclaimer["latest_report_date"] = metrics["latest_report_date"]
                p = _parse_amz_csv(ab, mapping)
                if not p.empty:
                    amz_rar_parts.append(p)
            if amz_blobs:
                debug["amz_disclaimer"] = amz_disclaimer
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
        _all_src = [
            "OMS_Inventory", "Buffer_Stock", "Amazon_Inventory", "Flipkart_Inventory",
            "Myntra_Other_Inventory", "Meesho_Inventory",
        ]
        agg_cols = [c for c in _all_src if c in combined_oms.columns]
        oms_part = combined_oms.groupby("OMS_SKU")[agg_cols].sum().reset_index()
        inv_dfs.insert(0, oms_part)
        debug["oms"] = f"{len(oms_part)} SKUs"

        # If OMS CSV provides marketplace columns, they are authoritative (same source as OMS UI).
        # Remove those columns from separately parsed sources to avoid double-counting.
        oms_mkt_cols = {
            c for c in [
                "Amazon_Inventory", "Flipkart_Inventory", "Myntra_Other_Inventory", "Meesho_Inventory",
            ]
            if c in oms_part.columns and oms_part[c].sum() > 0
        }
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
            df["OMS_SKU"] = (
                df["OMS_SKU"]
                .str.strip()
                .str.upper()
                .map(collapse_duplicate_trailing_size_suffix)
            )
            num_cols = [c for c in df.columns if c != "OMS_SKU"]
            df = df.groupby("OMS_SKU")[num_cols].sum().reset_index()
            inv_dfs[i] = df

    # ── Outer-merge all sources on OMS_SKU ───────────────────
    consolidated = inv_dfs[0]
    for d in inv_dfs[1:]:
        consolidated = pd.merge(consolidated, d, on="OMS_SKU", how="outer")

    inv_cols = inventory_source_columns(consolidated)
    if inv_cols:
        consolidated[inv_cols] = consolidated[inv_cols].fillna(0)

    consolidated = recompute_inventory_totals(consolidated)

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
_EXTRA_MKT_COLS = frozenset({"Manual_InTransit", "Not_In_Inventory_Qty"})


def inventory_source_columns(df: pd.DataFrame) -> list[str]:
    """Physical stock columns (excludes computed totals like Total_Inventory)."""
    cols: list[str] = []
    for c in df.columns:
        if c in _COMPUTED_COLS or c == "OMS_SKU":
            continue
        if (
            c in _EXTRA_MKT_COLS
            or c.endswith("_Live")
            or c.endswith("_InTransit")
            or c.endswith("_Inventory")
            or c == "Buffer_Stock"
        ):
            cols.append(c)
    return cols


def marketplace_total_columns(df: pd.DataFrame) -> list[str]:
    """Columns summed into Marketplace_Total (OMS warehouse + buffer are excluded)."""
    return [
        c
        for c in inventory_source_columns(df)
        if c != "Buffer_Stock" and "oms" not in c.lower()
    ]


def recompute_inventory_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Marketplace_Total and Total_Inventory from source columns.

    Buffer_Stock is informational only. Manual in-transit and not-in-inventory qty
    count toward marketplace total. Never sum the existing Total_Inventory column
    into Marketplace_Total (that would double-count).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    src_cols = inventory_source_columns(out)
    if src_cols:
        for col in src_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    mkt_cols = marketplace_total_columns(out)
    out["Marketplace_Total"] = out[mkt_cols].sum(axis=1) if mkt_cols else 0
    if "OMS_Inventory" in out.columns:
        oms_inv = pd.to_numeric(out["OMS_Inventory"], errors="coerce").fillna(0)
    else:
        oms_inv = 0
    out["Total_Inventory"] = oms_inv + out["Marketplace_Total"]
    return out


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
    result = recompute_inventory_totals(result)

    return result[result["Total_Inventory"] > 0].reset_index(drop=True)
