"""Parse marketplace return reports (CSV / Excel / RAR / ZIP) for PO return overlay."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .po_engine import canonical_oms_key
from .po_raise_import import pick_csv_column

_log = logging.getLogger(__name__)

_RAR_MAGIC = b"Rar!\x1a\x07"
_ARCHIVE_EXTS = (".rar", ".zip")
_DATA_EXTS = (".csv", ".xlsx", ".xls", ".txt")


def _skip_return_archive_member(inner_name: str) -> bool:
    """Skip non-return exports bundled in daily Return Data.rar (e.g. Meesho Lost)."""
    base = inner_name.split("/")[-1].lower()
    if base.startswith("help"):
        return True
    if "meesho" in base and "lost" in base:
        return True
    return False

_SKU_CANDS = (
    "oms_sku",
    "sku",
    "oms sku",
    "item_sku",
    "seller_sku",
    "seller_sku_code",
    "asin",
    "(child) asin",
    "child asin",
    "style_id",
    "product_id",
    "returned_sku",
    "myntra_sku_code",
)
_QTY_CANDS = (
    "return_units",
    "return_qty",
    "returned_qty",
    "return quantity",
    "returns",
    "units refunded",
    "units refunded – b2b",
    "units refunded - b2b",
    "qty",
    "quantity",
    "units",
)


def _strip_flipkart_sku(val: str) -> str:
    s = str(val or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    return re.sub(r"^SKU:\s*", "", s, flags=re.IGNORECASE).strip()


def _extract_zip_members(raw: bytes, prefix: str = "") -> List[Tuple[str, bytes]]:
    """Recursively pull CSV/Excel members out of a ZIP (and any ZIPs nested inside it)."""
    out: List[Tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner_name = info.filename.replace("\\", "/")
                full_name = f"{prefix}{inner_name}"
                if inner_name.lower().endswith(".zip"):
                    out.extend(_extract_zip_members(zf.read(info), prefix=f"{full_name}/"))
                elif inner_name.lower().endswith(_DATA_EXTS):
                    out.append((full_name, zf.read(info)))
    except Exception:
        _log.exception("Nested ZIP extract failed for return upload")
    return out


def _extract_rar_members_local(raw: bytes) -> List[Tuple[str, bytes]]:
    """Extract RAR (incl. ZIPs nested inside it) without importing the upload router.

    ``unar`` is tried first: ``bsdtar``/libarchive silently misreads some RAR5
    archives that bundle a top-level CSV alongside several nested ZIP entries —
    it "sees" only the last ZIP's contents and drops everything else (including
    the Meesho/Amazon return files we actually need).
    """
    if raw[:6] != _RAR_MAGIC:
        return []
    out: List[Tuple[str, bytes]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="return-rar-") as td:
            rar_path = os.path.join(td, "upload.rar")
            with open(rar_path, "wb") as fh:
                fh.write(raw)
            extract_dir = os.path.join(td, "out")
            os.makedirs(extract_dir, exist_ok=True)
            extracted = False
            unar = shutil.which("unar")
            if unar:
                try:
                    subprocess.run(
                        [unar, "-q", "-o", extract_dir, "-f", rar_path],
                        check=True,
                        capture_output=True,
                        timeout=120,
                    )
                    extracted = True
                except Exception:
                    _log.exception("unar RAR extract failed for return upload")
            if not extracted:
                bsdtar = shutil.which("bsdtar")
                if bsdtar:
                    try:
                        subprocess.run(
                            [bsdtar, "-xf", rar_path, "-C", extract_dir],
                            check=True,
                            capture_output=True,
                            timeout=120,
                        )
                        extracted = True
                    except Exception:
                        _log.exception("bsdtar RAR extract failed for return upload")
            if not extracted:
                return []
            for root, _dirs, files in os.walk(extract_dir):
                for fname in files:
                    if fname == "upload.rar":
                        continue
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, extract_dir).replace("\\", "/")
                    if rel.lower().endswith(".zip"):
                        with open(full, "rb") as fh:
                            out.extend(_extract_zip_members(fh.read(), prefix=f"{rel}/"))
                        continue
                    if not rel.lower().endswith(_DATA_EXTS):
                        continue
                    with open(full, "rb") as fh:
                        out.append((rel, fh.read()))
    except Exception:
        _log.exception("RAR extract failed for return upload")
    return out


def _expand_upload_to_member_files(raw: bytes, filename: str) -> List[Tuple[str, bytes]]:
    """Single data file, or all CSV/Excel members inside a RAR/ZIP archive."""
    name = (filename or "").lower().strip()
    # .xlsx/.xls are ZIP containers — never treat them as multi-file archives.
    if name.endswith(_DATA_EXTS):
        return [(filename or "upload.csv", raw)]
    if name.endswith(_ARCHIVE_EXTS) or raw[:6] == _RAR_MAGIC or raw[:2] == b"PK":
        members: List[Tuple[str, bytes]] = []
        if raw[:6] == _RAR_MAGIC or name.endswith(".rar"):
            members = _extract_rar_members_local(raw)
            if not members:
                try:
                    from ..routers.upload import _extract_rar_files

                    members = _extract_rar_files(raw)
                except Exception:
                    _log.exception("RAR extract failed for return upload")
        if not members and (raw[:2] == b"PK" or name.endswith(".zip")):
            members = _extract_zip_members(raw)
        if members:
            return [
                (n, b)
                for n, b in members
                if n.lower().endswith(_DATA_EXTS)
                and not n.split("/")[-1].startswith(".")
                and not _skip_return_archive_member(n)
            ]
        return []
    return [(filename or "upload.csv", raw)]


def _read_tabular(raw: bytes, filename: str) -> Tuple[pd.DataFrame, Optional[str]]:
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(raw)), None
        return pd.read_csv(BytesIO(raw), encoding_errors="replace"), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def _parse_myntra_seller_returns_csv(
    raw: bytes,
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Myntra Seller Returns Report — one row per return with return_created_date."""
    try:
        df = pd.read_csv(BytesIO(raw), encoding_errors="replace")
    except Exception as e:
        return pd.DataFrame(), str(e)
    if df is None or df.empty:
        return pd.DataFrame(), "Myntra return CSV is empty."
    cols = {str(c).strip().lower(): c for c in df.columns}
    sku_key = None
    for cand in ("seller_sku_code", "seller sku code", "oms_sku", "sku"):
        if cand in cols:
            sku_key = cols[cand]
            break
    if sku_key is None and "myntra_sku_code" in cols:
        sku_key = cols["myntra_sku_code"]
    if sku_key is None:
        return pd.DataFrame(), "Myntra return CSV: missing seller_sku_code / SKU column."
    qty_key = cols.get("quantity") or cols.get("qty")
    if not qty_key:
        return pd.DataFrame(), "Myntra return CSV: missing quantity column."
    if not any(c in cols for c in ("return_created_date", "refunded_date", "order_rto_date")):
        return pd.DataFrame(), "Myntra return CSV: missing return date columns."

    work = pd.DataFrame()
    work["OMS_SKU"] = df[sku_key].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    work["Return_Units"] = pd.to_numeric(df[qty_key], errors="coerce").fillna(0).astype(int)
    work["Return_Date"] = _coalesce_return_date_columns(
        df,
        ("return_created_date", "refunded_date", "order_rto_date"),
    )
    work["Return_Platform"] = "myntra"
    work = work[
        work["OMS_SKU"].str.len().gt(0)
        & work["Return_Units"].gt(0)
        & work["Return_Date"].map(_valid_return_iso)
    ]
    if work.empty:
        return pd.DataFrame(), "No dated Myntra return rows found."
    return (
        work.groupby(
            ["OMS_SKU", "Return_Platform", "Return_Date"], as_index=False
        )["Return_Units"]
        .sum(),
        None,
    )


def _parse_meesho_panel_csv(raw: bytes) -> Tuple[pd.DataFrame, Optional[str]]:
    for skip in range(0, 16):
        try:
            df = pd.read_csv(BytesIO(raw), skiprows=skip, encoding_errors="replace")
        except Exception:
            continue
        if df is None or df.empty:
            continue
        cols = [str(c).strip() for c in df.columns]
        sku_col = pick_csv_column(cols, ("sku", "seller_sku", "oms_sku"))
        qty_col = pick_csv_column(cols, ("qty", "quantity", "return_units"))
        if sku_col and qty_col:
            return df, None
    return pd.DataFrame(), "Meesho return CSV: could not find SKU/Qty header row."


def _parse_meesho_return_table(
    df: pd.DataFrame,
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Meesho Supplier Panel return sheet — dated by per-row "Return Created Date"."""
    sku_col = pick_csv_column(list(df.columns), _SKU_CANDS)
    qty_col = pick_csv_column(list(df.columns), _QTY_CANDS)
    if not sku_col or not qty_col:
        return pd.DataFrame(), None
    out = pd.DataFrame()
    out["OMS_SKU"] = df[sku_col].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    out["Return_Units"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0).astype(int)
    out["Return_Date"] = _coalesce_return_date_columns(
        df, ("return created date", "dispatch date")
    )
    out["Return_Platform"] = "meesho"
    out = out[
        out["OMS_SKU"].str.len().gt(0)
        & out["Return_Units"].gt(0)
        & out["Return_Date"].map(_valid_return_iso)
    ]
    if out.empty:
        return pd.DataFrame(), "No dated Meesho return rows found."
    return (
        out.groupby(
            ["OMS_SKU", "Return_Platform", "Return_Date"], as_index=False
        )["Return_Units"]
        .sum(),
        None,
    )


def _valid_return_iso(d: str) -> bool:
    """Reject epoch placeholders and pre-2018 junk from marketplace exports."""
    s = str(d or "").strip()[:10]
    return bool(re.match(r"^20\d{2}-\d{2}-\d{2}$", s)) and s >= "2018-01-01"


def _coalesce_return_date_columns(df: pd.DataFrame, col_names: tuple[str, ...]) -> pd.Series:
    """Pick first parseable reporting date per row (Flipkart / Myntra return exports)."""
    from .karigar_attendance import _cell_to_report_date

    cols = {str(c).strip().lower(): c for c in df.columns}
    dates = pd.Series([""] * len(df), index=df.index, dtype=str)
    for low in col_names:
        key = cols.get(low)
        if not key:
            continue
        parsed = df[key].map(_cell_to_report_date)
        good = parsed.map(_valid_return_iso)
        fill = dates.str.len().ne(10) & good
        dates = dates.where(~fill, parsed)
    return dates


def _parse_flipkart_returns_xlsx(raw: bytes) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        xl = pd.ExcelFile(BytesIO(raw))
    except Exception as e:
        return pd.DataFrame(), str(e)
    sheet = "Returns" if "Returns" in xl.sheet_names else xl.sheet_names[0]
    try:
        df = pd.read_excel(BytesIO(raw), sheet_name=sheet)
    except Exception as e:
        return pd.DataFrame(), str(e)
    if df is None or df.empty:
        return pd.DataFrame(), "Flipkart Returns sheet is empty."
    cols = [str(c).strip().lower() for c in df.columns]
    if "sku" not in cols or "quantity" not in cols:
        return pd.DataFrame(), "Flipkart Returns sheet missing sku/quantity columns."
    out = pd.DataFrame()
    out["OMS_SKU"] = df["sku"].map(_strip_flipkart_sku)
    out["Return_Units"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    out["Return_Date"] = _coalesce_return_date_columns(
        df,
        (
            "return_completion_date",
            "return_approval_date",
            "return_complete_by_date",
            "return_requested_date",
        ),
    )
    out["Return_Platform"] = "flipkart"
    out = out[
        out["OMS_SKU"].str.len().gt(0)
        & out["Return_Units"].gt(0)
        & out["Return_Date"].map(_valid_return_iso)
    ]
    if out.empty:
        return pd.DataFrame(), "No dated Flipkart return rows found."
    return (
        out.groupby(
            ["OMS_SKU", "Return_Platform", "Return_Date"], as_index=False
        )["Return_Units"]
        .sum(),
        None,
    )


def _parse_amazon_business_return(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    sku_col = pick_csv_column(list(df.columns), ("(child) asin", "child asin", "asin", "sku"))
    qty_col = pick_csv_column(
        list(df.columns),
        ("units refunded", "units refunded – b2b", "units refunded - b2b"),
    )
    if not sku_col or not qty_col:
        return pd.DataFrame(), None
    work = df.copy()
    work["_qty"] = (
        pd.to_numeric(
            work[qty_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    work = work[work["_qty"] > 0]
    if work.empty:
        return pd.DataFrame(), "No Amazon Units Refunded > 0."
    out = pd.DataFrame()
    out["OMS_SKU"] = work[sku_col].astype(str).str.strip()
    out["Return_Units"] = work["_qty"]
    return out.groupby("OMS_SKU", as_index=False)["Return_Units"].sum(), None


def _parse_amazon_mtr_returns(
    df: pd.DataFrame,
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Amazon MTR (B2B/B2C) GST report — only ``Transaction Type == Refund`` rows are returns.

    These reports list every transaction (Shipment, Refund, Cancel, ...) for the
    period; summing ``Quantity`` across all rows would count regular sales as
    returns, so refund rows must be filtered out first.
    """
    cols = {str(c).strip().lower(): c for c in df.columns}
    txn_col = cols.get("transaction type")
    sku_col = cols.get("sku")
    qty_col = cols.get("quantity")
    if not (txn_col and sku_col and qty_col):
        return pd.DataFrame(), None
    work = df[df[txn_col].astype(str).str.strip().str.casefold() == "refund"].copy()
    if work.empty:
        return pd.DataFrame(), "No Amazon MTR Refund rows found."
    work["Return_Units"] = pd.to_numeric(work[qty_col], errors="coerce").fillna(0).astype(int)
    work = work[work["Return_Units"] > 0]
    if work.empty:
        return pd.DataFrame(), "No Amazon MTR Refund rows with positive quantity."
    out = pd.DataFrame()
    out["OMS_SKU"] = work[sku_col].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    out["Return_Units"] = work["Return_Units"]
    out["Return_Date"] = _coalesce_return_date_columns(work, ("invoice date", "order date"))
    out["Return_Platform"] = "amazon"
    out = out[
        out["OMS_SKU"].str.len().gt(0)
        & out["Return_Units"].gt(0)
        & out["Return_Date"].map(_valid_return_iso)
    ]
    if out.empty:
        return pd.DataFrame(), "No dated Amazon MTR refund rows found."
    return (
        out.groupby(
            ["OMS_SKU", "Return_Platform", "Return_Date"], as_index=False
        )["Return_Units"]
        .sum(),
        None,
    )


def _parse_amazon_fba_returns(
    df: pd.DataFrame,
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Amazon FBA "Manage Returns" export — return-date, order-id, sku, ..., quantity."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    sku_col = cols.get("sku")
    qty_col = cols.get("quantity")
    date_col = cols.get("return-date") or cols.get("return date")
    if not (sku_col and qty_col and date_col):
        return pd.DataFrame(), None
    out = pd.DataFrame()
    out["OMS_SKU"] = df[sku_col].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    out["Return_Units"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0).astype(int)
    out["Return_Date"] = _coalesce_return_date_columns(df, (str(date_col).strip().lower(),))
    out["Return_Platform"] = "amazon"
    out = out[
        out["OMS_SKU"].str.len().gt(0)
        & out["Return_Units"].gt(0)
        & out["Return_Date"].map(_valid_return_iso)
    ]
    if out.empty:
        return pd.DataFrame(), "No dated Amazon FBA return rows found."
    return (
        out.groupby(
            ["OMS_SKU", "Return_Platform", "Return_Date"], as_index=False
        )["Return_Units"]
        .sum(),
        None,
    )


def _infer_return_platform_from_filename(filename: str) -> str:
    """Map return export filename to Tier-3 platform key (amazon, myntra, …)."""
    low = (filename or "").lower()
    if "amazon" in low or "businessreport" in low.replace(" ", ""):
        return "amazon"
    if "myntra" in low:
        return "myntra"
    if "meesho" in low:
        return "meesho"
    if "flipkart" in low:
        return "flipkart"
    if "snapdeal" in low:
        return "snapdeal"
    return "unknown"


def _infer_return_brand_from_filename(filename: str) -> str:
    """Best-effort brand/company label from return export filename."""
    import re

    low = (filename or "").lower().replace("_", " ")
    if "akiko" in low:
        return "Akiko"
    if "yash gallery" in low or re.search(r"\byg\b", low):
        return "YG"
    if "raisinghani" in low:
        return "Raisinghani"
    return ""


def _normalize_source_filename(filename: str) -> str:
    import os

    return os.path.basename(str(filename or "").strip())


def _stamp_overlay_source_file(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["Source_File"] = _normalize_source_filename(filename)
    return out


def aggregate_return_overlay_for_use(df: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse stored per-file rows into SKU+platform+date totals for sales/PO."""
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    return _finalize_return_overlay_df(df)


def _overlay_aggregate_totals(df: pd.DataFrame | None) -> tuple[int, int]:
    agg = aggregate_return_overlay_for_use(df)
    if agg is None or agg.empty:
        return 0, 0
    return int(len(agg)), int(pd.to_numeric(agg["Return_Units"], errors="coerce").fillna(0).sum())


def summarize_return_overlay_by_platform(df: pd.DataFrame | None) -> list[dict]:
    agg = aggregate_return_overlay_for_use(df)
    if agg is None or agg.empty or "Return_Platform" not in agg.columns:
        return []
    plat = agg["Return_Platform"].astype(str).str.strip().str.lower()
    plat = plat.replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    work = agg.copy()
    work["Return_Platform"] = plat
    rows: list[dict] = []
    for platform, part in work.groupby("Return_Platform", sort=True):
        rows.append(
            {
                "platform": str(platform),
                "skus": int(part["OMS_SKU"].nunique()),
                "units": int(pd.to_numeric(part["Return_Units"], errors="coerce").fillna(0).sum()),
            }
        )
    return rows


def _source_entry_from_overlay(
    filename: str,
    overlay_part: pd.DataFrame,
    *,
    uploaded_at: str,
) -> dict:
    skus, units = _overlay_aggregate_totals(overlay_part)
    return {
        "filename": _normalize_source_filename(filename),
        "uploaded_at": uploaded_at,
        "platform": _infer_return_platform_from_filename(filename),
        "brand": _infer_return_brand_from_filename(filename),
        "skus": skus,
        "units": units,
    }


def rebuild_return_overlay_sources(sess) -> list[dict]:
    """Rebuild upload registry from stored overlay rows (legacy or after disk restore)."""
    ov = getattr(sess, "po_return_overlay_df", None)
    if ov is None or getattr(ov, "empty", True):
        return []
    uploaded_at = str(getattr(sess, "return_overlay_uploaded_at", "") or "")
    if "Source_File" in ov.columns:
        out: list[dict] = []
        for fn in sorted(ov["Source_File"].astype(str).unique()):
            if not str(fn).strip():
                continue
            part = ov[ov["Source_File"].astype(str) == fn]
            out.append(_source_entry_from_overlay(fn, part, uploaded_at=uploaded_at))
        return out
    fn = str(getattr(sess, "return_overlay_filename", "") or "").strip() or "Return data"
    return [_source_entry_from_overlay(fn, ov, uploaded_at=uploaded_at)]


def _attach_return_platform(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "Return_Platform" in out.columns:
        existing = out["Return_Platform"].astype(str).str.strip().str.lower()
        if (existing != "").all() and not existing.isin(["nan", "none"]).any():
            return out
    out["Return_Platform"] = _infer_return_platform_from_filename(filename)
    return out


def _finalize_return_overlay_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Return_Platform" not in out.columns:
        out["Return_Platform"] = "unknown"
    out["Return_Platform"] = out["Return_Platform"].astype(str).str.strip().str.lower()
    out.loc[out["Return_Platform"].isin(["", "nan", "none"]), "Return_Platform"] = "unknown"
    if "Return_Date" in out.columns:
        out["Return_Date"] = out["Return_Date"].astype(str).str.strip().str[:10]
        out = out[out["Return_Date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
        if out.empty:
            return pd.DataFrame()
        keys = ["OMS_SKU", "Return_Platform", "Return_Date"]
        return out.groupby(keys, as_index=False)["Return_Units"].sum()
    if "Return_Platform" in out.columns:
        out["Return_Platform"] = out["Return_Platform"].astype(str).str.strip().str.lower()
        out.loc[out["Return_Platform"].isin(["", "nan", "none"]), "Return_Platform"] = "unknown"
        return (
            out.groupby(["OMS_SKU", "Return_Platform"], as_index=False)["Return_Units"]
            .sum()
        )
    return out.groupby("OMS_SKU", as_index=False)["Return_Units"].sum()


def _parse_generic_return_table(
    df: pd.DataFrame,
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
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
    return out.groupby("OMS_SKU", as_index=False)["Return_Units"].sum(), None


def _parse_single_return_file(
    raw: bytes,
    filename: str,
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if not raw:
        return pd.DataFrame(), "Empty file."
    low = (filename or "").lower()

    def _finish(part: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        if part is None or part.empty:
            return pd.DataFrame(), None
        part = _stamp_return_filename_fallback(part, filename)
        return _attach_return_platform(part, filename), None

    if "flipkart" in low and low.endswith((".xlsx", ".xls")):
        partial, err = _parse_flipkart_returns_xlsx(raw)
        if partial is not None and not partial.empty:
            if sku_mapping:
                partial["OMS_SKU"] = partial["OMS_SKU"].map(
                    lambda s: canonical_oms_key(s, sku_mapping)
                )
            return _finish(partial)
        if err:
            return pd.DataFrame(), err

    if "meesho" in low and low.endswith(".csv"):
        df, err = _parse_meesho_panel_csv(raw)
        if err:
            return pd.DataFrame(), err
        part, err = _parse_meesho_return_table(df, sku_mapping=sku_mapping)
        if part is not None and not part.empty:
            return _finish(part)
        part, err = _parse_generic_return_table(df, sku_mapping=sku_mapping)
        if err:
            return pd.DataFrame(), err
        return _finish(part)

    if low.endswith(".csv") and (
        "seller_returns" in low or ("myntra" in low and "return" in low)
    ):
        part, err = _parse_myntra_seller_returns_csv(raw, sku_mapping=sku_mapping)
        if part is not None and not part.empty:
            return part, None
        if err and "missing" not in err.lower():
            return pd.DataFrame(), err

    df, read_err = _read_tabular(raw, filename)
    if read_err:
        return pd.DataFrame(), f"Could not read file: {read_err}"
    if df is None or df.empty:
        return pd.DataFrame(), "No rows in file."

    col_low = {str(c).strip().lower() for c in df.columns}
    if "return_created_date" in col_low and (
        "seller_sku_code" in col_low or "myntra_sku_code" in col_low
    ):
        part, err = _parse_myntra_seller_returns_csv(raw, sku_mapping=sku_mapping)
        if part is not None and not part.empty:
            return part, None
        if err:
            return pd.DataFrame(), err

    if {"transaction type", "sku", "quantity"} <= col_low:
        partial, err = _parse_amazon_mtr_returns(df, sku_mapping=sku_mapping)
        if partial is not None and not partial.empty:
            return _finish(partial)
        if err:
            return pd.DataFrame(), err

    if {"return-date", "sku", "quantity"} <= col_low:
        partial, err = _parse_amazon_fba_returns(df, sku_mapping=sku_mapping)
        if partial is not None and not partial.empty:
            return _finish(partial)
        if err:
            return pd.DataFrame(), err

    if "amazon" in low or "businessreport" in low.replace(" ", ""):
        partial, err = _parse_amazon_business_return(df)
        if partial is not None and not partial.empty:
            if sku_mapping:
                partial["OMS_SKU"] = partial["OMS_SKU"].map(
                    lambda s: canonical_oms_key(s, sku_mapping)
                )
            return _finish(partial)
        if err:
            return pd.DataFrame(), err

    if low.endswith((".xlsx", ".xls")):
        try:
            xl = pd.ExcelFile(BytesIO(raw))
            if "Returns" in xl.sheet_names:
                partial, err = _parse_flipkart_returns_xlsx(raw)
                if partial is not None and not partial.empty:
                    if sku_mapping:
                        partial["OMS_SKU"] = partial["OMS_SKU"].map(
                            lambda s: canonical_oms_key(s, sku_mapping)
                        )
                    return _finish(partial)
        except Exception:
            pass

    part, err = _parse_generic_return_table(df, sku_mapping=sku_mapping)
    if err:
        return pd.DataFrame(), err
    return _finish(part)


def parse_return_upload_bytes(
    raw: bytes,
    filename: str,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return DataFrame with OMS_SKU, Return_Units (summed per SKU).

    Accepts a single CSV/Excel, or a RAR/ZIP containing marketplace return exports
    (Amazon Business Report, Myntra, Meesho, Flipkart, etc.).
    """
    if not raw:
        return pd.DataFrame(), "Empty file."

    members = _expand_upload_to_member_files(raw, filename)
    if not members:
        low = (filename or "").lower()
        if low.endswith(_ARCHIVE_EXTS) or raw[:6] == _RAR_MAGIC or raw[:2] == b"PK":
            return (
                pd.DataFrame(),
                "Could not extract return files from archive. On the server, install unar, bsdtar, or p7zip — or upload CSV/Excel directly.",
            )
        members = [(filename or "upload.csv", raw)]

    frames: List[pd.DataFrame] = []
    errors: List[str] = []
    for inner_name, inner_raw in members:
        if _skip_return_archive_member(inner_name):
            continue
        # Keep archive path (e.g. "Akiko Flipkart Return/foo.xlsx") for platform detection.
        part, err = _parse_single_return_file(
            inner_raw, inner_name.replace("\\", "/"), sku_mapping=sku_mapping
        )
        if part is not None and not part.empty:
            frames.append(part)
        elif err:
            errors.append(f"{inner_name.split('/')[-1]}: {err}")

    if not frames:
        detail = "; ".join(errors[:4]) if errors else "No return rows found."
        return pd.DataFrame(), detail

    out = pd.concat(frames, ignore_index=True)
    out = _finalize_return_overlay_df(out)

    if group_by_parent:
        from .helpers import get_parent_sku

        out["OMS_SKU"] = out["OMS_SKU"].map(
            lambda s: str(get_parent_sku(s) or s).strip().upper()
        )
        out = _finalize_return_overlay_df(out)

    sources = len(frames)
    _log.info(
        "Return import: %d SKU(s), %d units from %d file(s)",
        len(out),
        int(out["Return_Units"].sum()),
        sources,
    )
    return out, None


def _stamp_return_filename_fallback(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """When a return table has no per-row dates, anchor rows to the export period in the filename."""
    if df is None or df.empty:
        return df
    if "Return_Date" in df.columns and df["Return_Date"].map(_valid_return_iso).any():
        return df
    as_of = _return_report_date_from_filename(filename)
    if not as_of:
        return df
    out = df.copy()
    out["Return_Date"] = as_of
    return out


def _return_report_date_from_filename(filename: str) -> str:
    """
    Reporting date embedded in return export names.
    Avoids mis-parsing ISO ``2026-06-01`` as DD-MM (→ 2001-06-26).
    """
    fn = (filename or "").strip()
    if not fn:
        return ""
    low = fn.lower()
    iso_dates = re.findall(r"(20\d{2}-\d{2}-\d{2})", fn)
    if "seller_returns" in low or "seller returns" in low:
        # Period export (…_2026-03-01_2026-03-31.csv) — use per-row Return_Date instead.
        return ""
    if iso_dates:
        return iso_dates[-1]
    m = re.search(r"(\d{1,2})[-/_\.](\d{1,2})[-/_\.](\d{2,4})", fn)
    if not m:
        return ""
    d, mth, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        y = "20" + y
    try:
        return pd.to_datetime(f"{d}-{mth}-{y}", dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return ""


def infer_return_overlay_as_of(filename: str, overlay_df: pd.DataFrame | None = None) -> str:
    """Pick reporting date from archive/filename or return table — not upload day."""
    from .karigar_attendance import _cell_to_report_date

    if overlay_df is not None and not overlay_df.empty:
        if "Return_Date" in overlay_df.columns:
            dated = overlay_df["Return_Date"].astype(str).str.strip()
            if dated.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False).any():
                return ""
        for col in ("Report_Date", "Date", "Return_Date", "TxnDate", "Order_Date"):
            if col not in overlay_df.columns:
                continue
            for raw in overlay_df[col].dropna().head(20):
                found = _cell_to_report_date(raw)
                if found:
                    return found
    as_of = _return_report_date_from_filename(filename or "")
    if as_of:
        return as_of
    return ""


def return_overlay_meta_bundle(sess) -> dict:
    ov = getattr(sess, "po_return_overlay_df", None)
    skus, units = _overlay_aggregate_totals(ov)
    sources = list(getattr(sess, "return_overlay_sources", None) or [])
    if not sources and ov is not None and not getattr(ov, "empty", True):
        sources = rebuild_return_overlay_sources(sess)
    return {
        "return_overlay_uploaded_at": str(getattr(sess, "return_overlay_uploaded_at", "") or ""),
        "return_overlay_filename": str(getattr(sess, "return_overlay_filename", "") or ""),
        "return_overlay_skus": skus,
        "return_overlay_units": units,
        "return_overlay_sources": sources,
    }


def persist_return_overlay_meta(sess) -> None:
    """Write return overlay upload metadata next to the warm-cache parquet."""
    import json
    import os

    meta = return_overlay_meta_bundle(sess)
    if not meta.get("return_overlay_skus"):
        return
    try:
        import backend.main as _main

        if not isinstance(_main._warm_cache, dict):
            _main._warm_cache = {}
        _main._warm_cache[_main._RETURN_OVERLAY_META_WARM_KEY] = dict(meta)
        root = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
        path = os.path.join(root, "return_overlay_meta.json")
        os.makedirs(root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)
    except Exception:
        _log.exception("persist return_overlay_meta failed")


def load_return_overlay_meta_from_disk() -> dict:
    import json
    import os

    path = os.path.join(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"), "return_overlay_meta.json")
    if not os.path.isfile(path):
        return {}
    try:
        data = json.loads(open(path, encoding="utf-8").read())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def apply_return_overlay_meta_to_session(sess, meta: dict | None) -> None:
    if not meta:
        return
    if meta.get("return_overlay_uploaded_at"):
        sess.return_overlay_uploaded_at = str(meta["return_overlay_uploaded_at"])
    if meta.get("return_overlay_filename"):
        sess.return_overlay_filename = str(meta["return_overlay_filename"])
    raw_sources = meta.get("return_overlay_sources")
    if isinstance(raw_sources, list):
        sess.return_overlay_sources = [s for s in raw_sources if isinstance(s, dict)]


def ensure_return_overlay_meta_hydrated(sess) -> None:
    """Fill upload timestamp/filename when overlay exists but session meta is missing."""
    ov = getattr(sess, "po_return_overlay_df", None)
    if ov is None or getattr(ov, "empty", True):
        return
    if not list(getattr(sess, "return_overlay_sources", None) or []):
        sess.return_overlay_sources = rebuild_return_overlay_sources(sess)
    if str(getattr(sess, "return_overlay_uploaded_at", "") or "").strip():
        return
    meta = load_return_overlay_meta_from_disk()
    apply_return_overlay_meta_to_session(sess, meta)
    if str(getattr(sess, "return_overlay_uploaded_at", "") or "").strip():
        return
    import json
    import os
    from datetime import datetime, timezone

    root = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
    parquet = os.path.join(root, "po_return_overlay_df.parquet")
    ts: float | None = None
    if os.path.isfile(parquet):
        try:
            ts = os.path.getmtime(parquet)
        except OSError:
            ts = None
    if ts is None:
        manifest_path = os.path.join(root, "_manifest.json")
        if os.path.isfile(manifest_path):
            try:
                manifest = json.loads(open(manifest_path, encoding="utf-8").read())
                saved_at = str(manifest.get("saved_at") or "").strip()
                if saved_at:
                    dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        from zoneinfo import ZoneInfo

                        dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
                    sess.return_overlay_uploaded_at = dt.astimezone(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    )
                    return
            except Exception:
                pass
    if ts is not None:
        sess.return_overlay_uploaded_at = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        )


def clear_return_overlay_meta(sess) -> None:
    sess.return_overlay_uploaded_at = ""
    sess.return_overlay_filename = ""
    sess.return_overlay_sources = []
    import os

    try:
        import backend.main as _main

        if isinstance(_main._warm_cache, dict):
            _main._warm_cache.pop(_main._RETURN_OVERLAY_META_WARM_KEY, None)
    except Exception:
        pass
    path = os.path.join(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"), "return_overlay_meta.json")
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            _log.warning("failed to remove return_overlay_meta.json")


def apply_return_overlay_import(
    sess,
    overlay_df: pd.DataFrame,
    *,
    replace: bool = False,
    filename: str = "",
) -> dict:
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    if overlay_df is None or overlay_df.empty:
        return {"ok": False, "message": "No return rows to import."}
    prepared = _stamp_overlay_source_file(_finalize_return_overlay_df(overlay_df), filename)
    file_key = _normalize_source_filename(filename)
    uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if replace:
        sess.po_return_overlay_df = prepared.copy()
        sess.return_overlay_sources = [
            _source_entry_from_overlay(file_key, prepared, uploaded_at=uploaded_at)
        ]
    else:
        base = getattr(sess, "po_return_overlay_df", pd.DataFrame())
        sources = list(getattr(sess, "return_overlay_sources", None) or [])
        if base is not None and not getattr(base, "empty", True) and file_key:
            if "Source_File" in base.columns:
                base = base[base["Source_File"].astype(str) != file_key]
            sources = [
                s
                for s in sources
                if _normalize_source_filename(str(s.get("filename") or "")) != file_key
            ]
        sess.po_return_overlay_df = pd.concat([base, prepared], ignore_index=True)
        sources.append(_source_entry_from_overlay(file_key, prepared, uploaded_at=uploaded_at))
        sess.return_overlay_sources = sources
    as_of = infer_return_overlay_as_of(filename, overlay_df)
    if not as_of and (
        overlay_df is None
        or "Return_Date" not in overlay_df.columns
        or overlay_df["Return_Date"].astype(str).str.len().lt(10).all()
    ):
        # Legacy bundle with no per-row dates — anchor refunds to prior IST day.
        as_of = (datetime.now(ZoneInfo("Asia/Kolkata")).date() - timedelta(days=1)).isoformat()
    sess.return_overlay_as_of = as_of or ""
    sess.return_overlay_uploaded_at = uploaded_at
    sess.return_overlay_filename = file_key
    file_skus, file_units = _overlay_aggregate_totals(prepared)
    n, units = _overlay_aggregate_totals(sess.po_return_overlay_df)
    try:
        persist_return_overlay_meta(sess)
    except Exception:
        _log.exception("persist_return_overlay_meta after import failed")
    sess._quarterly_cache.clear()
    action = "Replaced" if replace else "Added"
    return {
        "ok": True,
        "message": (
            f"{action} {file_key or 'return file'}: {file_skus} SKU(s), {file_units:,} units. "
            f"Combined total: {n} SKU(s), {units:,} return units across all uploads."
        ),
        "skus": n,
        "total_units": units,
        "file_skus": file_skus,
        "file_units": file_units,
    }
