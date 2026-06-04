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


def _extract_rar_members_local(raw: bytes) -> List[Tuple[str, bytes]]:
    """Extract RAR via bsdtar without importing the upload router."""
    if raw[:6] != _RAR_MAGIC:
        return []
    bsdtar = shutil.which("bsdtar")
    if not bsdtar:
        return []
    out: List[Tuple[str, bytes]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="return-rar-") as td:
            rar_path = os.path.join(td, "upload.rar")
            with open(rar_path, "wb") as fh:
                fh.write(raw)
            subprocess.run(
                [bsdtar, "-xf", rar_path, "-C", td],
                check=True,
                capture_output=True,
                timeout=120,
            )
            for root, _dirs, files in os.walk(td):
                for fname in files:
                    if fname == "upload.rar":
                        continue
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, td).replace("\\", "/")
                    if not rel.lower().endswith(_DATA_EXTS):
                        continue
                    with open(full, "rb") as fh:
                        out.append((rel, fh.read()))
    except Exception:
        _log.exception("bsdtar RAR extract failed for return upload")
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
            try:
                with zipfile.ZipFile(BytesIO(raw)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        inner_name = info.filename.replace("\\", "/")
                        if not inner_name.lower().endswith(_DATA_EXTS):
                            continue
                        members.append((inner_name, zf.read(info)))
            except Exception:
                _log.exception("ZIP extract failed for return upload")
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
    from .karigar_attendance import _cell_to_report_date

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
    date_key = None
    for cand in ("return_created_date", "refunded_date", "order_rto_date"):
        if cand in cols:
            date_key = cols[cand]
            break
    if not date_key:
        return pd.DataFrame(), "Myntra return CSV: missing return_created_date."

    work = pd.DataFrame()
    work["OMS_SKU"] = df[sku_key].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    work["Return_Units"] = pd.to_numeric(df[qty_key], errors="coerce").fillna(0).astype(int)
    work["Return_Date"] = df[date_key].map(_cell_to_report_date)
    work["Return_Platform"] = "myntra"
    work = work[
        work["OMS_SKU"].str.len().gt(0)
        & work["Return_Units"].gt(0)
        & work["Return_Date"].str.len().eq(10)
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
    out = out[out["OMS_SKU"].str.len() > 0]
    out = out[out["Return_Units"] > 0]
    if out.empty:
        return pd.DataFrame(), "No positive Flipkart return rows."
    return out.groupby("OMS_SKU", as_index=False)["Return_Units"].sum(), None


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


def _attach_return_platform(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
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


def apply_return_overlay_import(
    sess,
    overlay_df: pd.DataFrame,
    *,
    replace: bool = True,
    filename: str = "",
) -> dict:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    if overlay_df is None or overlay_df.empty:
        return {"ok": False, "message": "No return rows to import."}
    overlay_df = _finalize_return_overlay_df(overlay_df)
    if replace:
        sess.po_return_overlay_df = overlay_df.copy()
    else:
        base = getattr(sess, "po_return_overlay_df", pd.DataFrame())
        merged = pd.concat([base, overlay_df], ignore_index=True)
        sess.po_return_overlay_df = _finalize_return_overlay_df(merged)
    as_of = infer_return_overlay_as_of(filename, overlay_df)
    if not as_of and (
        overlay_df is None
        or "Return_Date" not in overlay_df.columns
        or overlay_df["Return_Date"].astype(str).str.len().lt(10).all()
    ):
        # Legacy bundle with no per-row dates — anchor refunds to prior IST day.
        as_of = (datetime.now(ZoneInfo("Asia/Kolkata")).date() - timedelta(days=1)).isoformat()
    sess.return_overlay_as_of = as_of or ""
    n = int(len(sess.po_return_overlay_df))
    units = int(sess.po_return_overlay_df["Return_Units"].sum())
    sess._quarterly_cache.clear()
    return {
        "ok": True,
        "message": (
            f"Return sheet: {n} SKU(s), {units:,} return units. "
            "Dashboard net sales and PO return overlay updated."
        ),
        "skus": n,
        "total_units": units,
    }
