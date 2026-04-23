"""
Upload router — all file ingestion endpoints.
"""
import gc
import io
import os
import re
import zipfile
import shutil
import subprocess
import tempfile
from typing import List, Optional, Set

import logging
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..session import store, resume_auto_data_restore
from ..models.schemas import UploadResponse
from ..services.sku_mapping import parse_sku_mapping
from ..services.mtr import load_mtr_from_zip, load_mtr_from_extracted_files, parse_mtr_csv
from ..services.myntra import load_myntra_from_zip
from ..services.meesho import (
    load_meesho_from_zip,
    looks_like_meesho_order_export,
    parse_meesho_order_export_xlsx,
)
from ..services.flipkart import load_flipkart_from_zip
from ..services.snapdeal import load_snapdeal_from_zip
from ..services.inventory import load_inventory_consolidated
from ..services.sales import build_sales_df, list_sku_mapping_gaps
from ..services.existing_po import parse_existing_po
from ..services.github_cache import save_cache_to_drive
from ..services.daily_store import save_daily_file, merge_platform_data as _merge_platform_data
from ..services.helpers import apply_dsr_segment_from_upload_filename


router = APIRouter()

_log = logging.getLogger(__name__)


def _session_data_changed(sess) -> None:
    """Undo pause_auto_data_restore after uploads / builds so Tier-3 merge and warm cache work again."""
    resume_auto_data_restore(sess)

_RAR_MAGIC = b"Rar!\x1a\x07"


def _extract_rar_files(rar_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    Extract all files from a RAR archive (incl. RAR5). Returns [(relative path, bytes), ...].

    Tries, in order: **bsdtar** (libarchive), **unar**, **7zz** / **7z**. Many Linux images
    have no ``bsdtar``, or libarchive without RAR — then Meesho CSVs inside a RAR never
    reach the dashboard. ``unar`` / p7zip are common Homebrew / apt packages.
    """
    tmpdir = tempfile.mkdtemp(prefix="rar_daily_")
    rar_path = os.path.join(tmpdir, "upload.rar")
    extract_root = os.path.join(tmpdir, "_out")

    def _collect(out_root: str) -> list[tuple[str, bytes]]:
        result: list[tuple[str, bytes]] = []
        if not os.path.isdir(out_root):
            return result
        for walk_root, _dirs, files in os.walk(out_root):
            for fname in files:
                if fname == "upload.rar":
                    continue
                full = os.path.join(walk_root, fname)
                try:
                    rel = os.path.relpath(full, out_root).replace("\\", "/")
                except ValueError:
                    continue
                with open(full, "rb") as fh:
                    result.append((rel, fh.read()))
        return result

    def _reset_out() -> None:
        shutil.rmtree(extract_root, ignore_errors=True)
        os.makedirs(extract_root, exist_ok=True)

    try:
        with open(rar_path, "wb") as f:
            f.write(rar_bytes)

        os.makedirs(extract_root, exist_ok=True)
        errors: list[str] = []

        bsdtar = shutil.which("bsdtar")
        if bsdtar:
            try:
                r = subprocess.run(
                    [bsdtar, "xf", rar_path, "-C", extract_root],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                got = _collect(extract_root)
                if r.returncode == 0 and got:
                    _log.info("RAR extract: bsdtar OK (%d files)", len(got))
                    return got
                err = (r.stderr or r.stdout or "").strip()
                if r.returncode != 0:
                    errors.append(f"bsdtar exit {r.returncode}: {err[:320]}")
                else:
                    errors.append("bsdtar: 0 files extracted")
            except subprocess.TimeoutExpired:
                errors.append("bsdtar: timeout")
            except Exception as e:
                errors.append(f"bsdtar: {e}")
            _reset_out()

        unar = shutil.which("unar")
        if unar:
            try:
                r = subprocess.run(
                    [unar, "-o", extract_root, "-f", rar_path],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                got = _collect(extract_root)
                if got:
                    _log.info("RAR extract: unar OK (%d files)", len(got))
                    return got
                err = (r.stderr or r.stdout or "").strip()
                errors.append(f"unar exit {r.returncode}: {err[:320]}")
            except subprocess.TimeoutExpired:
                errors.append("unar: timeout")
            except Exception as e:
                errors.append(f"unar: {e}")
            _reset_out()

        for bin_name in ("7zz", "7z"):
            seven = shutil.which(bin_name)
            if not seven:
                continue
            try:
                r = subprocess.run(
                    [seven, "x", "-y", f"-o{extract_root}", rar_path],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                got = _collect(extract_root)
                if got:
                    _log.info("RAR extract: %s OK (%d files)", bin_name, len(got))
                    return got
                err = (r.stderr or r.stdout or "").strip()
                errors.append(f"{bin_name} exit {r.returncode}: {err[:320]}")
            except subprocess.TimeoutExpired:
                errors.append(f"{bin_name}: timeout")
            except Exception as e:
                errors.append(f"{bin_name}: {e}")
            _reset_out()

        hint = (
            "Install one of: bsdtar (libarchive with RAR), unar (`brew install unar`), "
            "or p7zip (`apt install p7zip-full` / `brew install p7zip`). "
            "Or extract the RAR locally and upload the Meesho `Orders_*.csv` files as a ZIP."
        )
        raise ValueError(
            "Cannot extract RAR — " + (" | ".join(errors) if errors else "no files produced") + ". " + hint
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _auto_save_cache(sess) -> None:
    """Run in background after build-sales: silently push session to GitHub Releases."""
    session_data = {
        "sales_df":             sess.sales_df,
        "mtr_df":               sess.mtr_df,
        "meesho_df":            sess.meesho_df,
        "myntra_df":            sess.myntra_df,
        "flipkart_df":          sess.flipkart_df,
        "snapdeal_df":          sess.snapdeal_df,
        "sku_mapping":          sess.sku_mapping,
        "inventory_df_variant": sess.inventory_df_variant,
        "inventory_df_parent":  sess.inventory_df_parent,
    }
    ok, msg = save_cache_to_drive(session_data)
    if ok:
        _log.info("Auto-save cache succeeded: %s", msg)
    else:
        _log.warning("Auto-save cache skipped/failed: %s", msg)

    sid = getattr(sess, "_persist_sid", None)
    if sid:
        try:
            from ..db.forecast_session_pg import persist_session_bundle

            if persist_session_bundle(sid, sess):
                _log.info("PostgreSQL session snapshot saved (%s…)", sid[:8])
        except Exception:
            _log.exception("PostgreSQL session snapshot failed")


def _get_session(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")
    return sess


# ── SKU Mapping ───────────────────────────────────────────────

@router.post("/sku-mapping", response_model=UploadResponse)
async def upload_sku_mapping(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    sess = _get_session(request)
    try:
        file_bytes = await file.read()
        mapping = parse_sku_mapping(file_bytes)
        sess.sku_mapping = mapping

        had_platform = (
            not sess.mtr_df.empty
            or not sess.myntra_df.empty
            or not sess.meesho_df.empty
            or not sess.flipkart_df.empty
            or not sess.snapdeal_df.empty
        )
        if had_platform:
            sess.sales_df = build_sales_df(
                mtr_df=sess.mtr_df,
                myntra_df=sess.myntra_df,
                meesho_df=sess.meesho_df,
                flipkart_df=sess.flipkart_df,
                snapdeal_df=sess.snapdeal_df,
                sku_mapping=mapping,
            )
            background_tasks.add_task(_auto_save_cache, sess)

        gaps = list_sku_mapping_gaps(sess.sales_df, mapping)
        msg = f"SKU mapping loaded: {len(mapping):,} entries"
        if had_platform:
            msg += f"; sales rebuilt ({len(sess.sales_df):,} rows)"
        if gaps:
            msg += (
                f". Warning: {len(gaps)} SKU(s) in sales are not in this map "
                f"(as seller key or OMS value) — add them to the master or fix typos."
            )

        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=msg,
            sku_count=len(mapping),
            unmapped_skus=gaps or None,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse SKU mapping: {e}")


# ── Amazon MTR ────────────────────────────────────────────────

@router.post("/mtr", response_model=UploadResponse)
async def upload_mtr(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        raw = await file.read()
        fn = (file.filename or "").lower()
        if raw[:6] == _RAR_MAGIC or fn.endswith(".rar"):
            inner = _extract_rar_files(raw)
            df, csv_count, skipped = load_mtr_from_extracted_files(inner)
        else:
            df, csv_count, skipped = load_mtr_from_zip(raw)
        del raw
        gc.collect()

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No valid CSV files found. Issues: {'; '.join(skipped[:5])}",
            )

        save_daily_file("amazon", file.filename or "mtr-upload.zip", df)
        sess.mtr_df = _merge_platform_data(
            sess.mtr_df, df, "amazon", source_filename=file.filename or None,
        )
        total = len(sess.mtr_df)
        years = sorted(sess.mtr_df["Date"].dt.year.dropna().unique().astype(int).tolist())
        background_tasks.add_task(_auto_save_cache, sess)
        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=f"Amazon MTR: added {len(df):,} rows ({csv_count} files). Total: {total:,} rows.",
            rows=total,
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse MTR archive (ZIP/RAR): {e}")


# ── Myntra ────────────────────────────────────────────────────

@router.post("/myntra", response_model=UploadResponse)
async def upload_myntra(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return UploadResponse(ok=False, message="Upload SKU Mapping first.")
    try:
        zip_bytes = await file.read()
        df, csv_count, skipped = load_myntra_from_zip(
            zip_bytes, sess.sku_mapping, file.filename or None,
        )
        del zip_bytes
        gc.collect()

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
            )

        save_daily_file("myntra", file.filename or "myntra-upload.zip", df)
        sess.myntra_df = _merge_platform_data(
            sess.myntra_df, df, "myntra", source_filename=file.filename or None,
        )
        total = len(sess.myntra_df)
        years = sorted(sess.myntra_df["Date"].dt.year.dropna().unique().astype(int).tolist())
        background_tasks.add_task(_auto_save_cache, sess)
        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=f"Myntra: added {len(df):,} rows ({csv_count} CSVs). Total: {total:,} rows.",
            rows=total,
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Myntra ZIP: {e}")


# ── Meesho ────────────────────────────────────────────────────

@router.post("/meesho", response_model=UploadResponse)
async def upload_meesho(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        raw_bytes = await file.read()
        fname = (file.filename or "").lower()

        # Unified sales Excel export (TxnDate, Sku, Sku.1 listing fallback) — must run before PK/ZIP logic
        if fname.endswith((".xlsx", ".xls")):
            df, msg = parse_meesho_order_export_xlsx(raw_bytes)
            if not df.empty:
                df = apply_dsr_segment_from_upload_filename(
                    df, file.filename or None, "Meesho",
                )
                save_daily_file("meesho", file.filename or "meesho-order.xlsx", df)
                sess.meesho_df = _merge_platform_data(
                    sess.meesho_df, df, "meesho", source_filename=file.filename or None,
                )
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df,
                    myntra_df=sess.myntra_df,
                    meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df,
                    snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping,
                )
                total = len(sess.meesho_df)
                years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
                skus = int((sess.meesho_df["SKU"].astype(str).str.strip() != "").sum())
                background_tasks.add_task(_auto_save_cache, sess)
                _session_data_changed(sess)
                return UploadResponse(
                    ok=True,
                    message=f"Meesho order export (Excel): added {len(df):,} rows ({skus:,} with SKU). Total: {total:,} rows.",
                    rows=total,
                    years=years,
                )
            return UploadResponse(
                ok=False,
                message=(
                    f"Excel is not a recognised Meesho sales export ({msg}). "
                    "Expected columns TxnDate, Transaction Type, Sku, Quantity (and optional Sku.1 listing). "
                    "Or upload supplier Order CSV / monthly ZIP."
                ),
            )

        # Auto-detect: CSV order report vs ZIP (TCS / monthly archive)
        is_csv = fname.endswith(".csv") or (not fname.endswith(".zip") and raw_bytes[:3] != b"PK\x03")

        if is_csv:
            from ..services.meesho import parse_meesho_csv
            df, msg = parse_meesho_csv(raw_bytes)
            del raw_bytes
            gc.collect()
            if df.empty:
                return UploadResponse(ok=False, message=f"Meesho CSV parse error: {msg}")
            df = apply_dsr_segment_from_upload_filename(df, file.filename or None, "Meesho")
            save_daily_file("meesho", file.filename or "meesho-orders.csv", df)
            sess.meesho_df = _merge_platform_data(
                sess.meesho_df, df, "meesho", source_filename=file.filename or None,
            )
            sess.sales_df  = build_sales_df(
                mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                sku_mapping=sess.sku_mapping,
            )
            total = len(sess.meesho_df)
            years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
            skus  = int((sess.meesho_df["SKU"].astype(str).str.strip() != "").sum()) if "SKU" in sess.meesho_df.columns else 0
            background_tasks.add_task(_auto_save_cache, sess)
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=f"Meesho Order CSV: added {len(df):,} rows ({skus:,} with SKU). Total: {total:,} rows.",
                rows=total,
                years=years,
            )
        else:
            df, zip_count, skipped = load_meesho_from_zip(
                raw_bytes, source_filename=file.filename or None,
            )
            del raw_bytes
            gc.collect()
            if df.empty:
                return UploadResponse(
                    ok=False,
                    message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )
            save_daily_file("meesho", file.filename or "meesho-upload.zip", df)
            sess.meesho_df = _merge_platform_data(
                sess.meesho_df, df, "meesho", source_filename=file.filename or None,
            )
            sess.sales_df  = build_sales_df(
                mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                sku_mapping=sess.sku_mapping,
            )
            total = len(sess.meesho_df)
            years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
            background_tasks.add_task(_auto_save_cache, sess)
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=f"Meesho: added {len(df):,} rows ({zip_count} monthly ZIPs). Total: {total:,} rows.",
                rows=total,
                years=years,
            )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Meesho file: {e}")


# ── Flipkart ──────────────────────────────────────────────────

@router.post("/flipkart", response_model=UploadResponse)
async def upload_flipkart(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return UploadResponse(ok=False, message="Upload SKU Mapping first.")
    tmp_path: Optional[str] = None
    try:
        # Stream to disk so multi-year Tier-1 ZIPs are not held wholly in RAM (reduces OOM → 502).
        _fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(_fd)
        chunk_size = 8 * 1024 * 1024
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
        df, xlsx_count, skipped = load_flipkart_from_zip(
            tmp_path, sess.sku_mapping, source_filename=file.filename or None,
        )
        gc.collect()

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
            )

        save_daily_file("flipkart", file.filename or "flipkart-upload.zip", df)
        sess.flipkart_df = _merge_platform_data(
            sess.flipkart_df, df, "flipkart", source_filename=file.filename or None,
        )
        total = len(sess.flipkart_df)
        years = sorted(sess.flipkart_df["Date"].dt.year.dropna().unique().astype(int).tolist())
        background_tasks.add_task(_auto_save_cache, sess)
        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=f"Flipkart: added {len(df):,} rows ({xlsx_count} files). Total: {total:,} rows.",
            rows=total,
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Flipkart ZIP: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Snapdeal ──────────────────────────────────────────────────

@router.post("/snapdeal", response_model=UploadResponse)
async def upload_snapdeal(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        zip_bytes = await file.read()
        df, file_count, skipped, parse_info = load_snapdeal_from_zip(zip_bytes, sess.sku_mapping, file.filename or "upload")
        del zip_bytes
        gc.collect()

        # Store parse diagnostics (merge with existing)
        sess.snapdeal_parse_info.update(parse_info)

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
            )

        save_daily_file("snapdeal", file.filename or "snapdeal-upload.zip", df)

        # Snapdeal may not have OrderId — dedup on all columns
        if sess.snapdeal_df.empty:
            sess.snapdeal_df = df
        else:
            sess.snapdeal_df = pd.concat([sess.snapdeal_df, df], ignore_index=True).drop_duplicates()

        # Rebuild sales_df if SKU mapping is loaded
        if sess.sku_mapping:
            from ..services.sales import build_sales_df
            sess.sales_df = build_sales_df(
                mtr_df=sess.mtr_df,
                myntra_df=sess.myntra_df,
                meesho_df=sess.meesho_df,
                flipkart_df=sess.flipkart_df,
                snapdeal_df=sess.snapdeal_df,
                sku_mapping=sess.sku_mapping,
            )
            sess._quarterly_cache.clear()

        years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())
        background_tasks.add_task(_auto_save_cache, sess)
        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=f"Snapdeal loaded: {len(df):,} rows from {file_count} file(s)"
                    + (f". Warnings: {'; '.join(skipped[:3])}" if skipped else ""),
            rows=len(df),
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Snapdeal ZIP: {e}")


# ── Inventory ─────────────────────────────────────────────────

def _detect_inventory_type(filename: str, content_bytes: bytes) -> str:
    """
    Guess inventory file type from filename and first ~2 KB of content.
    Returns: 'oms', 'flipkart', 'myntra', 'amazon', 'rar'
    """
    fn = filename.lower()

    # RAR by magic bytes or extension
    if content_bytes[:6] == _RAR_MAGIC or fn.endswith(".rar"):
        return "rar"

    # Filename hints
    if "myntra" in fn:
        return "myntra"
    if "flipkart" in fn or fn.startswith("fk"):
        return "flipkart"
    if "amazon" in fn or fn.startswith("amz"):
        return "amazon"

    # Content-based detection
    try:
        # XLSX files are ZIP containers and won't decode to useful text.
        # Probe headers via pandas to classify Flipkart/Myntra/OMS uploads.
        if filename.lower().endswith((".xlsx", ".xls")) or content_bytes[:4] == b"PK\x03\x04":
            try:
                xdf = pd.read_excel(io.BytesIO(content_bytes), nrows=5)
                cols = [str(c).strip().lower() for c in xdf.columns]
                joined = " | ".join(cols)
                if "seller sku code" in joined or ("style id" in joined and "inventory count" in joined):
                    return "myntra"
                if "live on website" in joined or ("sku" in joined and "available to promise" in joined):
                    return "flipkart"
                if "item skucode" in joined or "buffer stock" in joined:
                    return "oms"
            except Exception:
                pass

        text = content_bytes[:2000].decode("utf-8", errors="ignore").lower()
        if "item skucode" in text or "buffer stock" in text:
            return "oms"
        if "combo sku code" in text and "combo qty stock" in text:
            return "oms"
        if "seller sku code" in text or ("style id" in text and "inventory count" in text):
            return "myntra"
        if "live on website" in text:
            return "flipkart"
        if "msku" in text and "ending warehouse balance" in text:
            return "amazon"
        if "merchant sku" in text and "shipped" in text:
            return "amazon"
    except Exception:
        pass

    return "oms"  # safe default


@router.post("/inventory-auto")
async def upload_inventory_auto(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Drop any mix of inventory files — type auto-detected from filename/content.
    Accepts: OMS inventory CSV/XLSX, Flipkart CSV, Myntra CSV, Amazon RAR/CSV.
    """
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    oms_bytes_list: list[bytes] = []
    fk_bytes_list: list[bytes] = []
    myntra_bytes_list: list[bytes] = []
    amz_bytes = None
    detected: list[str] = []

    for file in files:
        raw = await file.read()
        fname = file.filename or ""
        inv_type = _detect_inventory_type(fname, raw)

        if inv_type == "rar":
            amz_bytes = raw
            detected.append(f"RAR archive ({fname})")
        elif inv_type == "flipkart":
            fk_bytes_list.append(raw)
            detected.append(f"Flipkart ({fname})")
        elif inv_type == "myntra":
            myntra_bytes_list.append(raw)
            detected.append(f"Myntra ({fname})")
        elif inv_type == "amazon":
            amz_bytes = raw
            detected.append(f"Amazon ({fname})")
        else:  # oms or unknown
            oms_bytes_list.append(raw)
            detected.append(f"OMS ({fname})")

    fk_bytes = fk_bytes_list or None
    myntra_bytes = myntra_bytes_list if myntra_bytes_list else None

    if not any([oms_bytes_list, fk_bytes, myntra_bytes, amz_bytes]):
        return JSONResponse(content={"ok": False, "message": "No files provided."})

    try:
        df_variant, debug = load_inventory_consolidated(
            oms_bytes_list or None, fk_bytes, myntra_bytes, amz_bytes, sess.sku_mapping,
            group_by_parent=False, return_debug=True,
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Parse error: {e}"})

    # Always replace inventory snapshot for this upload batch.
    # This prevents stale values after accidental duplicate file uploads.
    # Users can upload the full current set again to refresh all sources cleanly.

    # Rebuild parent view from merged variant DF (group by parent SKU)
    try:
        from ..services.helpers import get_parent_sku
        df_parent = df_variant.copy()
        inv_cols = [
            c for c in df_parent.columns
            if c.endswith("_Inventory") or c.endswith("_Live") or c.endswith("_InTransit")
               or c in ("Buffer_Stock", "Marketplace_Total", "Total_Inventory")
        ]
        df_parent["Parent_SKU"] = df_parent["OMS_SKU"].apply(get_parent_sku)
        df_parent = (
            df_parent.groupby("Parent_SKU")[inv_cols].sum()
            .reset_index()
            .rename(columns={"Parent_SKU": "OMS_SKU"})
        )
    except Exception:
        df_parent = df_variant  # fallback

    sess.inventory_df_variant = df_variant
    sess.inventory_df_parent  = df_parent
    sess.inventory_debug      = debug   # persist so /data/inventory can expose it
    _session_data_changed(sess)
    background_tasks.add_task(_auto_save_cache, sess)

    parts = [f"{len(df_variant):,} total SKUs"]
    for src, info in debug.items():
        parts.append(f"{src}: {info}")
    return JSONResponse(content={
        "ok":      True,
        "message": " | ".join(parts),
        "rows":    len(df_variant),
        "debug":   debug,
        "detected": detected,
        "v":       "inv-v8",   # RAR: all Myntra/FK/OMS/Amazon CSVs + byte dedupe; multi-Myntra auto-upload
    })


@router.post("/inventory")
async def upload_inventory(
    request: Request,
    background_tasks: BackgroundTasks,
    oms_file:    List[UploadFile] = File(default=[]),
    fk_file:     Optional[UploadFile] = File(None),
    myntra_file: Optional[UploadFile] = File(None),
    amz_file:    Optional[UploadFile] = File(None),
):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    oms_b_list = [await f.read() for f in oms_file] if oms_file else []
    fk_b     = await fk_file.read()    if fk_file     else None
    myntra_b = await myntra_file.read() if myntra_file else None
    amz_b    = await amz_file.read()   if amz_file    else None

    if not any([oms_b_list, fk_b, myntra_b, amz_b]):
        return JSONResponse(content={"ok": False, "message": "No files provided."})

    try:
        df_variant, debug = load_inventory_consolidated(
            oms_b_list or None, fk_b, myntra_b, amz_b, sess.sku_mapping,
            group_by_parent=False, return_debug=True,
        )
        df_parent = load_inventory_consolidated(
            oms_b_list or None, fk_b, myntra_b, amz_b, sess.sku_mapping, group_by_parent=True
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Parse error: {e}"})

    sess.inventory_df_variant = df_variant
    sess.inventory_df_parent  = df_parent
    _session_data_changed(sess)
    background_tasks.add_task(_auto_save_cache, sess)

    # Build human-readable breakdown
    parts = [f"{len(df_variant):,} total SKUs"]
    for src, info in debug.items():
        parts.append(f"{src}: {info}")
    return JSONResponse(content={
        "ok": True,
        "message": " | ".join(parts),
        "rows": len(df_variant),
        "debug": debug,
    })


# ── Amazon B2C (single CSV) ────────────────────────────────────

@router.post("/amazon-b2c", response_model=UploadResponse)
async def upload_amazon_b2c(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        csv_bytes = await file.read()
        df, msg = parse_mtr_csv(csv_bytes, file.filename or "b2c.csv")
        del csv_bytes
        gc.collect()

        if df.empty:
            return UploadResponse(ok=False, message=f"B2C parse failed: {msg}")

        sess.mtr_df = _merge_platform_data(
            sess.mtr_df, df, "amazon", source_filename=file.filename or None,
        )
        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=f"Amazon B2C loaded: {len(df):,} rows. {msg if msg != 'OK' else ''}".strip(),
            rows=len(df),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse B2C CSV: {e}")


# ── Amazon B2B (single CSV) ────────────────────────────────────

@router.post("/amazon-b2b", response_model=UploadResponse)
async def upload_amazon_b2b(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        csv_bytes = await file.read()
        df, msg = parse_mtr_csv(csv_bytes, file.filename or "b2b.csv")
        del csv_bytes
        gc.collect()

        if df.empty:
            return UploadResponse(ok=False, message=f"B2B parse failed: {msg}")

        sess.mtr_df = _merge_platform_data(
            sess.mtr_df, df, "amazon", source_filename=file.filename or None,
        )
        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=f"Amazon B2B loaded: {len(df):,} rows. {msg if msg != 'OK' else ''}".strip(),
            rows=len(df),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse B2B CSV: {e}")


# ── Existing PO Sheet ──────────────────────────────────────────

@router.post("/existing-po", response_model=UploadResponse)
async def upload_existing_po(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        file_bytes = await file.read()
        df = parse_existing_po(file_bytes, file.filename or "existing_po.xlsx")
        sess.existing_po_df = df
        _session_data_changed(sess)
        return UploadResponse(
            ok=True,
            message=f"Existing PO loaded: {len(df):,} SKUs with pipeline quantities.",
            rows=len(df),
        )
    except Exception as e:
        _log.warning("existing-po parse failed: %s", e, exc_info=True)
        return UploadResponse(ok=False, message=f"Failed to parse Existing PO: {e}")


@router.post("/cogs", response_model=UploadResponse)
async def upload_cogs(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        from ..services.finance import parse_cogs_sheet
        file_bytes = await file.read()
        df = parse_cogs_sheet(file_bytes, file.filename or "cogs.xlsx")
        sess.cogs_df = df
        _session_data_changed(sess)
        return UploadResponse(ok=True, message=f"COGS sheet loaded: {len(df):,} SKUs.", rows=len(df))
    except ValueError as e:
        return UploadResponse(ok=False, message=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse COGS sheet: {e}")


# ── Daily Orders — Auto-detect (drop all files, we figure it out) ─

def _zip_member_paths(file_bytes: bytes) -> tuple[list[str], str]:
    """Non-directory ZIP members (no __MACOSX); returns (paths, space-joined lowercase)."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = [
                n for n in zf.namelist()
                if "__MACOSX" not in n and not n.endswith("/")
            ]
        return names, " ".join(n.lower() for n in names)
    except Exception:
        return [], ""


def _zip_is_meesho_monthly(file_bytes: bytes, outer_fn: str) -> bool:
    if "meesho" in outer_fn.lower():
        return True
    _names, joined = _zip_member_paths(file_bytes)
    return "tcs_sales" in joined or "forwardreports" in joined


def _zip_is_amazon_mtr_master(file_bytes: bytes) -> bool:
    try:
        probe, n_csv, _sk = load_mtr_from_zip(file_bytes)
        return not probe.empty and n_csv > 0
    except Exception:
        return False


def _detect_platform_zip(file_bytes: bytes, fn: str) -> str:
    """
    Distinguish Meesho monthly ZIP vs Amazon MTR master ZIP (nested CSV/ZIPs).
    Used only when the **entire** archive is passed to ``_handle_one`` (see daily-auto ZIP routing).
    """
    if "meesho" in fn:
        return "meesho"
    _names, joined = _zip_member_paths(file_bytes)
    if "tcs_sales" in joined or "forwardreports" in joined:
        return "meesho"
    try:
        probe, n_csv, _sk = load_mtr_from_zip(file_bytes)
        if not probe.empty and n_csv > 0:
            return "amazon_mtr_zip"
    except Exception:
        pass
    return "meesho"


def _zip_is_flipkart_spreadsheet_bundle(file_bytes: bytes, outer_fn: str) -> bool:
    """ZIP of XLSX/XLSB only (typical Flipkart seller export) — no CSV members."""
    if "flipkart" in outer_fn.lower():
        return True
    names, _joined = _zip_member_paths(file_bytes)
    if not names:
        return False
    has_sheet = any(n.lower().endswith((".xlsx", ".xls", ".xlsb")) for n in names)
    has_csv = any(n.lower().endswith(".csv") for n in names)
    return has_sheet and not has_csv


def _detect_platform(filename: str, file_bytes: bytes) -> str:
    """
    Guess platform from path/filename + file contents.
    Returns: 'amazon_b2c', 'amazon_b2b', 'amazon_mtr_zip', 'myntra', 'meesho',
    'meesho_csv', 'meesho_order_xlsx', 'flipkart', 'snapdeal', 'unknown'
    """
    fn = (filename or "").lower().replace("\\", "/")

    if "snapdeal" in fn:
        return "snapdeal"

    if fn.endswith(".zip"):
        return _detect_platform_zip(file_bytes, fn)

    if fn.endswith(".xlsb"):
        return "flipkart"

    if fn.endswith((".xlsx", ".xls")):
        try:
            peek = pd.read_excel(io.BytesIO(file_bytes), nrows=40)
            if looks_like_meesho_order_export(peek):
                src = next((c for c in peek.columns if str(c).strip().lower() == "source"), None)
                if src is None or peek[src].astype(str).str.strip().str.lower().eq("meesho").any():
                    return "meesho_order_xlsx"
        except Exception:
            pass
        return "flipkart"

    header_bytes = file_bytes[:3000]
    # CSV — filename hints first
    if "myntra" in fn or "ppmp" in fn or "seller_orders" in fn or "seller orders" in fn or "my ppmp" in fn:
        return "myntra"
    if "b2b" in fn:
        return "amazon_b2b"
    if "b2c" in fn or "mtr" in fn or "merchant" in fn or "tax report" in fn:
        return "amazon_b2c"
    if "meesho" in fn:
        return "meesho_csv"
    # Meesho supplier panel CSV: Orders_YYYY-MM-DD_... (strict — avoid *orders_report* false positives)
    if fn.endswith(".csv") and "seller_orders" not in fn and re.search(
        r"(?:^|/)orders_\d{4}-\d{2}-\d{2}", fn
    ):
        try:
            head = file_bytes[:4000].decode("utf-8", errors="ignore").lower()
            if "reason for credit entry" in head or (
                "sub order no" in head and "customer state" in head
            ):
                return "meesho_csv"
        except Exception:
            pass
    # "Amz ..." or "amz ..." filenames → Amazon FBA
    if fn.startswith("amz ") or fn.startswith("amz_"):
        return "amazon_b2c"
    # Purely numeric filename (e.g. 937788020504.csv) → Amazon FBA/MTR export
    if re.match(r"^\d+\.csv$", fn):
        return "amazon_b2c"

    # Content-based detection (first 3 KB)
    try:
        text = header_bytes[:3000].decode("utf-8", errors="ignore").lower()
        if "buyer name" in text or "customer bill to gstid" in text:
            return "amazon_b2b"
        # Meesho daily CSV report ("Orders_..." filenames from Meesho Supplier Panel)
        if "reason for credit entry" in text or (
                "sub order no" in text and "customer state" in text):
            return "meesho_csv"
        # Amazon FBA Shipment Report
        if "customer shipment date" in text and "merchant sku" in text:
            return "amazon_b2c"
        # Amazon Order Report format (dash-separated headers)
        if "amazon-order-id" in text or "purchase-date" in text or "merchant-order-id" in text:
            return "amazon_b2c"
        if "order_created_date" in text or "product_mrp" in text or "sub_order_no" in text:
            return "myntra"
        # Myntra Seller Orders Report (space-separated headers)
        if "sub order id" in text and ("myntra" in text or "sku id" in text):
            return "myntra"
        if "sub_order_num" in text and "meesho" in text:
            return "meesho_csv"
        if "shipment date" in text or "invoice date" in text or "transaction type" in text:
            return "amazon_b2c"
        if "sales report" in text or "buyer invoice date" in text:
            return "flipkart"
    except Exception:
        pass

    return "unknown"


@router.post("/daily-auto")
async def upload_daily_auto(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """
    Drop any mix of daily report files — platform is auto-detected per file.
    Also accepts a RAR archive — each file inside is extracted and processed.
    Appends to existing session data and rebuilds sales_df automatically.
    """
    sess = _get_session(request)

    detected: list[str] = []
    warnings: list[str] = []

    from ..services.daily_store import load_platform_data as _load_platform_data

    def _flush_deferred_platforms(defer: Set[str]) -> None:
        if not defer:
            return
        if "amazon" in defer:
            sess.mtr_df = _load_platform_data("amazon")
        if "myntra" in defer:
            sess.myntra_df = _load_platform_data("myntra")
        if "meesho" in defer:
            sess.meesho_df = _load_platform_data("meesho")
        if "flipkart" in defer:
            sess.flipkart_df = _load_platform_data("flipkart")
        sess.daily_restored = False

    def _handle_one(
        fname: str,
        raw: bytes,
        defer_reload: Optional[Set[str]] = None,
    ) -> None:
        """Process a single (non-RAR) file and mutate detected/warnings."""
        from ..services.daily_store import load_platform_data as _load_platform

        def _touch_platform(p: str) -> None:
            if defer_reload is not None:
                defer_reload.add(p)
            else:
                if p == "amazon":
                    sess.mtr_df = _load_platform("amazon")
                elif p == "myntra":
                    sess.myntra_df = _load_platform("myntra")
                elif p == "meesho":
                    sess.meesho_df = _load_platform("meesho")
                elif p == "flipkart":
                    sess.flipkart_df = _load_platform("flipkart")
                sess.daily_restored = False

        platform = _detect_platform(fname, raw)
        try:
            if platform == "amazon_mtr_zip":
                df_mtr, _n, sk_mtr = load_mtr_from_zip(raw)
                if not df_mtr.empty:
                    save_daily_file("amazon", fname, df_mtr)
                    _touch_platform("amazon")
                    detected.append(f"Amazon MTR ({fname})")
                    if sk_mtr:
                        warnings.append(f"{fname}: {'; '.join(sk_mtr[:2])}")
                else:
                    warnings.append(f"{fname}: No Amazon MTR CSVs — {'; '.join(sk_mtr[:3])}")

            elif platform == "amazon_b2c":
                df, msg = parse_mtr_csv(raw, fname)
                if not df.empty:
                    save_daily_file("amazon", fname, df)
                    _touch_platform("amazon")
                    detected.append(f"Amazon ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "amazon_b2b":
                df, msg = parse_mtr_csv(raw, fname)
                if not df.empty:
                    save_daily_file("amazon", fname, df)
                    _touch_platform("amazon")
                    detected.append(f"Amazon B2B ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "myntra":
                from ..services.myntra import _parse_myntra_csv
                df, msg = _parse_myntra_csv(raw, fname, sess.sku_mapping)
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(df, fname, "Myntra")
                    save_daily_file("myntra", fname, df)
                    _touch_platform("myntra")
                    detected.append(f"Myntra ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "meesho":
                df, _count, _skipped = load_meesho_from_zip(raw, source_filename=fname)
                if not df.empty:
                    save_daily_file("meesho", fname, df)
                    _touch_platform("meesho")
                    detected.append(f"Meesho ({fname})")
                    if _skipped:
                        warnings.append(f"{fname}: {'; '.join(_skipped[:2])}")
                else:
                    warnings.append(f"{fname}: No data extracted")

            elif platform == "meesho_csv":
                from ..services.meesho import parse_meesho_csv
                df, msg = parse_meesho_csv(raw)
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(df, fname, "Meesho")
                    save_daily_file("meesho", fname, df)
                    _touch_platform("meesho")
                    detected.append(f"Meesho ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "meesho_order_xlsx":
                df, msg = parse_meesho_order_export_xlsx(raw)
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(df, fname, "Meesho")
                    save_daily_file("meesho", fname, df)
                    _touch_platform("meesho")
                    detected.append(f"Meesho order export ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "snapdeal":
                df_sd, _fc, skipped_sd, parse_info = load_snapdeal_from_zip(
                    raw, sess.sku_mapping or {}, fname,
                )
                sess.snapdeal_parse_info.update(parse_info)
                if df_sd.empty:
                    warnings.append(f"{fname}: Snapdeal — {'; '.join(skipped_sd[:3])}")
                else:
                    if sess.snapdeal_df.empty:
                        sess.snapdeal_df = df_sd
                    else:
                        sess.snapdeal_df = pd.concat(
                            [sess.snapdeal_df, df_sd], ignore_index=True,
                        ).drop_duplicates()
                    sess.daily_restored = False
                    detected.append(f"Snapdeal ({fname})")

            elif platform == "flipkart":
                from ..services.flipkart import (
                    _parse_flipkart_xlsx, _parse_flipkart_orders_sheet,
                    _parse_flipkart_earn_more, _parse_flipkart_xlsb,
                )
                if fname.lower().endswith(".xlsb"):
                    df = _parse_flipkart_xlsb(raw, fname, sess.sku_mapping)
                    if df.empty:
                        warnings.append(f"{fname}: No data extracted from Flipkart XLSB file")
                        return
                else:
                    try:
                        xl_sheets = pd.ExcelFile(io.BytesIO(raw)).sheet_names
                    except Exception:
                        xl_sheets = []
                    if "Sales Report" in xl_sheets:
                        df = _parse_flipkart_xlsx(raw, fname, sess.sku_mapping)
                    elif "Orders" in xl_sheets:
                        df = _parse_flipkart_orders_sheet(raw, fname, sess.sku_mapping)
                    elif "earn_more_report" in xl_sheets:
                        df = _parse_flipkart_earn_more(raw, fname, sess.sku_mapping)
                    else:
                        df = _parse_flipkart_xlsx(raw, fname, sess.sku_mapping)
                        if df.empty:
                            warnings.append(f"{fname}: Skipped — no Sales Report, Orders, or earn_more_report sheet (sheets: {', '.join(xl_sheets[:4])})")
                            return
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(df, fname, "Flipkart")
                    save_daily_file("flipkart", fname, df)
                    _touch_platform("flipkart")
                    detected.append(f"Flipkart ({fname})")
                else:
                    warnings.append(f"{fname}: No data extracted from Flipkart file")

            else:
                warnings.append(f"{fname}: Could not detect platform (unknown format)")

        except Exception as e:
            warnings.append(f"{fname}: {e}")

    for fobj in files:
        fname = fobj.filename or "upload"
        raw = await fobj.read()
        try:
            fl = fname.lower()
            # RAR archive — extract and process each file inside
            if raw[:6] == _RAR_MAGIC or fl.endswith(".rar"):
                try:
                    inner_files = _extract_rar_files(raw)
                    if not inner_files:
                        warnings.append(f"{fname}: RAR archive extracted to zero files — check archive is not corrupt.")
                    defer_rar: Set[str] = set()
                    for inner_name, inner_bytes in inner_files:
                        if not inner_bytes:
                            continue
                        norm = inner_name.replace("\\", "/").strip()
                        base = norm.rsplit("/", 1)[-1] if norm else ""
                        if not base or base.endswith("/"):
                            continue
                        _handle_one(inner_name, inner_bytes, defer_rar)
                    _flush_deferred_platforms(defer_rar)
                except Exception as e:
                    warnings.append(f"{fname} (RAR extract): {e}")
            elif fl.endswith(".zip"):
                if "snapdeal" in fl:
                    _handle_one(fname, raw)
                elif _zip_is_meesho_monthly(raw, fname):
                    _handle_one(fname, raw)
                elif _zip_is_amazon_mtr_master(raw):
                    _handle_one(fname, raw)
                elif _zip_is_flipkart_spreadsheet_bundle(raw, fname):
                    try:
                        df_fk, n_fc, skipped_fk = load_flipkart_from_zip(
                            raw,
                            sess.sku_mapping or {},
                            source_filename=fname,
                        )
                        if not df_fk.empty:
                            save_daily_file("flipkart", fname, df_fk)
                            sess.flipkart_df = _load_platform_data("flipkart")
                            sess.daily_restored = False
                            detected.append(
                                f"Flipkart ZIP ({fname}, {n_fc} file(s))",
                            )
                            if skipped_fk:
                                warnings.append(
                                    f"{fname}: {'; '.join(skipped_fk[:2])}",
                                )
                        else:
                            defer_z: Set[str] = set()
                            try:
                                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                                    for info in zf.infolist():
                                        if info.is_dir():
                                            continue
                                        n = info.filename
                                        if "__MACOSX" in n:
                                            continue
                                        base = n.rsplit("/", 1)[-1]
                                        if base.startswith("."):
                                            continue
                                        _handle_one(n, zf.read(n), defer_z)
                                _flush_deferred_platforms(defer_z)
                            except Exception as e:
                                warnings.append(f"{fname} (ZIP expand): {e}")
                            fk_hits = [d for d in detected if "Flipkart" in d]
                            if not fk_hits:
                                hint = (
                                    "; ".join(skipped_fk[:3])
                                    if skipped_fk
                                    else "no parsable sheets"
                                )
                                warnings.append(
                                    f"{fname}: Flipkart ZIP — no rows ({hint})",
                                )
                    except Exception as e:
                        warnings.append(f"{fname} (Flipkart ZIP): {e}")
                else:
                    defer_z: Set[str] = set()
                    try:
                        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                            for info in zf.infolist():
                                if info.is_dir():
                                    continue
                                n = info.filename
                                if "__MACOSX" in n:
                                    continue
                                base = n.rsplit("/", 1)[-1]
                                if base.startswith("."):
                                    continue
                                _handle_one(n, zf.read(n), defer_z)
                        _flush_deferred_platforms(defer_z)
                    except Exception as e:
                        warnings.append(f"{fname} (ZIP): {e}")
            else:
                _handle_one(fname, raw)
        finally:
            del raw
            gc.collect()

    if not detected:
        warn_str = "; ".join(warnings) if warnings else "No valid files found."
        return JSONResponse(content={"ok": False, "message": warn_str})

    # Track loaded platforms & auto-rebuild sales_df
    # Always rebuild. Amazon merges even with an empty user map; bundled/Yash master fills gaps when set.
    sess.daily_sales_sources = list(set(sess.daily_sales_sources + detected))
    try:
        sess.sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            snapdeal_df=sess.snapdeal_df,
            sku_mapping=sess.sku_mapping,
        )
        sess._quarterly_cache.clear()
    except Exception as e:
        warnings.append(f"Sales rebuild warning: {e}")

    _session_data_changed(sess)
    msg_parts = [f"Loaded {len(detected)} file(s): {', '.join(d.split('(')[0].strip() for d in detected)}."]
    if not sess.sales_df.empty:
        msg_parts.append(f"Sales rebuilt ({len(sess.sales_df):,} rows).")
    if warnings:
        msg_parts.append(f"Warnings: {'; '.join(warnings)}")
    return JSONResponse(content={
        "ok": True,
        "message": " ".join(msg_parts),
        "detected_platforms": detected,
    })


# ── Daily Orders (multi-platform CSVs) ────────────────────────

@router.post("/daily")
async def upload_daily(
    request: Request,
    amz_b2c:  Optional[UploadFile] = File(None),
    amz_b2b:  Optional[UploadFile] = File(None),
    myntra:   Optional[UploadFile] = File(None),
    meesho:   Optional[UploadFile] = File(None),
    flipkart: Optional[UploadFile] = File(None),
):
    """
    Upload daily order report CSVs from one or more platforms.
    Each is parsed and appended to existing session data.
    """
    sess = _get_session(request)

    detected: list[str] = []
    warnings: list[str] = []

    # Amazon B2C / B2B
    for fobj, label in [(amz_b2c, "Amazon B2C"), (amz_b2b, "Amazon B2B")]:
        if fobj is not None:
            try:
                raw = await fobj.read()
                df, msg = parse_mtr_csv(raw, fobj.filename or f"{label}.csv")
                del raw
                if not df.empty:
                    sess.mtr_df = _merge_platform_data(
                        sess.mtr_df, df, "amazon", source_filename=fobj.filename or None,
                    )
                    detected.append(label)
                    if msg != "OK":
                        warnings.append(f"{label}: {msg}")
                else:
                    warnings.append(f"{label}: {msg}")
            except Exception as e:
                warnings.append(f"{label}: {e}")

    # Myntra (single-file PPMP CSV)
    if myntra is not None:
        try:
            from ..services.myntra import _parse_myntra_csv
            raw = await myntra.read()
            df, msg = _parse_myntra_csv(raw, myntra.filename or "myntra.csv", sess.sku_mapping)
            del raw
            if not df.empty:
                df = apply_dsr_segment_from_upload_filename(
                    df, myntra.filename or None, "Myntra",
                )
                sess.myntra_df = _merge_platform_data(
                    sess.myntra_df, df, "myntra", source_filename=myntra.filename or None,
                )
                detected.append("Myntra")
                if msg != "OK":
                    warnings.append(f"Myntra: {msg}")
            else:
                warnings.append(f"Myntra: {msg}")
        except Exception as e:
            warnings.append(f"Myntra: {e}")

    # Meesho (single monthly ZIP)
    if meesho is not None:
        try:
            from ..services.meesho import load_meesho_from_zip
            raw = await meesho.read()
            df, _count, _skipped = load_meesho_from_zip(
                raw, source_filename=meesho.filename or None,
            )
            del raw
            if not df.empty:
                sess.meesho_df = _merge_platform_data(
                    sess.meesho_df, df, "meesho", source_filename=meesho.filename or None,
                )
                detected.append("Meesho")
                if _skipped:
                    warnings.append(f"Meesho: {'; '.join(_skipped[:3])}")
            else:
                warnings.append(f"Meesho: No data. {'; '.join(_skipped[:3])}")
        except Exception as e:
            warnings.append(f"Meesho: {e}")

    # Flipkart (single-file XLSX)
    if flipkart is not None:
        try:
            from ..services.flipkart import _parse_flipkart_xlsx
            raw = await flipkart.read()
            df = _parse_flipkart_xlsx(raw, flipkart.filename or "flipkart.xlsx", sess.sku_mapping)
            del raw
            if not df.empty:
                df = apply_dsr_segment_from_upload_filename(
                    df, flipkart.filename or None, "Flipkart",
                )
                sess.flipkart_df = _merge_platform_data(
                    sess.flipkart_df, df, "flipkart", source_filename=flipkart.filename or None,
                )
                detected.append("Flipkart")
            else:
                warnings.append("Flipkart: No data extracted")
        except Exception as e:
            warnings.append(f"Flipkart: {e}")

    gc.collect()

    if not detected:
        warn_str = "; ".join(warnings) if warnings else "No valid files provided."
        return JSONResponse(content={"ok": False, "message": warn_str})

    _session_data_changed(sess)
    msg_parts = [f"Daily data loaded — {', '.join(detected)}."]
    if warnings:
        msg_parts.append(f"Warnings: {'; '.join(warnings)}")
    return JSONResponse(content={
        "ok": True,
        "message": " ".join(msg_parts),
        "detected_platforms": detected,
    })


# ── Build Sales (merge all platforms into unified sales_df) ───

@router.post("/build-sales")
async def build_sales(request: Request, background_tasks: BackgroundTasks):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    try:
        sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            snapdeal_df=sess.snapdeal_df,
            sku_mapping=sess.sku_mapping,
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Build error: {e}"})

    sess.sales_df = sales_df
    sess._quarterly_cache.clear()  # invalidate quarterly cache on new sales data
    _session_data_changed(sess)
    background_tasks.add_task(_auto_save_cache, sess)
    gaps = list_sku_mapping_gaps(sales_df, sess.sku_mapping)
    msg = f"Sales built: {len(sales_df):,} rows. Saving to cache in background…"
    if gaps:
        msg += f" Warning: {len(gaps)} SKU(s) not found as map key or OMS value — see Upload → SKU Mapping list."
    return JSONResponse(
        content={
            "ok": True,
            "message": msg,
            "rows": len(sales_df),
            "unmapped_skus": gaps or None,
        }
    )


# ── Clear platform data ────────────────────────────────────────

_PLATFORM_CLEAR = {
    "mtr":      lambda sess, pd: setattr(sess, "mtr_df",      pd.DataFrame()),
    "myntra":   lambda sess, pd: setattr(sess, "myntra_df",   pd.DataFrame()),
    "meesho":   lambda sess, pd: setattr(sess, "meesho_df",   pd.DataFrame()),
    "flipkart": lambda sess, pd: setattr(sess, "flipkart_df", pd.DataFrame()),
    "snapdeal": lambda sess, pd: setattr(sess, "snapdeal_df", pd.DataFrame()),
    "sales":    lambda sess, pd: setattr(sess, "sales_df",    pd.DataFrame()),
}

@router.delete("/clear/{platform}")
async def clear_platform(platform: str, request: Request):
    """Clear a specific platform's data from the session."""
    sess = _get_session(request)
    cleaner = _PLATFORM_CLEAR.get(platform)
    if not cleaner:
        return JSONResponse(content={"ok": False, "message": f"Unknown platform: {platform}"}, status_code=400)
    import pandas as pd
    cleaner(sess, pd)
    if platform != "sales":
        sess.sales_df = pd.DataFrame()   # invalidate combined sales
        sess._quarterly_cache.clear()
    return JSONResponse(content={"ok": True, "message": f"{platform} data cleared."})
