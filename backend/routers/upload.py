"""
Upload router — all file ingestion endpoints.
"""
import gc
import io
import re
from typing import List, Optional

import logging
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..session import store
from ..models.schemas import UploadResponse
from ..services.sku_mapping import parse_sku_mapping
from ..services.mtr import load_mtr_from_zip, parse_mtr_csv
from ..services.myntra import load_myntra_from_zip
from ..services.meesho import load_meesho_from_zip
from ..services.flipkart import load_flipkart_from_zip
from ..services.snapdeal import load_snapdeal_from_zip
from ..services.inventory import load_inventory_consolidated
from ..services.sales import build_sales_df
from ..services.existing_po import parse_existing_po
from ..services.github_cache import save_cache_to_drive
from ..services.daily_store import save_daily_file

router = APIRouter()

_log = logging.getLogger(__name__)


def _auto_save_cache(sess) -> None:
    """Run in background after build-sales: silently push session to GitHub Releases."""
    session_data = {
        "sales_df":    sess.sales_df,
        "mtr_df":      sess.mtr_df,
        "meesho_df":   sess.meesho_df,
        "myntra_df":   sess.myntra_df,
        "flipkart_df": sess.flipkart_df,
        "snapdeal_df": sess.snapdeal_df,
        "sku_mapping": sess.sku_mapping,
    }
    ok, msg = save_cache_to_drive(session_data)
    if ok:
        _log.info("Auto-save cache succeeded: %s", msg)
    else:
        _log.warning("Auto-save cache skipped/failed: %s", msg)


def _get_session(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")
    return sess


# ── SKU Mapping ───────────────────────────────────────────────

@router.post("/sku-mapping", response_model=UploadResponse)
async def upload_sku_mapping(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        file_bytes = await file.read()
        mapping = parse_sku_mapping(file_bytes)
        sess.sku_mapping = mapping
        return UploadResponse(
            ok=True,
            message=f"SKU mapping loaded: {len(mapping):,} entries",
            sku_count=len(mapping),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse SKU mapping: {e}")


# ── Amazon MTR ────────────────────────────────────────────────

@router.post("/mtr", response_model=UploadResponse)
async def upload_mtr(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        zip_bytes = await file.read()
        df, csv_count, skipped = load_mtr_from_zip(zip_bytes)
        del zip_bytes
        gc.collect()

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No valid CSV files found. Issues: {'; '.join(skipped[:5])}",
            )

        sess.mtr_df = df
        years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())
        return UploadResponse(
            ok=True,
            message=f"MTR loaded: {len(df):,} rows from {csv_count} files",
            rows=len(df),
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse MTR ZIP: {e}")


# ── Myntra ────────────────────────────────────────────────────

@router.post("/myntra", response_model=UploadResponse)
async def upload_myntra(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return UploadResponse(ok=False, message="Upload SKU Mapping first.")
    try:
        zip_bytes = await file.read()
        df, csv_count, skipped = load_myntra_from_zip(zip_bytes, sess.sku_mapping)
        del zip_bytes
        gc.collect()

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
            )

        sess.myntra_df = df
        years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())
        return UploadResponse(
            ok=True,
            message=f"Myntra loaded: {len(df):,} rows from {csv_count} CSVs",
            rows=len(df),
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Myntra ZIP: {e}")


# ── Meesho ────────────────────────────────────────────────────

@router.post("/meesho", response_model=UploadResponse)
async def upload_meesho(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        zip_bytes = await file.read()
        df, zip_count, skipped = load_meesho_from_zip(zip_bytes)
        del zip_bytes
        gc.collect()

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
            )

        sess.meesho_df = df
        years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())
        return UploadResponse(
            ok=True,
            message=f"Meesho loaded: {len(df):,} rows from {zip_count} monthly ZIPs",
            rows=len(df),
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Meesho ZIP: {e}")


# ── Flipkart ──────────────────────────────────────────────────

@router.post("/flipkart", response_model=UploadResponse)
async def upload_flipkart(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return UploadResponse(ok=False, message="Upload SKU Mapping first.")
    try:
        zip_bytes = await file.read()
        df, xlsx_count, skipped = load_flipkart_from_zip(zip_bytes, sess.sku_mapping)
        del zip_bytes
        gc.collect()

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
            )

        sess.flipkart_df = df
        years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())
        return UploadResponse(
            ok=True,
            message=f"Flipkart loaded: {len(df):,} rows from {xlsx_count} files",
            rows=len(df),
            years=years,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Flipkart ZIP: {e}")


# ── Snapdeal ──────────────────────────────────────────────────

@router.post("/snapdeal", response_model=UploadResponse)
async def upload_snapdeal(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        zip_bytes = await file.read()
        df, file_count, skipped, parse_info = load_snapdeal_from_zip(zip_bytes, sess.sku_mapping)
        del zip_bytes
        gc.collect()

        # Store parse diagnostics (merge with existing)
        sess.snapdeal_parse_info.update(parse_info)

        if df.empty:
            return UploadResponse(
                ok=False,
                message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
            )

        import pandas as pd
        sess.snapdeal_df = pd.concat(
            [sess.snapdeal_df, df], ignore_index=True
        ) if not sess.snapdeal_df.empty else df

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

@router.post("/inventory")
async def upload_inventory(
    request: Request,
    oms_file:    Optional[UploadFile] = File(None),
    fk_file:     Optional[UploadFile] = File(None),
    myntra_file: Optional[UploadFile] = File(None),
    amz_file:    Optional[UploadFile] = File(None),
):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    oms_b    = await oms_file.read()    if oms_file    else None
    fk_b     = await fk_file.read()    if fk_file     else None
    myntra_b = await myntra_file.read() if myntra_file else None
    amz_b    = await amz_file.read()   if amz_file    else None

    if not any([oms_b, fk_b, myntra_b, amz_b]):
        return JSONResponse(content={"ok": False, "message": "No files provided."})

    try:
        df_variant = load_inventory_consolidated(
            oms_b, fk_b, myntra_b, amz_b, sess.sku_mapping, group_by_parent=False
        )
        df_parent = load_inventory_consolidated(
            oms_b, fk_b, myntra_b, amz_b, sess.sku_mapping, group_by_parent=True
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Parse error: {e}"})

    sess.inventory_df_variant = df_variant
    sess.inventory_df_parent  = df_parent
    return JSONResponse(content={
        "ok": True,
        "message": f"Inventory loaded: {len(df_variant):,} SKUs.",
        "rows": len(df_variant),
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

        import pandas as pd
        sess.mtr_df = pd.concat([sess.mtr_df, df], ignore_index=True) if not sess.mtr_df.empty else df
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

        import pandas as pd
        sess.mtr_df = pd.concat([sess.mtr_df, df], ignore_index=True) if not sess.mtr_df.empty else df
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
        return UploadResponse(
            ok=True,
            message=f"Existing PO loaded: {len(df):,} SKUs with pipeline quantities.",
            rows=len(df),
        )
    except ValueError as e:
        return UploadResponse(ok=False, message=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Existing PO: {e}")


@router.post("/cogs", response_model=UploadResponse)
async def upload_cogs(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        from ..services.finance import parse_cogs_sheet
        file_bytes = await file.read()
        df = parse_cogs_sheet(file_bytes, file.filename or "cogs.xlsx")
        sess.cogs_df = df
        return UploadResponse(ok=True, message=f"COGS sheet loaded: {len(df):,} SKUs.", rows=len(df))
    except ValueError as e:
        return UploadResponse(ok=False, message=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse COGS sheet: {e}")


# ── Daily Orders — Auto-detect (drop all files, we figure it out) ─

def _detect_platform(filename: str, header_bytes: bytes) -> str:
    """
    Guess platform from filename + first ~3 KB of content.
    Returns one of: 'amazon_b2c', 'amazon_b2b', 'myntra', 'meesho', 'flipkart', 'unknown'
    """
    fn = (filename or "").lower()

    # ZIP → Meesho monthly ZIP
    if fn.endswith(".zip"):
        return "meesho"

    # XLSX → Flipkart Sales Report
    if fn.endswith((".xlsx", ".xls")):
        return "flipkart"

    # CSV — filename hints first
    if "myntra" in fn or "ppmp" in fn or "seller_orders" in fn or "seller orders" in fn:
        return "myntra"
    if "b2b" in fn:
        return "amazon_b2b"
    if "b2c" in fn or "mtr" in fn or "merchant" in fn or "tax report" in fn:
        return "amazon_b2c"
    if "meesho" in fn:
        return "meesho_csv"
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
    Appends to existing session data and rebuilds sales_df automatically.
    """
    sess = _get_session(request)
    import pandas as pd

    detected: list[str] = []
    warnings: list[str] = []

    for fobj in files:
        fname = fobj.filename or "upload"
        raw = await fobj.read()
        platform = _detect_platform(fname, raw)

        try:
            if platform == "amazon_b2c":
                df, msg = parse_mtr_csv(raw, fname)
                if not df.empty:
                    sess.mtr_df = pd.concat([sess.mtr_df, df], ignore_index=True) if not sess.mtr_df.empty else df
                    save_daily_file("amazon", fname, df)
                    detected.append(f"Amazon ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "amazon_b2b":
                df, msg = parse_mtr_csv(raw, fname)
                if not df.empty:
                    sess.mtr_df = pd.concat([sess.mtr_df, df], ignore_index=True) if not sess.mtr_df.empty else df
                    save_daily_file("amazon", fname, df)
                    detected.append(f"Amazon B2B ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "myntra":
                from ..services.myntra import _parse_myntra_csv
                df, msg = _parse_myntra_csv(raw, fname, sess.sku_mapping)
                if not df.empty:
                    sess.myntra_df = pd.concat([sess.myntra_df, df], ignore_index=True) if not sess.myntra_df.empty else df
                    save_daily_file("myntra", fname, df)
                    detected.append(f"Myntra ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "meesho":
                df, _count, _skipped = load_meesho_from_zip(raw)
                if not df.empty:
                    sess.meesho_df = pd.concat([sess.meesho_df, df], ignore_index=True) if not sess.meesho_df.empty else df
                    save_daily_file("meesho", fname, df)
                    detected.append(f"Meesho ({fname})")
                    if _skipped:
                        warnings.append(f"{fname}: {'; '.join(_skipped[:2])}")
                else:
                    warnings.append(f"{fname}: No data extracted")

            elif platform == "meesho_csv":
                from ..services.meesho import parse_meesho_csv
                df, msg = parse_meesho_csv(raw)
                if not df.empty:
                    sess.meesho_df = pd.concat([sess.meesho_df, df], ignore_index=True) if not sess.meesho_df.empty else df
                    save_daily_file("meesho", fname, df)
                    detected.append(f"Meesho ({fname})")
                    if msg != "OK":
                        warnings.append(f"{fname}: {msg}")
                else:
                    warnings.append(f"{fname}: {msg}")

            elif platform == "flipkart":
                from ..services.flipkart import _parse_flipkart_xlsx, _parse_flipkart_orders_sheet
                try:
                    import pandas as _pd
                    xl_sheets = _pd.ExcelFile(io.BytesIO(raw)).sheet_names
                except Exception:
                    xl_sheets = []
                if "Sales Report" in xl_sheets:
                    df = _parse_flipkart_xlsx(raw, fname, sess.sku_mapping)
                elif "Orders" in xl_sheets:
                    df = _parse_flipkart_orders_sheet(raw, fname, sess.sku_mapping)
                else:
                    warnings.append(f"{fname}: Skipped — no Sales Report or Orders sheet (sheets: {', '.join(xl_sheets[:4])})")
                    continue
                if not df.empty:
                    sess.flipkart_df = pd.concat([sess.flipkart_df, df], ignore_index=True) if not sess.flipkart_df.empty else df
                    save_daily_file("flipkart", fname, df)
                    detected.append(f"Flipkart ({fname})")
                else:
                    warnings.append(f"{fname}: No data extracted from Flipkart file")

            else:
                warnings.append(f"{fname}: Could not detect platform (unknown format)")

        except Exception as e:
            warnings.append(f"{fname}: {e}")
        finally:
            del raw
            gc.collect()

    if not detected:
        warn_str = "; ".join(warnings) if warnings else "No valid files found."
        return JSONResponse(content={"ok": False, "message": warn_str})

    # Track loaded platforms & auto-rebuild sales_df
    sess.daily_sales_sources = list(set(sess.daily_sales_sources + detected))

    if sess.sku_mapping:
        try:
            sess.sales_df = build_sales_df(
                mtr_df=sess.mtr_df,
                myntra_df=sess.myntra_df,
                meesho_df=sess.meesho_df,
                flipkart_df=sess.flipkart_df,
                snapdeal_df=sess.snapdeal_df,
                sku_mapping=sess.sku_mapping,
            )
            sess._quarterly_cache.clear()  # invalidate quarterly cache
        except Exception as e:
            warnings.append(f"Sales rebuild warning: {e}")

    msg_parts = [f"Loaded {len(detected)} file(s): {', '.join(d.split('(')[0].strip() for d in detected)}."]
    if sess.sku_mapping:
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
    import pandas as pd

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
                    sess.mtr_df = pd.concat([sess.mtr_df, df], ignore_index=True) if not sess.mtr_df.empty else df
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
                sess.myntra_df = pd.concat([sess.myntra_df, df], ignore_index=True) if not sess.myntra_df.empty else df
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
            df, _count, _skipped = load_meesho_from_zip(raw)
            del raw
            if not df.empty:
                sess.meesho_df = pd.concat([sess.meesho_df, df], ignore_index=True) if not sess.meesho_df.empty else df
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
                sess.flipkart_df = pd.concat([sess.flipkart_df, df], ignore_index=True) if not sess.flipkart_df.empty else df
                detected.append("Flipkart")
            else:
                warnings.append("Flipkart: No data extracted")
        except Exception as e:
            warnings.append(f"Flipkart: {e}")

    gc.collect()

    if not detected:
        warn_str = "; ".join(warnings) if warnings else "No valid files provided."
        return JSONResponse(content={"ok": False, "message": warn_str})

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
    background_tasks.add_task(_auto_save_cache, sess)
    return JSONResponse(content={
        "ok": True,
        "message": f"Sales built: {len(sales_df):,} rows. Saving to cache in background…",
        "rows": len(sales_df),
    })
