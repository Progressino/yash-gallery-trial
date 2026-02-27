"""
Upload router — all file ingestion endpoints.
"""
import gc
import traceback
from typing import Optional

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from ..session import store
from ..models.schemas import UploadResponse
from ..services.sku_mapping import parse_sku_mapping
from ..services.mtr import load_mtr_from_zip
from ..services.myntra import load_myntra_from_zip
from ..services.meesho import load_meesho_from_zip
from ..services.flipkart import load_flipkart_from_zip
from ..services.inventory import load_inventory_consolidated
from ..services.sales import build_sales_df

router = APIRouter()


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


# ── Build Sales (merge all platforms into unified sales_df) ───

@router.post("/build-sales")
async def build_sales(request: Request):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    try:
        sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            sku_mapping=sess.sku_mapping,
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Build error: {e}"})

    sess.sales_df = sales_df
    return JSONResponse(content={
        "ok": True,
        "message": f"Sales built: {len(sales_df):,} rows.",
        "rows": len(sales_df),
    })
