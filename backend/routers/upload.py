"""
Upload router — all file ingestion endpoints.
POST /api/upload/sku-mapping  → parse Excel, store in session
POST /api/upload/mtr          → parse MTR ZIP, store in session
POST /api/upload/myntra       → (placeholder)
POST /api/upload/meesho       → (placeholder)
POST /api/upload/flipkart     → (placeholder)
"""
import gc
import traceback

from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from ..session import store
from ..models.schemas import UploadResponse
from ..services.sku_mapping import parse_sku_mapping
from ..services.mtr import load_mtr_from_zip

router = APIRouter()


def _get_session(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")
    return sess


# ── SKU Mapping ───────────────────────────────────────────────

@router.post("/sku-mapping", response_model=UploadResponse)
async def upload_sku_mapping(
    request: Request,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Expected an Excel file (.xlsx)")

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
async def upload_mtr(
    request: Request,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Expected a ZIP file")

    sess = _get_session(request)
    try:
        zip_bytes = await file.read()
        df, csv_count, skipped = load_mtr_from_zip(zip_bytes)
        del zip_bytes
        gc.collect()

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail=f"No valid CSV files found. Issues: {'; '.join(skipped[:5])}",
            )

        sess.mtr_df = df
        years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())

        return UploadResponse(
            ok=True,
            message=f"MTR loaded: {len(df):,} rows from {csv_count} files",
            rows=len(df),
            years=years,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse MTR ZIP: {e}\n{traceback.format_exc()}",
        )


# ── Placeholder endpoints for the remaining sources ───────────
# Each follows the exact same pattern as /mtr; full logic extracted next.

@router.post("/myntra", response_model=UploadResponse)
async def upload_myntra(request: Request, file: UploadFile = File(...)):
    _get_session(request)
    return UploadResponse(ok=False, message="Myntra upload not yet implemented")


@router.post("/meesho", response_model=UploadResponse)
async def upload_meesho(request: Request, file: UploadFile = File(...)):
    _get_session(request)
    return UploadResponse(ok=False, message="Meesho upload not yet implemented")


@router.post("/flipkart", response_model=UploadResponse)
async def upload_flipkart(request: Request, file: UploadFile = File(...)):
    _get_session(request)
    return UploadResponse(ok=False, message="Flipkart upload not yet implemented")


@router.post("/amazon-b2c", response_model=UploadResponse)
async def upload_amazon_b2c(request: Request, file: UploadFile = File(...)):
    _get_session(request)
    return UploadResponse(ok=False, message="Amazon B2C upload not yet implemented")


@router.post("/amazon-b2b", response_model=UploadResponse)
async def upload_amazon_b2b(request: Request, file: UploadFile = File(...)):
    _get_session(request)
    return UploadResponse(ok=False, message="Amazon B2B upload not yet implemented")


@router.post("/inventory", response_model=UploadResponse)
async def upload_inventory(request: Request, file: UploadFile = File(...)):
    _get_session(request)
    return UploadResponse(ok=False, message="Inventory upload not yet implemented")
