"""Pydantic response models shared across all routers."""
from typing import Any, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    ok: bool
    message: str
    rows: Optional[int] = None
    years: Optional[list[int]] = None
    sku_count: Optional[int] = None
    detected_platforms: Optional[list[str]] = None


class CoverageResponse(BaseModel):
    sku_mapping: bool
    mtr: bool
    sales: bool
    myntra: bool
    meesho: bool
    flipkart: bool
    inventory: bool
    daily_orders: bool
    # row counts for loaded datasets
    mtr_rows: int = 0
    sales_rows: int = 0
    myntra_rows: int = 0
    meesho_rows: int = 0
    flipkart_rows: int = 0


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None
