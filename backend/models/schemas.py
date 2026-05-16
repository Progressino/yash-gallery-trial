"""Pydantic response models shared across all routers."""
from typing import Any, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    ok: bool
    message: str
    rows: Optional[int] = None
    # Import quality diagnostics (optional; set by parsers that can measure row drops)
    parsed_rows: Optional[int] = None
    kept_rows: Optional[int] = None
    dropped_rows: Optional[int] = None
    dropped_reasons: Optional[list[str]] = None
    validation_warnings: Optional[list[str]] = None
    years: Optional[list[int]] = None
    sku_count: Optional[int] = None
    detected_platforms: Optional[list[str]] = None
    # After SKU master upload: tokens in sales not present as map key (incl. PL-alias) or OMS value
    unmapped_skus: Optional[list[str]] = None


class CoverageResponse(BaseModel):
    sku_mapping: bool
    mtr: bool
    sales: bool
    myntra: bool
    meesho: bool
    flipkart: bool
    snapdeal: bool = False
    inventory: bool
    daily_orders: bool
    existing_po: bool = False
    sku_status_lead: bool = False
    daily_inventory_history: bool = False
    po_raise_ledger: bool = False
    # row counts for loaded datasets
    mtr_rows: int = 0
    sales_rows: int = 0
    myntra_rows: int = 0
    meesho_rows: int = 0
    flipkart_rows: int = 0
    snapdeal_rows: int = 0
    sku_status_lead_rows: int = 0
    daily_inventory_history_rows: int = 0
    daily_inventory_history_skus: int = 0
    po_raise_ledger_rows: int = 0
    # After full wipe: True until user uploads or clicks Load Cache (blocks auto-restore)
    pause_auto_data_restore: bool = False
    # Tier-3 daily-auto background sales rebuild (idle | running | done | error)
    sales_rebuild: str = "idle"
    sales_rebuild_message: str = ""


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None
