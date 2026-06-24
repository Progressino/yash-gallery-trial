"""Pydantic response models shared across all routers."""
from typing import Any, Optional
from pydantic import BaseModel, Field


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
    existing_po_uploaded_at: Optional[str] = None
    existing_po_generation: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Fast background-job snapshot — no restore or Tier-3 side effects."""
    server_time: str
    warm_cache: bool = False
    warm_cache_generation: int = 0
    upload_memory_lock_held: bool = False
    daily_auto_ingest_status: str = "idle"
    daily_auto_ingest_message: str = ""
    sales_rebuild_status: str = "idle"
    sales_rebuild_message: str = ""
    session_restore_status: str = "idle"
    inventory_upload_status: str = "idle"
    daily_inventory_upload_status: str = "idle"
    tier1_bulk_status: str = "idle"
    tier1_bulk_message: str = ""


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
    manual_intransit_sheet: bool = False
    po_raise_ledger: bool = False
    return_sheet: bool = False
    # row counts for loaded datasets
    mtr_rows: int = 0
    sales_rows: int = 0
    myntra_rows: int = 0
    meesho_rows: int = 0
    flipkart_rows: int = 0
    snapdeal_rows: int = 0
    inventory_rows: int = 0
    sku_status_lead_rows: int = 0
    daily_inventory_history_rows: int = 0
    daily_inventory_history_skus: int = 0
    daily_inventory_history_min_date: Optional[str] = None
    daily_inventory_history_max_date: Optional[str] = None
    daily_inventory_history_uploaded_at: Optional[str] = None
    daily_inventory_history_filename: Optional[str] = None
    inventory_snapshot_stale: bool = False
    inventory_snapshot_lag_days: Optional[int] = None
    daily_inventory_history_stale: bool = False
    daily_inventory_history_lag_days: Optional[int] = None
    inventory_staleness_warnings: Optional[list[str]] = None
    manual_intransit_skus: int = 0
    manual_intransit_units: int = 0
    manual_not_in_inventory_units: int = 0
    manual_intransit_uploaded_at: Optional[str] = None
    manual_intransit_filename: Optional[str] = None
    manual_intransit_parse_report: Optional[dict] = None
    finishing_receipt_uploaded_at: Optional[str] = None
    finishing_receipt_filename: Optional[str] = None
    finishing_receipt_report: Optional[dict] = None
    po_raise_ledger_rows: int = 0
    return_sheet_skus: int = 0
    return_sheet_units: int = 0
    return_overlay_uploaded_at: Optional[str] = None
    return_overlay_filename: Optional[str] = None
    return_overlay_sources: Optional[list[dict]] = None
    return_overlay_by_platform: Optional[list[dict]] = None
    # After full wipe: True until user uploads or clicks Load Cache (blocks auto-restore)
    pause_auto_data_restore: bool = False
    # Return overlay import (Upload → Returns for PO) — async RAR/CSV parse
    returns_import_status: str = "idle"
    returns_import_message: str = ""
    returns_import_progress: int = 0
    returns_import_warnings: Optional[list[str]] = None
    # Tier-3 daily-auto background sales rebuild (idle | running | done | error)
    sales_rebuild: str = "idle"
    sales_rebuild_message: str = ""
    sales_data_revision: int = 0
    # Tier-3 daily-auto background ingest (RAR / large multi-file) before sales rebuild
    daily_auto_ingest_status: str = "idle"
    daily_auto_ingest_message: str = ""
    # Populated after background daily-auto ingest completes (see session.daily_auto_ingest_result).
    daily_auto_ingest_detected_platforms: Optional[list[str]] = None
    daily_auto_ingest_warnings: Optional[list[str]] = None
    daily_auto_ingest_processed_files: Optional[int] = None
    daily_auto_ingest_detected_files: Optional[int] = None
    daily_auto_ingest_unknown_files: Optional[int] = None
    daily_auto_ingest_expanded_files: Optional[int] = None
    daily_auto_ingest_saved_files: Optional[int] = None
    daily_auto_ingest_file_results: Optional[list[dict]] = None
    # Tier-1 bulk history ZIP/RAR (MTR, Myntra PPMP, …) — async parse
    tier1_bulk_status: str = "idle"
    tier1_bulk_message: str = ""
    tier1_bulk_platform: str = ""
    # Snapshot inventory-auto background job
    inventory_upload_status: str = "idle"
    inventory_upload_message: str = ""
    inventory_upload_progress: Optional[int] = None
    inventory_upload_rows: Optional[int] = None
    inventory_upload_warnings: Optional[list[str]] = None
    inventory_upload_file_results: Optional[list[dict]] = None
    inventory_upload_sources: Optional[list[str]] = None
    inventory_upload_amz_disclaimer: Optional[dict] = None
    inventory_snapshot_date: Optional[str] = None
    inventory_snapshot_date_label: Optional[str] = None
    inventory_snapshot_date_sources: Optional[list[str]] = None
    inventory_snapshot_uploaded_at: Optional[str] = None
    existing_po_uploaded_at: Optional[str] = None
    existing_po_filename: Optional[str] = None
    existing_po_generation: int = 0
    existing_po_rows: int = 0
    existing_po_needs_recalc: bool = False
    existing_po_per_size_skus: int = 0
    existing_po_looks_aggregated: bool = False
    existing_po_upload_status: str = "idle"
    existing_po_upload_message: str = ""
    existing_po_upload_progress: Optional[int] = None
    # Wide daily inventory matrix (PO) background parse
    daily_inventory_upload_status: str = "idle"
    daily_inventory_upload_message: str = ""
    # Background full session restore (Upload → Restore all from server)
    session_restore_status: str = "idle"
    session_restore_message: str = ""
    session_restore_step: str = ""
    session_restore_progress: int = 0
    # PO readiness (decoupled from non-critical background jobs like sales_rebuild)
    data_ready: bool = False
    po_ready: bool = False
    background_tasks: dict[str, bool] = Field(default_factory=dict)
    critical_restore_running: bool = False
    # Intelligence dashboard (stronger than 8/8 alone)
    platforms_loaded: bool = False
    hydration_complete: bool = False
    intelligence_ready: bool = False
    dashboard_ready: bool = False


class IntelligenceReadinessResponse(BaseModel):
    intelligence_ready: bool
    dashboard_ready: bool = False
    precomputed_bundle_ready: bool = False
    tier3_platforms_in_window: list[str] = Field(default_factory=list)
    data_ready: bool = False
    platforms_loaded: bool = False
    hydration_complete: bool = False
    hydration_inflight: bool = False
    sales_available: bool = False
    inventory_available: bool = False
    sales_rows: int = 0
    inventory_rows: int = 0
    platform_rows: dict[str, int] = Field(default_factory=dict)
    data_source: str = "warm_cache"
    background_jobs: list[str] = Field(default_factory=list)
    background_tasks: dict[str, bool] = Field(default_factory=dict)


class DashboardSummaryResponse(BaseModel):
    source: str = "none"
    platforms: dict[str, dict] = Field(default_factory=dict)
    platform_summary: list[dict] = Field(default_factory=list)
    top_skus: list[dict] = Field(default_factory=list)
    sales_summary: dict = Field(default_factory=dict)
    data_completeness: str = "partial"
    message: str = ""
    version: str = ""
    stale: bool = False
    refresh_queued: bool = False


class PoReadinessResponse(BaseModel):
    po_ready: bool
    data_ready: bool = False
    sales_rows: int = 0
    inventory_rows: int = 0
    data_source: str = "warm_cache"
    hydration: str = "unknown"
    background_jobs: list[str] = Field(default_factory=list)
    background_tasks: dict[str, bool] = Field(default_factory=dict)
    critical_restore_running: bool = False


class RestoreFullResponse(CoverageResponse):
    """POST /data/restore-full — full warm + disk + Tier-3 + GitHub recovery."""

    ok: bool = True
    message: str = ""
    missing_platforms: list[str] = []
    restore_steps: list[str] = []
    restore_async: bool = False


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None
