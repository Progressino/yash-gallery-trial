/**
 * Zustand store — tracks what the server has loaded for this session.
 * Synced from GET /api/data/coverage after every successful upload.
 */
import { create } from 'zustand'
import type { CoverageResponse } from '../api/client'
import { persistLocalSessionHint } from '../lib/localSessionHint'

interface SessionState extends CoverageResponse {
  setCoverage: (c: CoverageResponse) => void
}

const empty: CoverageResponse = {
  sku_mapping: false,
  mtr: false,
  sales: false,
  myntra: false,
  meesho: false,
  flipkart: false,
  snapdeal: false,
  inventory: false,
  daily_orders: false,
  existing_po: false,
  sku_status_lead: false,
  daily_inventory_history: false,
  po_raise_ledger: false,
  return_sheet: false,
  mtr_rows: 0,
  sales_rows: 0,
  myntra_rows: 0,
  meesho_rows: 0,
  flipkart_rows: 0,
  snapdeal_rows: 0,
  sku_status_lead_rows: 0,
  daily_inventory_history_rows: 0,
  daily_inventory_history_skus: 0,
  po_raise_ledger_rows: 0,
  return_sheet_skus: 0,
  return_sheet_units: 0,
  return_overlay_uploaded_at: undefined,
  return_overlay_filename: undefined,
  return_overlay_sources: undefined,
  return_overlay_by_platform: undefined,
  returns_import_progress: 0,
  pause_auto_data_restore: false,
  sales_rebuild: 'idle',
  sales_rebuild_message: '',
  sales_data_revision: 0,
  session_restore_status: 'idle',
  session_restore_message: '',
  session_restore_step: '',
  session_restore_progress: 0,
  daily_auto_ingest_status: 'idle',
  daily_auto_ingest_message: '',
  tier1_bulk_status: 'idle',
  tier1_bulk_message: '',
  tier1_bulk_platform: '',
  inventory_upload_status: 'idle',
  inventory_upload_message: '',
  inventory_upload_progress: 0,
  inventory_snapshot_date: undefined,
  inventory_snapshot_date_label: undefined,
  inventory_snapshot_date_sources: undefined,
  inventory_snapshot_uploaded_at: undefined,
  existing_po_uploaded_at: undefined,
  existing_po_filename: undefined,
  existing_po_generation: 0,
  existing_po_rows: 0,
  existing_po_needs_recalc: false,
  existing_po_per_size_skus: 0,
  existing_po_looks_aggregated: false,
  finishing_receipt_uploaded_at: undefined,
  finishing_receipt_filename: undefined,
  finishing_receipt_report: undefined,
  daily_inventory_upload_status: 'idle',
  daily_inventory_upload_message: '',
}

export const useSession = create<SessionState>((set) => ({
  ...empty,
  setCoverage: (c) => {
    set(c)
    persistLocalSessionHint(c)
  },
}))
