/**
 * Zustand store — tracks what the server has loaded for this session.
 * Synced from GET /api/data/coverage after every successful upload.
 */
import { create } from 'zustand'
import type { CoverageResponse } from '../api/client'

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
  pause_auto_data_restore: false,
  sales_rebuild: 'idle',
  sales_rebuild_message: '',
  daily_auto_ingest_status: 'idle',
  daily_auto_ingest_message: '',
}

export const useSession = create<SessionState>((set) => ({
  ...empty,
  setCoverage: (c) => set(c),
}))
