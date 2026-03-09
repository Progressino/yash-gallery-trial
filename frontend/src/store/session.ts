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
  mtr_rows: 0,
  sales_rows: 0,
  myntra_rows: 0,
  meesho_rows: 0,
  flipkart_rows: 0,
  snapdeal_rows: 0,
}

export const useSession = create<SessionState>((set) => ({
  ...empty,
  setCoverage: (c) => set(c),
}))
