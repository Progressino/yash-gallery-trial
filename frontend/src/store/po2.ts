import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/** PO Engine 2 — isolated params (app-data target-cover mode only). */
export interface PO2Params {
  period_days: number
  lead_time: number
  target_days: number
  grace_days: number
  demand_basis: 'Sold' | 'Net'
  enforce_two_size_minimum: boolean
  urgent_all_sizes_days: number
}

export interface PO2Row {
  OMS_SKU: string
  [key: string]: string | number | undefined
}

export interface PO2Result {
  ok: boolean
  message?: string
  rows?: PO2Row[]
  columns?: string[]
  summary?: {
    new_po_qty_sum?: number
    new_po_sku_count?: number
    pipeline_qty_sum?: number
    pipeline_sku_count?: number
  }
}

interface PO2State {
  params: PO2Params
  setParams: (p: PO2Params) => void
}

export const PO2_DEFAULT_PARAMS: PO2Params = {
  period_days: 30,
  lead_time: 60,
  target_days: 180,
  grace_days: 0,
  demand_basis: 'Sold',
  enforce_two_size_minimum: true,
  urgent_all_sizes_days: 45,
}

export const usePO2Store = create<PO2State>()(
  persist(
    (set) => ({
      params: PO2_DEFAULT_PARAMS,
      setParams: (p) => set({ params: p }),
    }),
    {
      name: 'po2-store-v1',
      partialize: (s) => ({ params: s.params }),
    },
  ),
)
