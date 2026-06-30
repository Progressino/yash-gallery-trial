import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { POCalculateResult } from '../api/client'

export type POFreshTab = 'po' | 'dashboard' | 'quarterly'

export type POFreshParams = {
  period_days: number
  lead_time: number
  target_days: number
  grace_days: number
  demand_basis: 'Sold' | 'Net'
  group_by_parent: boolean
  safety_pct: number
  use_seasonality: boolean
  use_ly_fallback: boolean
  use_oms_inventory_only: boolean
  enforce_two_size_minimum: boolean
  raise_view_date: string
  raise_ledger_lookback_days: number
}

export const DEFAULT_PO_FRESH_PARAMS: POFreshParams = {
  period_days: 30,
  lead_time: 45,
  target_days: 180,
  grace_days: 0,
  demand_basis: 'Sold',
  group_by_parent: false,
  safety_pct: 0,
  use_seasonality: true,
  use_ly_fallback: true,
  enforce_two_size_minimum: true,
  use_oms_inventory_only: false,
  raise_view_date: '2026-05-16',
  raise_ledger_lookback_days: 45,
}

type QuarterlyResult = {
  loaded?: boolean
  status?: string
  progress?: number
  message?: string
  columns?: string[]
  rows?: Record<string, string | number | undefined>[]
}

type POFreshState = {
  params: POFreshParams
  tab: POFreshTab
  result: POCalculateResult | null
  quarterly: QuarterlyResult | null
  search: string
  priorityFilter: 'all' | 'urgent' | 'with_po' | 'blocked'
  sortByPriority: boolean
  page: number
  fromSharedCache: boolean
  setParams: (p: POFreshParams | ((prev: POFreshParams) => POFreshParams)) => void
  setTab: (t: POFreshTab) => void
  setResult: (r: POCalculateResult | null) => void
  setQuarterly: (q: QuarterlyResult | null) => void
  setSearch: (s: string) => void
  setPriorityFilter: (f: POFreshState['priorityFilter']) => void
  setSortByPriority: (v: boolean) => void
  setPage: (p: number) => void
  setFromSharedCache: (v: boolean) => void
  clearResult: () => void
}

export const usePOFreshStore = create<POFreshState>()(
  persist(
    set => ({
      params: DEFAULT_PO_FRESH_PARAMS,
      tab: 'po',
      result: null,
      quarterly: null,
      search: '',
      priorityFilter: 'all',
      sortByPriority: true,
      page: 0,
      fromSharedCache: false,
      setParams: p =>
        set(s => ({
          params: typeof p === 'function' ? p(s.params) : p,
        })),
      setTab: t => set({ tab: t }),
      setResult: r => set({ result: r, fromSharedCache: Boolean(r?.from_shared_cache) }),
      setQuarterly: q => set({ quarterly: q }),
      setSearch: s => set({ search: s }),
      setPriorityFilter: f => set({ priorityFilter: f }),
      setSortByPriority: v => set({ sortByPriority: v }),
      setPage: p => set({ page: p }),
      setFromSharedCache: v => set({ fromSharedCache: v }),
      clearResult: () => set({ result: null, quarterly: null, fromSharedCache: false }),
    }),
    {
      name: 'po-fresh-store-v1',
      partialize: s => ({
        params: s.params,
        tab: s.tab,
        sortByPriority: s.sortByPriority,
        priorityFilter: s.priorityFilter,
      }),
    },
  ),
)
