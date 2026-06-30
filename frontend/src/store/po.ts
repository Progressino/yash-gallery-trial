import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface PORow {
  OMS_SKU: string
  [key: string]: string | number | undefined
}

interface POResult {
  ok: boolean
  message?: string
  rows?: PORow[]
  columns?: string[]
}

interface QuarterlyRow {
  OMS_SKU: string
  [key: string]: string | number | undefined
}

interface QuarterlyResult {
  loaded: boolean
  status?: 'warming' | 'error'
  progress?: number
  message?: string
  columns?: string[]
  rows?: QuarterlyRow[]
}

export type Tab = 'po' | 'dashboard' | 'quarterly' | 'shipment'

interface POParams {
  period_days: number
  lead_time: number
  target_days: number
  demand_basis: string
  use_seasonality: boolean
  seasonal_weight: number
  group_by_parent: boolean
  grace_days: number
  safety_pct: number
  enforce_two_size_minimum: boolean
  urgent_all_sizes_days: number
  enforce_lead_time_release_gate: boolean
  use_oms_inventory_only: boolean
}

interface POState {
  activeTab: Tab
  params: POParams
  result: POResult | null
  quarterly: QuarterlyResult | null
  search: string
  sortByPriority: boolean
  editedQty: Record<string, number>
  selected: Set<string>
  qSearch: string
  groupedView: boolean
  collapsedParents: Set<string>
  /** After Existing PO upload, next Calculate PO skips shared cache once. */
  skipSharedCacheOnce: boolean

  setActiveTab: (t: Tab) => void
  setParams: (p: POParams) => void
  setResult: (r: POResult | null) => void
  setQuarterly: (q: QuarterlyResult | null) => void
  setSearch: (s: string) => void
  setSortByPriority: (v: boolean) => void
  setEditedQty: (q: Record<string, number>) => void
  setSelected: (s: Set<string>) => void
  setQSearch: (s: string) => void
  setGroupedView: (v: boolean) => void
  setCollapsedParents: (s: Set<string>) => void
  setSkipSharedCacheOnce: (v: boolean) => void
}

export const usePOStore = create<POState>()(
  persist(
    (set) => ({
  activeTab: 'po',
  params: {
    period_days: 30,
    lead_time: 45,
    target_days: 180,
    demand_basis: 'Sold',
    use_seasonality: false,
    seasonal_weight: 0.5,
    group_by_parent: false,
    grace_days: 0,
    safety_pct: 0,
    enforce_two_size_minimum: true,
    urgent_all_sizes_days: 45,
    enforce_lead_time_release_gate: false,
    use_oms_inventory_only: false,
  },
  result: null,
  quarterly: null,
  search: '',
  sortByPriority: true,
  editedQty: {},
  selected: new Set(),
  qSearch: '',
  groupedView: false,
  collapsedParents: new Set(),
  skipSharedCacheOnce: false,

  setActiveTab:        (t) => set({ activeTab: t }),
  setParams:           (p) => set({ params: p }),
  setResult:           (r) => set({ result: r }),
  setQuarterly:        (q) => set({ quarterly: q }),
  setSearch:           (s) => set({ search: s }),
  setSortByPriority:   (v) => set({ sortByPriority: v }),
  setEditedQty:        (q) => set({ editedQty: q }),
  setSelected:         (s) => set({ selected: s }),
  setQSearch:          (s) => set({ qSearch: s }),
  setGroupedView:      (v) => set({ groupedView: v }),
  setCollapsedParents: (s) => set({ collapsedParents: s }),
  setSkipSharedCacheOnce: (v) => set({ skipSharedCacheOnce: v }),
    }),
    {
      name: 'po-store-v1',
      version: 6,
      migrate: (persisted, fromVersion) => {
        const p = persisted as Partial<POState> | undefined
        if (!p) return persisted as POState
        if (fromVersion < 3) {
          p.quarterly = null
        }
        const params = p.params as POParams | undefined
        if (params && params.enforce_lead_time_release_gate === undefined) {
          params.enforce_lead_time_release_gate = false
        }
        // App-only PO mode: keep lead-time gate off by default.
        if (fromVersion < 5 && params) {
          params.enforce_lead_time_release_gate = false
        }
        if (fromVersion < 6 && params && params.use_oms_inventory_only === undefined) {
          params.use_oms_inventory_only = false
        }
        return p as POState
      },
      // Persist only user preferences/params; never persist large results or Sets.
      partialize: (s) => ({
        activeTab: s.activeTab,
        params: s.params,
        sortByPriority: s.sortByPriority,
        groupedView: s.groupedView,
      }),
    }
  )
)
