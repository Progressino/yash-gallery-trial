import { create } from 'zustand'

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
  columns?: string[]
  rows?: QuarterlyRow[]
}

type Tab = 'po' | 'quarterly' | 'shipment'

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
}

export const usePOStore = create<POState>((set) => ({
  activeTab: 'po',
  params: {
    period_days: 90,
    lead_time: 30,
    target_days: 210,
    demand_basis: 'Sold',
    use_seasonality: false,
    seasonal_weight: 0.5,
    group_by_parent: false,
    grace_days: 0,
    safety_pct: 0,
    enforce_two_size_minimum: true,
  },
  result: null,
  quarterly: null,
  search: '',
  sortByPriority: true,
  editedQty: {},
  selected: new Set(),
  qSearch: '',
  groupedView: true,
  collapsedParents: new Set(),

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
}))
