import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'demands' | 'orders' | 'reports' | 'tracker' | 'settings'
type ReportView = 'sku-pending' | 'delivery-due' | 'buyer' | 'source' | 'demand-coverage'
type SettingsTab = 'buyers' | 'warehouses' | 'teams'

interface Item {
  id: number; item_code: string; item_name: string
  parent_id: number | null; size_label: string; uom: string
  variant_count?: number
  variants?: { id: number; item_code: string; item_name: string; size_label: string; uom: string }[]
}
interface Demand {
  id: number; demand_number: string; demand_date: string
  demand_source: string; buyer: string; priority: string; status: string
  notes: string; lines: DemandLine[]
}
interface DemandLine { id: number; sku: string; sku_name: string; demand_qty: number; delivered_qty: number }
interface SalesOrder {
  id: number; so_number: string; so_date: string; buyer: string
  warehouse: string; delivery_date: string; status: string; notes: string
  source_type: string; ref_demand: string; sales_team: string; payment_terms: string 
dispatch_date: string; ref_number: string; ref_date: string
  lines: SOLine[]
}
interface SOLine {
  id: number; sku: string; sku_name: string; qty: number
  produced_qty: number; dispatch_qty: number; received_qty: number; unit: string
  rate: number; delivery_date: string; remarks: string
}

const SOURCES = ['Sales Team', 'Forecasting System', 'Buyer Indent', 'Marketing Team']
const PRIORITIES = ['Normal', 'High', 'Urgent']
const DEMAND_STATUSES = ['Draft', 'Submitted', 'Approved', 'In Production', 'Closed', 'Cancelled']
const SO_STATUSES = ['Draft', 'Submitted', 'Approved', 'In Production', 'Partially Dispatched', 'Dispatched', 'Closed', 'Cancelled']
const SOURCE_TYPES = ['Sales Team Demand', 'Buyer PO', 'Offline Order']
const UOM_OPTIONS = ['PCS', 'SET', 'PAIR', 'BOX', 'MTR', 'KG']

const DEFAULT_BUYERS = ['Myntra', 'Flipkart', 'Amazon', 'Reliance', 'Direct']
const DEFAULT_WAREHOUSES = ['Main Warehouse', 'Finished Goods Store', 'Export Warehouse']
const DEFAULT_TEAMS = ['Team Alpha', 'Team Beta', 'North Zone', 'South Zone', 'Export Desk']

const statusColor = (s: string) => {
  if (['Approved', 'Dispatched', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (['Draft', 'Submitted'].includes(s)) return 'bg-yellow-100 text-yellow-700'
  if (s === 'In Production') return 'bg-blue-100 text-blue-700'
  if (['Cancelled'].includes(s)) return 'bg-red-100 text-red-700'
  return 'bg-gray-100 text-gray-600'
}
const priorityColor = (p: string) => {
  if (p === 'Urgent') return 'bg-red-100 text-red-700'
  if (p === 'High') return 'bg-orange-100 text-orange-700'
  return 'bg-gray-100 text-gray-600'
}

// ── SKU Autocomplete Picker ───────────────────────────────────────────────────
function SkuPicker({ value, onChange }: {
  value: string
  onChange: (sku: string, skuName: string, uom: string) => void
}) {
  const [q, setQ] = useState(value)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const { data: results = [] } = useQuery<Item[]>({
    queryKey: ['sku-search', q],
    queryFn: () => api.get(`/items/search?q=${encodeURIComponent(q)}`).then(r => r.data),
    enabled: q.length >= 2,
    staleTime: 30_000,
  })

  useEffect(() => { setQ(value) }, [value])

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div ref={ref} className="relative flex-1">
      <input
        value={q}
        onChange={e => { setQ(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        placeholder="Search SKU…"
        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm"
      />
      {open && results.length > 0 && (
        <div className="absolute z-50 bg-white border border-gray-200 rounded-lg shadow-lg mt-1 max-h-48 overflow-y-auto w-72">
          {results.map(item => (
            <button key={item.id} type="button"
              className="w-full text-left px-3 py-2 hover:bg-blue-50 text-sm border-b border-gray-50 last:border-0"
              onClick={() => {
                onChange(item.item_code, item.item_name, item.uom || 'PCS')
                setQ(item.item_code)
                setOpen(false)
              }}>
              <span className="font-mono font-medium text-[#002B5B]">{item.item_code}</span>
              {item.size_label && <span className="ml-1 text-xs text-gray-400">({item.size_label})</span>}
              <span className="ml-2 text-gray-500 text-xs truncate">{item.item_name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Parent SKU Picker with size expansion ─────────────────────────────────────
function ParentSkuPicker({ onAddLines }: {
  onAddLines: (lines: { sku: string; sku_name: string; demand_qty: number }[]) => void
}) {
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const { data: results = [] } = useQuery<Item[]>({
    queryKey: ['sku-search-parent', q],
    queryFn: () => api.get(`/items?search=${encodeURIComponent(q)}&parent_only=true`).then(r => r.data),
    enabled: q.length >= 2,
    staleTime: 30_000,
  })

  const { data: itemDetail } = useQuery<Item>({
    queryKey: ['sku-search-detail', q],
    queryFn: async () => {
      const found = results.find(r => r.item_code === q)
      if (!found) return null as any
      return api.get(`/items/${found.id}`).then(r => r.data)
    },
    enabled: results.some(r => r.item_code === q),
    staleTime: 30_000,
  })

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  function selectItem(item: Item) {
    setQ(item.item_code)
    setOpen(false)
  }

  function addSizeWise() {
    if (!itemDetail) return
    const variants = itemDetail.variants || []
    if (variants.length > 0) {
      onAddLines(variants.map(v => ({ sku: v.item_code, sku_name: v.item_name + (v.size_label ? ` (${v.size_label})` : ''), demand_qty: 0 })))
    } else {
      const found = results.find(r => r.item_code === q)
      if (found) onAddLines([{ sku: found.item_code, sku_name: found.item_name, demand_qty: 0 }])
    }
    setQ('')
  }

  const selected = results.find(r => r.item_code === q)

  return (
    <div ref={ref} className="flex gap-2 items-center mb-2">
      <div className="relative flex-1">
        <input value={q} onChange={e => { setQ(e.target.value); setOpen(true) }}
          onFocus={() => setOpen(true)}
          placeholder="Search parent SKU to add size-wise lines…"
          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm" />
        {open && results.length > 0 && (
          <div className="absolute z-50 bg-white border border-gray-200 rounded-lg shadow-lg mt-1 max-h-48 overflow-y-auto w-full">
            {results.map(item => (
              <button key={item.id} type="button"
                className="w-full text-left px-3 py-2 hover:bg-blue-50 text-sm border-b border-gray-50"
                onClick={() => selectItem(item)}>
                <span className="font-mono font-medium text-[#002B5B]">{item.item_code}</span>
                <span className="ml-2 text-gray-500 text-xs">{item.item_name}</span>
                {item.variant_count != null && item.variant_count > 0 &&
                  <span className="ml-2 text-blue-500 text-xs">{item.variant_count} sizes</span>}
              </button>
            ))}
          </div>
        )}
      </div>
      {selected && (
        <button type="button" onClick={addSizeWise}
          className="text-xs bg-[#002B5B] text-white px-3 py-1.5 rounded hover:bg-blue-800">
          + Add Size-wise
        </button>
      )}
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function SalesOrders() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [showNewDemand, setShowNewDemand] = useState(false)
  const [showNewSO, setShowNewSO] = useState(false)
  const [expandedDemand, setExpandedDemand] = useState<number | null>(null)
  const [expandedSO, setExpandedSO] = useState<number | null>(null)
  const [filterStatus, setFilterStatus] = useState('')
  const [editingLineId, setEditingLineId] = useState<number | null>(null)
  const [editLineForm, setEditLineForm] = useState({ qty: 0, rate: 0, delivery_date: '', remarks: '', produced_qty: 0, dispatch_qty: 0, received_qty: 0 })

  // Reports & settings state
  const [reportView, setReportView] = useState<ReportView>('sku-pending')
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('buyers')
  const [newSettingVal, setNewSettingVal] = useState('')
  const [customBuyers, setCustomBuyers] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('erp_buyers') || 'null') || DEFAULT_BUYERS } catch { return DEFAULT_BUYERS }
  })
  const [customWarehouses, setCustomWarehouses] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('erp_warehouses') || 'null') || DEFAULT_WAREHOUSES } catch { return DEFAULT_WAREHOUSES }
  })
  const [customTeams, setCustomTeams] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('erp_teams') || 'null') || DEFAULT_TEAMS } catch { return DEFAULT_TEAMS }
  })

  // Demand form
  const [dForm, setDForm] = useState({ demand_source: 'Sales Team', buyer: '', priority: 'Normal', notes: '' })
  const [dLines, setDLines] = useState<{ sku: string; sku_name: string; demand_qty: number }[]>([])

  // SO form
  const [soForm, setSOForm] = useState({
    buyer: '', warehouse: '', sales_team: '', source_type: 'Sales Team Demand',
    ref_demand: '', delivery_date: '', payment_terms: '', notes: '', dispatch_date: '', ref_number: '', ref_date: ''
  })
  const [soLines, setSOLines] = useState<{
    sku: string; sku_name: string; qty: number; unit: string; rate: number; remarks: string
    hsn_code: string; gst_pct: number; merchant_code: string; priority: string; line_delivery_date: string
  }[]>([])
  const [fetchingDemand, setFetchingDemand] = useState(false)

  const { data: demands = [] } = useQuery<Demand[]>({
    queryKey: ['demands', filterStatus],
    queryFn: () => api.get('/sales/demands' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data)
  })
  const { data: orders = [] } = useQuery<SalesOrder[]>({
    queryKey: ['sales-orders', filterStatus],
    queryFn: () => api.get('/sales/orders' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data)
  })

  // All orders (unfiltered) for reports/tracker
  const { data: allOrders = [] } = useQuery<SalesOrder[]>({
    queryKey: ['sales-orders-all'],
    queryFn: () => api.get('/sales/orders').then(r => r.data),
    enabled: tab === 'reports' || tab === 'tracker'
  })
  const { data: allDemands = [] } = useQuery<Demand[]>({
    queryKey: ['demands-all'],
    queryFn: () => api.get('/sales/demands').then(r => r.data),
    enabled: tab === 'reports'
  })

  const createDemandMut = useMutation({
    mutationFn: (body: object) => api.post('/sales/demands', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['demands'] })
      setShowNewDemand(false); setDLines([])
      setDForm({ demand_source: 'Sales Team', buyer: '', priority: 'Normal', notes: '' })
    }
  })
  const createSOMut = useMutation({
    mutationFn: (body: object) => api.post('/sales/orders', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sales-orders'] }); qc.invalidateQueries({ queryKey: ['sales-orders-all'] })
      setShowNewSO(false); setSOLines([])
      setSOForm({ buyer: '', warehouse: '', sales_team: '', source_type: 'Sales Team Demand', ref_demand: '', delivery_date: '', payment_terms: '', notes: '', dispatch_date: '', ref_number: '', ref_date: '' })
    }
  })
  const updateDemandMut = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/sales/demands/${id}/status`, { status }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['demands'] })
  })
  const updateSOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/sales/orders/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['sales-orders'] }); qc.invalidateQueries({ queryKey: ['sales-orders-all'] }) }
  })
  const updateSOLineMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/sales/orders/lines/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['sales-orders'] }); qc.invalidateQueries({ queryKey: ['sales-orders-all'] }); setEditingLineId(null) }
  })

  function addDLine() { setDLines(l => [...l, { sku: '', sku_name: '', demand_qty: 0 }]) }
  const blankSOLine = () => ({
    sku: '', sku_name: '', qty: 0, unit: 'PCS', rate: 0, remarks: '',
    hsn_code: '', gst_pct: 0, merchant_code: '', priority: 'Normal', line_delivery_date: soForm.delivery_date
  })
  function addSOLine() { setSOLines(l => [...l, blankSOLine()]) }

  async function fetchDemandLines() {
    if (!soForm.ref_demand.trim()) return
    setFetchingDemand(true)
    try {
      const { data } = await api.get(`/sales/demands/by-number/${encodeURIComponent(soForm.ref_demand.trim())}`)
      const lines = (data.lines || []).map((l: DemandLine) => ({
        ...blankSOLine(), sku: l.sku, sku_name: l.sku_name, qty: l.demand_qty,
      }))
      setSOLines(lines)
      if (data.buyer && !soForm.buyer) setSOForm(f => ({ ...f, buyer: data.buyer }))
    } catch {
      // demand not found — lines stay as-is
    } finally { setFetchingDemand(false) }
  }

  // Settings helpers
  function saveSetting(key: string, list: string[], setList: (l: string[]) => void) {
    const v = newSettingVal.trim()
    if (!v || list.includes(v)) return
    const updated = [...list, v]
    setList(updated)
    localStorage.setItem(key, JSON.stringify(updated))
    setNewSettingVal('')
  }
  function deleteSetting(key: string, val: string, list: string[], setList: (l: string[]) => void) {
    const updated = list.filter(x => x !== val)
    setList(updated)
    localStorage.setItem(key, JSON.stringify(updated))
  }

  // Computed report data
  const now = new Date()
  const openOrdersForReport = allOrders.filter(o => !['Closed', 'Cancelled'].includes(o.status))

  // SKU Pending
  const skuPendingMap = openOrdersForReport.flatMap(o => o.lines).reduce((acc, l) => {
    if (!acc[l.sku]) acc[l.sku] = { sku: l.sku, sku_name: l.sku_name, ordered: 0, produced: 0, dispatched: 0 }
    acc[l.sku].ordered += l.qty
    acc[l.sku].produced += l.produced_qty
    acc[l.sku].dispatched += l.dispatch_qty
    return acc
  }, {} as Record<string, { sku: string; sku_name: string; ordered: number; produced: number; dispatched: number }>)
  const skuPending = Object.values(skuPendingMap).sort((a, b) => (b.ordered - b.dispatched) - (a.ordered - a.dispatched))

  // Delivery Due
  const deliveryDue = openOrdersForReport
    .filter(o => o.delivery_date)
    .map(o => {
      const dl = Math.ceil((new Date(o.delivery_date).getTime() - now.getTime()) / (1000 * 60 * 60 * 24))
      const totalQty = o.lines.reduce((s, l) => s + l.qty, 0)
      const dispatched = o.lines.reduce((s, l) => s + l.dispatch_qty, 0)
      return { ...o, daysLeft: dl, totalQty, dispatched }
    })
    .sort((a, b) => a.daysLeft - b.daysLeft)

  // Buyer Summary
  const buyerSummary = allOrders
    .filter(o => o.status !== 'Cancelled')
    .reduce((acc, o) => {
      const b = o.buyer || '(none)'
      if (!acc[b]) acc[b] = { buyer: b, so_count: 0, total_qty: 0, dispatched_qty: 0, total_value: 0 }
      acc[b].so_count++
      o.lines.forEach(l => {
        acc[b].total_qty += l.qty
        acc[b].dispatched_qty += l.dispatch_qty
        acc[b].total_value += l.qty * (l.rate || 0)
      })
      return acc
    }, {} as Record<string, { buyer: string; so_count: number; total_qty: number; dispatched_qty: number; total_value: number }>)

  // Source Summary
  const sourceSummary = allOrders
    .filter(o => o.status !== 'Cancelled')
    .reduce((acc, o) => {
      const s = o.source_type || '(none)'
      if (!acc[s]) acc[s] = { source: s, so_count: 0, total_qty: 0, total_value: 0 }
      acc[s].so_count++
      o.lines.forEach(l => {
        acc[s].total_qty += l.qty
        acc[s].total_value += l.qty * (l.rate || 0)
      })
      return acc
    }, {} as Record<string, { source: string; so_count: number; total_qty: number; total_value: number }>)

  // Demand coverage
  const demandCoverage = allDemands.map(d => {
    const linkedSOs = allOrders.filter(o => o.ref_demand === d.demand_number)
    const demandQty = d.lines.reduce((s, l) => s + l.demand_qty, 0)
    const soQty = linkedSOs.flatMap(o => o.lines).reduce((s, l) => s + l.qty, 0)
    const pct = demandQty > 0 ? Math.round(soQty / demandQty * 100) : 0
    return { ...d, demandQty, soQty, pct, linkedCount: linkedSOs.length }
  })

  const openDemands = demands.filter(d => !['Closed', 'Cancelled'].includes(d.status)).length
  const openOrders = orders.filter(o => !['Closed', 'Cancelled'].includes(o.status)).length
  const urgentDemands = demands.filter(d => d.priority === 'Urgent' && !['Closed', 'Cancelled'].includes(d.status)).length
  const TABS: [Tab, string][] = [
    ['dashboard', '📊 Dashboard'], ['demands', '📋 Demands'], ['orders', '🧾 Sales Orders'],
    ['tracker', '📈 Tracker'], ['reports', '📉 Reports'], ['settings', '⚙️ Settings']
  ]

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800">Sales Orders & Demand</h1>
          <p className="text-sm text-gray-500">Manage demands, sales orders and tracking</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit flex-wrap">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* ── Dashboard ── */}
      {tab === 'dashboard' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'OPEN DEMANDS', value: openDemands, color: 'text-blue-600' },
              { label: 'OPEN ORDERS', value: openOrders, color: 'text-green-600' },
              { label: 'URGENT', value: urgentDemands, color: 'text-red-600' },
              { label: 'TOTAL DEMANDS', value: demands.length, color: 'text-gray-700' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
              </div>
            ))}
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent Demands</h3>
              {demands.slice(0, 5).map(d => (
                <div key={d.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                  <div>
                    <p className="text-sm font-medium text-gray-700">{d.demand_number}</p>
                    <p className="text-xs text-gray-400">{d.buyer} · {d.demand_date}</p>
                  </div>
                  <div className="flex gap-1">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${priorityColor(d.priority)}`}>{d.priority}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(d.status)}`}>{d.status}</span>
                  </div>
                </div>
              ))}
              {demands.length === 0 && <p className="text-xs text-gray-400">No demands yet</p>}
            </div>
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent Sales Orders</h3>
              {orders.slice(0, 5).map(o => (
                <div key={o.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                  <div>
                    <p className="text-sm font-medium text-gray-700">{o.so_number}</p>
                    <p className="text-xs text-gray-400">{o.buyer} · {o.delivery_date || '—'}</p>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(o.status)}`}>{o.status}</span>
                </div>
              ))}
              {orders.length === 0 && <p className="text-xs text-gray-400">No orders yet</p>}
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={() => { setTab('demands'); setShowNewDemand(true) }}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
              + New Demand
            </button>
            <button onClick={() => { setTab('orders'); setShowNewSO(true) }}
              className="px-4 py-2 border border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-medium hover:bg-gray-50">
              + New Sales Order
            </button>
          </div>
        </div>
      )}

      {/* ── Demands ── */}
      {tab === 'demands' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Statuses</option>
              {DEMAND_STATUSES.map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowNewDemand(true)}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
              + New Demand
            </button>
          </div>

          {showNewDemand && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-4">
              <h3 className="font-semibold text-gray-700">New Demand</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="text-xs text-gray-500">Source</label>
                  <select value={dForm.demand_source} onChange={e => setDForm(f => ({ ...f, demand_source: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {SOURCES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Buyer</label>
                  <input list="buyers-list" value={dForm.buyer} onChange={e => setDForm(f => ({ ...f, buyer: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  <datalist id="buyers-list">{customBuyers.map(b => <option key={b} value={b} />)}</datalist>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Priority</label>
                  <select value={dForm.priority} onChange={e => setDForm(f => ({ ...f, priority: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {PRIORITIES.map(p => <option key={p}>{p}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Notes</label>
                  <input value={dForm.notes} onChange={e => setDForm(f => ({ ...f, notes: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>

              {/* SKU Lines */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-600">SKU Lines</p>
                  <button onClick={addDLine} className="text-xs text-blue-600 hover:underline">+ Add Line Manually</button>
                </div>

                <ParentSkuPicker onAddLines={lines => setDLines(l => [...l, ...lines])} />

                {dLines.length > 0 && (
                  <div className="flex gap-2 mb-1 px-1">
                    <span className="text-xs text-gray-400 flex-1">SKU Code</span>
                    <span className="text-xs text-gray-400 flex-1">SKU Name</span>
                    <span className="text-xs text-gray-400 w-24">Qty</span>
                    <span className="w-6" />
                  </div>
                )}

                {dLines.map((ln, i) => (
                  <div key={i} className="flex gap-2 mb-2 items-center">
                    <SkuPicker
                      value={ln.sku}
                      onChange={(sku, skuName) => setDLines(l => l.map((x, j) => j === i ? { ...x, sku, sku_name: skuName } : x))}
                    />
                    <input placeholder="SKU Name" value={ln.sku_name}
                      onChange={e => setDLines(l => l.map((x, j) => j === i ? { ...x, sku_name: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm flex-1" />
                    <input type="number" placeholder="Qty" value={ln.demand_qty}
                      onChange={e => setDLines(l => l.map((x, j) => j === i ? { ...x, demand_qty: +e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-24" />
                    <button onClick={() => setDLines(l => l.filter((_, j) => j !== i))}
                      className="text-red-400 hover:text-red-600 text-sm px-1">✕</button>
                  </div>
                ))}
                {dLines.length === 0 && (
                  <p className="text-xs text-gray-400 mt-1">Search a parent SKU above to add size-wise lines, or click "+ Add Line Manually".</p>
                )}
              </div>

              <div className="flex gap-2">
                <button onClick={() => createDemandMut.mutate({ ...dForm, lines: dLines })}
                  disabled={createDemandMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                  {createDemandMut.isPending ? 'Saving…' : 'Create Demand'}
                </button>
                <button onClick={() => setShowNewDemand(false)}
                  className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-gray-50">Cancel</button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {demands.map(d => (
              <div key={d.id} className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer"
                  onClick={() => setExpandedDemand(expandedDemand === d.id ? null : d.id)}>
                  <div className="flex items-center gap-3">
                    <div>
                      <p className="font-semibold text-gray-800 text-sm">{d.demand_number}</p>
                      <p className="text-xs text-gray-500">{d.buyer} · {d.demand_source} · {d.demand_date}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${priorityColor(d.priority)}`}>{d.priority}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(d.status)}`}>{d.status}</span>
                    <select value={d.status} onChange={e => updateDemandMut.mutate({ id: d.id, status: e.target.value })}
                      onClick={e => e.stopPropagation()}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {DEMAND_STATUSES.map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expandedDemand === d.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expandedDemand === d.id && (
                  <div className="border-t border-gray-50 px-4 pb-4">
                    <table className="w-full text-xs mt-3">
                      <thead>
                        <tr className="text-gray-400 uppercase">
                          <th className="text-left py-1">SKU</th>
                          <th className="text-left py-1">Name</th>
                          <th className="text-right py-1">Qty Demanded</th>
                          <th className="text-right py-1">Delivered</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.lines.map(l => (
                          <tr key={l.id} className="border-t border-gray-50">
                            <td className="py-1.5 font-mono font-medium text-gray-700">{l.sku}</td>
                            <td className="py-1.5 text-gray-600">{l.sku_name}</td>
                            <td className="py-1.5 text-right text-gray-700">{l.demand_qty.toLocaleString()}</td>
                            <td className="py-1.5 text-right text-green-600">{l.delivered_qty.toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {d.notes && <p className="text-xs text-gray-500 mt-2">📝 {d.notes}</p>}
                    <button
                      onClick={() => { setTab('orders'); setShowNewSO(true); setSOForm(f => ({ ...f, ref_demand: d.demand_number, buyer: d.buyer })) }}
                      className="mt-3 text-xs text-blue-600 hover:underline">
                      → Create SO from this Demand
                    </button>
                  </div>
                )}
              </div>
            ))}
            {demands.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No demands found. Create your first demand.</p>}
          </div>
        </div>
      )}

      {/* ── Sales Orders ── */}
      {tab === 'orders' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Statuses</option>
              {SO_STATUSES.map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowNewSO(true)}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
              + New Sales Order
            </button>
          </div>

          {showNewSO && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-4">
              <h3 className="font-semibold text-gray-700">New Sales Order</h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="text-xs text-gray-500">Buyer</label>
                  <input list="so-buyers-list" value={soForm.buyer} onChange={e => setSOForm(f => ({ ...f, buyer: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  <datalist id="so-buyers-list">{customBuyers.map(b => <option key={b} value={b} />)}</datalist>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Warehouse</label>
                  <input list="so-wh-list" value={soForm.warehouse} onChange={e => setSOForm(f => ({ ...f, warehouse: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  <datalist id="so-wh-list">{customWarehouses.map(w => <option key={w} value={w} />)}</datalist>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Sales Team</label>
                  <input list="so-teams-list" value={soForm.sales_team} onChange={e => setSOForm(f => ({ ...f, sales_team: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  <datalist id="so-teams-list">{customTeams.map(t => <option key={t} value={t} />)}</datalist>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Payment Terms</label>
                  <input value={soForm.payment_terms} onChange={e => setSOForm(f => ({ ...f, payment_terms: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Delivery Date</label>
                  <input type="date" value={soForm.delivery_date} onChange={e => setSOForm(f => ({ ...f, delivery_date: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div>
  <label className="text-xs text-gray-500">Planned Dispatch Date</label>
  <input type="date" value={soForm.dispatch_date}
    onChange={e => setSOForm(f => ({ ...f, dispatch_date: e.target.value }))}
    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
</div>
<div>
  <label className="text-xs text-gray-500">Buyer PO / Ref Number</label>
  <input value={soForm.ref_number}
    onChange={e => setSOForm(f => ({ ...f, ref_number: e.target.value }))}
    placeholder="e.g. BPO/2024/001"
    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
</div>
<div>
  <label className="text-xs text-gray-500">Ref Date</label>
  <input type="date" value={soForm.ref_date}
    onChange={e => setSOForm(f => ({ ...f, ref_date: e.target.value }))}
    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
</div>
                <div>
                  <label className="text-xs text-gray-500">Source Type</label>
                  <select value={soForm.source_type} onChange={e => setSOForm(f => ({ ...f, source_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {SOURCE_TYPES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Notes</label>
                  <input value={soForm.notes} onChange={e => setSOForm(f => ({ ...f, notes: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>

              {/* Demand auto-fetch */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 space-y-2">
                <p className="text-xs font-semibold text-blue-700 uppercase tracking-wide">Link to Demand (Auto-fetch Lines)</p>
                <div className="flex gap-2 items-center">
                  <select value={soForm.ref_demand}
                    onChange={e => setSOForm(f => ({ ...f, ref_demand: e.target.value }))}
                    className="flex-1 border border-blue-200 rounded px-2 py-1.5 text-sm bg-white">
                    <option value="">— Select Demand —</option>
                    {demands.filter(d => !['Closed', 'Cancelled'].includes(d.status)).map(d => (
                      <option key={d.id} value={d.demand_number}>
                        {d.demand_number} · {d.buyer} · {d.priority}
                      </option>
                    ))}
                  </select>
                  <button onClick={fetchDemandLines} disabled={!soForm.ref_demand || fetchingDemand}
                    className="px-3 py-1.5 bg-blue-600 text-white rounded text-xs font-medium hover:bg-blue-700 disabled:opacity-50">
                    {fetchingDemand ? 'Fetching…' : 'Fetch Lines'}
                  </button>
                </div>
                {soLines.length > 0 && soForm.ref_demand && (
                  <p className="text-xs text-blue-600">✓ {soLines.length} line(s) fetched from {soForm.ref_demand}</p>
                )}
              </div>

              {/* SO Lines — Streamlit-style card layout */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-semibold text-gray-700">SKU Lines
                    {soLines.length > 0 && <span className="ml-2 text-xs text-gray-400 font-normal">({soLines.length} line{soLines.length > 1 ? 's' : ''})</span>}
                  </p>
                  <button onClick={addSOLine}
                    className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-medium hover:bg-blue-700">
                    + Add SKU Line
                  </button>
                </div>

                {soLines.map((ln, i) => (
                  <div key={i} className="border border-gray-200 rounded-xl p-4 bg-gray-50 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Line {i + 1}</span>
                      <button onClick={() => setSOLines(l => l.filter((_, j) => j !== i))}
                        className="text-red-400 hover:text-red-600 text-xs px-2 py-0.5 rounded hover:bg-red-50">✕ Remove</button>
                    </div>

                    {/* Row 1: SKU + UOM + HSN */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">SKU / Style-Size *</label>
                        <SkuPicker
                          value={ln.sku}
                          onChange={async (sku, skuName, uom) => {
                            // Auto-fetch item details for HSN, GST%, merchant, rate
                            let hsn = '', gst = 0, merchant = '', rate = ln.rate
                            try {
                              const { data: itm } = await api.get(`/items/items?search=${encodeURIComponent(sku)}&limit=1`)
                              const item = itm?.[0]
                              if (item) {
                                hsn = item.hsn_code || ''
                                gst = item.gst_rate || 0
                                merchant = item.merchant_code || ''
                                rate = item.selling_price || ln.rate
                              }
                            } catch { /* ignore */ }
                            setSOLines(l => l.map((x, j) => j === i
                              ? { ...x, sku, sku_name: skuName, unit: uom, hsn_code: hsn, gst_pct: gst, merchant_code: merchant, rate }
                              : x))
                          }}
                        />
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">UOM</label>
                        <select value={ln.unit}
                          onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, unit: e.target.value } : x))}
                          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm bg-white">
                          {UOM_OPTIONS.map(u => <option key={u}>{u}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">HSN Code</label>
                        <input value={ln.hsn_code}
                          onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, hsn_code: e.target.value } : x))}
                          placeholder="e.g. 61041300"
                          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm" />
                      </div>
                    </div>

                    {/* Row 2: Qty + Rate + Delivery Date */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">Quantity *</label>
                        <div className="flex items-center border border-gray-200 rounded bg-white overflow-hidden">
                          <button onClick={() => setSOLines(l => l.map((x, j) => j === i ? { ...x, qty: Math.max(0, x.qty - 1) } : x))}
                            className="px-3 py-1.5 text-gray-500 hover:bg-gray-100 font-bold text-sm">−</button>
                          <input type="number" value={ln.qty}
                            onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, qty: +e.target.value } : x))}
                            className="flex-1 text-center text-sm py-1.5 focus:outline-none border-0 bg-transparent" />
                          <button onClick={() => setSOLines(l => l.map((x, j) => j === i ? { ...x, qty: x.qty + 1 } : x))}
                            className="px-3 py-1.5 text-gray-500 hover:bg-gray-100 font-bold text-sm">+</button>
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">Rate (₹)</label>
                        <div className="flex items-center border border-gray-200 rounded bg-white overflow-hidden">
                          <button onClick={() => setSOLines(l => l.map((x, j) => j === i ? { ...x, rate: Math.max(0, +(x.rate - 1).toFixed(2)) } : x))}
                            className="px-3 py-1.5 text-gray-500 hover:bg-gray-100 font-bold text-sm">−</button>
                          <input type="number" value={ln.rate}
                            onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))}
                            className="flex-1 text-center text-sm py-1.5 focus:outline-none border-0 bg-transparent" />
                          <button onClick={() => setSOLines(l => l.map((x, j) => j === i ? { ...x, rate: +(x.rate + 1).toFixed(2) } : x))}
                            className="px-3 py-1.5 text-gray-500 hover:bg-gray-100 font-bold text-sm">+</button>
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">Line Delivery Date</label>
                        <input type="date" value={ln.line_delivery_date}
                          onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, line_delivery_date: e.target.value } : x))}
                          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm" />
                      </div>
                    </div>

                    {/* Row 3: Priority + GST% + Merchant Code + Remarks */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">Priority</label>
                        <select value={ln.priority}
                          onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, priority: e.target.value } : x))}
                          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm bg-white">
                          {['Normal','High','Urgent'].map(p => <option key={p}>{p}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">GST %</label>
                        <select value={String(ln.gst_pct)}
                          onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, gst_pct: +e.target.value } : x))}
                          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm bg-white">
                          {[0,5,12,18,28].map(r => <option key={r} value={r}>{r}%</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">Merchant Code</label>
                        <input value={ln.merchant_code}
                          onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, merchant_code: e.target.value } : x))}
                          placeholder="Merchant code"
                          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm" />
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-0.5 block">Remarks</label>
                        <input value={ln.remarks}
                          onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, remarks: e.target.value } : x))}
                          placeholder="Notes"
                          className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm" />
                      </div>
                    </div>
                    

                    {/* Line total */}
                    {(ln.qty > 0 && ln.rate > 0) && (
                      <div className="flex justify-end gap-4 text-xs text-gray-500 pt-1 border-t border-gray-100">
                        <span>Taxable: <strong className="text-gray-700">₹{(ln.qty * ln.rate).toLocaleString('en-IN', {minimumFractionDigits:2})}</strong></span>
                        {ln.gst_pct > 0 && <span>GST ({ln.gst_pct}%): <strong className="text-gray-700">₹{(ln.qty * ln.rate * ln.gst_pct / 100).toLocaleString('en-IN', {minimumFractionDigits:2})}</strong></span>}
                        <span>Total: <strong className="text-[#002B5B]">₹{(ln.qty * ln.rate * (1 + ln.gst_pct/100)).toLocaleString('en-IN', {minimumFractionDigits:2})}</strong></span>
                      </div>
                    )}
                  </div>
                ))}

                {soLines.length === 0 && (
                  <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 text-center text-gray-400 text-sm">
                    No SKU lines added. Fetch from a demand or click "+ Add SKU Line".
                  </div>
                )}

                {/* Order summary */}
                {soLines.length > 0 && (
                  <div className="bg-[#002B5B] text-white rounded-xl px-4 py-3 flex justify-between items-center text-sm">
                    <span>{soLines.reduce((s,l)=>s+l.qty,0).toLocaleString()} total units · {soLines.length} SKU{soLines.length>1?'s':''}</span>
                    <span className="font-bold">₹{soLines.reduce((s,l)=>s+(l.qty*l.rate*(1+l.gst_pct/100)),0).toLocaleString('en-IN',{minimumFractionDigits:2})}</span>
                  </div>
                )}
              </div>

              <div className="flex gap-2 pt-2">
                <button onClick={() => createSOMut.mutate({ ...soForm, status: 'Draft', lines: soLines })}
                  disabled={createSOMut.isPending}
                  className="px-4 py-2 border-2 border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-semibold hover:bg-blue-50 disabled:opacity-50">
                  {createSOMut.isPending ? 'Saving…' : '💾 Save as Draft'}
                </button>
                <button onClick={() => createSOMut.mutate({ ...soForm, status: 'Submitted', lines: soLines })}
                  disabled={createSOMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-semibold hover:bg-blue-800 disabled:opacity-50">
                  {createSOMut.isPending ? 'Saving…' : '✅ Save & Submit'}
                </button>
                <button onClick={() => { setSOLines([]); setShowNewSO(false) }}
                  className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-gray-50">Cancel</button>
              </div>
            </div>
          )}

          {/* Orders list */}
          <div className="space-y-2">
            {orders.map(o => (
              <div key={o.id} className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer"
                  onClick={() => setExpandedSO(expandedSO === o.id ? null : o.id)}>
                  <div>
                    <p className="font-semibold text-gray-800 text-sm">{o.so_number}</p>
                    <p className="text-xs text-gray-500">{o.buyer} · Delivery: {o.delivery_date || '—'} · {o.source_type || ''}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Progress quick view */}
                    {o.lines.length > 0 && (() => {
                      const total = o.lines.reduce((s, l) => s + l.qty, 0)
                      const disp = o.lines.reduce((s, l) => s + l.dispatch_qty, 0)
                      const pct = total > 0 ? Math.round(disp / total * 100) : 0
                      return <span className="text-xs text-gray-400">{pct}% dispatched</span>
                    })()}
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(o.status)}`}>{o.status}</span>
                    <select value={o.status}
                      onChange={e => updateSOMut.mutate({ id: o.id, data: { status: e.target.value } })}
                      onClick={e => e.stopPropagation()}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {SO_STATUSES.map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expandedSO === o.id ? '▲' : '▼'}</span>
                  </div>
                </div>

                {expandedSO === o.id && (
                  <div className="border-t border-gray-50 px-4 pb-4">
                    <div className="flex gap-4 text-xs text-gray-400 mt-2 mb-3">
                      {o.ref_demand && <span>Demand: <span className="text-blue-600 font-medium">{o.ref_demand}</span></span>}
                      {o.dispatch_date && <span>Dispatch: {o.dispatch_date}</span>}
{o.ref_number && <span>Ref: <span className="font-medium text-gray-600">{o.ref_number}</span></span>}
                      {o.warehouse && <span>WH: {o.warehouse}</span>}
                      {o.payment_terms && <span>Terms: {o.payment_terms}</span>}
                      {o.sales_team && <span>Team: {o.sales_team}</span>}
                    </div>
                    <table className="w-full text-xs mt-1">
                      <thead>
                        <tr className="text-gray-400 uppercase">
                          <th className="text-left py-1">SKU</th>
                          <th className="text-left py-1">Name</th>
                          <th className="text-right py-1">Ordered</th>
                          <th className="text-right py-1">Rate ₹</th>
                          <th className="text-right py-1">Produced</th>
                          <th className="text-right py-1">Dispatched</th>
                          <th className="text-right py-1">Pending</th>
                          <th className="text-left py-1 pl-2">Remarks</th>
                          <th className="py-1" />
                        </tr>
                      </thead>
                      <tbody>
                        {o.lines.map(l => (
                          editingLineId === l.id ? (
                            <tr key={l.id} className="border-t border-blue-100 bg-blue-50/40">
                              <td className="py-1.5 font-mono font-medium text-gray-700">{l.sku}</td>
                              <td className="py-1.5 text-gray-600">{l.sku_name}</td>
                              <td className="py-1.5 text-right">
                                <input type="number" value={editLineForm.qty}
                                  onChange={e => setEditLineForm(f => ({ ...f, qty: +e.target.value }))}
                                  className="w-16 border border-gray-300 rounded px-1 py-0.5 text-xs text-right" />
                              </td>
                              <td className="py-1.5 text-right">
                                <input type="number" value={editLineForm.rate}
                                  onChange={e => setEditLineForm(f => ({ ...f, rate: +e.target.value }))}
                                  className="w-16 border border-gray-300 rounded px-1 py-0.5 text-xs text-right" />
                              </td>
                              <td className="py-1.5 text-right">
                                <input type="number" value={editLineForm.produced_qty}
                                  onChange={e => setEditLineForm(f => ({ ...f, produced_qty: +e.target.value }))}
                                  className="w-16 border border-gray-300 rounded px-1 py-0.5 text-xs text-right bg-blue-50" />
                              </td>
                              <td className="py-1.5 text-right">
                                <input type="number" value={editLineForm.dispatch_qty}
                                  onChange={e => setEditLineForm(f => ({ ...f, dispatch_qty: +e.target.value }))}
                                  className="w-16 border border-gray-300 rounded px-1 py-0.5 text-xs text-right bg-green-50" />
                              </td>
                              <td className="py-1.5 text-right text-orange-600">{Math.max(0, editLineForm.qty - editLineForm.dispatch_qty).toLocaleString()}</td>
                              <td className="py-1.5 pl-2">
                                <input value={editLineForm.remarks}
                                  onChange={e => setEditLineForm(f => ({ ...f, remarks: e.target.value }))}
                                  className="w-full border border-gray-300 rounded px-1 py-0.5 text-xs" />
                              </td>
                              <td className="py-1.5 pl-2">
                                <div className="flex gap-1">
                                  <button onClick={() => updateSOLineMut.mutate({ id: l.id, data: editLineForm })}
                                    className="text-xs bg-green-600 text-white px-2 py-0.5 rounded hover:bg-green-700">Save</button>
                                  <button onClick={() => setEditingLineId(null)}
                                    className="text-xs text-gray-400 hover:text-gray-600 px-1">✕</button>
                                </div>
                              </td>
                            </tr>
                          ) : (
                            <tr key={l.id} className="border-t border-gray-50 hover:bg-gray-50/50">
                              <td className="py-1.5 font-mono font-medium text-gray-700">{l.sku}</td>
                              <td className="py-1.5 text-gray-600">{l.sku_name}</td>
                              <td className="py-1.5 text-right text-gray-700">{l.qty.toLocaleString()}</td>
                              <td className="py-1.5 text-right text-gray-600">{l.rate > 0 ? `₹${l.rate.toLocaleString()}` : '—'}</td>
                              <td className="py-1.5 text-right text-blue-600">{l.produced_qty.toLocaleString()}</td>
                              <td className="py-1.5 text-right text-green-600">{l.dispatch_qty.toLocaleString()}</td>
                              <td className="py-1.5 text-right text-orange-600">{Math.max(0, l.qty - l.dispatch_qty).toLocaleString()}</td>
                              <td className="py-1.5 pl-2 text-gray-400 max-w-[120px] truncate">{l.remarks || '—'}</td>
                              <td className="py-1.5 pl-2">
                                <button onClick={() => { setEditingLineId(l.id); setEditLineForm({ qty: l.qty, rate: l.rate || 0, delivery_date: l.delivery_date || '', remarks: l.remarks || '', produced_qty: l.produced_qty || 0, dispatch_qty: l.dispatch_qty || 0, received_qty: l.received_qty || 0 }) }}
                                  className="text-xs text-blue-500 hover:text-blue-700 hover:underline">Edit</button>
                              </td>
                            </tr>
                          )
                        ))}
                      </tbody>
                    </table>
                    {/* SO value summary */}
                    {o.lines.length > 0 && (() => {
                      const totalValue = o.lines.reduce((s, l) => s + l.qty * (l.rate || 0), 0)
                      const totalQty = o.lines.reduce((s, l) => s + l.qty, 0)
                      return totalValue > 0 ? (
                        <div className="mt-2 flex gap-4 text-xs text-gray-500">
                          <span>Total Qty: <strong>{totalQty.toLocaleString()}</strong></span>
                          <span>Order Value: <strong className="text-gray-700">₹{totalValue.toLocaleString()}</strong></span>
                        </div>
                      ) : null
                    })()}
                  </div>
                )}
              </div>
            ))}
            {orders.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No sales orders found.</p>}
          </div>
        </div>
      )}

      {/* ── Master Tracker ── */}
      {tab === 'tracker' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-x-auto">
            <div className="px-4 py-3 border-b border-gray-50 flex items-center justify-between">
              <h3 className="font-semibold text-gray-700 text-sm">Master SO Tracker</h3>
              <span className="text-xs text-gray-400">{allOrders.length} orders</span>
            </div>
            <table className="w-full text-sm min-w-[900px]">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>
                  {['SO #', 'Buyer', 'Source', 'Delivery', 'Total Qty', 'Produced', 'Dispatched', 'Pending', 'Value ₹', 'Progress', 'Status'].map(h => (
                    <th key={h} className="text-left px-3 py-2">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {allOrders.map(o => {
                  const totalQty = o.lines.reduce((s, l) => s + l.qty, 0)
                  const produced = o.lines.reduce((s, l) => s + l.produced_qty, 0)
                  const dispatched = o.lines.reduce((s, l) => s + l.dispatch_qty, 0)
                  const pending = Math.max(0, totalQty - dispatched)
                  const value = o.lines.reduce((s, l) => s + l.qty * (l.rate || 0), 0)
                  const pct = totalQty > 0 ? Math.round(dispatched / totalQty * 100) : 0
                  const isOverdue = o.delivery_date && new Date(o.delivery_date) < now && !['Closed', 'Cancelled'].includes(o.status)
                  return (
                    <tr key={o.id} className={`border-t border-gray-50 hover:bg-gray-50/50 ${isOverdue ? 'bg-red-50/30' : ''}`}>
                      <td className="px-3 py-2 font-medium text-gray-700 font-mono">{o.so_number}</td>
                      <td className="px-3 py-2 text-gray-600">{o.buyer || '—'}</td>
                      <td className="px-3 py-2 text-gray-500 text-xs">{o.source_type || '—'}</td>
                      <td className={`px-3 py-2 text-xs font-medium ${isOverdue ? 'text-red-600' : 'text-gray-600'}`}>
                        {o.delivery_date || '—'}
                        {isOverdue && <span className="ml-1 text-red-500">⚠</span>}
                      </td>
                      <td className="px-3 py-2 text-gray-700">{totalQty.toLocaleString()}</td>
                      <td className="px-3 py-2 text-blue-600">{produced.toLocaleString()}</td>
                      <td className="px-3 py-2 text-green-600">{dispatched.toLocaleString()}</td>
                      <td className="px-3 py-2 text-orange-600 font-medium">{pending.toLocaleString()}</td>
                      <td className="px-3 py-2 text-gray-700">{value > 0 ? `₹${value.toLocaleString()}` : '—'}</td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-200 rounded-full h-1.5">
                            <div className="h-1.5 rounded-full bg-green-500" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="text-xs text-gray-500">{pct}%</span>
                        </div>
                      </td>
                      <td className="px-3 py-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(o.status)}`}>{o.status}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {allOrders.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No orders yet.</p>}
          </div>

          {/* Summary cards */}
          {allOrders.length > 0 && (() => {
            const totalValue = allOrders.flatMap(o => o.lines).reduce((s, l) => s + l.qty * (l.rate || 0), 0)
            const totalQty = allOrders.flatMap(o => o.lines).reduce((s, l) => s + l.qty, 0)
            const totalDisp = allOrders.flatMap(o => o.lines).reduce((s, l) => s + l.dispatch_qty, 0)
            const overdue = allOrders.filter(o => o.delivery_date && new Date(o.delivery_date) < now && !['Closed', 'Cancelled'].includes(o.status)).length
            return (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'TOTAL ORDER VALUE', value: `₹${totalValue.toLocaleString()}`, color: 'text-gray-700' },
                  { label: 'TOTAL QTY ORDERED', value: totalQty.toLocaleString(), color: 'text-blue-600' },
                  { label: 'TOTAL DISPATCHED', value: totalDisp.toLocaleString(), color: 'text-green-600' },
                  { label: 'OVERDUE SOs', value: overdue, color: 'text-red-600' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                    <p className={`text-xl font-bold ${color}`}>{value}</p>
                    <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
                  </div>
                ))}
              </div>
            )
          })()}
        </div>
      )}

      {/* ── Reports ── */}
      {tab === 'reports' && (
        <div className="space-y-4">
          <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit flex-wrap">
            {([
              ['sku-pending', 'SKU Pending'],
              ['delivery-due', 'Delivery Due'],
              ['buyer', 'Buyer Summary'],
              ['source', 'Source Summary'],
              ['demand-coverage', 'Demand Coverage'],
            ] as [ReportView, string][]).map(([key, label]) => (
              <button key={key} onClick={() => setReportView(key)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${reportView === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                {label}
              </button>
            ))}
          </div>

          {/* SKU Pending */}
          {reportView === 'sku-pending' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="font-semibold text-gray-700 text-sm">SKU Pending Report</p>
                <p className="text-xs text-gray-400">Aggregated across all open sales orders</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>
                    {['SKU Code', 'SKU Name', 'Total Ordered', 'Produced', 'Dispatched', 'Pending', 'Fill %'].map(h => (
                      <th key={h} className="text-left px-4 py-2">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {skuPending.map(row => {
                    const pending = Math.max(0, row.ordered - row.dispatched)
                    const pct = row.ordered > 0 ? Math.round(row.dispatched / row.ordered * 100) : 0
                    return (
                      <tr key={row.sku} className="border-t border-gray-50 hover:bg-gray-50/50">
                        <td className="px-4 py-2 font-mono font-medium text-gray-700">{row.sku}</td>
                        <td className="px-4 py-2 text-gray-600">{row.sku_name}</td>
                        <td className="px-4 py-2 text-gray-700">{row.ordered.toLocaleString()}</td>
                        <td className="px-4 py-2 text-blue-600">{row.produced.toLocaleString()}</td>
                        <td className="px-4 py-2 text-green-600">{row.dispatched.toLocaleString()}</td>
                        <td className={`px-4 py-2 font-semibold ${pending > 0 ? 'text-orange-600' : 'text-green-600'}`}>{pending.toLocaleString()}</td>
                        <td className="px-4 py-2">
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-gray-200 rounded-full h-1.5">
                              <div className="h-1.5 rounded-full bg-green-500" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-xs text-gray-500">{pct}%</span>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {skuPending.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No open orders with SKU lines found.</p>}
            </div>
          )}

          {/* Delivery Due */}
          {reportView === 'delivery-due' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="font-semibold text-gray-700 text-sm">Delivery Due Report</p>
                <p className="text-xs text-gray-400">Open orders sorted by delivery date</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['SO #', 'Buyer', 'Delivery Date', 'Days Left', 'Total Qty', 'Dispatched', 'Pending', 'Status'].map(h => (
                    <th key={h} className="text-left px-4 py-2">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {deliveryDue.map(o => {
                    const isOverdue = o.daysLeft < 0
                    const isUrgent = o.daysLeft >= 0 && o.daysLeft <= 7
                    return (
                      <tr key={o.id} className={`border-t border-gray-50 ${isOverdue ? 'bg-red-50/40' : isUrgent ? 'bg-yellow-50/40' : ''}`}>
                        <td className="px-4 py-2 font-mono font-medium text-gray-700">{o.so_number}</td>
                        <td className="px-4 py-2 text-gray-600">{o.buyer || '—'}</td>
                        <td className="px-4 py-2 text-gray-600">{o.delivery_date}</td>
                        <td className={`px-4 py-2 font-semibold ${isOverdue ? 'text-red-600' : isUrgent ? 'text-yellow-600' : 'text-green-600'}`}>
                          {isOverdue ? `${Math.abs(o.daysLeft)}d overdue` : `${o.daysLeft}d left`}
                        </td>
                        <td className="px-4 py-2 text-gray-700">{o.totalQty.toLocaleString()}</td>
                        <td className="px-4 py-2 text-green-600">{o.dispatched.toLocaleString()}</td>
                        <td className="px-4 py-2 text-orange-600 font-medium">{(o.totalQty - o.dispatched).toLocaleString()}</td>
                        <td className="px-4 py-2">
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(o.status)}`}>{o.status}</span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {deliveryDue.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No open orders with delivery dates.</p>}
            </div>
          )}

          {/* Buyer Summary */}
          {reportView === 'buyer' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="font-semibold text-gray-700 text-sm">Buyer Summary Report</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['Buyer', 'SOs', 'Total Qty', 'Dispatched', 'Fill %', 'Order Value ₹'].map(h => (
                    <th key={h} className="text-left px-4 py-2">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {Object.values(buyerSummary).sort((a, b) => b.total_value - a.total_value).map(row => {
                    const pct = row.total_qty > 0 ? Math.round(row.dispatched_qty / row.total_qty * 100) : 0
                    return (
                      <tr key={row.buyer} className="border-t border-gray-50 hover:bg-gray-50/50">
                        <td className="px-4 py-2 font-medium text-gray-700">{row.buyer}</td>
                        <td className="px-4 py-2 text-gray-600">{row.so_count}</td>
                        <td className="px-4 py-2 text-gray-700">{row.total_qty.toLocaleString()}</td>
                        <td className="px-4 py-2 text-green-600">{row.dispatched_qty.toLocaleString()}</td>
                        <td className="px-4 py-2">
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-gray-200 rounded-full h-1.5">
                              <div className="h-1.5 rounded-full bg-blue-500" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-xs text-gray-500">{pct}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-2 font-medium text-gray-700">{row.total_value > 0 ? `₹${row.total_value.toLocaleString()}` : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {Object.keys(buyerSummary).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No order data available.</p>}
            </div>
          )}

          {/* Source Summary */}
          {reportView === 'source' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="font-semibold text-gray-700 text-sm">Source Summary Report</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['Source Type', 'SO Count', 'Total Qty', 'Order Value ₹'].map(h => (
                    <th key={h} className="text-left px-4 py-2">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {Object.values(sourceSummary).sort((a, b) => b.so_count - a.so_count).map(row => (
                    <tr key={row.source} className="border-t border-gray-50 hover:bg-gray-50/50">
                      <td className="px-4 py-2 font-medium text-gray-700">{row.source}</td>
                      <td className="px-4 py-2 text-gray-600">{row.so_count}</td>
                      <td className="px-4 py-2 text-gray-700">{row.total_qty.toLocaleString()}</td>
                      <td className="px-4 py-2 font-medium text-gray-700">{row.total_value > 0 ? `₹${row.total_value.toLocaleString()}` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {Object.keys(sourceSummary).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No order data available.</p>}
            </div>
          )}

          {/* Demand Coverage */}
          {reportView === 'demand-coverage' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="font-semibold text-gray-700 text-sm">Demand vs SO Coverage</p>
                <p className="text-xs text-gray-400">How well each demand is covered by created SOs</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['Demand #', 'Buyer', 'Date', 'Status', 'Demand Qty', 'SO Qty', 'Coverage', 'Linked SOs'].map(h => (
                    <th key={h} className="text-left px-4 py-2">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {demandCoverage.map(d => (
                    <tr key={d.id} className="border-t border-gray-50 hover:bg-gray-50/50">
                      <td className="px-4 py-2 font-mono font-medium text-gray-700">{d.demand_number}</td>
                      <td className="px-4 py-2 text-gray-600">{d.buyer || '—'}</td>
                      <td className="px-4 py-2 text-gray-400 text-xs">{d.demand_date}</td>
                      <td className="px-4 py-2"><span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(d.status)}`}>{d.status}</span></td>
                      <td className="px-4 py-2 text-gray-700">{d.demandQty.toLocaleString()}</td>
                      <td className="px-4 py-2 text-blue-600">{d.soQty.toLocaleString()}</td>
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-gray-200 rounded-full h-1.5">
                            <div className={`h-1.5 rounded-full ${d.pct >= 100 ? 'bg-green-500' : d.pct > 0 ? 'bg-yellow-400' : 'bg-gray-300'}`} style={{ width: `${Math.min(100, d.pct)}%` }} />
                          </div>
                          <span className={`text-xs font-medium ${d.pct >= 100 ? 'text-green-600' : d.pct > 0 ? 'text-yellow-600' : 'text-gray-400'}`}>{d.pct}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-2 text-gray-500">{d.linkedCount} SO{d.linkedCount !== 1 ? 's' : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {demandCoverage.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No demands found.</p>}
            </div>
          )}
        </div>
      )}

      {/* ── Settings ── */}
      {tab === 'settings' && (
        <div className="space-y-4">
          <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
            {([['buyers', '🛒 Buyers'], ['warehouses', '🏭 Warehouses'], ['teams', '👥 Sales Teams']] as [SettingsTab, string][]).map(([key, label]) => (
              <button key={key} onClick={() => setSettingsTab(key)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${settingsTab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                {label}
              </button>
            ))}
          </div>

          {settingsTab === 'buyers' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-3">
              <h3 className="font-semibold text-gray-700 text-sm">Buyers</h3>
              <p className="text-xs text-gray-400">These appear as suggestions in buyer fields across the SO module.</p>
              <div className="flex gap-2">
                <input value={newSettingVal} onChange={e => setNewSettingVal(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') saveSetting('erp_buyers', customBuyers, setCustomBuyers) }}
                  placeholder="Add buyer name…"
                  className="flex-1 border border-gray-200 rounded px-3 py-1.5 text-sm" />
                <button onClick={() => saveSetting('erp_buyers', customBuyers, setCustomBuyers)}
                  className="px-4 py-1.5 bg-[#002B5B] text-white rounded text-sm font-medium hover:bg-blue-800">Add</button>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {customBuyers.map(b => (
                  <div key={b} className="flex items-center gap-1 bg-gray-100 rounded-full px-3 py-1 text-sm">
                    <span>{b}</span>
                    <button onClick={() => deleteSetting('erp_buyers', b, customBuyers, setCustomBuyers)}
                      className="text-gray-400 hover:text-red-500 ml-1 text-xs">✕</button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {settingsTab === 'warehouses' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-3">
              <h3 className="font-semibold text-gray-700 text-sm">Warehouses</h3>
              <p className="text-xs text-gray-400">These appear as suggestions in warehouse fields across the SO module.</p>
              <div className="flex gap-2">
                <input value={newSettingVal} onChange={e => setNewSettingVal(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') saveSetting('erp_warehouses', customWarehouses, setCustomWarehouses) }}
                  placeholder="Add warehouse name…"
                  className="flex-1 border border-gray-200 rounded px-3 py-1.5 text-sm" />
                <button onClick={() => saveSetting('erp_warehouses', customWarehouses, setCustomWarehouses)}
                  className="px-4 py-1.5 bg-[#002B5B] text-white rounded text-sm font-medium hover:bg-blue-800">Add</button>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {customWarehouses.map(w => (
                  <div key={w} className="flex items-center gap-1 bg-gray-100 rounded-full px-3 py-1 text-sm">
                    <span>{w}</span>
                    <button onClick={() => deleteSetting('erp_warehouses', w, customWarehouses, setCustomWarehouses)}
                      className="text-gray-400 hover:text-red-500 ml-1 text-xs">✕</button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {settingsTab === 'teams' && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-3">
              <h3 className="font-semibold text-gray-700 text-sm">Sales Teams</h3>
              <p className="text-xs text-gray-400">These appear as suggestions in sales team fields across the SO module.</p>
              <div className="flex gap-2">
                <input value={newSettingVal} onChange={e => setNewSettingVal(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') saveSetting('erp_teams', customTeams, setCustomTeams) }}
                  placeholder="Add team name…"
                  className="flex-1 border border-gray-200 rounded px-3 py-1.5 text-sm" />
                <button onClick={() => saveSetting('erp_teams', customTeams, setCustomTeams)}
                  className="px-4 py-1.5 bg-[#002B5B] text-white rounded text-sm font-medium hover:bg-blue-800">Add</button>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {customTeams.map(t => (
                  <div key={t} className="flex items-center gap-1 bg-gray-100 rounded-full px-3 py-1 text-sm">
                    <span>{t}</span>
                    <button onClick={() => deleteSetting('erp_teams', t, customTeams, setCustomTeams)}
                      className="text-gray-400 hover:text-red-500 ml-1 text-xs">✕</button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
