import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'demands' | 'orders'

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
  lines: SOLine[]
}
interface SOLine {
  id: number; sku: string; sku_name: string; qty: number
  produced_qty: number; dispatch_qty: number; unit: string
  rate: number; delivery_date: string; remarks: string
}

const SOURCES = ['Sales Team', 'Forecasting System', 'Buyer Indent', 'Marketing Team']
const PRIORITIES = ['Normal', 'High', 'Urgent']
const DEMAND_STATUSES = ['Draft', 'Submitted', 'Approved', 'In Production', 'Closed', 'Cancelled']
const SO_STATUSES = ['Draft', 'Submitted', 'Approved', 'In Production', 'Partially Dispatched', 'Dispatched', 'Closed', 'Cancelled']
const SOURCE_TYPES = ['Sales Team Demand', 'Buyer PO', 'Offline Order']
const UOM_OPTIONS = ['PCS', 'SET', 'PAIR', 'BOX', 'MTR', 'KG']

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
  const [editLineForm, setEditLineForm] = useState({ qty: 0, rate: 0, delivery_date: '', remarks: '' })

  // Demand form
  const [dForm, setDForm] = useState({ demand_source: 'Sales Team', buyer: '', priority: 'Normal', notes: '' })
  const [dLines, setDLines] = useState<{ sku: string; sku_name: string; demand_qty: number }[]>([])

  // SO form
  const [soForm, setSOForm] = useState({
    buyer: '', warehouse: '', sales_team: '', source_type: 'Sales Team Demand',
    ref_demand: '', delivery_date: '', payment_terms: '', notes: ''
  })
  const [soLines, setSOLines] = useState<{ sku: string; sku_name: string; qty: number; unit: string; rate: number; remarks: string }[]>([])
  const [fetchingDemand, setFetchingDemand] = useState(false)

  const { data: demands = [] } = useQuery<Demand[]>({
    queryKey: ['demands', filterStatus],
    queryFn: () => api.get('/sales/demands' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data)
  })
  const { data: orders = [] } = useQuery<SalesOrder[]>({
    queryKey: ['sales-orders', filterStatus],
    queryFn: () => api.get('/sales/orders' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data)
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
      qc.invalidateQueries({ queryKey: ['sales-orders'] })
      setShowNewSO(false); setSOLines([])
      setSOForm({ buyer: '', warehouse: '', sales_team: '', source_type: 'Sales Team Demand', ref_demand: '', delivery_date: '', payment_terms: '', notes: '' })
    }
  })
  const updateDemandMut = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/sales/demands/${id}/status`, { status }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['demands'] })
  })
  const updateSOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/sales/orders/${id}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['sales-orders'] })
  })
  const updateSOLineMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/sales/orders/lines/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['sales-orders'] }); setEditingLineId(null) }
  })

  function addDLine() { setDLines(l => [...l, { sku: '', sku_name: '', demand_qty: 0 }]) }
  function addSOLine() { setSOLines(l => [...l, { sku: '', sku_name: '', qty: 0, unit: 'PCS', rate: 0, remarks: '' }]) }

  async function fetchDemandLines() {
    if (!soForm.ref_demand.trim()) return
    setFetchingDemand(true)
    try {
      const { data } = await api.get(`/sales/demands/by-number/${encodeURIComponent(soForm.ref_demand.trim())}`)
      const lines = (data.lines || []).map((l: DemandLine) => ({
        sku: l.sku, sku_name: l.sku_name, qty: l.demand_qty, unit: 'PCS', rate: 0, remarks: ''
      }))
      setSOLines(lines)
      if (data.buyer && !soForm.buyer) setSOForm(f => ({ ...f, buyer: data.buyer }))
    } catch {
      // demand not found — lines stay as-is
    } finally { setFetchingDemand(false) }
  }

  const openDemands = demands.filter(d => !['Closed', 'Cancelled'].includes(d.status)).length
  const openOrders = orders.filter(o => !['Closed', 'Cancelled'].includes(o.status)).length
  const urgentDemands = demands.filter(d => d.priority === 'Urgent' && !['Closed', 'Cancelled'].includes(d.status)).length
  const TABS: [Tab, string][] = [['dashboard', '📊 Dashboard'], ['demands', '📋 Demands'], ['orders', '🧾 Sales Orders']]

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800">Sales Orders & Demand</h1>
          <p className="text-sm text-gray-500">Manage demands, sales orders and tracking</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
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
                    <p className="text-xs text-gray-400">{o.buyer} · {o.delivery_date}</p>
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
                  <input value={dForm.buyer} onChange={e => setDForm(f => ({ ...f, buyer: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
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

                {/* Parent SKU size-wise picker */}
                <ParentSkuPicker onAddLines={lines => setDLines(l => [...l, ...lines])} />

                {/* Table header */}
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

              {/* Header fields */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="text-xs text-gray-500">Buyer</label>
                  <input value={soForm.buyer} onChange={e => setSOForm(f => ({ ...f, buyer: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Warehouse</label>
                  <input value={soForm.warehouse} onChange={e => setSOForm(f => ({ ...f, warehouse: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Sales Team</label>
                  <input value={soForm.sales_team} onChange={e => setSOForm(f => ({ ...f, sales_team: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
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

              {/* SO Lines */}
              <div>
                <div className="flex justify-between mb-2">
                  <p className="text-sm font-medium text-gray-600">SKU Lines</p>
                  <button onClick={addSOLine} className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                </div>

                {soLines.length > 0 && (
                  <div className="flex gap-2 mb-1 px-1 text-xs text-gray-400">
                    <span className="flex-1">SKU Code</span>
                    <span className="flex-1">SKU Name</span>
                    <span className="w-20">Qty</span>
                    <span className="w-16">Rate ₹</span>
                    <span className="w-18">UOM</span>
                    <span className="w-28">Remarks</span>
                    <span className="w-6" />
                  </div>
                )}

                {soLines.map((ln, i) => (
                  <div key={i} className="flex gap-2 mb-2 items-center">
                    <SkuPicker
                      value={ln.sku}
                      onChange={(sku, skuName, uom) => setSOLines(l => l.map((x, j) => j === i ? { ...x, sku, sku_name: skuName, unit: uom } : x))}
                    />
                    <input placeholder="SKU Name" value={ln.sku_name}
                      onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, sku_name: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm flex-1" />
                    <input type="number" placeholder="Qty" value={ln.qty}
                      onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, qty: +e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-20" />
                    <input type="number" placeholder="Rate" value={ln.rate}
                      onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-20" />
                    <select value={ln.unit}
                      onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, unit: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-18">
                      {UOM_OPTIONS.map(u => <option key={u}>{u}</option>)}
                    </select>
                    <input placeholder="Remarks" value={ln.remarks}
                      onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, remarks: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-28" />
                    <button onClick={() => setSOLines(l => l.filter((_, j) => j !== i))}
                      className="text-red-400 hover:text-red-600 text-sm px-1">✕</button>
                  </div>
                ))}
                {soLines.length === 0 && (
                  <p className="text-xs text-gray-400">Fetch from a demand or click "+ Add Line".</p>
                )}
              </div>

              <div className="flex gap-2">
                <button onClick={() => createSOMut.mutate({ ...soForm, lines: soLines })}
                  disabled={createSOMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                  {createSOMut.isPending ? 'Saving…' : 'Create Sales Order'}
                </button>
                <button onClick={() => setShowNewSO(false)}
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
                    <p className="text-xs text-gray-500">{o.buyer} · Delivery: {o.delivery_date || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
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
                    <table className="w-full text-xs mt-3">
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
                              <td className="py-1.5 text-right text-blue-600">{l.produced_qty.toLocaleString()}</td>
                              <td className="py-1.5 text-right text-green-600">{l.dispatch_qty.toLocaleString()}</td>
                              <td className="py-1.5 text-right text-orange-600">{Math.max(0, editLineForm.qty - l.dispatch_qty).toLocaleString()}</td>
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
                                <button onClick={() => { setEditingLineId(l.id); setEditLineForm({ qty: l.qty, rate: l.rate || 0, delivery_date: l.delivery_date || '', remarks: l.remarks || '' }) }}
                                  className="text-xs text-blue-500 hover:text-blue-700 hover:underline">Edit</button>
                              </td>
                            </tr>
                          )
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
            {orders.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No sales orders found.</p>}
          </div>
        </div>
      )}
    </div>
  )
}
